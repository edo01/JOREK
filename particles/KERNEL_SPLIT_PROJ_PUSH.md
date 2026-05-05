# Kernel Split: proj + push with Double-Buffering and HIP Graphs

**File modified:** `jorek/particles/kernel_re_evolution.hip.cpp`
**Date:** 2026-05-05

---

## Motivation

The original `evolve_REs_kernel` was a monolithic kernel that performed two
logically independent operations per particle per kinetic step:

1. **Projection** — reads the particle's pre-step state (position `x`, momentum
   `pm`, local coordinates `st`, element index `i_elm`), interpolates E/B
   fields at that position, and accumulates three feedback quantities
   (`P_par`, `P_perp`, `j_phi`) to `feedback_rhs` via `atomicAdd`.

2. **Push** — reads the same pre-step state, integrates the particle trajectory
   one kinetic timestep forward using the Volume-Preserving Algorithm (VPA),
   and writes updated `x`, `pm`, `st`, `i_elm` back to global memory in-place.

Because both phases read the same input and write to entirely disjoint outputs
(projection → `feedback_rhs` only; push → updated particle arrays only), they
are data-independent and can run concurrently on the GPU.

The split also opens the door to double-buffering: push writes the new particle
state into an alternate buffer while projection still reads the old one, so no
read-after-write hazard exists even if the two kernels overlap in time.

---

## Key Correctness Property

In the original kernel, **projection always used the pre-step state** — values
were loaded into registers before the push call, and the push modified only
those local registers (then stored at the very end). The split preserves this
exactly: both new kernels receive the same `_curr` input buffers, which are
never modified by either kernel during the step.

---

## What Changed

### 1. `evolve_proj_kernel` (new, replaces the projection half)

```
Reads:   p_x, p_p, p_st, p_i_elm, p_weight  (all const — particle state unchanged)
         nl_values, nl_deltas, nl_x, el_vertex, el_size  (field data)
Writes:  feedback_rhs  (via atomicAdd — same accumulation as before)
```

- All particle input arrays carry `const` qualifier, making the no-write
  contract explicit and enabling better compiler alias analysis.
- `el_neighbours` is omitted (not needed for projection; only the push uses it
  for element-hopping during particle tracking).
- `tstep_part_adj` is omitted (not used in the projection computation).
- The `__launch_bounds__(BLOCK_SIZE, 2)` hint is preserved so the compiler
  keeps the register budget at the same level as the original kernel.
- The `#if LUT_VALUES_DELTAS` shared-memory LUT block is preserved unchanged
  for when that path is re-enabled.

### 2. `evolve_push_kernel` (new, replaces the push half)

```
Reads:   p_x_in, p_p_in, p_st_in, p_i_elm_in  (current/input buffers, const)
         nl_values, nl_deltas, nl_x, el_vertex, el_neighbours, el_size
Writes:  p_x_out, p_p_out, p_st_out, p_i_elm_out  (alternate/output buffers)
```

- Reads come from the "current" buffers (`_in`); writes go to the "alternate"
  buffers (`_out`). These two sets never alias, so `__restrict__` is correct
  on all pointers.
- `p_weight` is not accepted because weight is never modified by the push.
- `feedback_rhs` is not accepted.
- The output write is **unconditional**: even a lost particle (`i_elm <= 0`,
  for which the push call is skipped) copies its input state to the output
  buffer. This ensures the alternate buffer is always a fully valid copy of
  particle state after each step, regardless of particle status.
- The `#if LUT_VALUES_DELTAS` block is preserved for the push's two
  `calc_EBpsiU` calls inside `volume_preserving_push`.

### 3. `evolve_REs_kernel` (removed)

The original monolithic kernel is deleted entirely. It is no longer called.

### 4. Double-buffer allocation in `launch_evolve_REs`

Three sets of particle buffers now exist simultaneously on the device:

| Name | Role |
|---|---|
| Allocation A (`d_x_orig` …) | Primary `hipMalloc` — holds initial H2D data |
| Allocation B (`d_x_sorted` …) | Sort scratch — used by `sort_particles_by_i_elm_gpu` |
| Allocation C (`d_x_push_alt` …) | Push alternate — written by `evolve_push_kernel` |

`d_x_curr` (the tracking pointer passed to kernels) always points to whichever
of A/B/C currently holds the valid particle state. After each kinetic step the
host swaps `d_x_curr ↔ d_x_push_alt` via `std::swap`. After each sort step
`sort_particles_by_i_elm_gpu` swaps `d_x_curr ↔ d_x_sorted` internally.

The three allocations are always distinct: the sort function never touches C,
and the push kernels never touch B.

**Original handles saved:** immediately after `hipMalloc`, constant pointers
`d_x_orig`, `d_p_orig`, `d_st_orig`, `d_i_elm_orig` (for A) and
`d_x_push_alt_orig` etc. (for C) are saved. The free block always frees via
these original handles, never via the tracking pointers, avoiding any
double-free regardless of how many swaps have occurred.

### 5. Two HIP streams

```cpp
hipStream_t stream_proj, stream_push;
```

`evolve_proj_kernel` is always dispatched on `stream_proj`;
`evolve_push_kernel` on `stream_push`. Since they are independent, the GPU
scheduler can run them concurrently when SM resources allow.

### 6. HIP graphs — two alternating graphs per sort interval

Direct kernel launches cost ~1–2 µs of CPU driver overhead each. With
`nstep_particles` potentially in the thousands, this adds up. HIP graphs
reduce that per-step overhead to ~0.1 µs by replaying a pre-compiled launch
sequence.

**The graph invalidation problem:** HIP graphs capture kernel argument values
(device pointers) at capture time. The pointer-swap after each push step means
the captured addresses would be stale in the next replay. Additionally, the
sort function swaps `d_x_curr ↔ d_x_sorted` at every sort boundary, again
invalidating any previously captured graph.

**Solution: two alternating graphs, re-captured once per sort interval.**

Within a sort interval of N_SORTING steps, the current and alternate buffers
alternate between exactly two pointer configurations:

- **Even step** within the interval: `curr = P`, `alt = Q` → `graph_even`
- **Odd step** within the interval: `curr = Q`, `alt = P` → `graph_odd`

Both graphs are captured once at the start of each sort interval (right after
the sort completes, when pointer values are stable). For the remaining
N_SORTING − 1 steps in the interval, `hipGraphLaunch` replays the appropriate
graph at minimal overhead. At the next sort boundary, both graphs are destroyed
and re-captured.

**Graph structure (fork-join pattern):**

```
stream_proj ──[fork_event]──→ evolve_proj_kernel ──┐
                                                    [join_event]──→ (exit)
stream_push ──[fork_event]──→ evolve_push_kernel ──┘
```

- `fork_event` (recorded on `stream_proj`, waited by `stream_push`) ensures
  both streams start from the same dependency point in the graph.
- `join_event` (recorded on `stream_push`, waited by `stream_proj`) creates a
  single exit node so that `hipStreamSynchronize(stream_proj)` after the graph
  launch is sufficient to know both kernels are complete.
- Both events use `hipEventDisableTiming` to avoid the overhead of hardware
  timestamp collection.

**Capture lambda** (`capture_graph`): a local lambda in `launch_evolve_REs`
encapsulates the capture sequence (BeginCapture → fork → proj launch → push
launch → join → EndCapture → Instantiate) to avoid code duplication between
the even and odd captures.

### 7. Main loop structure

```
for k = 0 .. nstep_particles-1:

    if k % N_SORTING == 0:          // sort step
        sync stream_push, stream_proj
        destroy graphs (if valid)
        sort_particles_by_i_elm_gpu(...)
        interval_step = 0

    if not graph_valid:             // first step of each sort interval
        capture_graph(curr→alt) → graph_even / graph_exec_even
        capture_graph(alt→curr) → graph_odd  / graph_exec_odd
        graph_valid = true

    cur_exec = graph_exec_even if interval_step is even else graph_exec_odd
    hipGraphLaunch(cur_exec, stream_proj)
    hipStreamSynchronize(stream_proj)   // waits for both proj and push

    swap(d_x_curr, d_x_push_alt)       // host tracking swap
    swap(d_p_curr, d_p_push_alt)
    swap(d_st_curr, d_st_push_alt)
    swap(d_i_elm_curr, d_i_elm_push_alt)
    ++interval_step
```

### 8. Post-loop cleanup

After the loop, both streams are synchronized before the D2H copies.
Graphs and streams are destroyed before the `hipFree` block.

The free block uses the saved original allocation handles (A, B, C) rather
than the tracking pointers, guaranteeing each `hipMalloc` is freed exactly
once.

---

## Memory overhead

The only additional GPU memory is allocation C (the push alternate buffer):

| Array | Size |
|---|---|
| `d_x_push_alt` | `3 * num_particles * sizeof(double)` |
| `d_p_push_alt` | `3 * num_particles * sizeof(double)` |
| `d_st_push_alt` | `2 * num_particles * sizeof(double)` |
| `d_i_elm_push_alt` | `num_particles * sizeof(int)` |
| **Total** | `≈ 8 * num_particles * sizeof(double)` (weight not duplicated) |

This is exactly double the particle state memory (minus weight). Field data,
`feedback_rhs`, and sort buffers are unchanged.

---

## Shared memory note

When `LUT_VALUES_DELTAS=0` (the current default in `optimization_defines.h`),
both new kernels are purely register-based. Their individual register footprints
are smaller than the monolithic kernel's, which can improve occupancy.
`__launch_bounds__(BLOCK_SIZE, 2)` is kept on both kernels to communicate the
register budget to the compiler.

When `LUT_VALUES_DELTAS=1`, each kernel calls `lut_build_cooperative`
independently. In the original monolithic kernel the LUT was built once and
shared between projection and push; after the split, two builds are required
per step (one per kernel). This is an accepted regression for that currently-
disabled code path.

---

## Verification

1. **Single-step sanity** (`nstep_particles=1`, large `N_SORTING` to skip
   sorting): compare `h_feedback_rhs` and final particle arrays
   (`x`, `p`, `st`, `i_elm`) against a reference run of the original kernel.
   Expected: bitwise-identical results.

2. **Full jet benchmark** at default settings: compare feedback and particle
   output to a reference. Expected: floating-point match (any difference
   indicates a pointer aliasing or ordering bug).

3. **Graph capture log**: the `printf("[launch_evolve_REs rank %d] recapturing
   graphs at k=%d\n", ...)` line fires at each sort boundary. With
   `N_SORTING=100` and `nstep_particles=1000`, expect 10 recaptures at
   k = 0, 100, 200, …, 900.

4. **Timing**: compare the "nstep_particles loop" printf against the baseline.
   Expected improvement proportional to the overlap between proj and push
   runtimes.
