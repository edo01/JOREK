# optimization_defines.h

Single place to toggle GPU build flags and particle ordering strategy.
Included via `#include "optimization_defines.h"` in both Fortran and HIP C++ sources.
After editing, recompile: `make cleanall && make`

## USE_GPU

0 = CPU-only build, 1 = enable GPU (HIP) code paths in Fortran and HIP C++ kernel.

## BLOCK_SIZE

Number of GPU threads per block for the RE evolution kernel.
Only meaningful when USE_GPU = 1.

## GPU_DEBUG

0 = disabled, 1 = verbose GPU debug output (host + kernel printf, hipDeviceSynchronize after each launch).
Only meaningful when USE_GPU = 1.

## ORDERING_TYPE

Particle ordering strategy before the coupling scheme loop:

- 0 — no ordering (sort_particles is never called)
- 1 — global sort (particle_global_sort)
- 2 — SIMD-lane sort (particle_simd_lane_sort)

## STEPS_PER_BATCH

Number of kinetic time steps executed per GPU kernel launch (batch size).
The total kinetic steps (`nstep_part_adj`) are split into batches of this size;
the last batch may be smaller. Between batches, particles are re-sorted by element
index to improve memory locality.

- 0 — batching disabled; all kinetic steps run in a single kernel launch with no intermediate sort
- N > 0 — one kernel launch every N kinetic steps, with a particle re-sort before each launch

Only meaningful when USE_GPU = 1 and ORDERING_TYPE > 0.

## LUT_VALUES_DELTAS

Enable a look-up table (LUT) that caches the `nl_values` / `nl_deltas` field data of
a small set of mesh elements in shared memory, so particles sharing those elements
read from the on-chip cache instead of global memory. Both field evaluations per
kinetic step use it: the PROJ phase (`calc_B_only`, psi only) and the PUSH phase
(`calc_EBpsiU`, psi + U).

The cache is built **once per kernel launch** (i.e. once per batch). Because particles
are counting-sorted by element index before each batch, a block holds very few distinct
elements at the first step; the build extracts those distinct "base" elements with a
cheap parallel run-length scan of the sorted block (it depends on that sort — the LUT is
only meaningful with batch sorting enabled). With `LUT_NEIGHBOR_PRELOAD`, any leftover
slots are filled with the base elements' mesh neighbours, to capture the particles that
drift into neighbouring elements by the second step of the batch.

- 0 — disabled
- 1 — enabled

Only meaningful when USE_GPU = 1.

## LUT_N_SLOTS

Number of LUT cache slots (distinct elements cached) per GPU thread block. This is the
single capacity knob: base (step-0) elements fill slots first, then neighbour preload
uses any leftover slots. More slots cache more elements (more step-1 neighbour coverage)
at the cost of shared memory; if the distinct elements exceed the slots, the overflow
falls back to direct global-memory interpolation (still correct, just not cached).
The shared-memory cost is `LUT_N_SLOTS * 512 * N_TOR` bytes; a compile-time
`static_assert` keeps it within the 2-blocks-per-CU occupancy budget (reduce
LUT_N_SLOTS or N_TOR if it fires). Only meaningful when LUT_VALUES_DELTAS = 1.

## LUT_NEIGHBOR_PRELOAD

0 = cache only the distinct base (step-0) elements; 1 = additionally fill any leftover
slots with the deduplicated mesh neighbours of the base elements. Preloading neighbours
captures step-1 drift (particles move mostly into neighbouring elements) so the second
batch step still hits the cache. Boundary edges (neighbour = 0) are skipped. Base
elements are always inserted first and never evicted.
Only meaningful when LUT_VALUES_DELTAS = 1.

## LUT_DEBUG

0 = disabled, 1 = instrument the LUT with hit/miss counters printed per batch (counts
both PROJ and PUSH lookups). Only meaningful when LUT_VALUES_DELTAS = 1.

## NODES_FIRST

Controls the memory layout of node-list arrays (`nl_x`, `nl_values`, `nl_deltas`) passed to the GPU.

- 0 — `n_nodes` is the **last** (slowest-changing) dimension: `(…, n_nodes)`. Matches Fortran column-major convention.
- 1 — `n_nodes` is the **first** (fastest-changing) dimension: `(n_nodes, …)`. Fortran packs accordingly.

## ELEMENTS_FIRST

Controls the memory layout of element-list arrays (`el_vertex`, `el_neighbours`, `el_size`) passed to the GPU.

- 0 — `n_elements` is the **last** (slowest-changing) dimension: `(…, n_elements)`.
- 1 — `n_elements` is the **first** (fastest-changing) dimension: `(n_elements, …)`.

## FB_ELEMENTS_FIRST

Controls the memory layout of the `feedback_rhs` buffer exchanged between Fortran and the GPU kernel.

- 0 — `n_elements` is the **last** (slowest-changing) dimension: `(NDEG, NV, n_elements, N_TOR, NVAR)`. Matches Fortran column-major convention.
- 1 — `n_elements` is the **first** (fastest-changing) dimension: `(n_elements, NDEG, NV, N_TOR, NVAR)`. Improves GPU warp coalescing.
