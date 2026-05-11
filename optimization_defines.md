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

Enable a look-up table (LUT) that caches interpolated field values and deltas
to avoid redundant memory reads for particles sharing the same element.

- 0 — disabled
- 1 — enabled

Only meaningful when USE_GPU = 1.

## LUT_N_SLOTS

Number of LUT cache slots (entries) available per GPU thread block.
Only meaningful when LUT_VALUES_DELTAS = 1.

## LUT_MIN_OCCUPANCY

Minimum number of particles that must map to a LUT slot before it is considered
worth caching. Slots with fewer hits fall back to direct interpolation.
Only meaningful when LUT_VALUES_DELTAS = 1.

## LUT_REFRESH_INTERVAL

Number of kinetic steps between LUT refreshes within a batch.
Only meaningful when LUT_VALUES_DELTAS = 1 and STEPS_PER_BATCH > 0.
