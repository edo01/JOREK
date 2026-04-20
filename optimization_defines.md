# optimization_defines.h

Single place to toggle GPU build flags and particle ordering strategy.
Included via `#include "optimization_defines.h"` in both Fortran and HIP C++ sources.
After editing, recompile: `make cleanall && make`

## USE_GPU

0 = CPU-only build, 1 = enable GPU (HIP) code paths in Fortran and HIP C++ kernel.

## GPU_DEBUG

0 = disabled, 1 = verbose GPU debug output (host + kernel printf, hipDeviceSynchronize after each launch).
Only meaningful when USE_GPU = 1.

## ORDERING_TYPE

Particle ordering strategy before the coupling scheme loop:

- 0 — no ordering (sort_particles is never called)
- 1 — global sort (particle_global_sort)
- 2 — SIMD-lane sort (particle_simd_lane_sort) [default]
