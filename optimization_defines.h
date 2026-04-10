! =============================================================================
! optimization_defines.h -- Particle ordering strategy selector
!
! Set ORDERING_TYPE to one of:
!   0  -- no ordering (sort_particles is never called)
!   1  -- global sort (particle_global_sort)
!   2  -- SIMD-lane sort (particle_simd_lane_sort)   [default]
!
! After editing this file, recompile the affected sources:
!   make particles/mod_particle_evolution.o   (or simply: make)
! =============================================================================

#define ORDERING_TYPE 1
