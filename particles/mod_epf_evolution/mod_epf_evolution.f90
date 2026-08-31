!> Particle evolution for the 'epf' coupling scheme: energetic particles.
!> Called from evolve_particle_group (mod_particle_evolution.f90).
module mod_epf_evolution
    use particle_tracer
    use coupling_variables
    use phys_module, only: use_manual_random_seed, proj_collection_period
    use mod_pcg32_rng, only: pcg32_rng
    !$ use omp_lib

    implicit none
    private
    public :: evolve_epf

    !> The six anisotropic-pressure components and the density, in the order
    !> jorek::epf_var gives them on the other side of the seam.
    integer, parameter :: n_epf_var = 7

    !> Interface only -- the bodies are in C++. One signature, because the two
    !> entry points are one kernel; which one is bound is decided at build time.
    abstract interface
      subroutine jgx_evolve_epf_iface(part_base, n_particles, &
                                      el_base, n_elements,    &
                                      nd_base, n_nodes,       &
                                      interp_base,            &
                                      rhs_data, rhs_ext,      &
                                      proj_idx,               &
                                      mass, time, timestep,   &
                                      nstep, proj_period,     &
                                      phi_search,             &
                                      not_found, nf_R, nf_Z,  & !> debug
                                      bad_i_from, bad_i_to) bind(C)
        use, intrinsic :: iso_c_binding, only: c_double, c_int32_t, c_ptr
        implicit none
        type(c_ptr),        value, intent(in)    :: part_base, el_base, nd_base, interp_base
        integer(c_int32_t), value, intent(in)    :: n_particles, n_elements, n_nodes
        real(c_double),            intent(inout) :: rhs_data(*)
        integer(c_int32_t),        intent(in)    :: rhs_ext(5)
        integer(c_int32_t),        intent(in)    :: proj_idx(*)
        integer(c_int32_t), value, intent(in)    :: nstep, proj_period
        real(c_double),     value, intent(in)    :: mass, time, timestep, phi_search
        integer(c_int32_t),        intent(out)   :: not_found, bad_i_from, bad_i_to
        real(c_double),            intent(out)   :: nf_R, nf_Z
      end subroutine
    end interface

    procedure(jgx_evolve_epf_iface), &
      bind(C, name="jgx_host_epf_evolution_evolve_epf")        :: evolve_epf_host
#ifdef JGX_HAS_DEVICE
    procedure(jgx_evolve_epf_iface), &
      bind(C, name="jgx_host_epf_evolution_evolve_epf_device") :: evolve_epf_device
#endif
contains

  !> Gathers the energetic-particle projections of a particle group and pushes
  !> its particles.
  !>
  !> Facade only -- the body is in C++ (epf_evolution.h next door).
  subroutine evolve_epf(sim, group_num, feedback_rhs, rng, tstep_part_adj, nstep_part_adj)
    use mod_fields_linear, only: jorek_fields_interp_linear
    use mod_find_rz_nearby, only: find_RZ_nearby_phi_search, find_RZ_nearby_report
    use mod_particle_types, only: particle_kinetic_leapfrog
    use, intrinsic :: iso_c_binding, only: c_double, c_int32_t, c_loc

    implicit none
    class(particle_sim), target, intent(inout)                :: sim
    integer, intent(in)                                       :: group_num
    real*8, allocatable, intent(inout)                        :: feedback_rhs(:,:,:,:,:)
    !> Unused: the scheme draws no random numbers. Kept because
    !> evolve_particle_group calls every scheme through the same argument list.
    type(pcg32_rng), dimension(:), allocatable, intent(inout) :: rng
    real*8, intent(in)                                        :: tstep_part_adj
    integer, intent(in)                                       :: nstep_part_adj

    integer(c_int32_t) :: rhs_ext(5), proj_idx(n_epf_var)
    integer(c_int32_t) :: not_found, bad_i_from, bad_i_to
    real(c_double)     :: nf_R, nf_Z
    integer            :: i
    integer            :: iterations !> number of tsteps for which projection quantities are collected
    procedure(jgx_evolve_epf_iface), pointer :: evolve_epf_kernel

#if defined(fullmhd) || STELLARATOR_MODEL
    error stop "evolve_epf: the C++ kernel is reduced MHD only (calc_EBpsiU_reduced)"
#else
    ! One kernel, two entry points: a device build runs on the device, a host
    ! build over the Fortran storage in place. Decided by the build, not by input.
#ifdef JGX_HAS_DEVICE
    evolve_epf_kernel => evolve_epf_device
#else
    evolve_epf_kernel => evolve_epf_host
#endif

    if (nstep_part_adj < proj_collection_period) then
      proj_collection_period = nstep_part_adj
    endif

    do i = 1, 5
      rhs_ext(i) = int(size(feedback_rhs, i), c_int32_t)
    end do

    !> 0-based, in jorek::epf_var order -- the one place the two orders meet.
    proj_idx = int([PI_RR_idx_kin, PI_ZZ_idx_kin, PI_PHIPHI_idx_kin, &
                    PI_RZ_idx_kin, PI_RPHI_idx_kin, PI_ZPHI_idx_kin, &
                    rho_ep_idx_kin] - 1, c_int32_t)

    select type (particles => sim%groups(group_num)%particles)
    type is (particle_kinetic_leapfrog)
      if (size(particles,1) > 0) then

        if (use_manual_random_seed) then
          !$ call omp_set_schedule(omp_sched_static,10)
        else
          !$ call omp_set_schedule(omp_sched_dynamic,10)
        end if

        select type (fi => sim%fields%interp)
        type is (jorek_fields_interp_linear)
          call evolve_epf_kernel(                                               &
                 c_loc(particles(1)), int(size(particles,1), c_int32_t),         &
                 c_loc(sim%fields%element_list%element(1)),                      &
                 int(sim%fields%element_list%n_elements, c_int32_t),             &
                 c_loc(sim%fields%node_list%node(1)),                            &
                 int(sim%fields%node_list%n_nodes, c_int32_t),                   &
                 c_loc(fi),                                                      &
                 feedback_rhs, rhs_ext, proj_idx,                                &
                 sim%groups(group_num)%mass, sim%time, tstep_part_adj,           &
                 int(nstep_part_adj, c_int32_t),                                 &
                 int(proj_collection_period, c_int32_t),                         &
                 find_RZ_nearby_phi_search(0.d0),                                &
                 not_found, nf_R, nf_Z, bad_i_from, bad_i_to)

          ! The kernel cannot print; find_RZ_nearby's diagnostics come back instead.
          call find_RZ_nearby_report(int(not_found), int(bad_i_from), int(bad_i_to), nf_R, nf_Z)
        class default
          error stop "evolve_epf: the C++ kernel needs jorek_fields_interp_linear"
        end select
      endif
    end select

    !> Renormalise by number of timesteps the projection quantities were collected for. Need integer division
    !> Ie if nstep = 201, proj_col_period = 10, then iterations = 20 (we only collected projections 20 times)
    iterations   = nstep_part_adj / proj_collection_period
    feedback_rhs = feedback_rhs/iterations
#endif
  end subroutine evolve_epf

end module mod_epf_evolution
