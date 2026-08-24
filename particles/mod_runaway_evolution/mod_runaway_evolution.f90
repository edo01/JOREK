!> Particle evolution for the 'rep' coupling scheme: runaway electrons.
!> Called from evolve_particle_group (mod_particle_evolution.f90).
module mod_runaway_evolution
    use particle_tracer
    use coupling_variables
    use mod_ccoll_relativistic, only: ccoll_data, ccoll_init
    use phys_module, only: use_manual_random_seed, part_group_configs
    use, intrinsic :: iso_c_binding, only: c_int64_t
    !$ use omp_lib

    implicit none
    private
    public :: evolve_REs !< public for regtesting, used by evolve_particle_group

    !> Spacing between the generator stream ids of two MPI ranks.
    !>
    !> A rank seeds one stream per particle at stream_base + particle index, so
    !> the ranks must not overlap. A fixed stride rather than the particle count
    !> keeps a rank's ids where they were when the count changes between fluid
    !> steps, and 1e12 is far more slots than any rank will ever hold.
    integer(c_int64_t), parameter :: rng_rank_stride = 1000000000000_c_int64_t

    !> The collision table, read once per run rather than once per call.
    !>
    !> ccoll_init reads an HDF5 file and builds the ion data; nothing in it
    !> depends on the step, so re-reading it every fluid step would be pure I/O.
    !> Kept here rather than in the sim object because it is a property of the
    !> plasma composition, which the sim does not carry.
    type(ccoll_data), save, target :: re_ccoll_data   !< target: c_loc takes its four tables
    logical,          save :: re_ccoll_ready = .false.

    !> Interface only -- the bodies are in C++. One signature, because the two
    !> entry points are one kernel; which one is bound is a build decision.
    abstract interface
      subroutine jgx_evolve_REs_iface(part_base, n_particles, &
                                      el_base, n_elements,    &
                                      nd_base, n_nodes,       &
                                      interp_base,            &
                                      rhs_data, rhs_ext,      &
                                      i_P_par, i_P_perp,      &
                                      i_j_Phi,                &
                                      mass, time, timestep,   &
                                      nstep, phi_search,      &
                                      use_ccoll, use_radreact,&
                                      ccoll_nu, ccoll_nth,    &
                                      ccoll_u, ccoll_theta,   &
                                      ccoll_L0, ccoll_L1,     &
                                      ccoll_mi, ccoll_Z0,     &
                                      rng_seed, rng_stream_base, &
                                      not_found, nf_R, nf_Z,  & !> debug
                                      bad_i_from, bad_i_to) bind(C)
        use, intrinsic :: iso_c_binding, only: c_double, c_int32_t, c_int64_t, c_ptr
        implicit none
        type(c_ptr),        value, intent(in)  :: part_base, el_base, nd_base, interp_base
        integer(c_int32_t), value, intent(in)  :: n_particles, n_elements, n_nodes
        real(c_double),            intent(inout) :: rhs_data(*)
        integer(c_int32_t),        intent(in)  :: rhs_ext(5)
        integer(c_int32_t), value, intent(in)  :: i_P_par, i_P_perp, i_j_Phi, nstep
        real(c_double),     value, intent(in)  :: mass, time, timestep, phi_search
        integer(c_int32_t), value, intent(in)  :: use_ccoll, use_radreact
        integer(c_int32_t), value, intent(in)  :: ccoll_nu, ccoll_nth
        type(c_ptr),        value, intent(in)  :: ccoll_u, ccoll_theta, ccoll_L0, ccoll_L1
        real(c_double),     value, intent(in)  :: ccoll_mi, ccoll_Z0
        integer(c_int64_t), value, intent(in)  :: rng_seed, rng_stream_base
        integer(c_int32_t),        intent(out) :: not_found, bad_i_from, bad_i_to
        real(c_double),            intent(out) :: nf_R, nf_Z
      end subroutine
    end interface

    procedure(jgx_evolve_REs_iface), &
      bind(C, name="jgx_host_runaway_evolution_evolve_REs")     :: evolve_REs_host
#ifdef JGX_HAS_DEVICE
    procedure(jgx_evolve_REs_iface), &
      bind(C, name="jgx_host_runaway_evolution_evolve_REs_device") :: evolve_REs_device
#endif
contains

  !> Gathers the runaway-electron projections of a particle group and pushes its
  !> particles.
  !>
  !> Facade only -- the body is in C++ (runaway_evolution.h next door).
  subroutine evolve_REs(sim, group_num, feedback_rhs, tstep_part_adj, nstep_part_adj)
    use mpi
    use mod_fields_linear, only: jorek_fields_interp_linear
    use mod_find_rz_nearby, only: find_RZ_nearby_phi_search, find_RZ_nearby_report
    use mod_particle_types, only: particle_kinetic_relativistic
    use mod_random_seed, only: random_seed
    use, intrinsic :: iso_c_binding, only: c_double, c_int32_t, c_int64_t, c_loc, c_null_ptr, c_ptr

    implicit none
    class(particle_sim), target, intent(inout) :: sim
    integer, intent(in)                        :: group_num
    real*8, allocatable, intent(inout)         :: feedback_rhs(:,:,:,:,:)
    real*8,  intent(in)                        :: tstep_part_adj
    integer, intent(in)                        :: nstep_part_adj

    integer(c_int32_t) :: rhs_ext(5), not_found, bad_i_from, bad_i_to
    integer(c_int32_t) :: c_use_ccoll, c_use_radreact, ccoll_nu, ccoll_nth
    type(c_ptr)        :: ccoll_u, ccoll_theta, ccoll_L0, ccoll_L1
    real(c_double)     :: ccoll_mi, ccoll_Z0, nf_R, nf_Z
    integer(c_int64_t) :: rng_seed
    integer            :: i, seed, ierr
    logical            :: use_ccoll, use_radreact
    procedure(jgx_evolve_REs_iface), pointer :: evolve_REs_kernel

#if defined(fullmhd) || STELLARATOR_MODEL
    error stop "evolve_REs: the C++ kernel is reduced MHD only (calc_EBpsiU_reduced)"
#else
    ! One kernel, two entry points: a device build runs on the device, a host
    ! build over the Fortran storage in place. Decided by the build, not by input.
#ifdef JGX_HAS_DEVICE
    evolve_REs_kernel => evolve_REs_device
#else
    evolve_REs_kernel => evolve_REs_host
#endif

    use_ccoll    = part_group_configs(group_num)%use_ccoll
    use_radreact = part_group_configs(group_num)%use_radreact

    ! The collision operator covers one background electron species and one main
    ! ion species at the same temperature -- the with_impurities = .false.,
    ! with_TiTe = .false. branch of calc_NjTj. The partial-screening operator is
    ! not ported at all. Guarded here rather than in the kernel, which cannot
    ! report anything.
    if (use_ccoll) then
#if defined(WITH_TiTe) || defined(WITH_Impurities)
      error stop "evolve_REs: use_ccoll needs with_TiTe = .f. and with_impurities = .f."
#endif
      if (.not. re_ccoll_ready) then
        call ccoll_init('ccolldata', re_ccoll_data)
        re_ccoll_ready = .true.
      end if
      if (size(re_ccoll_data%mi) /= 1) &
        error stop "evolve_REs: the C++ collision operator takes exactly one ion species"
    end if

    ! One seed for the whole run, drawn on rank 0 and broadcast, as every other
    ! generator in JOREK is set up. The stream itself is chosen per particle
    ! inside the kernel -- see jorek::re_seed_stream.
    rng_seed = 0_c_int64_t
    if (use_ccoll) then
      if (sim%my_id == 0) seed = random_seed()
      call MPI_Bcast(seed, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
      rng_seed = int(seed, c_int64_t)
    end if

    do i = 1, 5
      rhs_ext(i) = int(size(feedback_rhs, i), c_int32_t)
    end do

    ! The table crosses as its four arrays with their two extents, and the ion
    ! data as two scalars: ccoll_data holds allocatables, so it is not a record
    ! the registry could describe.
    ccoll_nu    = 0_c_int32_t
    ccoll_nth   = 0_c_int32_t
    ccoll_u     = c_null_ptr
    ccoll_theta = c_null_ptr
    ccoll_L0    = c_null_ptr
    ccoll_L1    = c_null_ptr
    ccoll_mi    = 0.d0
    ccoll_Z0    = 0.d0
    if (use_ccoll) then
      ccoll_nu    = int(size(re_ccoll_data%u),     c_int32_t)
      ccoll_nth   = int(size(re_ccoll_data%theta), c_int32_t)
      ccoll_u     = c_loc(re_ccoll_data%u(1))
      ccoll_theta = c_loc(re_ccoll_data%theta(1))
      ccoll_L0    = c_loc(re_ccoll_data%L0(1,1))
      ccoll_L1    = c_loc(re_ccoll_data%L1(1,1))
      ccoll_mi    = re_ccoll_data%mi(1)
      ccoll_Z0    = real(re_ccoll_data%Z0(1), 8)   ! integer*1 in ccoll_data
    end if

    ! Logicals cross as int32 and are tested against zero on the other side.
    c_use_ccoll    = merge(1_c_int32_t, 0_c_int32_t, use_ccoll)
    c_use_radreact = merge(1_c_int32_t, 0_c_int32_t, use_radreact)

    select type (particles => sim%groups(group_num)%particles)
    type is (particle_kinetic_relativistic)
      if (size(particles,1) <= 0) return

      if(use_manual_random_seed) then
        !$ call omp_set_schedule(omp_sched_static,10)
      else
        !$ call omp_set_schedule(omp_sched_dynamic,10)
      end if

      select type (fi => sim%fields%interp)
      type is (jorek_fields_interp_linear)
        call evolve_REs_kernel(                                                 &
               c_loc(particles(1)), int(size(particles,1), c_int32_t),          &
               c_loc(sim%fields%element_list%element(1)),                       &
               int(sim%fields%element_list%n_elements, c_int32_t),              &
               c_loc(sim%fields%node_list%node(1)),                             &
               int(sim%fields%node_list%n_nodes, c_int32_t),                    &
               c_loc(fi),                                                       &
               feedback_rhs, rhs_ext,                                           &
               int(P_par_idx_kin  - 1, c_int32_t),                              &
               int(P_perp_idx_kin - 1, c_int32_t),                              &
               int(j_Phi_idx_kin  - 1, c_int32_t),                              &
               sim%groups(group_num)%mass, sim%time, tstep_part_adj,            &
               int(nstep_part_adj, c_int32_t),                                  &
               find_RZ_nearby_phi_search(0.d0),                                 &
               c_use_ccoll, c_use_radreact,                                     &
               ccoll_nu, ccoll_nth,                                             &
               ccoll_u, ccoll_theta, ccoll_L0, ccoll_L1,                        &
               ccoll_mi, ccoll_Z0,                                              &
               rng_seed, int(sim%my_id, c_int64_t)*rng_rank_stride,             &
               not_found, nf_R, nf_Z, bad_i_from, bad_i_to)

        ! The kernel cannot print; find_RZ_nearby's diagnostics come back instead.
        call find_RZ_nearby_report(int(not_found), int(bad_i_from), int(bad_i_to), nf_R, nf_Z)
      class default
        error stop "evolve_REs: the C++ kernel needs jorek_fields_interp_linear"
      end select
    end select
#endif
  end subroutine evolve_REs

end module mod_runaway_evolution
