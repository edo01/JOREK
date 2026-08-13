!> Particle evolution for the 'rep' coupling scheme: runaway electrons.
!> Called from evolve_particle_group (mod_particle_evolution.f90).
module mod_runaway_evolution
    use particle_tracer
    use coupling_variables
    use phys_module, only: use_manual_random_seed
    !$ use omp_lib

    implicit none
    private
    public :: evolve_REs !< public for regtesting, used by evolve_particle_group

    !> Interface only -- the body is in C++ (runaway_evolution_shim.cpp next door).
    interface
      subroutine jgx_host_runaway_evolution_evolve_REs(part_base, n_particles, &
                                                       el_base, n_elements,    &
                                                       nd_base, n_nodes,       &
                                                       interp_base,            &
                                                       rhs_data, rhs_ext,      &
                                                       i_P_par, i_P_perp,      &
                                                       i_j_Phi,                &
                                                       mass, time, timestep,   &
                                                       nstep, phi_search,      &
                                                       not_found, nf_R, nf_Z,  &
                                                       bad_i_from, bad_i_to)   &
          bind(C, name="jgx_host_runaway_evolution_evolve_REs")
        use, intrinsic :: iso_c_binding, only: c_double, c_int32_t, c_ptr
        implicit none
        type(c_ptr),        value, intent(in)  :: part_base, el_base, nd_base, interp_base
        integer(c_int32_t), value, intent(in)  :: n_particles, n_elements, n_nodes
        real(c_double),            intent(inout) :: rhs_data(*)
        integer(c_int32_t),        intent(in)  :: rhs_ext(5)
        integer(c_int32_t), value, intent(in)  :: i_P_par, i_P_perp, i_j_Phi, nstep
        real(c_double),     value, intent(in)  :: mass, time, timestep, phi_search
        integer(c_int32_t),        intent(out) :: not_found, bad_i_from, bad_i_to
        real(c_double),            intent(out) :: nf_R, nf_Z
      end subroutine
    end interface
contains

  !> Gathers the runaway-electron projections of a particle group and pushes its
  !> particles.
  !>
  !> Facade only -- the body is in C++ (runaway_evolution.h next door).
  subroutine evolve_REs(sim, group_num, feedback_rhs, tstep_part_adj, nstep_part_adj)
    use mod_fields_linear, only: jorek_fields_interp_linear
    use mod_find_rz_nearby, only: find_RZ_nearby_phi_search, find_RZ_nearby_report
    use mod_particle_types, only: particle_kinetic_relativistic
    use, intrinsic :: iso_c_binding, only: c_double, c_int32_t, c_loc

    implicit none
    class(particle_sim), target, intent(inout) :: sim
    integer, intent(in)                        :: group_num
    real*8, allocatable, intent(inout)         :: feedback_rhs(:,:,:,:,:)
    real*8,  intent(in)                        :: tstep_part_adj
    integer, intent(in)                        :: nstep_part_adj

    integer(c_int32_t) :: rhs_ext(5), not_found, bad_i_from, bad_i_to
    real(c_double)     :: nf_R, nf_Z
    integer            :: i

#if defined(fullmhd) || STELLARATOR_MODEL
    error stop "evolve_REs: the C++ kernel is reduced MHD only (calc_EBpsiU_reduced)"
#else
    do i = 1, 5
      rhs_ext(i) = int(size(feedback_rhs, i), c_int32_t)
    end do

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
        call jgx_host_runaway_evolution_evolve_REs(                             &
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
