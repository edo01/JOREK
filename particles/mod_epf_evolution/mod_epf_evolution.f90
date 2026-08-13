!> Particle evolution for the 'epf' coupling scheme: energetic particles.
!> Called from evolve_particle_group (mod_particle_evolution.f90).
module mod_epf_evolution
    use mod_model_settings
    use particle_tracer
    use phys_module, only: CENTRAL_MASS, CENTRAL_DENSITY
    use phys_module, only: use_manual_random_seed, n_aux_var, part_kill_ratio, proj_collection_period
    use mod_coupling_settings
    use coupling_variables
    use mod_project_particles
    use mod_random_seed
    use mod_interp, only: mode_moivre, interp_0
    use mod_basisfunctions
    use mod_particle_types, only: copy_particle_kinetic_leapfrog
    use mod_sampling, only: boxmueller_transform,sample_chi_squared_3
    use mod_coordinate_transforms, only: vector_cartesian_to_cylindrical
    !$ use omp_lib

    implicit none
    private
    public :: evolve_epf

contains

  subroutine evolve_epf(sim, group_num, feedback_rhs, rng, tstep_part_adj, nstep_part_adj)
    use mod_project_particles
    use mod_random_seed
    use mod_interp, only: mode_moivre
    use mod_basisfunctions

    implicit none
    class(particle_sim), target, intent(inout)                :: sim
    integer, intent(in)                                       :: group_num
    real*8, allocatable, intent(inout)                        :: feedback_rhs(:,:,:,:,:)
    type(count_action)                                        :: counter
    type(pcg32_rng), dimension(:), allocatable, intent(inout) :: rng
    real*8, intent(in)                                        :: tstep_part_adj
    integer, intent(in)                                       :: nstep_part_adj

    !> local variables
    real*8   :: t, E(3), B(3), B_norm(3), psi, U
    real*8   :: rzp_old(3), st_old(2)
    real*8   :: v_tilde_r, v_tilde_z, v_par, p_par, v, p_perp, p_atrop, base
    real*8   :: HZ(n_tor), HH(4,4), HH_s(4,4), HH_t(4,4) !> Bezier basis functions
    integer  :: iterations                               !> number of tsteps for which projection quantities are collected

    integer  :: i, j, k, l, m, ifail, i_tor
    integer  :: i_elm, i_elm_old

    if (nstep_part_adj < proj_collection_period) then
      proj_collection_period = nstep_part_adj
    endif

    !> loop over all particles in group(group_num)
    select type (particles => sim%groups(group_num)%particles)
    type is (particle_kinetic_leapfrog)
      if (use_manual_random_seed) then
        !$ call omp_set_schedule(omp_sched_static,10)
      else
        !$ call omp_set_schedule(omp_sched_dynamic,10)
      end if
      !$omp parallel do default(none) &
      !$omp schedule(runtime) &
      !$omp private(j, k, l, m, HZ, HH, HH_s, HH_t, E, B, psi, U, rzp_old, st_old, i_elm_old, &
      !$omp i_elm, B_norm, v_tilde_r, v_tilde_z, v_par, p_perp, p_par, p_atrop, base, v, i_tor, ifail) &
      !$omp shared(nstep_part_adj, tstep_part_adj, sim, group_num, proj_collection_period, &
      !$omp PI_RR_idx_kin, PI_ZZ_idx_kin, PI_PHIPHI_idx_kin, PI_RZ_idx_kin, PI_RPHI_idx_kin, PI_ZPHI_idx_kin, rho_ep_idx_kin) &
      !$omp reduction(+:feedback_rhs)

      do j=1,size(particles,1)
        do k=1,nstep_part_adj

          !> if particle is lost, skip
          if (particles(j)%i_elm .le. 0) exit

          !> Determine E, B at particles location
          call sim%fields%calc_EBpsiU(sim%time, particles(j)%i_elm, particles(j)%st, particles(j)%x(3), E, B, psi, U)

          rzp_old   = particles(j)%x
          st_old    = particles(j)%st
          i_elm_old = particles(j)%i_elm

          !> push particles and find (R,Z)
          if (particles(j)%i_elm .gt. 0) then
            call boris_push_cylindrical(particles(j), sim%groups(group_num)%mass, E, B, tstep_part_adj)
            call find_RZ_nearby(sim%fields%node_list, sim%fields%element_list, rzp_old(1), rzp_old(2), st_old(1), st_old(2), i_elm_old, particles(j)%x(1), particles(j)%x(2), particles(j)%st(1), particles(j)%st(2), particles(j)%i_elm, ifail)
          endif

          !> may have pushed particle out of domain, check again if it is lost
          if (particles(j)%i_elm .le. 0) exit

          !> only collect projections every proj_collection_period number of timesteps
          if (mod(k,proj_collection_period) .ne. 0) cycle

          !> calc normalised B and orthonormal v cmpts
          B_norm    = B / norm2(B)
          v_par     = dot_product(B_norm, particles(j)%v)
          v_tilde_r = (-(B_norm(1))*particles(j)%v(3)+B_norm(3)*particles(j)%v(1))/sqrt(B_norm(1)**2+B_norm(3)**2)
          v_tilde_z = (particles(j)%v(2)-B_norm(2)*v_par)/sqrt(B_norm(1)**2+B_norm(3)**2)

          !> calc parallel and perpendicular pressures
          p_perp  = 1.d0/2.d0 * (v_tilde_r**2 + v_tilde_z**2)
          p_par   = v_par**2
          p_atrop = p_par - p_perp

          !> calc FEM basis functions
          call basisfunctions(particles(j)%st(1), particles(j)%st(2), HH, HH_s, HH_t)
          call mode_moivre(particles(j)%x(3), HZ)

          !> Gather particle pressures for jorek feedback
          i_elm = particles(j)%i_elm
          do l = 1,n_vertex_max
            do m = 1,n_order+1

              v = HH(l,m) * sim%fields%element_list%element(i_elm)%size(l,m)

              do i_tor = 1,n_tor
                !> expression used in all of the following
                base = HZ(i_tor)*v*particles(j)%weight*sim%groups(group_num)%mass*ATOMIC_MASS_UNIT*MU_ZERO

                !> PI_RR
                feedback_rhs(m,l,i_elm,i_tor,PI_RR_idx_kin) = feedback_rhs(m,l,i_elm,i_tor,PI_RR_idx_kin) + base*(p_perp+B_norm(1)**2*p_atrop)

                !> PI_ZZ
                feedback_rhs(m,l,i_elm,i_tor,PI_ZZ_idx_kin) = feedback_rhs(m,l,i_elm,i_tor,PI_ZZ_idx_kin) + base*(p_perp+B_norm(2)**2*p_atrop)

                !> PI_PHIPHI
                feedback_rhs(m,l,i_elm,i_tor,PI_PHIPHI_idx_kin) = feedback_rhs(m,l,i_elm,i_tor,PI_PHIPHI_idx_kin) + base*(p_perp+B_norm(3)**2*p_atrop)

                !> PI_RZ
                feedback_rhs(m,l,i_elm,i_tor,PI_RZ_idx_kin) = feedback_rhs(m,l,i_elm,i_tor,PI_RZ_idx_kin) + base*(B_norm(1)*B_norm(2)*p_atrop)

                !> PI_RPHI
                feedback_rhs(m,l,i_elm,i_tor,PI_RPHI_idx_kin) = feedback_rhs(m,l,i_elm,i_tor,PI_RPHI_idx_kin) + base*(B_norm(1)*B_norm(3)*p_atrop)

                !> PI_ZPHI
                feedback_rhs(m,l,i_elm,i_tor,PI_ZPHI_idx_kin) = feedback_rhs(m,l,i_elm,i_tor,PI_ZPHI_idx_kin) + base*(B_norm(2)*B_norm(3)*p_atrop)

                !> density
                feedback_rhs(m,l,i_elm,i_tor,rho_ep_idx_kin) = feedback_rhs(m,l,i_elm,i_tor,rho_ep_idx_kin) + HZ(i_tor)*v*particles(j)%weight

              end do !> toroidal harmonics
            end do   !> FEM basis order
          end do     !> verticies of element
        end do       !> particle timesteps
      end do         !> particles


      !$omp end parallel do
    end select

    !> Renormalise by number of timesteps the projection quantities were collected for. Need integer division
    !> Ie if nstep = 201, proj_col_period = 10, then iterations = 20 (we only collected projections 20 times)
    iterations   = nstep_part_adj / proj_collection_period
    feedback_rhs = feedback_rhs/iterations

  end subroutine evolve_epf

end module mod_epf_evolution
