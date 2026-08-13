!> this module contains the functionality related to performing physics on particles that needs to happen each particle step
!> the main functionality is all contained in evolve_particle_group, so below for more explanation
module mod_particle_evolution
    use mod_model_settings
    use particle_tracer
    use mod_coupling_settings
    use coupling_variables
    use mod_project_particles
    use mod_random_seed

    !> One module per coupling scheme; evolve_particle_group only dispatches.
    use mod_ncs_ics_evolution,  only: evolve_ncs_ics  !< 'ncs', 'ics'
    use mod_runaway_evolution,  only: evolve_REs      !< 'rep'
    use mod_epf_evolution,      only: evolve_epf      !< 'epf'

    implicit none
    private
    public :: evolve_particle_group

contains

  !> For each particle group, this function does the following:
  !> - performs coupling scheme specific physics (e.g. ionisation, radiation... etc)
  !> - creates the feedback rhs required for projections of kinetic variables
  !> - evolves the location and velocity of the particles based on the background plasma (pushing)
  subroutine evolve_particle_group(sim, group_num, jorek_feedback, rng, tstep_part_adj, nstep_part_adj)

    implicit none
    class(particle_sim), target, intent(inout)                :: sim
    integer, intent(in)                                       :: group_num
    type(projection), target, intent(inout)                   :: jorek_feedback
    type(count_action)                                        :: counter
    type(pcg32_rng), dimension(:), allocatable, intent(inout) :: rng
    real*8,  intent(in)                                       :: tstep_part_adj
    integer, intent(in)                                       :: nstep_part_adj
    
    real*8,allocatable :: feedback_rhs(:,:,:,:,:)
    type (type_node_list),         pointer :: feedback_nodelist
    type (type_element_list),      pointer :: feedback_element_list
    type (particle_group),         pointer :: part_group
    character(len=3) :: cs

    !> Coupling scheme specific
    integer :: imp_q_idx

    !> ================================ INITIALISATION =======================================
    part_group => sim%groups(group_num)
    if (sim%my_id .eq. 0) write(*,*) '---------- Evolving particle group: ', part_group%id, " ----------"

    !> if ics, determine index for impurity charge projection
    if (part_group%coupling_scheme == 'ics') then
      imp_q_idx = ics_indices_kin(part_group%ics_group_idx)
    endif

    !> Set up storage of feedback
    allocate(feedback_rhs,source=jorek_feedback%rhs)
    feedback_nodelist => jorek_feedback%node_list
    feedback_element_list => jorek_feedback%element_list
    feedback_rhs       = 0.d0
    
    !> count number of particles in system, and update sim%groups(...)%average_weight
    call with(sim, counter)
    
    !> ================================ COUPLING SPECIFIC LOOPS =======================================
    !> gathers feedback rhs per particle per tstep_part_adj and pushes particle
    !> this is where coupling specific physics such as ionisation, charge exchange... etc happens

    select case (part_group%coupling_scheme)
      case ('ncs')
        call evolve_ncs_ics(sim, group_num, feedback_rhs, feedback_nodelist, feedback_element_list, rng, tstep_part_adj, nstep_part_adj)
      case ('ics')
        call evolve_ncs_ics(sim, group_num, feedback_rhs, feedback_nodelist, feedback_element_list, rng, tstep_part_adj, nstep_part_adj, imp_q_idx)
      case ('rep')
        call evolve_REs(sim, group_num, feedback_rhs, tstep_part_adj, nstep_part_adj)
      case ('epf')
        call evolve_epf(sim, group_num, feedback_rhs, rng, tstep_part_adj, nstep_part_adj)
      case default
        write(*,*) "ERROR: Unknown coupling scheme: '", part_group%coupling_scheme, "' found for group '", part_group%id, "'"
        stop 1
    end select
    
    ! ================================= CONSTRUCT PROJECTION RHS =======================================
    !> enter gathered rhs into jorek_feedback
    if (part_group%coupling_scheme == 'ncs' .or. part_group%coupling_scheme == 'ics') then
      ! To get rates in the feedback, we need to divide the change by the time. Since we keep adding changes each evolve_particle_group call until 
      ! the rhs is reset to 0 when the jorek_feedback is projected (each fluid tstep), we should divide by sim%tstep_fluid_si
      jorek_feedback%rhs(:,:,:,:,mom_par_idx_kin) = jorek_feedback%rhs(:,:,:,:,mom_par_idx_kin) + feedback_rhs(:,:,:,:,mom_par_idx_kin) / sim%tstep_fluid_si
#ifdef WITH_TiTe
      jorek_feedback%rhs(:,:,:,:,E_Te_idx_kin)    = jorek_feedback%rhs(:,:,:,:,E_Te_idx_kin)    + feedback_rhs(:,:,:,:,E_Te_idx_kin)    / sim%tstep_fluid_si
      jorek_feedback%rhs(:,:,:,:,E_Ti_idx_kin)    = jorek_feedback%rhs(:,:,:,:,E_Ti_idx_kin)    + feedback_rhs(:,:,:,:,E_Ti_idx_kin)    / sim%tstep_fluid_si
#else
      jorek_feedback%rhs(:,:,:,:,E_idx_kin)       = jorek_feedback%rhs(:,:,:,:,E_idx_kin)       + feedback_rhs(:,:,:,:,E_idx_kin)       / sim%tstep_fluid_si
#endif
      !> ncs specific projections
      if (part_group%coupling_scheme == 'ncs') then
        jorek_feedback%rhs(:,:,:,:,rho_idx_kin)   = jorek_feedback%rhs(:,:,:,:,rho_idx_kin)     + feedback_rhs(:,:,:,:,rho_idx_kin)     / sim%tstep_fluid_si
        ! for the density, we should divide by the amount of times we will double count the same particle (=the number of particle steps in a fluid step)
        jorek_feedback%rhs(:,:,:,:,6)             = jorek_feedback%rhs(:,:,:,:,6)               + feedback_rhs(:,:,:,:,6)               / (sim%tstep_fluid_si/tstep_part_adj) !< extra diagnostic projection (density) 
      endif

      !> ics specific projections
      if (part_group%coupling_scheme == 'ics') then
        jorek_feedback%rhs(:,:,:,:,imp_q_idx)     = jorek_feedback%rhs(:,:,:,:,imp_q_idx)       + feedback_rhs(:,:,:,:,imp_q_idx)       / (sim%tstep_fluid_si/tstep_part_adj)
        jorek_feedback%rhs(:,:,:,:,7)             = jorek_feedback%rhs(:,:,:,:,7)               + feedback_rhs(:,:,:,:,7)               / sim%tstep_fluid_si                  !< extra projection (impurity radiated power)
        jorek_feedback%rhs(:,:,:,:,8)             = jorek_feedback%rhs(:,:,:,:,8)               + feedback_rhs(:,:,:,:,8)               / (sim%tstep_fluid_si/tstep_part_adj) !< extra projection (impurity density)
      endif
    endif

    !> rep specific projections
    if (part_group%coupling_scheme == 'rep') then
      feedback_rhs = feedback_rhs / real(nstep_part_adj,8) 
      jorek_feedback%rhs(:,:,:,:,P_par_idx_kin)   = jorek_feedback%rhs(:,:,:,:,P_par_idx_kin)   + feedback_rhs(:,:,:,:,P_par_idx_kin)
      jorek_feedback%rhs(:,:,:,:,P_perp_idx_kin)  = jorek_feedback%rhs(:,:,:,:,P_perp_idx_kin)  + feedback_rhs(:,:,:,:,P_perp_idx_kin)
      jorek_feedback%rhs(:,:,:,:,j_Phi_idx_kin)   = jorek_feedback%rhs(:,:,:,:,j_Phi_idx_kin)   + feedback_rhs(:,:,:,:,j_Phi_idx_kin)
    endif

    !> epf specific projection
    if (part_group%coupling_scheme == 'epf') then
      jorek_feedback%rhs(:,:,:,:,PI_RR_idx_kin)     = jorek_feedback%rhs(:,:,:,:,PI_RR_idx_kin)     + feedback_rhs(:,:,:,:,PI_RR_idx_kin)
      jorek_feedback%rhs(:,:,:,:,PI_ZZ_idx_kin)     = jorek_feedback%rhs(:,:,:,:,PI_ZZ_idx_kin)     + feedback_rhs(:,:,:,:,PI_ZZ_idx_kin)
      jorek_feedback%rhs(:,:,:,:,PI_PHIPHI_idx_kin) = jorek_feedback%rhs(:,:,:,:,PI_PHIPHI_idx_kin) + feedback_rhs(:,:,:,:,PI_PHIPHI_idx_kin)
      jorek_feedback%rhs(:,:,:,:,PI_RZ_idx_kin)     = jorek_feedback%rhs(:,:,:,:,PI_RZ_idx_kin)     + feedback_rhs(:,:,:,:,PI_RZ_idx_kin)
      jorek_feedback%rhs(:,:,:,:,PI_RPHI_idx_kin)   = jorek_feedback%rhs(:,:,:,:,PI_RPHI_idx_kin)   + feedback_rhs(:,:,:,:,PI_RPHI_idx_kin)
      jorek_feedback%rhs(:,:,:,:,PI_ZPHI_idx_kin)   = jorek_feedback%rhs(:,:,:,:,PI_ZPHI_idx_kin)   + feedback_rhs(:,:,:,:,PI_ZPHI_idx_kin)
      jorek_feedback%rhs(:,:,:,:,rho_ep_idx_kin)    = jorek_feedback%rhs(:,:,:,:,rho_ep_idx_kin)    + feedback_rhs(:,:,:,:,rho_ep_idx_kin)
    endif

    jorek_feedback%rhs_gather_time = 0.d0
    deallocate(feedback_rhs)
    
    if (sim%my_id .eq. 0) write(*,*) '---------- Finished evolving group: ', part_group%id, " ----------"
    
  end subroutine evolve_particle_group

end module mod_particle_evolution
