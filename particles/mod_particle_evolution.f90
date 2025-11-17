!> this module contains the functionality related to performing physics on particles that needs to happen each particle step
!> the main functionality is all contained in evolve_particle_group, so below for more explanation
module mod_particle_evolution
    use particle_tracer
    use phys_module, only: CENTRAL_MASS, CENTRAL_DENSITY
    use phys_module, only: nstep_particles, use_manual_random_seed, n_aux_var
    use mod_coupling_settings
    use coupling_variables
    use mod_project_particles
    use mod_random_seed
    use mod_interp, only: mode_moivre, interp_0
    use mod_basisfunctions
    use mod_particle_types, only: copy_particle_kinetic_leapfrog
    use mod_sampling, only: boxmueller_transform,sample_chi_squared_3
    use mod_coordinate_transforms, only: vector_cartesian_to_cylindrical
    use mod_particle_sorting
    !$ use omp_lib

    implicit none
    private
    public :: evolve_particle_group, evolve_REs, evolve_REs_gpu
contains

  !> For each particle group, this function does the following:
  !> - performs coupling scheme specific physics (e.g. ionisation, radiation... etc)
  !> - creates the feedback rhs required for projections of kinetic variables
  !> - evolves the location and velocity of the particles based on the background plasma (pushing)
  subroutine evolve_particle_group(sim, group_num, jorek_feedback, rng, tstep_part_adj)
    use mod_project_particles
    use mod_random_seed
    use mod_interp, only: mode_moivre
    use mod_basisfunctions
    use mod_particle_types, only: copy_particle_kinetic_leapfrog
    use mod_sampling, only: boxmueller_transform,sample_chi_squared_3
    use mpi

    
    implicit none
    class(particle_sim), target, intent(inout)                :: sim
    integer, intent(in)                                       :: group_num
    type(projection), target, intent(inout)                   :: jorek_feedback
    type(count_action)                                        :: counter
    type(pcg32_rng), dimension(:), allocatable, intent(inout) :: rng
    real*8, intent(in)                                        :: tstep_part_adj
    
    real*8,allocatable :: feedback_rhs(:,:,:,:,:)
#ifdef USE_RHS_INV
    real*8,allocatable :: feedback_rhs_inv(:,:,:,:,:)
    integer :: n, m, i_elm, i_tor
#endif
    type (type_node_list),         pointer :: feedback_nodelist
    type (type_element_list),      pointer :: feedback_element_list
    type (particle_group),         pointer :: part_group
    real*8 :: start_time, end_time, tot_time(3)

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
    jorek_feedback%rhs_gather_time = jorek_feedback%rhs_gather_time + nstep_particles * tstep_part_adj
    allocate(feedback_rhs,source=jorek_feedback%rhs)
    feedback_nodelist => jorek_feedback%node_list
    feedback_element_list => jorek_feedback%element_list
    feedback_rhs       = 0.d0

#ifdef USE_RHS_INV
    allocate(feedback_rhs_inv(sim%fields%element_list%n_elements, 8, n_tor, n_vertex_max, n_order+1), source=0.d0)
#endif

    !> count number of particles in system
    call with(sim, counter)
    
    !> ================================ COUPLING SPECIFIC LOOPS =======================================
    !> gathers feedback rhs per particle per tstep_part_adj and pushes particle
    !> this is where coupling specific physics such as ionisation, charge exchange... etc happens

    !> Start timing the section
    start_time = MPI_WTIME()

    select case (part_group%coupling_scheme)
      case ('ncs')
      call evolve_ncs_ics(sim, group_num, feedback_rhs, feedback_nodelist, feedback_element_list, rng, tstep_part_adj)
      case ('ics')
      call evolve_ncs_ics(sim, group_num, feedback_rhs, feedback_nodelist, feedback_element_list, rng, tstep_part_adj, imp_q_idx)
      case ('rep')
#ifdef USE_GPU
#ifdef USE_RHS_INV
      call evolve_REs_gpu(sim, group_num, feedback_rhs_inv, rng, tstep_part_adj)
#else
      call evolve_REs_gpu(sim, group_num, feedback_rhs, rng, tstep_part_adj)
#endif
#else
      call evolve_REs(sim, group_num, feedback_rhs, rng, tstep_part_adj)
#endif
      case default
      write(*,*) "ERROR: Unknown coupling scheme: '", part_group%coupling_scheme, "' found for group '", part_group%id, "'"
      stop 1
    end select

    !> End timing the section
    end_time = MPI_WTIME()
    tot_time = mpi_minmeanmax(end_time-start_time)
    if (sim%my_id .eq. 0) then
        write(*,"(A,3f10.3,A)") , "Time taken for coupling scheme  finished in (min/mean/max): ", tot_time, " seconds"
    endif

#ifdef USE_GPU_TEST
    !> count number of particles lost in the system
    call gpu_test(sim, group_num, feedback_rhs)
#endif

#ifdef USE_RHS_INV
    start_time = MPI_WTIME()
    do n=1,n_degrees
      do m=1,n_vertex_max
        do i_elm=1,sim%fields%element_list%n_elements
          do i_tor=1,n_tor
            feedback_rhs(n,m,i_elm,i_tor,P_par_idx_kin)  = feedback_rhs_inv(i_elm,P_par_idx_kin,i_tor,m,n)
            feedback_rhs(n,m,i_elm,i_tor,P_perp_idx_kin) = feedback_rhs_inv(i_elm,P_perp_idx_kin,i_tor,m,n)
            feedback_rhs(n,m,i_elm,i_tor,j_Phi_idx_kin)  = feedback_rhs_inv(i_elm,j_Phi_idx_kin,i_tor,m,n)
          enddo
        enddo
      enddo
    enddo
    end_time = MPI_WTIME()
    tot_time = mpi_minmeanmax(end_time-start_time)
    if (sim%my_id .eq. 0) then
      write(*,"(A,3f10.3,A)") , "Time taken for feedback_rhs reconstruction finished in (min/mean/max): ", tot_time, " seconds"
    endif

    deallocate(feedback_rhs_inv) 
#endif

    ! ================================= CONSTRUCT PROJECTION RHS =======================================
    !> enter gathered rhs into jorek_feedback
    if (part_group%coupling_scheme == 'ncs' .or. part_group%coupling_scheme == 'ics') then
      write(*,*) 'GATHER TIME : ',jorek_feedback%rhs_gather_time
      jorek_feedback%rhs(:,:,:,:,mom_par_idx_kin) = jorek_feedback%rhs(:,:,:,:,mom_par_idx_kin) + feedback_rhs(:,:,:,:,mom_par_idx_kin) / jorek_feedback%rhs_gather_time !* TWOPI
      jorek_feedback%rhs(:,:,:,:,E_idx_kin) = jorek_feedback%rhs(:,:,:,:,E_idx_kin) + feedback_rhs(:,:,:,:,E_idx_kin) / jorek_feedback%rhs_gather_time !* TWOPI

      !> ncs specific projections
      if (part_group%coupling_scheme == 'ncs') then
        jorek_feedback%rhs(:,:,:,:,rho_idx_kin) = jorek_feedback%rhs(:,:,:,:,rho_idx_kin) + feedback_rhs(:,:,:,:,rho_idx_kin) / jorek_feedback%rhs_gather_time !* TWOPI
        jorek_feedback%rhs(:,:,:,:,5) = jorek_feedback%rhs(:,:,:,:,5) + feedback_rhs(:,:,:,:,5)   !< extra diagnostic projection 
      endif

      !> ics specific projections
      if (part_group%coupling_scheme == 'ics') then
        jorek_feedback%rhs(:,:,:,:,imp_q_idx) = jorek_feedback%rhs(:,:,:,:,imp_q_idx) + feedback_rhs(:,:,:,:,imp_q_idx)
        jorek_feedback%rhs(:,:,:,:,6) = jorek_feedback%rhs(:,:,:,:,6) + feedback_rhs(:,:,:,:,6)   !< extra projection (impurity radiated power)
        jorek_feedback%rhs(:,:,:,:,7) = jorek_feedback%rhs(:,:,:,:,7) + feedback_rhs(:,:,:,:,7)   !< extra projection (impurity density)
      endif
    endif

    !> rep specific projections
    if (part_group%coupling_scheme == 'rep') then
      feedback_rhs = feedback_rhs / real(nstep_particles,8) 
      jorek_feedback%rhs(:,:,:,:,P_par_idx_kin) = jorek_feedback%rhs(:,:,:,:,P_par_idx_kin) + feedback_rhs(:,:,:,:,P_par_idx_kin) !* TWOPI
      jorek_feedback%rhs(:,:,:,:,P_perp_idx_kin) = jorek_feedback%rhs(:,:,:,:,P_perp_idx_kin) + feedback_rhs(:,:,:,:,P_perp_idx_kin) !* TWOPI
      jorek_feedback%rhs(:,:,:,:,j_Phi_idx_kin) = jorek_feedback%rhs(:,:,:,:,j_Phi_idx_kin) + feedback_rhs(:,:,:,:,j_Phi_idx_kin) !* TWOPI
    endif

    jorek_feedback%rhs_gather_time = 0.d0
    deallocate(feedback_rhs)
    
    if (sim%my_id .eq. 0) write(*,*) '---------- Finished evolving group: ', part_group%id, " ----------"
    
  end subroutine evolve_particle_group

  subroutine evolve_REs(sim, group_num, feedback_rhs, rng, tstep_part_adj)
    use mod_project_particles
    use mod_random_seed
    use mod_interp, only: mode_moivre
    use mod_basisfunctions
    use mod_particle_types, only: copy_particle_kinetic_leapfrog
    use mod_sampling, only: boxmueller_transform,sample_chi_squared_3
    
    implicit none
    class(particle_sim), target, intent(inout)                :: sim
    integer, intent(in)                                       :: group_num
    real*8,allocatable,          intent(inout)                :: feedback_rhs(:,:,:,:,:)
    type(count_action)                                        :: counter
    type(pcg32_rng), dimension(:), allocatable, intent(inout) :: rng
    real*8, intent(in)                                        :: tstep_part_adj
    
    character(len=3) :: cs

    !> RE specific variables
    real*8    :: n_norm, rho_norm
    real*8    :: HZ(n_tor), HH(4,4), HH_s(4,4), HH_t(4,4)
    real*8    :: v_n, v_jR, v_jZ, v_jPhi, v_Ppar, v_Pperp

    real*8    :: E(3), B(3), B_norm2(3), psi, U
    real*8    :: cylindrical_velocity(3), cylindrical_momentum(3)
    real*8    :: v_par, v_perp, gamma_m, proj_factor
    integer   :: j, k, m, n, ifail, i_tor, n_lost
    
    if (sim%my_id .eq. 0) then
      write(*,*) 'INFO: REs evolution executed on     : CPU'
      write(*,*) 'INFO: kernel                        : evolve_REs'
      write(*,*) 'INFO: number of timesteps           : ', nstep_particles
      write(*,*) 'INFO: feedback_rhs dimension        : ', size(feedback_rhs,1)*size(feedback_rhs,2)*size(feedback_rhs,3)*size(feedback_rhs,4)*size(feedback_rhs,5)*8 , ' Bytes'
    endif

    n_norm   = CENTRAL_DENSITY * 1.d20                              ! (number) density normalisation
    rho_norm = CENTRAL_MASS * MASS_PROTON * n_norm                  ! rho_SI = rho_norm * rho

    ! Loop over all particle groups
    n_lost = 0
    select type (particles => sim%groups(group_num)%particles)
    type is (particle_kinetic_relativistic)

      ! FIX initialization bug
      do j=1,size(particles,1)
        if(particles(j)%weight .eq. 0.0) then
              particles(j)%weight = 1.0
        endif
        if(particles(j)%q .eq. 0.0) then
              particles(j)%q = -1.0 !> default charge of the runaway electrons
        endif
      enddo

      if(use_manual_random_seed) then
        !$ call omp_set_schedule(omp_sched_static,10)
      else
        !$ call omp_set_schedule(omp_sched_dynamic,10)
      end if  
      !$omp parallel do default(none) &
      !$omp schedule(runtime)         &
      !$omp private(j, k, m, n, HZ, HH, HH_s, HH_t, E, B, psi, U,  &
      !$omp B_norm2, proj_factor, v_Ppar, v_Pperp, v_jPhi, v_n, i_tor, ifail, &
      !$omp cylindrical_velocity, cylindrical_momentum, v_par, v_perp, gamma_m ) &
      !$omp shared (particles, nstep_particles, tstep_part_adj, sim, group_num, rho_norm, &
      !$omp P_par_idx_kin, P_perp_idx_kin, j_phi_idx_kin) &
      !$omp reduction(+:feedback_rhs)
  
      do j=1,size(particles,1)
        do k=1,nstep_particles
          if (j .eq. 0) write(*,*) "k = ", k
          if (particles(j)%i_elm .le. 0) exit
  
          call basisfunctions(particles(j)%st(1), particles(j)%st(2), HH, HH_s, HH_t)
          call mode_moivre(particles(j)%x(3), HZ)
  
          ! Determines velocity in cylindrical coordinates
          cylindrical_momentum = vector_cartesian_to_cylindrical(particles(j)%x(3), particles(j)%p)
          cylindrical_velocity = cylindrical_momentum / &
                                  sqrt(dot_product(cylindrical_momentum,cylindrical_momentum)/SPEED_OF_LIGHT**2 + sim%groups(group_num)%mass**2)
  
          ! Uncomment for PCS
          call sim%fields%calc_EBpsiU(sim%time, particles(j)%i_elm, particles(j)%st, particles(j)%x(3), E, B, psi, U)
          B_norm2 = B/norm2(B)
  
          v_par   = dot_product(cylindrical_velocity, B_norm2)
          v_perp  = norm2(cylindrical_velocity - v_par * B_norm2)
          gamma_m = sqrt(MASS_ELECTRON**2 + dot_product(cylindrical_momentum,cylindrical_momentum)*ATOMIC_MASS_UNIT**2/SPEED_OF_LIGHT**2)
  
          do n=1,n_degrees
            do m=1,n_vertex_max
  
              proj_factor = HH(m,n) * sim%fields%element_list%element(particles(j)%i_elm)%size(m,n) * particles(j)%weight
  

              ! PCS for REs
              v_Ppar  = proj_factor * gamma_m * v_par**2 * MU_ZERO
              v_Pperp = proj_factor * gamma_m * v_perp**2 / 2.d0 * MU_ZERO
  
              v_jPhi  = - proj_factor * real(particles(j)%q, 8) * EL_CHG * cylindrical_velocity(3) * particles(j)%x(1) * MU_ZERO
              
              v_n     = proj_factor

              do i_tor = 1,n_tor
                feedback_rhs(n,m,particles(j)%i_elm,i_tor,P_par_idx_kin) = feedback_rhs(n,m,particles(j)%i_elm,i_tor,P_par_idx_kin) + HZ(i_tor)*v_Ppar
                feedback_rhs(n,m,particles(j)%i_elm,i_tor,P_perp_idx_kin) = feedback_rhs(n,m,particles(j)%i_elm,i_tor,P_perp_idx_kin) + HZ(i_tor)*v_Pperp
                feedback_rhs(n,m,particles(j)%i_elm,i_tor,j_Phi_idx_kin) = feedback_rhs(n,m,particles(j)%i_elm,i_tor,j_Phi_idx_kin) + HZ(i_tor)*v_jPhi

                !> inverse implementation from Hannes (TODO: reverse data structure of the projections generally)
                ! feedback_rhs_inv(P_par_idx_kin,i_tor,particles(j)%i_elm,m,n) = feedback_rhs_inv(P_par_idx_kin,i_tor,particles(j)%i_elm,m,n) + HZ(i_tor)*v_Ppar
                ! feedback_rhs_inv(P_perp_idx_kin,i_tor,particles(j)%i_elm,m,n) = feedback_rhs_inv(P_perp_idx_kin,i_tor,particles(j)%i_elm,m,n) + HZ(i_tor)*v_Pperp
                ! feedback_rhs_inv(j_Phi_idx_kin,i_tor,particles(j)%i_elm,m,n) = feedback_rhs_inv(j_Phi_idx_kin,i_tor,particles(j)%i_elm,m,n) + HZ(i_tor)*v_jPhi

              enddo
            enddo
          enddo
  
          call volume_preserving_push_jorek(particles(j),sim%fields,sim%groups(group_num)%mass,sim%time,tstep_part_adj,ifail)
  
  
        end do !< steps
      end do !< particles
      !$omp end parallel do 
  
    end select
    
  end subroutine evolve_REs

!> Kernel to evolve REs on GPU
subroutine evolve_REs_gpu(sim, group_num, feedback_rhs, rng, tstep_part_adj)
    use mod_project_particles
    use mod_random_seed
    use mod_interp, only: mode_moivre
    use mod_basisfunctions
    use mod_particle_types, only: copy_particle_kinetic_leapfrog !> not used
    use mod_sampling, only: boxmueller_transform,sample_chi_squared_3
    use mod_fields_linear, only: jorek_fields_interp_linear, calc_EBpsiU_linear_gpu_inlined
    use phys_module, only: F0, mode, central_mass, central_density, tstep
    use hdf5_io_module
    use mod_particle_io
    ! timing
    use mpi
    
    implicit none
    class(particle_sim), target, intent(inout)                :: sim
    integer, intent(in)                                       :: group_num
    real*8,allocatable,          intent(inout)                :: feedback_rhs(:,:,:,:,:)
    type(count_action)                                        :: counter
    type(pcg32_rng), dimension(:), allocatable, intent(inout) :: rng
    real*8, intent(in)                                        :: tstep_part_adj
    
    character(len=3) :: cs !> TODO: not used
    integer :: particle_lost  
  
    !> RE specific variables
    real*8    :: n_norm, rho_norm
    real*8    :: HZ(n_tor), HH(4,4), HH_s(4,4), HH_t(4,4)
    real*8    :: v_n, v_jR, v_jZ, v_jPhi, v_Ppar, v_Pperp

    real*8    :: E(3), B(3), B_norm2(3), psi, U
    real*8    :: cylindrical_velocity(3), cylindrical_momentum(3)
    real*8    :: v_par, v_perp, gamma_m, proj_factor
    integer   :: j, k, m, n, ifail, i_tor
    real*8    :: group_mass, sim_time
    integer   :: num_particles
    
    !> SoA
    real*8, allocatable           :: p_x(:,:), p_p(:,:), p_st(:,:), p_weight(:)
    integer, allocatable          :: p_i_elm(:) 
    integer(kind=1),allocatable   :: p_q(:)

#ifdef CALC_EBPSIU_ENHANCED
    !> enhanced calc_EBpsiU
    real*8,  allocatable          :: nodes_x(:,:,:,:), nodes_values(:,:,:,:), nodes_deltas(:,:,:,:)
    integer, allocatable          :: elm_vertex(:,:), elm_neighbours(:,:)
    real*8,  allocatable          :: elm_size(:,:,:)
    integer                       :: n_elements, n_nodes
#endif
    
    !> sorting 
    integer                       :: n_sort, s !< Multiple sorting
    integer                       :: alive_particle_count !< last particle alive index, used to avoid evolving dead particles
    real*8                        :: start_time, end_time, tot_time(3)
#ifdef USE_ORDERING
    integer, allocatable          :: sorted_indices(:) !< indices of the particles sorted by i_elm
    integer                       :: ierr
#endif

    if (sim%my_id .eq. 0) then
      write(*,*) 'INFO: REs evolution executed on     : GPU'
      write(*,*) 'INFO: kernel                        : evolve_REs_gpu_SoA'
      write(*,*) 'INFO: number of timesteps           : ', nstep_particles
      write(*,*) 'INFO: feedback_rhs dimension        : ', size(feedback_rhs,1)*size(feedback_rhs,2)*size(feedback_rhs,3)*size(feedback_rhs,4)*size(feedback_rhs,5)*8.d0/(1024*1024) , ' MB'
      write(*,*) 'INFO: feedback_rhs shape            : (', size(feedback_rhs,1), ',', size(feedback_rhs,2), ',', size(feedback_rhs,3), ',', size(feedback_rhs,4), ',', size(feedback_rhs,5),')' 
#ifdef USE_ORDERING
      write(*,*) 'INFO: using particle ordering'
#endif
#ifdef USE_RHS_INV
      write(*,*) 'INFO: using feedback_rhs_inv'
#endif
#ifdef CALC_EBPSIU_ENHANCED
      write(*,*) 'INFO: using enhanced calc_EBpsiU'
#endif
    endif

    ! TODO: not used
    n_norm   = CENTRAL_DENSITY * 1.d20                              ! (number) density normalisation
    rho_norm = CENTRAL_MASS * MASS_PROTON * n_norm                  ! rho_SI = rho_norm * rho

    !> save simulation parameters
    group_mass      = sim%groups(group_num)%mass
    sim_time        = sim%time
    num_particles   = size(sim%groups(group_num)%particles,1)
    
    !> if the particles are not sorted, then the number of alive particles is the total number of particles
    alive_particle_count = num_particles 

    !> -----------------------------------------
    !> ---------- GPU DATA STRUCTURES ----------
    !> -----------------------------------------
    if (sim%my_id .eq. 0) write(*,*) 'INFO: Preparing GPU data structures...'

#ifdef CALC_EBPSIU_ENHANCED
    allocate(elm_vertex(sim%fields%element_list%n_elements, n_vertex_max))
    allocate(elm_size(n_vertex_max, n_degrees,sim%fields%element_list%n_elements))
    ! copy the element vertex and element size into elm_vertex and elm_size
    do j = 1,sim%fields%element_list%n_elements
      elm_vertex(j,:) = sim%fields%element_list%element(j)%vertex(:)
      elm_size(:,:,j) = sim%fields%element_list%element(j)%size(:,:)
    enddo

    allocate(nodes_x(n_coord_tor,n_degrees,n_dim,sim%fields%node_list%n_nodes))
    allocate(nodes_values(n_tor,n_degrees,n_var,sim%fields%node_list%n_nodes))
    allocate(nodes_deltas(n_tor,n_degrees,n_var,sim%fields%node_list%n_nodes))
    ! copy the node x,values and deltas into nodes_x, nodes_values, nodes_deltas
    do j = 1,sim%fields%node_list%n_nodes
      nodes_x(:,:,:,j)       = sim%fields%node_list%node(j)%x(:,:,:)
      nodes_values(:,:,:,j)  = sim%fields%node_list%node(j)%values(:,:,:)
      nodes_deltas(:,:,:,j)  = sim%fields%node_list%node(j)%deltas(:,:,:)
    enddo
#endif

    select type (particles => sim%groups(group_num)%particles)
    type is (particle_kinetic_relativistic)

    !> allocate the SoA data structures
    !> TODO: move them in the main and update them before the kernel execution
    allocate(p_x(3,alive_particle_count))
    allocate(p_p(3,alive_particle_count))
    allocate(p_st(2,alive_particle_count)) 
    allocate(p_weight(alive_particle_count))
    allocate(p_i_elm(alive_particle_count))
    allocate(p_q(alive_particle_count))

    !> Breaks the evolution in n_sort stage, and at each stage it orders the particles by i_elm to improve data locality
    n_sort = 1 !10

    do s=1,n_sort

#ifdef USE_ORDERING
    !> Start timing the sorting section
    start_time = MPI_WTIME()

    call i_elm_sort_global(sorted_indices, particles, alive_particle_count)

    !> End timing the sorting section
    end_time = MPI_WTIME()
    tot_time = mpi_minmeanmax(end_time-start_time)
    if (sim%my_id .eq. 0) then
        write(*,"(A,3f10.3,A)") , "Time taken for ordering (min/mean/max): ", tot_time, " seconds"
    endif

#endif

    !> Populate the SoA data structures
    do j=1,alive_particle_count
#ifdef USE_ORDERING
      p_x(:,j) = particles(sorted_indices(j))%x
      p_p(:,j) = particles(sorted_indices(j))%p
      p_st(:,j) = particles(sorted_indices(j))%st
      p_weight(j) = particles(sorted_indices(j))%weight
      p_i_elm(j) = particles(sorted_indices(j))%i_elm
      p_q(j) = particles(sorted_indices(j))%q
#else 
      p_x(:,j) = particles(j)%x
      p_p(:,j) = particles(j)%p
      p_st(:,j) = particles(j)%st
      p_weight(j) = particles(j)%weight
      p_i_elm(j) = particles(j)%i_elm
      p_q(j) = particles(j)%q
#endif
    enddo

    select type (fields => sim%fields)
    type is (jorek_fields_interp_linear)

    !> Start timing the kernel section
    start_time = MPI_WTIME()

#ifdef USE_GPU
      !> TODO: the optimal way would be to allocate the SoA structures in the main, and update them before and after
      !>       the kernel execution. 
      !$omp target data                                                                     &
      !$omp map(tofrom: feedback_rhs, p_x, p_p, p_st, p_i_elm)                              &
      !$omp map(to: fields, fields%element_list, fields%node_list, fields%node_list%node,   &      
      !$omp         group_mass, tstep_part_adj, nstep_particles,                            &  
      !$omp         P_par_idx_kin, P_perp_idx_kin,j_phi_idx_kin,                            &
#ifdef CALC_EBPSIU_ENHANCED
      !$omp         elm_vertex, elm_size, nodes_x, nodes_values, nodes_deltas,              &
#endif
      !$omp         p_weight, p_q)                                  

      !$omp target teams distribute parallel do default(none)                       &
      !$omp shared(p_x, p_p, p_st, p_weight, p_i_elm, p_q,                          &
      !$omp        fields,  feedback_rhs,                                           &
      !$omp        sim_time, group_mass, tstep_part_adj, nstep_particles,           & 
      !$omp        P_par_idx_kin, P_perp_idx_kin, j_phi_idx_kin,                    &
#ifdef CALC_EBPSIU_ENHANCED
      !$omp        elm_vertex, elm_size, nodes_x, nodes_values, nodes_deltas,       &
#endif
      !$omp        alive_particle_count, n_sort)                                    &
      !$omp private(ifail, j, k)                                                    &
      !$omp private(E, B, psi, U, HH, HH_s, HH_t, HZ,                               &
      !$omp         cylindrical_momentum, B_norm2, v_par, v_perp,                   &
      !$omp         cylindrical_velocity, gamma_m, proj_factor,                     &
      !$omp         v_Ppar, v_Pperp, v_jPhi, m, n, i_tor)                           &
      !$omp device(sim%gpu_id)                                                                    
#endif
      do j=1, alive_particle_count
        do k=1,nstep_particles/n_sort
        
          if (p_i_elm(j) .le. 0) cycle

          ! HH_s and HH_t are not used
          call basisfunctions(p_st(1,j), p_st(2,j), HH, HH_s, HH_t)
          call mode_moivre(p_x(3,j), HZ)

          ! Determines velocity in cylindrical coordinates
          cylindrical_momentum = vector_cartesian_to_cylindrical(p_x(3,j), p_p(:,j))
          cylindrical_velocity = cylindrical_momentum / &
                                  sqrt(dot_product(cylindrical_momentum,cylindrical_momentum)/SPEED_OF_LIGHT**2 + group_mass**2)

          ! Uncomment for PCS
          !> TODO: check if a function specific for the linear fields is needed  
          !> TODO: when calling interp_PRZ, exp and complex builtins are called. some optimization?
          !> TODO: a lot of computation is done in this function, can we optimize it? 
#ifdef CALC_EBPSIU_ENHANCED
          call calc_EBpsiU_linear_gpu_inlined_enhanced(nodes_x, nodes_values, nodes_deltas, elm_vertex, elm_size, fields%time_now, &
                                      fields%time_prev, fields%static, fields%flag_zero_dpsidt,                                    &
                                        sim_time, p_i_elm(j), p_st(:,j), p_x(3,j), E, B, psi, U)
#else
          call calc_EBpsiU_linear_gpu_inlined(fields%node_list, fields%element_list, fields%time_now,   &
                                      fields%time_prev, fields%static, fields%flag_zero_dpsidt,         &
                                        sim_time, p_i_elm(j), p_st(:,j), p_x(3,j), E, B, psi, U)
#endif
          ! norm2 is really slow on GPU
          !B_norm2 = B/norm2(B)
          B_norm2  = B / sqrt(dot_product(B,B)) 
          
          v_par   = dot_product(cylindrical_velocity, B_norm2)
          !v_perp  = norm2(cylindrical_velocity - v_par * B_norm2)
          v_perp = sqrt(dot_product(cylindrical_velocity - v_par * B_norm2, cylindrical_velocity - v_par * B_norm2))
          gamma_m = sqrt(MASS_ELECTRON**2 + dot_product(cylindrical_momentum,cylindrical_momentum)*ATOMIC_MASS_UNIT**2/SPEED_OF_LIGHT**2)

          v_Ppar  = gamma_m * v_par**2 * MU_ZERO
          v_Pperp = gamma_m * v_perp**2 / 2.d0 * MU_ZERO
          v_jPhi  = - real(p_q(j), 8) * EL_CHG * cylindrical_velocity(3) * p_x(1,j) * MU_ZERO
              
              
          do n=1,n_degrees
            do m=1,n_vertex_max
              proj_factor = HH(m,n) * fields%element_list%element(p_i_elm(j))%size(m,n) * p_weight(j)

              do i_tor = 1,n_tor
#ifdef USE_RHS_INV
                !$omp atomic update
                feedback_rhs(p_i_elm(j),P_par_idx_kin,i_tor,m,n) =  feedback_rhs(p_i_elm(j),P_par_idx_kin,i_tor,m,n) + HZ(i_tor)*v_Ppar*proj_factor
                !$omp end atomic
                !$omp atomic update
                feedback_rhs(p_i_elm(j),P_perp_idx_kin,i_tor,m,n) =  feedback_rhs(p_i_elm(j),P_perp_idx_kin,i_tor,m,n) + HZ(i_tor)*v_Pperp*proj_factor
                !$omp end atomic
                !$omp atomic update
                feedback_rhs(p_i_elm(j),j_phi_idx_kin,i_tor,m,n) =  feedback_rhs(p_i_elm(j),j_phi_idx_kin,i_tor,m,n) + HZ(i_tor)*v_jPhi*proj_factor
                !$omp end atomic
#else
                !$omp atomic update
                feedback_rhs(n,m,p_i_elm(j),i_tor,P_par_idx_kin)  =  feedback_rhs(n,m,p_i_elm(j),i_tor,P_par_idx_kin) + HZ(i_tor)*v_Ppar*proj_factor
                !$omp end atomic
                !$omp atomic update
                feedback_rhs(n,m,p_i_elm(j),i_tor,P_perp_idx_kin) =  feedback_rhs(n,m,p_i_elm(j),i_tor,P_perp_idx_kin) + HZ(i_tor)*v_Pperp*proj_factor
                !$omp end atomic
                !$omp atomic update
                feedback_rhs(n,m,p_i_elm(j),i_tor,j_Phi_idx_kin)  =  feedback_rhs(n,m,p_i_elm(j),i_tor,j_Phi_idx_kin) + HZ(i_tor)*v_jPhi*proj_factor
                !$omp end atomic
#endif
              enddo !< n_tor
            enddo !< n_vertex_max
          enddo !< degrees

#ifdef CALC_EBPSIU_ENHANCED
          call volume_preserving_push_jorek_gpu_SoA_enhanced(p_x(:,j), p_p(:,j), p_st(:,j), p_i_elm(j), p_q(j),  nodes_x, nodes_values, nodes_deltas, elm_vertex, elm_size, fields%node_list, &
                        fields%element_list, fields%time_now, fields%time_prev, fields%static, fields%flag_zero_dpsidt, &
                        group_mass, sim_time, tstep_part_adj, ifail)
#else
          call volume_preserving_push_jorek_gpu_SoA(p_x(:,j), p_p(:,j), p_st(:,j), p_i_elm(j), p_q(j), fields%node_list, &
                        fields%element_list, fields%time_now, fields%time_prev, fields%static, fields%flag_zero_dpsidt, &
                        group_mass, sim_time, tstep_part_adj, ifail)
#endif

        end do !< steps
      end do !< particles
#ifdef USE_GPU
      !$omp end target teams distribute parallel do

      !$omp end target data
#endif
    
    !> End timing the sorting section
    end_time = MPI_WTIME()
    tot_time = mpi_minmeanmax(end_time-start_time)
    if (sim%my_id .eq. 0) then
        write(*,"(A,3f10.3,A)") , "Time taken for the kernel (min/mean/max): ", tot_time, " seconds"
    endif

    end select ! select type (fields)
    
    
    !> ------------------COPY BACK---------------------
    !> copy the SoA data structures back to the particles
    do j=1,alive_particle_count
#ifdef USE_ORDERING
      particles(sorted_indices(j))%x = p_x(:,j)    
      particles(sorted_indices(j))%p = p_p(:,j)
      particles(sorted_indices(j))%st = p_st(:,j)
      particles(sorted_indices(j))%i_elm = p_i_elm(j)
#else
      particles(j)%x = p_x(:,j)    
      particles(j)%p = p_p(:,j)
      particles(j)%st = p_st(:,j)
      particles(j)%i_elm = p_i_elm(j)
#endif
    enddo !< copy

    enddo !< n_sort

    end select ! select type (particles)

    !> ------------------CLEANING---------------------
    !> deallocate SoA data structures
    deallocate(p_x);      deallocate(p_p);      deallocate(p_st)
    deallocate(p_weight); deallocate(p_i_elm);  deallocate(p_q)
#ifdef USE_ORDERING
    deallocate(sorted_indices)
#endif
#ifdef CALC_EBPSIU_ENHANCED
     !> deallocate SoA 
     deallocate(elm_vertex);  deallocate(elm_size)
     deallocate(nodes_x);     deallocate(nodes_values); deallocate(nodes_deltas);
#endif
#ifdef USE_COUNTERS
    !> deallocate counters
    deallocate(jumps_counter); deallocate(v_par_counter); deallocate(v_counter)
#endif

  end subroutine evolve_REs_gpu


  !> Internal function for gathering the feedback rhs values when using the ncs or ics coupling scheme
  !> The two coupling schemes are handled by the same function due to large degree of overlap in the physics 
  !> experienced by neutrals and impurities. 
  !> The pushing of the particle is also done here
  subroutine evolve_ncs_ics(sim, group_num, feedback_rhs, feedback_nodelist, feedback_element_list, rng, tstep_part_adj, imp_q_idx)
    use mod_collisions
    use mod_ionisation_recombination

    implicit none
    class(particle_sim),       target, intent(inout)          :: sim
    integer,                           intent(in)             :: group_num
    real*8,allocatable,                intent(inout)          :: feedback_rhs(:,:,:,:,:)
    type (type_node_list),    pointer, intent(in)             :: feedback_nodelist
    type (type_element_list), pointer, intent(in)             :: feedback_element_list
    type(pcg32_rng), dimension(:), allocatable, intent(inout) :: rng
    real*8,                            intent(in)             :: tstep_part_adj
    integer, optional,                 intent(in)             :: imp_q_idx


    real*8, parameter  :: H_binding_energy = 2.18d-18

    !> Diagnostics ----------------------------------- 
    real*8    :: n_lost_ion, n_lost_ion_all, p_lost_ion, p_lost_ion_all 
    real*8    :: p_lost_cx, p_lost_cx_all, p_lost_plt, p_lost_plt_all 
    integer   :: n_super_ionized, n_super_ionized_all

    !> Coupling --------------------------------------
    real*8    :: density_source, mom_par_source, energy_source
    real*8    :: density_fb, mom_par_fb, E_fb, imp_q_fb, imp_density_fb, imp_P_rad_fb, extra_proj 
    real*8    :: v_temp(3), B_norm(3)
    real*8    :: vvector(3), ran_norm(4)

    logical   :: limits, limits_coll
    real*8    :: ionize_rate, ionize_energy, ionize_source, ionize_prob
    real*8    :: cx_rate, cx_energy,  cx_source, cx_prob
    real*8    :: PLT, PRB, Srec, line_rad_energy
    real*8    :: ionize_ran(1), ionize_ran_imp(2), cx_ran(8), st_ran(2) 
    real*8    :: kinetic_energy, radiation_energy, binding_energy
    real*8    :: imp_charge_density ! impurity charge density in units of [e]

    !> impurity collision
    integer(kind=1) :: q_b
    integer, parameter :: n_coll=20
    real*8    :: ran(6), ran2(6,n_coll), q(3), m_b
    real*8    :: coulomb_log, kTb, n_b, v_b(3,n_coll) 
    real*8, dimension(1) :: P, P_s, P_t, P_phi, P_time

    !> System variables ------------------------------
    type(particle_kinetic_leapfrog) :: particle_tmp

    real*8    :: n_norm, rho_norm, t_norm, v_norm, E_norm, M_norm
    real*8    :: t, E(3), B(3), psi, U,n_i, n_e, T_e, grad_T_e(3), rz_old(2), st_old(2)
    real*8    :: Z_g, R_s, R_t, Z_s, Z_t, xjac, R, Z
    real*8    :: HZ(n_tor), HH(4,4), HH_s(4,4), HH_t(4,4)

    integer   :: i, j, k, l, m, n, i_elm_old, i_elm, q_old 
    integer   :: seed, i_rng, n_stream, ierr, nthreads
    integer   :: i_tor, i_elm_temp
    integer   :: n_particles, ifail 
    integer   :: imp_q_idx_temp

    !> Initialise diagnostics to 0
    n_lost_ion     = 0.d0; n_lost_ion_all = 0.d0; p_lost_ion     = 0.d0; p_lost_ion_all  = 0.d0; p_lost_plt          = 0.d0; 
    p_lost_plt_all = 0.d0; p_lost_cx      = 0.d0; p_lost_cx_all  = 0.d0; n_super_ionized = 0;    n_super_ionized_all = 0;

    !> Normalise variables 
    n_norm   = CENTRAL_DENSITY * 1.d20                              ! (number) density normalisation
    rho_norm = CENTRAL_MASS * MASS_PROTON * n_norm                  ! rho_SI = rho_norm * rho
    t_norm   = sqrt((MU_ZERO * rho_norm))                           ! t_SI   = t_norm * t_jorek
    v_norm   = 1.d0 / t_norm                                        ! V_SI   = v_norm * v_jorek
    E_norm   = 1.5d0 / MU_ZERO                                      ! E_SI   = E_norm * E_jorek
    M_norm   = rho_norm * v_norm                                    ! momentum normalisation

    if(use_manual_random_seed) then
      !$ call omp_set_schedule(omp_sched_static,10)
    else
      !$ call omp_set_schedule(omp_sched_dynamic,10)
    end if  
#ifdef __GFORTRAN__
    !$omp parallel do default(shared) & ! workaround for Error: �__vtab_mod_pcg32_rng_Pcg32_rng� not specified in enclosing �parallel�
#else
    !$omp parallel do default(none)                                                                       &
#endif
    !$omp schedule(runtime)                                                                               &
    !$omp shared(sim, group_num, nstep_particles, tstep_part_adj, rng,                                    &
    !$omp rho_norm, t_norm, v_norm, E_norm, M_norm, N_norm,                                               &    
    !$omp rho_idx_kin, mom_par_idx_kin, E_idx_kin, imp_q_idx, ics_indices_kin,                            &
    !$omp CENTRAL_DENSITY, CENTRAL_MASS, feedback_nodelist, feedback_element_list)                        &
    !$omp private(particle_tmp, i_rng, i, j, k, l, m, t, E, B, psi, U, rz_old, st_old,                    &
    !$omp i_elm_old, i_elm, n_i, n_e, T_e,imp_charge_density, PLT, PRB, Srec, grad_T_e, q_old,            &
    !$omp ionize_rate, ionize_prob, ionize_ran, ionize_ran_imp, ionize_source, ionize_energy,             &
    !$omp cx_rate, cx_prob, cx_source, cx_energy, cx_ran,                                                 &
    !$omp kinetic_energy, line_rad_energy, radiation_energy, binding_energy,                              &  
    !$omp R_s, R_t, Z_g, Z_s, Z_t, R, Z, xjac, HH, HH_s, HH_t, HZ, ifail,                                 &
    !$omp density_fb, E_fb, mom_par_fb,extra_proj, imp_q_fb, imp_density_fb, imp_P_rad_fb,                &
    !$omp density_source, mom_par_source, energy_source, v_temp,                                          &
    !$omp m_b, kTb,coulomb_log ,n_b,v_b, ran, ran2, q_b, q,                                               &
    !$omp P, P_s, P_t, P_phi, P_time, limits, limits_coll,                                                &
    !$omp vvector, ran_norm, imp_q_idx_temp)                                                              &
    !$omp reduction(+:feedback_rhs,n_lost_ion,p_lost_plt,p_lost_cx,p_lost_ion,n_super_ionized)
    do j=1,size(sim%groups(group_num)%particles,1)
      particle_tmp = sim%groups(group_num)%particles(j)
      !$ i_rng = omp_get_thread_num()+1
      do k=1,nstep_particles

        !> exit evolution loop if particle is outside domain
        if (particle_tmp%i_elm .le. 0) exit
        
        t = sim%time + (k-1)*tstep_part_adj

        ionize_source = 0.d0
        ionize_energy = 0.d0
        line_rad_energy = 0.d0
        radiation_energy = 0.d0
        cx_source = 0.d0
        cx_energy = 0.d0

        !> calculate local fields
        call sim%fields%calc_EBpsiU(t, particle_tmp%i_elm, particle_tmp%st, particle_tmp%x(3), E, B, psi, U)
        rz_old    = particle_tmp%x(1:2)
        st_old    = particle_tmp%st
        i_elm_old = particle_tmp%i_elm
        q_old     = particle_tmp%q 
        v_temp    = particle_tmp%v
        
        !> calculate ion density and electron temperature (jorek model assumption: n_e = n_i)
        call sim%fields%calc_NeTe(t, particle_tmp%i_elm, particle_tmp%st, particle_tmp%x(3), n_i, T_e, grad_T_e)

        !> loop over impurities groups and calculate their contribution to electron density
        imp_charge_density = 0.d0
        do n=1, size(sim%groups)
          if (trim(sim%groups(n)%coupling_scheme) == 'ics') then
            imp_q_idx_temp = ics_indices_kin(sim%groups(n)%ics_group_idx)
            call interp_0(feedback_nodelist, feedback_element_list, particle_tmp%i_elm, [imp_q_idx_temp], 1 , particle_tmp%st(1), particle_tmp%st(2), particle_tmp%x(3), P)
            imp_charge_density = imp_charge_density + P(1) ! charge density of impurities in units of [e]
          endif
        enddo
        
        !> adjust n_e based on impurity charge
        n_e = n_i + max(0.d0, imp_charge_density)
        
        limits = n_e .le. 1e14 .or. T_e * K_BOLTZ / EL_CHG .le. 1.d0 !ADAS limits
        
        !> check that particle weight is non negative
        if (particle_tmp%weight .lt. 0.0d0) write(*,*) "Negative particle weight p(j)%w=", particle_tmp%weight
        
        ! ============================================ NCS SPECIFIC PHYSICS ===========================================
        if (sim%groups(group_num)%coupling_scheme == 'ncs') then
          !> calculate fluid flow velocity [v_R, v_Z, v_phi] m/s
          call sim%fields%calc_vvector(t, particle_tmp%i_elm, particle_tmp%st, particle_tmp%x(3), vvector)

          !> RADIATION (only line radiation* for ncs)
          !>   (*Line radiation due to collisional excitation of the neutral's bound electrons with the 
          !>    electrons in the background plasma. The spectrum of these radiations are discrete.)
          if (sim%groups(group_num)%use_kin_radiation .and. .not. limits) then
            call sim%groups(group_num)%ad%PLT%interp(int(particle_tmp%q), log10(n_e), log10(T_e), PLT) ! [J m^3/s]
            line_rad_energy = n_e * particle_tmp%weight * PLT * tstep_part_adj
          endif ! RADIATION
          
          !> IONISATION (Neutrals)
          if (sim%groups(group_num)%use_kin_ionisation .and. .not. limits) then
            call sim%groups(group_num)%ad%SCD%interp(int(particle_tmp%q), log10(n_e), log10(T_e), ionize_rate) ! [m^3/s]
            ionize_prob = 1.d0 - exp(-ionize_rate * n_e * tstep_part_adj) ! [0] poisson point process, exponential 

            ! If the weight is to small throw away the particle with the probability, else reduce weight with ionising probability
            ionize_source = 0.d0
    
            if (particle_tmp%weight .le. 1.0d9) then
              call rng(i_rng)%next(ionize_ran)
              if (ionize_ran(1) .le. ionize_prob) then
                particle_tmp%i_elm  = 0
                ionize_source = particle_tmp%weight
                !superparticles ionized
                n_super_ionized = n_super_ionized +1
              else
                ionize_source = 0.d0
              endif
            else 
              ionize_source = particle_tmp%weight * ionize_prob
              particle_tmp%weight = particle_tmp%weight * (1.d0 - ionize_prob)
            endif 
    
            kinetic_energy = dot_product(particle_tmp%v,particle_tmp%v) * sim%groups(group_num)%mass * ATOMIC_MASS_UNIT /2.d0
            ionize_energy     = kinetic_energy - H_binding_energy
            !<including binding energy will make ionize_energy negative, so it becomes a sink for the plasma
          endif ! IONISATION
            
          !> CHARGE EXCHANGE
          ! It is assumed that we will have a exchange between hydrogen isotopes
          if (sim%groups(group_num)%use_kin_cx  .and. .not. limits) then !< CX uses adas as well. Te limit could be lower.
            call sim%groups(group_num)%ad%CCD%interp(int(particle_tmp%q+1), log10(n_e), log10(T_e), cx_rate) ! [m^3/s]
            CX_prob = 1.d0 - exp(-cx_rate * n_e * tstep_part_adj)
    
            call rng(i_rng)%next(cx_ran)
            if (cx_ran(1) .le. CX_prob) then
              !> ----- NEW CX PARTICLE ---------
              !Box-Mueller sample velocities with st.dev=1
              ran_norm = boxmueller_transform(cx_ran(2:5))
              !>v_temp = sqrt(kT/m) * ran_norm
              v_temp = sqrt(T_e * K_BOLTZ/(sim%groups(group_num)%mass * ATOMIC_MASS_UNIT))*ran_norm(2:4)
              !>add bulk fluid flow
              v_temp = v_temp + vvector 

              CX_source = particle_tmp%weight
              CX_energy   = 0.5d0 * sim%groups(group_num)%mass * ATOMIC_MASS_UNIT *  (dot_product(particle_tmp%v,particle_tmp%v) - dot_product(v_temp,v_temp))
            endif ! cx_ran
          endif ! CHARGE EXCHANGE
            
          !> check that the energy feedback is valid
          if ((ionize_source * ionize_energy + cx_source * cx_energy - line_rad_energy) /= (ionize_source * ionize_energy + cx_source * cx_energy - line_rad_energy)) then
            write(*,*) "ionize_energy", ionize_energy
            write(*,*) "cx_energy", cx_energy
            write(*,*) "line_rad_energy", line_rad_energy
            particle_tmp%i_elm  = 0
            CYCLE !< don't feed this particle into the feedback
          endif

          !> ----- CONSTRUCT FEEDBACK -----
          !> the feedback per particle per time step is accumulated which is then divided by gather time later
          density_source = ionize_source * sim%groups(group_num)%mass * ATOMIC_MASS_UNIT !< mass source in SI
          mom_par_source = ionize_source * dot_product(B, particle_tmp%v) * sim%groups(group_num)%mass * ATOMIC_MASS_UNIT &	
                + CX_source  * dot_product(B, particle_tmp%v - v_temp) * sim%groups(group_num)%mass * ATOMIC_MASS_UNIT 
          energy_source  = ionize_source * ionize_energy + cx_source * cx_energy - line_rad_energy
                  
          particle_tmp%v = v_temp 
          n_lost_ion = n_lost_ion + ionize_source	!< local sum #particles lost due to ionisation
          p_lost_ion = p_lost_ion + ionize_source * ionize_energy
          p_lost_plt = p_lost_plt + line_rad_energy
          p_lost_cx  = p_lost_cx + cx_source * cx_energy

          !> Calculate the projection of the ion source in real-time
          call basisfunctions(particle_tmp%st(1), particle_tmp%st(2), HH, HH_s, HH_t)
          call mode_moivre(particle_tmp%x(3), HZ)
                
          do l=1,n_vertex_max
            do m=1,n_order+1
  
              density_fb = HH(l,m) * sim%fields%element_list%element(i_elm_old)%size(l,m) * density_source     * t_norm / rho_norm
              mom_par_fb = HH(l,m) * sim%fields%element_list%element(i_elm_old)%size(l,m) * mom_par_source * t_norm / m_norm
              E_fb       = HH(l,m) * sim%fields%element_list%element(i_elm_old)%size(l,m) * energy_source       * t_norm / E_norm
              extra_proj = HH(l,m) * sim%fields%element_list%element(i_elm_old)%size(l,m) *particle_tmp%weight * 1.d0/real(nstep_particles,8) 
  
              do i_tor=1,n_tor
                feedback_rhs(m,l,i_elm_old,i_tor,rho_idx_kin) = feedback_rhs(m,l,i_elm_old,i_tor,rho_idx_kin) + HZ(i_tor) * density_fb
                feedback_rhs(m,l,i_elm_old,i_tor,E_idx_kin) = feedback_rhs(m,l,i_elm_old,i_tor,E_idx_kin) + HZ(i_tor) * E_fb
                feedback_rhs(m,l,i_elm_old,i_tor,mom_par_idx_kin) = feedback_rhs(m,l,i_elm_old,i_tor,mom_par_idx_kin) + HZ(i_tor) * mom_par_fb
                feedback_rhs(m,l,i_elm_old,i_tor,5) = feedback_rhs(m,l,i_elm_old,i_tor,5) + HZ(i_tor) * extra_proj
              enddo
            enddo
          enddo

        endif ! END OF NCS SPECIFIC PHYSICS

        ! ============================================ ICS SPECIFIC PHYSICS ===========================================

        if (sim%groups(group_num)%coupling_scheme == 'ics') then
          limits_coll = T_e > 0.d0 !< limits for collisions

          !> IONISATION & RECOMBINATION (Impurities)
          if (sim%groups(group_num)%use_kin_ionisation .and. .not. limits) then
            call rng(i_rng)%next(ionize_ran_imp)

            !> determines the new charge of the particle using ionisation/recombination rate coefficients
            particle_tmp%q = int(new_charge(int(q_old,4), sim%groups(group_num)%ad, log10(n_e), log10(T_e), tstep_part_adj, ionize_ran_imp(1:2)),1)
            
            if (particle_tmp%q .gt. q_old) then
              binding_energy = sim%groups(group_num)%ad%ionisation_energy(particle_tmp%q +1) * EL_CHG ! should this be q or q_old?
              ionize_energy     =  - binding_energy * particle_tmp%weight
              !< including binding energy will make ionize_energy negative, so it becomes a sink for the plasma
            endif
          endif ! IONISATION

          !> RADIATION (Line radiation + Bremsstrahlung + Recombination radiation)
          if (sim%groups(group_num)%use_kin_radiation .and. .not. limits) then !< before or after Ionisation and CX ??
            call sim%groups(group_num)%ad%PLT%interp(int(particle_tmp%q), log10(n_e), log10(T_e), PLT)  ! [J m^3/s] Line radiation
            call sim%groups(group_num)%ad%PRB%interp(int(particle_tmp%q), log10(n_e), log10(T_e), PRB)  ! [J m^3/s] Bremsstrahlung
            call sim%groups(group_num)%ad%ACD%interp(int(particle_tmp%q), log10(n_e), log10(T_e), Srec) ! [J m^3/s] Recomb radiation 
            binding_energy = sim%groups(group_num)%ad%ionisation_energy(particle_tmp%q+1) * EL_CHG ! should this be q or q_old?
            radiation_energy = - n_e * particle_tmp%weight * (PLT +PRB-Srec*binding_energy)* tstep_part_adj
          endif ! RADIATION
          
          !> COLLISIONS WITH THE BACKGROUND PLASMA (Neoclassical collisions)
          if (sim%groups(group_num)%use_kin_bg_collisions .and. .not. limits_coll) then
            if (particle_tmp%q .gt. 0) then
              ! Calculate collisions
              kTb = T_e*K_BOLTZ !/EL_CHG ! assume T_e == T_i
              n_b = n_e
              q_b = 1
              m_b = 2.d0
              !> Homma use temperature in [J] (kb [j/K]* T_e [K] or e [J/eV] * Te_eV [eV])
              q = q_homma2013(kTb, grad_T_e*K_BOLTZ, B, n_b, m_b, q_b) !EL_CHG/K_BOLTZ

              !> Calculate coulomb logarithm and limit it to reasonable values
              coulomb_log = coulomb_logarithm(kTb, n_b, particle_tmp%q, q_b, sim%groups(group_num)%mass, m_b)
              coulomb_log = max(10.d0, coulomb_log)
              coulomb_log = min(20.d0, coulomb_log)

              !> Get parallel flow velocity
              call sim%fields%interp_PRZ(t, particle_tmp%i_elm, [var_Vpar], 1, particle_tmp%st(1), particle_tmp%st(2), &
                  particle_tmp%x(3), P, P_s, P_t, P_phi, P_time, R, R_s, R_t, Z, Z_s, Z_t)
              
              do l=1,n_coll
                call rng(i_rng)%next(ran2(:,l))
              end do

              call sample_velocity_dist_magnetized(n_coll, ran2(1:6,:), kTb, q, n_b, m_b, q_b, P(1)*B/norm2(B)/sim%t_norm, v_b) !P(1)*B/norm2(B)/sim%t_norm
  
              do l=1,n_coll
                call rng(i_rng)%next(ran)
                call collide_particles(ran(1:3), particle_tmp%q, sim%groups(group_num)%mass, particle_tmp%v, &
                    q_b, m_b, v_b(:,l), n_b, coulomb_log, tstep_part_adj/real(n_coll,8))
              end do
            end if
          endif ! COLLISIONS
          
          !> check that the particle energy sources are valid
          if ((imp_charge_density + ionize_energy - radiation_energy) /= (imp_charge_density + ionize_energy - radiation_energy)) then
            write(*,*) "imp_charge_density", imp_charge_density
            write(*,*) "ionize_energy", ionize_energy
            write(*,*) "rad_energy", radiation_energy
            particle_tmp%i_elm  = 0
            CYCLE !< don't feed this particle into the feedback
          endif
      
          !> ----- CONSTRUCT FEEDBACK -----
          !> the feedback per particle per time step is accumulated which is then divided by gather time later
          energy_source  = ionize_energy + radiation_energy
          mom_par_source = -1.d0 * particle_tmp%weight * dot_product(B, particle_tmp%v-v_temp) * sim%groups(group_num)%mass * ATOMIC_MASS_UNIT !&	

          particle_tmp%v = v_temp 
          n_lost_ion = n_lost_ion
          p_lost_ion = p_lost_ion + ionize_energy
          p_lost_plt = p_lost_plt + radiation_energy
          p_lost_cx  = p_lost_cx + cx_source * cx_energy

          !> Calculate the projection of the ion source in real-time
          call basisfunctions(particle_tmp%st(1), particle_tmp%st(2), HH, HH_s, HH_t)
          call mode_moivre(particle_tmp%x(3), HZ)
          
          do l=1,n_vertex_max
            do m=1,n_order+1
  
              mom_par_fb     = HH(l,m) * sim%fields%element_list%element(i_elm_old)%size(l,m) * mom_par_source * t_norm / m_norm
              E_fb           = HH(l,m) * sim%fields%element_list%element(i_elm_old)%size(l,m) * energy_source  * t_norm / E_norm
              imp_q_fb       = HH(l,m) * sim%fields%element_list%element(i_elm_old)%size(l,m) * particle_tmp%weight * particle_tmp%q /real(nstep_particles,8)
              imp_density_fb = HH(l,m) * sim%fields%element_list%element(i_elm_old)%size(l,m) * particle_tmp%weight /real(nstep_particles,8)
              imp_P_rad_fb   = HH(l,m) * sim%fields%element_list%element(i_elm_old)%size(l,m) * radiation_energy / tstep_part_adj
              
              do i_tor=1,n_tor
                feedback_rhs(m,l,i_elm_old,i_tor,E_idx_kin) = feedback_rhs(m,l,i_elm_old,i_tor,E_idx_kin) + HZ(i_tor) * E_fb
                feedback_rhs(m,l,i_elm_old,i_tor,mom_par_idx_kin) = feedback_rhs(m,l,i_elm_old,i_tor,mom_par_idx_kin) + HZ(i_tor) * mom_par_fb
                feedback_rhs(m,l,i_elm_old,i_tor,imp_q_idx) = feedback_rhs(m,l,i_elm_old,i_tor,imp_q_idx) + HZ(i_tor) * imp_q_fb ! impurity charge density
                feedback_rhs(m,l,i_elm_old,i_tor,6) = feedback_rhs(m,l,i_elm_old,i_tor,6) + HZ(i_tor) * imp_P_rad_fb             ! impurity radiated power [to be moved to diag feedback]
                feedback_rhs(m,l,i_elm_old,i_tor,7) = feedback_rhs(m,l,i_elm_old,i_tor,7) + HZ(i_tor) * imp_density_fb           ! impurity density [to be moved to diag feedback]
              enddo
            enddo
          enddo

        endif ! END OF ICS SPECIFIC PHYSICS

        !> =============================== PUSH PARTICLE ====================================
        if (particle_tmp%i_elm .gt. 0) then
          call boris_push_cylindrical(particle_tmp, sim%groups(group_num)%mass, E, B, tstep_part_adj)
          call find_RZ_nearby(sim%fields%node_list, sim%fields%element_list, rz_old(1), rz_old(2), st_old(1), st_old(2), i_elm_old, &
                              particle_tmp%x(1), particle_tmp%x(2), particle_tmp%st(1), particle_tmp%st(2), particle_tmp%i_elm, ifail)
        end if
    
      end do ! steps 

      sim%groups(group_num)%particles(j) = particle_tmp
    end do   ! particles
    !$omp end parallel do
      
    !> ===================================== DIAGNOSTICS ===================================
    call MPI_REDUCE(n_lost_ion, n_lost_ion_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    call MPI_REDUCE(p_lost_ion, p_lost_ion_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    call MPI_REDUCE(p_lost_plt, p_lost_plt_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    call MPI_REDUCE(p_lost_cx, p_lost_cx_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    call MPI_REDUCE(n_super_ionized, n_super_ionized_all, 1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    
    if (sim%my_id .eq. 0) write(*,'(A46,E14.6,I6)') "Lost superparticles at t due to ionisation: ", sim%time, n_super_ionized_all
    if (sim%my_id .eq. 0) write(*,'(A46,2E14.6)') "Lost particles at t due to ionisation: ", sim%time, n_lost_ion_all
    p_lost_ion_all = p_lost_ion_all / (tstep_part_adj * nstep_particles)
    p_lost_plt_all = p_lost_plt_all / (tstep_part_adj * nstep_particles)
    p_lost_cx_all  = p_lost_cx_all / (tstep_part_adj * nstep_particles)
    if (sim%my_id .eq. 0) write(*,'(A46,2E14.6)') "Lost energy [W] at t due to ionisation: ", sim%time, p_lost_ion_all
    if (sim%my_id .eq. 0) write(*,'(A46,2E14.6)') "Lost energy [W] at t due to radiation: ", sim%time, p_lost_plt_all
    if (sim%my_id .eq. 0) write(*,'(A46,2E14.6)') "Lost energy [W] at t due to CX radiation: ", sim%time, p_lost_cx_all
    if (sim%my_id .eq. 0) write(*,'(A46,2E14.6)') "Total energy exchange to plasma [W]: ", sim%time, p_lost_ion_all -p_lost_plt_all+ p_lost_cx_all
  end subroutine evolve_ncs_ics

  subroutine gpu_test(sim, group_num, feedback_rhs)
    use hdf5_io_module
    use mpi

    implicit none
    class(particle_sim), target, intent(inout)                :: sim
    integer, intent(in)                                       :: group_num
    real*8,allocatable,          intent(inout)                :: feedback_rhs(:,:,:,:,:)
    real*8, allocatable :: feedback_rhs_linear(:)
    integer(HID_T) fid
    integer :: ierr
    integer :: particle_lost, particle_lost_all
    integer :: j, m, n, k

    !> ------------------TESTING---------------------
    particle_lost = 0
    do j=1,size(sim%groups(group_num)%particles,1)
      if (sim%groups(group_num)%particles(j)%i_elm .le. 0) then
        particle_lost = particle_lost + 1
      endif
    enddo    
    call MPI_REDUCE(particle_lost, particle_lost_all, 1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    if (sim%my_id .eq. 0) then
      write(*,'(A,I0)') "INFO: number of particles lost in group : ", particle_lost_all
    endif

    particle_lost = 0
    do j=1,size(sim%groups(group_num)%particles,1)
      if (sim%groups(group_num)%particles(j)%i_elm .eq. -1) then
        particle_lost = particle_lost + 1
      endif
    enddo
    call MPI_REDUCE(particle_lost, particle_lost_all, 1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    if (sim%my_id .eq. 0) then
      write(*,'(A,I0)') "INFO: number of particles lost in group NUMBER_OF_ITER : ", particle_lost_all
    endif

    particle_lost = 0
    do j=1,size(sim%groups(group_num)%particles,1)
      if (sim%groups(group_num)%particles(j)%i_elm .eq. -2) then
        particle_lost = particle_lost + 1
      endif
    enddo
    call MPI_REDUCE(particle_lost, particle_lost_all, 1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    if (sim%my_id .eq. 0) then
      write(*,'(A,I0)') "INFO: number of particles lost in group NAN : ", particle_lost_all
    endif

    particle_lost = 0
    do j=1,size(sim%groups(group_num)%particles,1)
      if (sim%groups(group_num)%particles(j)%i_elm .eq. -3) then
        particle_lost = particle_lost + 1
      endif
    enddo
    call MPI_REDUCE(particle_lost, particle_lost_all, 1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    if (sim%my_id .eq. 0) then
      write(*,'(A,I0)') "INFO: number of particles lost in an element crossing the axis: ", particle_lost_all
    endif
  end subroutine gpu_test

end module mod_particle_evolution