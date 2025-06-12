!> Module containing a datatype for simulation parameters
module mod_particle_sim
use mod_particle_types
use mod_fields
use mod_openadas
use mod_coronal
use basis_at_gaussian
implicit none
private
public particle_group, particle_sim, configure_particle_groups
public group_num_from_id, config_num_from_id

!> A group of particles, implemented as an allocatable array.
!> It must contain particles of the same species (charge number).
type :: particle_group
  integer            :: Z                                          !< Atomic number of al particles in the group (-1 for electrons, 0 for fieldline-following)
  real*8             :: mass                                       !< Mass of all the particles in the group
  type(ADF11_all)    :: ad                                         !< OPEN-ADAS datafiles for this species
  type(coronal)      :: cor                                        !< (coronal) equilibrium pre-calculation
  real*8             :: dt                                         !< timestep (if fixed for all particles in this group)
  character(len=3)   :: coupling_scheme                            !< coupling scheme to use for the group
  real*8             :: n_particles                                !< number of super/marker particles in group
  character(len=3)   :: id                                         !< unique identifier for the group (mainly used when restarting)
 
  ! ================ for neutrals and impurities =============
  logical            :: use_kin_ionisation       !< switch on ionisation for group         
  logical            :: use_kin_puffing          !< switch on particle puffing for group
  logical            :: use_kin_radiation        !< switch on line radiation for group

  ! --- neutrals only
  logical            :: use_kin_cx               !< switch on charge-exchange for group  
  logical            :: use_kin_recombination    !< switch on recombination for group       

  ! --- impurities only
  logical            :: use_kin_bg_collisions    !< switch only collisions with the background plasma
  integer            :: ics_group_idx            !< internal index given to this specific impurities group

  ! ==================== for runaway electrons =================
  real*8             :: av_weight                !< average weight of each super particle in the group
  integer            :: q                        !< charge of each runaway electron super particle

  class(particle_base), dimension(:), allocatable :: particles

end type particle_group

!> Particle simulation type, containing all variables pertaining to a simulation.
type :: particle_sim
  real*8                                          :: time = 0.d0 !< time of the simulation. Only accurate when in events with sync or at
  !< the start of the simulation
  class(fields_base), allocatable                 :: fields
  logical                                         :: stop_now = .false.
  real*8                                          :: t_norm !< JOREK normalisation factor
  type(particle_group), dimension(:), allocatable :: groups
  !< MPI settings
  integer :: my_id = 0
  integer :: n_mpi = 1 ! if not initialized, act as if there is no mpi
  real*8  :: wtime_start !< Clock time at the start of the program
contains
  procedure,pass(sim) :: finalize
  procedure,pass(sim) :: initialize
  procedure,pass(sim) :: set_t_norm  !< set the jorek time unit
  procedure,pass(sim) :: allocate_groups
  procedure,pass(sim) :: compute_group_size
  procedure,pass(sim) :: compute_particle_sizes
  procedure,pass(sim) :: find_particle_types
  procedure,pass(sim) :: find_active_particles_groups
end type particle_sim

contains

!> Loads the information from a type_part_group_config type to a particle_group type
subroutine configure_particle_groups(sim)
  use phys_module, only: n_part_groups, part_group_configs, type_part_group_config
  use phys_module, only: part_groups_in_use
  use mod_particle_group_id, only: matching_part_config_indices, matching_sim_groups_indices

  implicit none
  class(particle_sim), intent(inout)       :: sim
  integer                                  :: i,j
  type(type_part_group_config)             :: config

  do i=1, n_part_groups ! loop over groups defined in part_groups_in_use
    config = part_group_configs(matching_part_config_indices(i))

    sim%groups(i)%coupling_scheme = config%coupling_scheme

    sim%groups(i)%Z = config%Z
    sim%groups(i)%mass = config%mass
    sim%groups(i)%n_particles = config%n_particles
    sim%groups(i)%id = config%id

    ! === ncs and ics options
    if (sim%groups(i)%coupling_scheme == 'ncs' .or. sim%groups(i)%coupling_scheme == 'ics') then
        sim%groups(i)%use_kin_ionisation     =  config%use_kin_ionisation          
        sim%groups(i)%use_kin_puffing        =  config%use_kin_puffing        
        sim%groups(i)%use_kin_radiation      =  config%use_kin_radiation 

      ! --- ncs only
      sim%groups(i)%use_kin_cx             =  config%use_kin_cx
      sim%groups(i)%use_kin_recombination  =  config%use_kin_recombination         

      ! --- ics only
      sim%groups(i)%use_kin_bg_collisions  =  config%use_kin_bg_collisions
      sim%groups(i)%ics_group_idx          =  config%ics_group_idx
      
      if (len_trim(config%atom_data_suffix) > 0) then
        sim%groups(i)%ad =  read_adf11(sim%my_id, trim(part_group_configs(i)%atom_data_suffix))
      else
        if (trim(config%coupling_scheme) == 'ncs') write(*,*) "WARNING: No atom_data_suffix set for particle group ", i, "."
      endif
    endif

    ! === rep options
    if (sim%groups(i)%coupling_scheme == 'rep') then
      sim%groups(i)%av_weight  =  config%num_re / config%n_particles
      sim%groups(i)%q          =  -1 !< default charge of the runaway electrons
    endif

  enddo 

end subroutine configure_particle_groups

!> Actions to perform when setting up a simulation
!> inputs:
!>   sim:             (particle_sim) the particle simulation
!>   num_groups:      (integer) number of particle groups
!>   skip_jorek2help: (logical)(optional) call jorek2help if present
!>   my_id:           (integer)(optional) mpi rank
!>   n_mpi:           (integer)(optional) number of mpi tasks in the commworld
!> outputs:
!>   sim: (particle_sim) the particle simulation
subroutine initialize(sim,skip_jorek2help,my_id,n_mpi,do_jorek_init_in,skip_group_config)
  use mod_mpi_tools,     only: init_mpi_threads
  use mod_mpi_tools,     only: get_mpi_wtime
  use mod_parameters,    only: n_tor, n_period
  use phys_module,       only: mode, domm
  use basis_at_gaussian, only: initialise_basis
  use mod_chi,           only: init_chi_basis
  use data_structure,    only: init_threads, nbthreads
  use phys_module,       only: n_part_groups, n_part_groups_max
  !$ use omp_lib
  class(particle_sim), intent(inout) :: sim
  logical,intent(in), optional       :: skip_jorek2help,do_jorek_init_in,skip_group_config
  integer,intent(in),optional        :: my_id,n_mpi
  logical                            :: do_jorek_init
  integer                            :: ierr, i_tor,nthreads, group_num

  !> initialise the mpi comm world with threads if required
  if(present(my_id).and.present(n_mpi)) then
    sim%my_id = my_id; sim%n_mpi = n_mpi;
    sim%wtime_start = get_mpi_wtime()
  else
    call init_mpi_threads(sim%my_id,sim%n_mpi,ierr,sim%wtime_start)
  endif

 !> check if the initialisation of JOREK should be performed or not
  do_jorek_init = .true.
  if(present(do_jorek_init_in)) do_jorek_init = do_jorek_init_in  
  if(do_jorek_init) then 
    !> perform the initialisation if requried
    call init_threads()

    if (present(skip_jorek2help)) then
      if (sim%my_id .eq. 0 .and. .not. skip_jorek2help) call jorek2help(sim%n_mpi, nbthreads)
    end if

    ! Initialise mode numbers
    call det_modes()

    ! Initialise and broadcast parameters 
    call initialise_and_broadcast_parameters(sim%my_id, "__NO_FILENAME__", .true.)

    ! Set up normalisation factors
    call sim%set_t_norm()

    ! Initialise the gaussian points at basis functions
    call initialise_basis

    ! --- Initialize basis functions for the Dommaschk potentials
    if (domm) call init_chi_basis()
  endif

  ! Allocating groups
  call sim%allocate_groups(n_part_groups)

  ! configure particle groups with their characteristics
  if (.not. present(skip_group_config)) then 
    call configure_particle_groups(sim)
  else
    if (.not. skip_group_config) call configure_particle_groups(sim)
  endif

end subroutine

!> Actions to perform when stopping the simulation.
subroutine finalize(sim)
  use mod_mpi_tools, only: finalize_mpi_threads
  use mod_startup_teardown, only: jorek_finalize => finalize
  class(particle_sim), intent(in) :: sim
  integer :: ierr
  if (sim%stop_now) then
    write(*,"(A,g14.6,A)") "INFO: Stop requested at ", sim%time, " , exiting"
  else
    write(*,"(A,g14.6,A)") "INFO: End of events at ", sim%time, " , exiting"
  end if
  call finalize_mpi_threads(ierr)
end subroutine

!> set the t_norm value
!> inputs:
!>   sim: (particle_sim) the particle simulation
!> outputs:
!>   sim: (particle_sim) the particle simulation
subroutine set_t_norm(sim)
  use phys_module, only: central_mass, central_density
  use constants, only: MU_ZERO, MASS_PROTON
  implicit none
  ! input-outputs
  class(particle_sim), intent(inout) :: sim
  sim%t_norm = sqrt(MU_ZERO * central_mass * MASS_PROTON * central_density * 1.d20)
end subroutine set_t_norm

!> this function returns the size of the particle group
function compute_group_size(sim) result(n_groups)
  implicit none
  class(particle_sim),intent(inout) :: sim
  integer :: n_groups
  n_groups = size(sim%groups)
end function compute_group_size

!> return the particle sizes
subroutine compute_particle_sizes(sim,n_groups_in,n_particles)
  implicit none
  class(particle_sim),intent(inout) :: sim
  integer,intent(inout) :: n_groups_in
  integer,dimension(n_groups_in),intent(out) :: n_particles
  integer :: ii,n_groups
  n_groups = min(sim%compute_group_size(),n_groups_in)
  n_particles = 0
  do ii=1,n_groups
    n_particles(ii) = size(sim%groups(ii)%particles)
  enddo
end subroutine compute_particle_sizes

!> allocate groups, if allocated, deallocate groups first
!> except is there is not changes in the group size
subroutine allocate_groups(sim,n_groups)
  implicit none
  !> inputs-outpus
  class(particle_sim),intent(inout) :: sim
  !> inputs
  integer,intent(in) :: n_groups
  if(.not.allocated(sim%groups)) then
    allocate(sim%groups(n_groups))
  elseif(size(sim%groups).ne.n_groups) then
    deallocate(sim%groups); allocate(sim%groups(n_groups));
  endif
end subroutine allocate_groups

!> return the codified particle type of the particle list
!> Codification:
!>   0 -> default
!>   1 -> particle_fieldline
!>   2 -> particle_gc
!>   3 -> particle_gc_vpar
!>   4 -> particle_gc_Qin
!>   5 -> particle_kinetic
!>   6 -> particle_kinetic_leapfrog
!>   7 -> particle_realtivistic_kinetic
!>   8 -> particle_gc_relativistic
subroutine find_particle_types(sim,n_groups_in,p_types)
  use mod_particle_types, only: codify_particle_type
  implicit none
  class(particle_sim),intent(inout) :: sim
  integer,intent(in)                :: n_groups_in
  integer,dimension(n_groups_in),intent(out) :: p_types
  integer :: ii,n_groups
  n_groups = min(sim%compute_group_size(),n_groups_in)
  do ii=1,n_groups
    p_types(ii) = codify_particle_type(sim%groups(ii)%particles)
  enddo
end subroutine find_particle_types

!> returns number and ids of active particles for all groups
!> and for a specific type of particle (encoded)
subroutine find_active_particles_groups(sim,n_groups,n_particles_max,&
n_particles,n_active_particles,active_particle_id,n_p_type,p_type)
  use mod_particle_types, only: find_active_particle_id
  implicit none
  !> inputs-outputs
  class(particle_sim),intent(inout) :: sim
  !> inputs
  integer,intent(in)                       :: n_groups,n_particles_max
  integer,dimension(n_groups),intent(in)   :: n_particles
  integer,intent(in),optional              :: n_p_type
  integer,dimension(:),intent(in),optional :: p_type
  !> outputs
  integer,dimension(n_groups),intent(out) :: n_active_particles
  integer,dimension(n_particles_max,n_groups),intent(out) :: active_particle_id
  !> variables
  integer :: ii,jj
  n_active_particles = 0; active_particle_id = 0;
  if(present(n_p_type).and.present(p_type)) then
    if(size(p_type).eq.n_p_type) then
      do ii=1,n_groups
        do jj=1,n_p_type
          call find_active_particle_id(p_type(jj),n_particles(ii),&
          sim%groups(ii)%particles,n_active_particles(ii),&
          active_particle_id(1:n_particles(ii),ii))
          !> the particle list of each group contains particles
          !> of only one type hence if an active particle is found
          !> the search for th iith group can be stopped
          if(n_active_particles(ii).gt.0) exit
        enddo
      enddo
    endif
  else
    do ii=1,n_groups
      call find_active_particle_id(n_particles(ii),&
      sim%groups(ii)%particles,n_active_particles(ii),&
      active_particle_id(1:n_particles(ii),ii))
    enddo
  endif
end subroutine find_active_particles_groups

!> returns the group_num which satisfies sim%groups(group_num)%id = id
function group_num_from_id(sim,id) result(group_num)
  implicit none

  type(particle_sim), intent(in) :: sim
  character(len=3),   intent(in) :: id !< particle group %id
  integer :: group_num !< the number sim%groups(group_num)
  integer :: i

  group_num = -1

  do i=1,size(sim%groups,1)
    if(sim%groups(i)%id == id) then ! matching id is found
      group_num = i
      return
    end if
  end do

  ! if at the end the matching id is not found, then the input id is not actually a valid used id in the sim
  if(sim%my_id == 0) write(*,"(3A)") "ERROR: id ",id," not found among sim%groups(:)%id (group_num_from_id)"

end function group_num_from_id

!> returns the group_num which satisfies part_group_configs(config_num)%id = id
!> note that this particle group does not necessarily have to be in use!
function config_num_from_id(id) result(config_num)
  use phys_module, only: part_group_configs
  
  implicit none
  
  character(len=3),   intent(in) :: id !< particle group %id
  integer :: config_num !< the number part_group_configs(config_num)%id
  integer :: i

  config_num = -1

  do i=1,size(part_group_configs,1)
    if(part_group_configs(i)%id == id) then ! matching id is found
      config_num = i
      return
    end if
  end do

  ! if at the end the matching id is not found, then the input id is not actually a valid used id in the sim
  write(*,"(3A)") "ERROR: id ",id," not found among part_group_configs(:)%id (config_num_from_id)"

end function config_num_from_id

end module mod_particle_sim
