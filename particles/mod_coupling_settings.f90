!> variables and functions related to settings for the coupling between kinetic particles and the fluid
module mod_coupling_settings
use mod_model_settings
use phys_module, only: n_part_groups, n_part_groups_max, part_group_configs
use phys_module, only: n_aux_var, n_diag_var
use coupling_variables

implicit none
private
public  :: use_ncs, use_ics, use_rep, use_epp, use_epc, use_epf, use_kin_recomb_global, n_ics
public  :: check_compatibility_and_determine_coupling_schemes, determine_coupling_variables

! the variables below are global variables determined by scanning over particle groups, 
! and hence shoud NOT be modified manually
logical :: use_ncs               = .false. !< use kinetic neutral particles 
logical :: use_ics               = .false. !< use kinetic impurity particles
logical :: use_rep               = .false. !< use pressure coupling scheme for runaway electrons
logical :: use_epc               = .false. !< use current coupling scheme for energetic particles                          [PLACEHOLDER, NOT YET IMPLEMENTED]
logical :: use_epp               = .false. !< use pressure coupling scheme for energetic particles                         [PLACEHOLDER, NOT YET IMPLEMENTED]
logical :: use_epf               = .false. !< use full anisotropic pressure tensor coupling scheme for energetic particles [PLACEHOLDER, NOT YET IMPLEMENTED]
logical :: use_kin_recomb_global = .false. !< whether recombination is required (has effect on both fluid and kinetic side)
integer :: n_ics                 = 0       !< number of ics groups in the simulation
contains

    
!> Scans over all the particle groups and 
!> - ensures that the physics settings enabled are compatible with the coupling scheme
!> - determines which coupling schemes are in use and hence how the coupling scheme 
!>   booleans should be initialized
subroutine check_compatibility_and_determine_coupling_schemes()
  implicit none
  integer                    ::   group_num

  do group_num=1, n_part_groups
    select case (part_group_configs(group_num)%coupling_scheme)
      case ('ncs')
        call check_compatibility_ncs(group_num)
        use_ncs = .true.
      case ('ics')
        call check_compatibility_ics(group_num)
        use_ics = .true.
        n_ics = n_ics + 1
        part_group_configs(group_num)%ics_group_idx = n_ics
      case ('rep')
        use_rep = .true.
      case ('non')
        
      case default
        write(*,*) "ERROR: The coupling scheme '", part_group_configs(group_num)%coupling_scheme, "' is invalid."
        stop
    end select

    if (part_group_configs(group_num)%use_kin_recombination .eqv. .true.) then
      use_kin_recomb_global = .true.
    endif 
    
  enddo 
end subroutine check_compatibility_and_determine_coupling_schemes

!> checks that the physics enabled for particle group is compatible with the ncs coupling scheme
subroutine check_compatibility_ncs(group_num)
  implicit none
  integer :: group_num
  
  !> currently ncs particles must be of type 'particle_kinetic_leapfrog'
  if (trim(part_group_configs(group_num)%type) /= 'particle_kinetic_leapfrog') then
    write(*,*) "ERROR: incompatible setting enabled for group '", part_group_configs(group_num)%id, "': "
    write(*,*) "  Currently only type = 'particle_kinetic_leapfrog' is supported for"
    write(*,*) "  groups with coupling scheme 'ncs'"
    stop
  endif

  !> currently ncs particles are not compatible with fluid neutrals and fluid impurities
  if (with_neutrals .or. with_impurities) then
    write(*,*) "ERROR: incompatible setting enabled for group '", part_group_configs(group_num)%id, "': "
    write(*,*) "  Currently kinetic neutrals are not compatible with fluid neutrals/impurities."
    write(*,*) "  Please recompile with with_neutrals and with_impurities=.false."
    stop
  endif

  !> currently ncs particles are not compatible with two temperature
  if (with_TiTe) then
    write(*,*) "ERROR: incompatible setting enabled for group '", part_group_configs(group_num)%id, "': "
    write(*,*) "  Currently kinetic neutrals are not compatible with two temperature models, "
    write(*,*) "  Please recompile with with_TiTe=.false."
    stop
  endif
  
end subroutine check_compatibility_ncs

!> checks that the physics enabled for particle group is compatible with the ics coupling scheme
subroutine check_compatibility_ics(group_num)
  implicit none
  integer :: group_num
  
  !> ics not compatible with use_kin_recombination
  if (part_group_configs(group_num)%use_kin_recombination) then
    write(*,*) "ERROR: incompatible setting enabled for group '", part_group_configs(group_num)%id, "': "
    write(*,*) "  use_kin_recombination can only be .t. for groups with coupling scheme 'ncs'"
    stop
  endif

  !> currently ics particles must be of type 'particle_kinetic_leapfrog'
  if (trim(part_group_configs(group_num)%type) /= 'particle_kinetic_leapfrog') then
    write(*,*) "ERROR: incompatible setting enabled for group '", part_group_configs(group_num)%id, "': "
    write(*,*) "  Currently only type = 'particle_kinetic_leapfrog' is supported for"
    write(*,*) "  groups with coupling scheme 'ics'"
    stop
  endif

  !> currently ics particles are not compatible with fluid neutrals and fluid impurities
  if (with_neutrals .or. with_impurities) then
    write(*,*) "ERROR: incompatible setting enabled for group '", part_group_configs(group_num)%id, "': "
    write(*,*) "  Currently kinetic impurities are not compatible with fluid neutrals/impurities."
    write(*,*) "  Please recompile with with_neutrals and with_impurities=.false."
    stop
  endif

  !> currently ics particles are not compatible with two temperature
  if (with_TiTe) then
    write(*,*) "ERROR: incompatible setting enabled for group '", part_group_configs(group_num)%id, "': "
    write(*,*) "  Currently kinetic impurities are not compatible with two temperature models, "
    write(*,*) "  Please recompile with with_TiTe=.false."
    stop
  endif

end subroutine check_compatibility_ics

!> compares the name of a given coupling variable associated with a coupling scheme (i.e. assessed_var) 
!> with the list of coupling variables already used by the simulation (i.e. coupling_vars). If the 
!> assessed_var is unique it will be appended to the list 
subroutine assess_and_accumulate_variable(assessed_var, coupling_var_idx, coupling_vars)
  implicit none
  character(len=var_name_len), intent(in) :: assessed_var
  integer, intent(inout)                  :: coupling_var_idx
  character(len=var_name_len), dimension(n_aux_var_max), intent(inout) :: coupling_vars

  if (.not. (any(coupling_vars == assessed_var))) then
    coupling_var_idx = coupling_var_idx + 1
    coupling_vars(coupling_var_idx) = assessed_var

    if (coupling_var_idx > n_aux_var_max) then
      write(*,*) "ERROR: The number of coupling variables required for kinetic-fluid coupling "
      write(*,*) "  exceeds the hardcoded n_aux_var_max. Consider increasing n_aux_var_max."
      stop
    endif
  endif
end subroutine assess_and_accumulate_variable

!> determines the list of unique coupling variables required by the simulation and assigns their corresponding indices
subroutine determine_coupling_variables()
  implicit none
  integer :: i, j
  integer :: coupling_var_idx, final_var_idx

  coupling_vars = ""
  coupling_var_idx = 0
  final_var_idx    = 0

  !> construct a list of unique coupling variables required
  if (use_ncs) then 
    do i=1, size(ncs_var_names)
      call assess_and_accumulate_variable(ncs_var_names(i), coupling_var_idx, coupling_vars)
    enddo
  endif

  if (use_ics) then 
    do i=1, size(ics_var_names)
      call assess_and_accumulate_variable(ics_var_names(i), coupling_var_idx, coupling_vars)
    enddo

    !> handling impurity group specific coupling variables:
    !> these variables are not used in mod_elt_matrix_fft but are required for coupling
    !> on the kinetic side
    do j=1, n_ics
      coupling_var_idx = coupling_var_idx + 1
      coupling_vars(coupling_var_idx) = "imp_q"          !< impurity charge density
      ics_indices_kin(j) = coupling_var_idx
    enddo
  endif
    
  if (use_rep) then
    do i=1, size(rep_var_names)
      call assess_and_accumulate_variable(rep_var_names(i), coupling_var_idx, coupling_vars)
    enddo
  endif

  !> additional coupling schemes will be added here in future PRs (e.g. use_epp, use_epf)  
    
  !> assign indices to the coupling variables and determine n_aux_var
  write(*,*) "===== Indices of coupling variables ====="
  do i=1, coupling_var_idx
    final_var_idx = final_var_idx + 1
    select case (trim(coupling_vars(i)))
      case ("rho")
        rho_idx_kin = final_var_idx
      case ("mom_par")
        mom_par_idx_kin = final_var_idx
      case ("E")
        E_idx_kin = final_var_idx
      case ("P_par")
        P_par_idx_kin  = final_var_idx
      case ("P_perp")
        P_perp_idx_kin = final_var_idx
      case ("j_Phi")
        j_Phi_idx_kin = final_var_idx
      case ("imp_q")
        continue       !< do nothing as already handled above in use_ics loop
      case default
        write(*,*) "Error: no match found for coupling variable: ", coupling_vars(i),", please check coupling_variables.f90 and recompile"
        stop
    end select
    write(*,"(2X,A12,' = ', I3)") coupling_vars(i), final_var_idx
  enddo
  write(*,*) "========================================="

  n_aux_var = final_var_idx
  n_aux_var = n_aux_var + 5 ! temporary as diag projections not yet created

end subroutine determine_coupling_variables

end module mod_coupling_settings