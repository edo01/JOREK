!> storage of the variables required in each of the coupling schemes
!> used to construct the indices of the feedback from the particle evolution
!> and the indices of the aux_node_list
!> PLEASE CAREFULLY OBSERVE EXISTING PARAMETERS TO AVOID OVERLAP WHEN ADDING A NEW COUPLING SCHEME
module coupling_variables
  implicit none

  integer, parameter :: n_aux_var_max = 100

  ! ====== variables names and number of coupling schemes ====== !

  integer, parameter :: var_name_len  = 15
  character(len=var_name_len), dimension(n_aux_var_max) :: coupling_vars

  ! NCS (Coupling scheme for neutral particles) 
  character(len=var_name_len), dimension(3) :: ncs_var_names = [character(len=var_name_len) :: &
    "rho",      & !> density
    "mom_par",  & !> parallel momentum
    "E"         & !> energy
  ]

  ! ICS (Coupling scheme for impurity particles)
  !> These are only the base variables, there is also the impurity charge density, unique to each impurity group
  character(len=var_name_len), dimension(2) :: ics_var_names = [character(len=var_name_len) :: &
    "mom_par",  & !> parallel momentum
    "E"         & !> energy
  ]

  ! REP (Pressure coupling scheme for runaway electrons)
  character(len=var_name_len), dimension(3) :: rep_var_names = [character(len=var_name_len) :: &
    "P_par",    & !> parallel component of dynamic pressure tensor
    "P_perp",   & !> perpendicular component of dynamic pressure tensor
    "j_Phi"     & !> Phi compoment of current  
  ]

  ! =========== Storage variables for kinetic coupling indices ======== !

  !> variables indices
  integer :: rho_idx_kin      = 0
  integer :: mom_par_idx_kin  = 0
  integer :: E_idx_kin        = 0
  integer :: P_par_idx_kin    = 0
  integer :: P_perp_idx_kin   = 0
  integer :: j_Phi_idx_kin    = 0

  !> index of coupling variables specific to each impurity group
  integer :: ics_indices_kin(n_aux_var_max) = -1

end module coupling_variables