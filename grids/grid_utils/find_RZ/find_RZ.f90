subroutine find_RZ(node_list,element_list,R_find,Z_find,R_out,Z_out,ielm_out,s_out,t_out,ifail)
  use data_structure, only: type_node_list, type_element_list
  use phys_module, only: i_plane_rtree
  use constants, only: PI
  use mod_parameters, only: n_period, n_plane

  implicit none 

  type (type_node_list),    intent(in)    :: node_list
  type (type_element_list), intent(in)    :: element_list
  real*8,                   intent(in)    :: R_find, Z_find
  real*8,                   intent(out)   :: R_out,Z_out,s_out,t_out
  integer,                  intent(inout) :: ielm_out
  integer,                  intent(out)   :: ifail

  real*8 :: phi
  integer :: checked_elms

  phi = 2.d0*pi*float(i_plane_rtree - 1)/float(n_period*n_plane)

  call find_RZ_general(node_list,element_list,R_find,Z_find,phi,R_out,Z_out,ielm_out,s_out,t_out,ifail,checked_elms)
end subroutine find_RZ


subroutine find_RZP(node_list,element_list,R_find,Z_find,phi_find,R_out,Z_out,ielm_out,s_out,t_out,ifail,checked_elms)
  use data_structure, only: type_node_list, type_element_list
  use constants, only: PI
  use mod_parameters, only: n_coord_period
  
  implicit none

  type (type_node_list),    intent(in)    :: node_list
  type (type_element_list), intent(in)    :: element_list
  real*8,                   intent(in)    :: R_find, Z_find, phi_find
  real*8,                   intent(out)   :: R_out,Z_out,s_out,t_out
  integer,                  intent(inout) :: ielm_out
  integer,                  intent(out)   :: ifail, checked_elms

  real*8 :: phi 

  phi = phi_find - (PI * 2.d0 / n_coord_period) * floor(phi_find / (PI * 2.d0 / n_coord_period))

  call find_RZ_general(node_list,element_list,R_find,Z_find,phi,R_out,Z_out,ielm_out,s_out,t_out,ifail,checked_elms)
end subroutine

!-------------------------------------------------------------------------
!< Facade only -- the body is in C++ (find_RZ/find_RZ_shim.cpp), together with
!< the find_RZ_single it looped over.
!<
!< The C++ scans every element rather than asking a tree which ones could
!< contain the point: the r-tree has not crossed the seam yet. See the @warning
!< in find_RZ.h for what that costs and what it changes.
!-------------------------------------------------------------------------
subroutine find_RZ_general(node_list,element_list,R_find,Z_find,phi_find,R_out,Z_out,ielm_out,s_out,t_out,ifail,checked_elms)
use data_structure, only: type_node_list, type_element_list
use, intrinsic :: iso_c_binding, only: c_double, c_int32_t, c_ptr, c_loc

implicit none

type (type_node_list),    target, intent(in) :: node_list
type (type_element_list), target, intent(in) :: element_list
real*8, intent(in)     :: R_find, Z_find, phi_find
real*8, intent(out)    :: R_out,Z_out,s_out,t_out
integer, intent(inout) :: ielm_out
integer, intent(out)   :: ifail, checked_elms

!> Interface only -- the body is in C++ (find_RZ/find_RZ_shim.cpp).
interface
  subroutine jgx_host_find_RZ_general(el_base, n_elements, nd_base, n_nodes, &
                                      R_find, Z_find, phi_find,              &
                                      R_out, Z_out, ielm_out, s_out, t_out,  &
                                      ifail, checked_elms)                   &
      bind(C, name="jgx_host_find_RZ_general")
    import :: c_double, c_int32_t, c_ptr
    implicit none
    type(c_ptr),        value, intent(in)  :: el_base, nd_base
    integer(c_int32_t), value, intent(in)  :: n_elements, n_nodes
    real(c_double),     value, intent(in)  :: R_find, Z_find, phi_find
    real(c_double),           intent(out)  :: R_out, Z_out, s_out, t_out
    integer(c_int32_t),       intent(out)  :: ielm_out, ifail, checked_elms
  end subroutine
end interface

integer(c_int32_t) :: c_ielm_out, c_ifail, c_checked_elms

call jgx_host_find_RZ_general(c_loc(element_list%element(1)),          &
                              int(element_list%n_elements, c_int32_t), &
                              c_loc(node_list%node(1)),                &
                              int(node_list%n_nodes, c_int32_t),       &
                              R_find, Z_find, phi_find,                &
                              R_out, Z_out, c_ielm_out, s_out, t_out,  &
                              c_ifail, c_checked_elms)

ielm_out     = int(c_ielm_out)
ifail        = int(c_ifail)
checked_elms = int(c_checked_elms)
end subroutine find_RZ_general
