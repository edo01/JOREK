module mod_find_rz_nearby
  implicit none
  private
  public :: find_rz_nearby
  !> For callers that reach the kernel directly rather than through the facade
  !> below -- see particles/pushers/mod_kinetic_relativistic.f90.
  public :: find_RZ_nearby_phi_search, find_RZ_nearby_report

  !> Interface only -- the body is in C++ (mod_find_rz_nearby/find_rz_nearby_shim.cpp).
  interface
    subroutine jgx_host_find_RZ_nearby(el_base, n_elements, nd_base, n_nodes,   &
                                       R_old, Z_old, s_old, t_old, i_elm_old,   &
                                       R_new, Z_new, p, phi_search,             &
                                       s_new, t_new, i_elm_new, ifail,          &
                                       not_found, bad_i_from, bad_i_to)         &
        bind(C, name="jgx_host_find_RZ_nearby")
      use, intrinsic :: iso_c_binding, only: c_double, c_int32_t, c_ptr
      implicit none
      type(c_ptr),        value, intent(in)  :: el_base, nd_base
      integer(c_int32_t), value, intent(in)  :: n_elements, n_nodes, i_elm_old
      real(c_double),     value, intent(in)  :: R_old, Z_old, s_old, t_old
      real(c_double),     value, intent(in)  :: R_new, Z_new, p, phi_search
      real(c_double),           intent(out)  :: s_new, t_new
      integer(c_int32_t),       intent(out)  :: i_elm_new, ifail
      integer(c_int32_t),       intent(out)  :: not_found, bad_i_from, bad_i_to
    end subroutine
  end interface

contains
!> Optimized subroutine to find st coordinates corresponding to x_new=[R_new, Z_new] using
!! The previous values x_old=[R_old, Z_old], st_old(2), i_elm_old and checking adjacent elements first
!! It first checks the current element, using newton iteration to find st_try corresponding to x_new
!! Once this is found it calls check_element_boundary to see if we have crossed an element boundary
!! If this is the case, the search is restarted within the new element.
!! This procedure continues until the required tolerance is reached, or element_try_max is reached.
!!
!! Example:
!! (o is the starting point, T is the target, x is the guess)
!! |^^^^^^|^^^^^^|  |^^^^^^|^^^^^^|  |^^^^^^|^^^^^^|
!! |1     |2     |  |1     |2     |  |1     |2     |
!! |      |  T   |  |      |  T   |  |      | xT   |
!! |  o   |      |  |  o   |x     |  |  o   |      |
!! |______|______|  |______|______|  |______|______|
!! We start out knowing the old position (R,Z) x_old, the new position (R,Z) x_new,
!! the old position and element number st_old, i_elm_old.
!!
!! We are now in element 1, and element_try_index=1. Using 2D newton iteration
!! with backtracking we find the (s,t) position st_try corresponding to x_new, starting
!! from x_old, in the basis functions of the current element. (Which are not
!! guaranteed to be nice outside of the element)
!!
!! When we have found the position st_try, we check whether it is outside of the
!! current element. (middle figure) If so, we switch to this element (element 2), perform a linear transform
!! of the coordinates and start the process again.
!! If it is inside of the element and we have found a good st_try the routine is done.
!!
!! There are some cases where this method does not work, notably near element
!! boundaries and the magnetic axis. Here, overshoot of the newton's method near
!! the element boundary causes errors. If too many elements are crossed, i.e.
!! element_try_max is exceeded, or if too many iterations are needed, i.e.
!! find_RZ_nearby_iter is exceeded, we stop the routine and call find_RZ, which is
!! extremely slow but works for all positions in the domain.
!! In that case, ifail=2:5 is returned.
!! If ifail=-1 the particle is lost
!!
!> Facade only -- the body is in C++ (mod_find_rz_nearby/find_rz_nearby_shim.cpp),
!> together with the try_interp it used. The C++ serves both configurations; all
!> that differs is the angle the global fallback searches at, worked out below.
subroutine find_RZ_nearby(node_list, element_list, R_old, Z_old, s_old, t_old, i_elm_old, &
        R_new, Z_new, s_new, t_new, i_elm_new, ifail, phi)
use data_structure
use, intrinsic :: iso_c_binding, only: c_double, c_int32_t, c_ptr, c_loc
implicit none
!> Input parameters
type (type_node_list),    target, intent(in) :: node_list
type (type_element_list), target, intent(in) :: element_list
real*8,                   intent(in)    :: R_old, Z_old !< The old R,Z location
real*8,                   intent(in)    :: R_new, Z_new !< The new R,Z location
real*8,                   intent(in)    :: s_old, t_old !< The old st location (used to compute a guess)
integer,                  intent(in)    :: i_elm_old
real*8,                   intent(out)   :: s_new, t_new !< The found new coordinates
integer,                  intent(out)   :: i_elm_new
integer,                  intent(out)   :: ifail !< if ifail = -1 the position could not be found in the grid.
!< ifail > 0 indicates various other cases
real*8, optional,         intent(in)    :: phi

real*8             :: p, phi_search
integer(c_int32_t) :: c_i_elm_new, c_ifail, not_found, bad_i_from, bad_i_to

if (present(phi)) then
  p = phi
else
#if STELLARATOR_MODEL
  write(*,*) "Toroidal angle phi must be defined for stellarator models"
  stop
#endif
  p = 0.0
endif

phi_search = find_RZ_nearby_phi_search(p)

call jgx_host_find_RZ_nearby(c_loc(element_list%element(1)),          &
                             int(element_list%n_elements, c_int32_t), &
                             c_loc(node_list%node(1)),                &
                             int(node_list%n_nodes, c_int32_t),       &
                             R_old, Z_old, s_old, t_old,              &
                             int(i_elm_old, c_int32_t),               &
                             R_new, Z_new, p, phi_search,             &
                             s_new, t_new, c_i_elm_new, c_ifail,      &
                             not_found, bad_i_from, bad_i_to)

i_elm_new = int(c_i_elm_new)
ifail     = int(c_ifail)

call find_RZ_nearby_report(not_found, bad_i_from, bad_i_to, R_old, Z_old)
end subroutine find_RZ_nearby

!> The angle the global fallback searches at. find_RZ built it from the plane
!> index and ignored p; find_RZP wrapped p into one coordinate period. That is
!> the whole of the difference between the two configurations.
pure function find_RZ_nearby_phi_search(p) result(phi_search)
use constants, only: PI
use phys_module, only: i_plane_rtree
use mod_parameters, only: n_period, n_plane, n_coord_period
implicit none
real*8, intent(in) :: p !< the toroidal angle the local interpolation runs at
real*8             :: phi_search

#if STELLARATOR_MODEL
phi_search = p - (PI * 2.d0 / n_coord_period) * floor(p / (PI * 2.d0 / n_coord_period))
#else
phi_search = 2.d0*pi*float(i_plane_rtree - 1)/float(n_period*n_plane)
#endif
end function find_RZ_nearby_phi_search

!> The kernel cannot print, so it hands its two diagnostics back and whoever
!> called it writes them out here.
subroutine find_RZ_nearby_report(not_found, bad_i_from, bad_i_to, R_old, Z_old)
implicit none
integer, intent(in) :: not_found        !< the search ran out and the fallback failed
integer, intent(in) :: bad_i_from, bad_i_to !< the element pair that disagreed
real*8,  intent(in) :: R_old, Z_old     !< where the failed search started from

if (bad_i_to /= 0) write(*,"(A,i5,A,i5)") &
  "ERROR IN element_list%element(", bad_i_from, ")%neighbours to ", bad_i_to

if (not_found /= 0) then
  !$omp critical
    write(*,"(A,2f10.5,A)") "ERROR: issue in mod_find_rz_nearby; could not find ",R_old,Z_old
    write(*,"(A)") "This position is likely outside of the domain but find_RZ_nearby_iter (namelist input parameter) is not big"
    write(*,"(A)") "enough to find the domain boundary element closest to it. Consider using a larger find_RZ_nearby_iter."
  !$omp end critical
endif
end subroutine find_RZ_nearby_report
end module mod_find_rz_nearby
