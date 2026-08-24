!  -*-f90-*-  (for emacs)    vim:set filetype=fortran:  (for vim)
!
!> Module containing a collection of interpolation methods
!>
!> DUPLICATED IN C++ as the interp_methods namespace, in interp_methods.h next
!> door -- keep the two in step. The C++ clamps the cell index at the top edge of
!> the mesh, where interp_bilinear below reads one past the array.

module mod_interp_methods

  implicit none

  public :: interp_bilinear

  private

contains

  !> Bilinear interpolation method (only works with equidistant meshes)
  real(kind=8) function interp_bilinear(x, y, f, xq, yq) 
    implicit none
    
    real*8, intent(in) :: x(:), y(:) !< abscissae
    real*8, intent(in) :: f(:,:)     !< function values
    real*8, intent(in) :: xq, yq     !< queried point

    real*8  :: dx, dy
    integer :: ix, iy

    dx = x(2) - x(1)
    dy = y(2) - y(1)
    
    ix = floor( ( xq - x(1) ) / dx ) + 1 
    iy = floor( ( yq - y(1) ) / dy ) + 1 

    interp_bilinear = ( f(ix, iy)         * ( x(ix + 1) - xq ) * ( y(iy + 1) - yq ) & 
                    + f(ix + 1, iy)     * ( xq - x(ix) )     * ( y(iy + 1) - yq ) & 
                    + f(ix, iy + 1)     * ( x(ix + 1) - xq ) * ( yq - y(iy) )     &   
                    + f(ix + 1, iy + 1) * ( xq - x(ix) )     * ( yq - y(iy) )     &   
                    ) / ( dx * dy )
    
  end function interp_bilinear

end module mod_interp_methods
