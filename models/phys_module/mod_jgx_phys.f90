!> The Fortran side of models/phys_module/phys.h: pushes phys_module's runtime
!> scalars across the seam so ported kernels can read them without a `use`.
!>
!> Call jgx_set_phys() wherever the field set it describes is (re)built.

!> it is a handful of scalars, so calling it more often than needed costs 
!> nothing.
module mod_jgx_phys
  use, intrinsic :: iso_c_binding, only: c_double, c_int32_t
  implicit none
  private
  public :: jgx_set_phys

  !> Interface only -- the body is in C++ (models/phys_module/phys.cpp).
  interface
    subroutine jgx_c_set_phys(F0, central_mass, central_density, tstep, &
                              find_RZ_nearby_iter, find_RZ_nearby_tol)      &
        bind(C, name="jgx_c_set_phys")
      import :: c_double, c_int32_t
      implicit none
      real(c_double),     value, intent(in) :: F0, central_mass, central_density, tstep
      integer(c_int32_t), value, intent(in) :: find_RZ_nearby_iter
      real(c_double),     value, intent(in) :: find_RZ_nearby_tol
    end subroutine
  end interface

contains

  !> Copy the current phys_module values over the seam.
  subroutine jgx_set_phys()
    use phys_module, only: F0, central_mass, central_density, tstep, &
                           find_RZ_nearby_iter, find_RZ_nearby_tol
    call jgx_c_set_phys(F0, central_mass, central_density, tstep, &
                        int(find_RZ_nearby_iter, c_int32_t), find_RZ_nearby_tol)
  end subroutine jgx_set_phys

end module mod_jgx_phys
