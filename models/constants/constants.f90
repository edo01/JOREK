!> Module containing constants which are used in the code
module constants
  implicit none
  public
  
#include "version.h"

  ! @name Mathematical and physical constants
  ! Declared from constants.def, which constants.h reads too, so
  ! the Fortran and the ported C++ kernels cannot disagree on a value.
  ! ! real*8,  parameter :: MASS_PROTON   = 1.67262178d-27         !< proton mass [kg] ! commented out because we should always use AMU
#define JOREK_CONSTANT(name, value) real(kind=8), parameter :: name = value##_8
#include "models/constants/constants.def"
#undef JOREK_CONSTANT

  !> Derived; constants.h spells the same expression out in C++.
  real*8,  parameter :: MU_ZERO       = 4.d-7*PI                 !< Magnetic constant  [Vs/Am]

  !> @name Constants which describe the domain of a certain position (used by function which_domain)
  integer, parameter :: DOMAIN_PLASMA         = 0    !< Plasma region
  integer, parameter :: DOMAIN_SOL            = 1    !< Scrape-off layer
  integer, parameter :: DOMAIN_OUTER_SOL      = 2    !< Outer scrape-off layer (double-null)
  integer, parameter :: DOMAIN_UPPER_PRIVATE  = 3    !< Upper private flux region
  integer, parameter :: DOMAIN_LOWER_PRIVATE  = 4    !< Lower private flux region

  !> @name Parameters which describe the X-point case
  integer, parameter :: LOWER_XPOINT          = 1
  integer, parameter :: UPPER_XPOINT          = 2
  integer, parameter :: DOUBLE_NULL           = 3
  integer, parameter :: SYMMETRIC_XPOINT      = 100  ! Used for grid construction purposes; do not use as value for xcase in the input file!

end module constants
