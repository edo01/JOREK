/* models/constants/constants.h -- the C++ view of the constants taken from
 * constants.def.
 *
 * The other half of models/constants/constants.f90: both files include the same
 * .def, so a constant cannot drift between the Fortran and the ported kernels.
 * Values are compile-time, so nothing crosses the seam to obtain them.
 */
#ifndef JOREK_CONSTANTS_H
#define JOREK_CONSTANTS_H

namespace jorek {

#define JOREK_CONSTANT(name, value) constexpr double name = value;
#include "models/constants/constants.def"
#undef JOREK_CONSTANT

/* Derived, as in constants.f90 (`4.d-7*PI`) */
constexpr double MU_ZERO = 4.0e-7 * PI;  /* Magnetic constant [Vs/Am] */

} /* namespace jorek */

#endif /* JOREK_CONSTANTS_H */
