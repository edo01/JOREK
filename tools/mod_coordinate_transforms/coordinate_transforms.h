/* tools/mod_coordinate_transforms/coordinate_transforms.h -- changes of basis
 * between (R,Z,phi) and (x,y,z).
 *
 * DUPLICATED FROM the real*8 specific procedures of mod_coordinate_transforms.f90,
 * next door, which are tagged to point back here. Keep the two in step.
 * Those keep their Fortran bodies: they are reached through generic interfaces
 * that also carry real*4 variants, and are called from dozens of sites that are
 * nowhere near a ported kernel. Same arrangement as jac() in
 * particles/mod_fields/fields_set.h.
 *
 * Written to be safe when the input and the output alias, which the Fortran
 * function form is for free.
 */
#ifndef JOREK_COORDINATE_TRANSFORMS_H
#define JOREK_COORDINATE_TRANSFORMS_H

#include <cmath>

#include "jgx/macros.h"

namespace coordinate_transforms {

/* mod_coordinate_transforms::cartesian_to_cylindrical, real*8 */
JGX_HD inline void cartesian_to_cylindrical(const double xyz[3], double cyl[3]) {
    const double x = xyz[0], y = xyz[1], z = xyz[2];

    cyl[0] = std::sqrt(x*x + y*y);
    cyl[1] = z;
    cyl[2] = std::atan2(-y, x);
}

/* mod_coordinate_transforms::cylindrical_to_cartesian, real*8 */
JGX_HD inline void cylindrical_to_cartesian(const double cyl[3], double xyz[3]) {
    const double R = cyl[0], Z = cyl[1], phi = cyl[2];

    xyz[0] = R*std::cos(-phi);
    xyz[1] = R*std::sin(-phi);
    xyz[2] = Z;
}

/* mod_coordinate_transforms::vector_cartesian_to_cylindrical -- a vector from
 * the (ex,ey,ez) basis into (eR,eZ,ephi). */
JGX_HD inline void vector_cartesian_to_cylindrical(const double phi,
                                                   const double a[3], double b[3]) {
    const double sin_phi = std::sin(phi), cos_phi = std::cos(phi);
    const double a1 = a[0], a2 = a[1], a3 = a[2];

    b[0] = a1*cos_phi - a2*sin_phi;
    b[1] = a3;
    b[2] = -1.0*(a1*sin_phi + a2*cos_phi);
}

/* mod_coordinate_transforms::vector_cylindrical_to_cartesian -- a vector from
 * the (eR,eZ,ephi) basis into (ex,ey,ez). */
JGX_HD inline void vector_cylindrical_to_cartesian(const double phi,
                                                   const double a[3], double b[3]) {
    const double sin_phi = std::sin(phi), cos_phi = std::cos(phi);
    const double a1 = a[0], a2 = a[1], a3 = a[2];

    b[0] = a1*cos_phi - a3*sin_phi;
    b[1] = -1.0*(a1*sin_phi + a3*cos_phi);
    b[2] = a2;
}

} // namespace coordinate_transforms

#endif // JOREK_COORDINATE_TRANSFORMS_H
