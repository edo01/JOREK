/* particles/pushers/mod_pusher_tools/pusher_tools.h -- the pusher helpers of
 * mod_pusher_tools.f90.
 *
 */
#ifndef JOREK_PUSHER_TOOLS_H
#define JOREK_PUSHER_TOOLS_H

#include "jgx/macros.h"

namespace pusher_tools {

/**
 * mod_pusher_tools::cayley_transform -- the right-handed Cayley transform of
 * alpha*vec,
 *
 *   cayley(alpha*V) = (I - alpha*V)^-1 * (I + alpha*V)
 *
 * with V the skew-symmetric cross-product matrix of vec.
 *
 * @param[out] M  the 3x3 transform. M[i][j] is the Fortran (i+1,j+1), so the
 *                Fortran's column assignments read transposed here.
 */
JGX_HD inline void cayley_transform(const double alpha, const double vec[3],
                                    double M[3][3]) {
    const double a  = alpha;
    const double v1 = vec[0], v2 = vec[1], v3 = vec[2];

    // (I + alpha*V)
    const double B[3][3] = {
        {      1.0,  a*v3, -a*v2 },
        {    -a*v3,   1.0,  a*v1 },
        {     a*v2, -a*v1,   1.0 }
    };

    // (I - alpha*V)^-1, up to the common denominator applied below
    const double A[3][3] = {
        { 1.0 + a*a*v1*v1,   a*(a*v2*v1 + v3),  a*(a*v3*v1 - v2) },
        { a*(a*v1*v2 - v3),   1.0 + a*a*v2*v2,  a*(a*v2*v3 + v1) },
        { a*(a*v3*v1 + v2),  a*(a*v3*v2 - v1),  1.0 + a*a*v3*v3 }
    };

    const double inv_den = 1.0/(1.0 + a*a*(v1*v1 + v2*v2 + v3*v3));

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            M[i][j] = (A[i][0]*B[0][j] + A[i][1]*B[1][j] + A[i][2]*B[2][j]) * inv_den;
} // cayley_transform

/* y = M*x, the matmul the Fortran applies the transform with. */
JGX_HD inline void matvec3(const double M[3][3], const double x[3], double y[3]) {
    const double x0 = x[0], x1 = x[1], x2 = x[2];
    for (int i = 0; i < 3; ++i) y[i] = M[i][0]*x0 + M[i][1]*x1 + M[i][2]*x2;
}

} // namespace pusher_tools

#endif // JOREK_PUSHER_TOOLS_H
