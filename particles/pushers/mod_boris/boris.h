/* particles/pushers/mod_boris/boris.h -- the cylindrical Boris push.
 *
 * G. L. Delzanno, E. Camporeale, JCP 253 (2013) 259-277.
 *
 */
#ifndef JOREK_BORIS_H
#define JOREK_BORIS_H

#include <cmath>
#include <cstddef>

#include "models/constants/constants.h"
#include "jgx/macros.h"

namespace boris {

/**
 * mod_boris::boris_push_cylindrical -- advance one particle's velocity from
 * v^(n-1/2) to v^(n+1/2) and its position from x^n to x^(n+1), in (R, Z, phi).
 *
 * The Fortran assigns whole arrays, so every right-hand side is evaluated
 * before any component of the left is written. Here that is the local `v`: the
 * rotation reads the post-electric-kick velocity three times, and the frame
 * correction at the end reads the pre-correction v(0) and v(2) -- writing
 * either in place would feed a half-updated vector back into its own formula.
 *
 * @param part, ip  the particle set and the particle in it
 * @param m         particle mass [AMU]
 * @param E, B      electric [V/m] and magnetic [T] field at the particle,
 *                  cylindrical
 * @param dt        time step [s]
 */
template<class PS>
JGX_HD inline void boris_push_cylindrical(PS& part, const std::size_t ip,
                                          const double m,
                                          const double E[3], const double B[3],
                                          const double dt) {
    const double eom = jorek::EL_CHG/(m*jorek::ATOMIC_MASS_UNIT);

    const double B2    = B[0]*B[0] + B[1]*B[1] + B[2]*B[2];
    const double Bnorm = std::sqrt(B2);

    /* The geometric factor f = tan(q/m dt/2 |B|)/|B| */
    const double q  = static_cast<double>(part.q(ip));
    const double fE = q*eom*dt*0.5;
    const double fB = std::tan(q*eom*dt*0.5*Bnorm)/Bnorm;

    double v[3] = { part.v(ip, 0), part.v(ip, 1), part.v(ip, 2) };

    /* the electric field update (v^(n-1/2) -> v-) */
    for (int k = 0; k < 3; ++k) v[k] += fE*E[k];

    /* the rotation */
    const double cross[3] = { v[1]*B[2] - v[2]*B[1],
                              v[2]*B[0] - v[0]*B[2],
                              v[0]*B[1] - v[1]*B[0] };
    const double vdotB = v[0]*B[0] + v[1]*B[1] + v[2]*B[2];
    const double rot   = 2.0*fB/(1.0 + fB*fB*B2);
    for (int k = 0; k < 3; ++k)
        v[k] = v[k] + rot*(cross[k] - fB*v[k]*B2 + fB*B[k]*vdotB);

    /* the second electric field update (v+ -> v^(n+1/2)) */
    for (int k = 0; k < 3; ++k) v[k] += fE*E[k];

    /* the position update from v^n to v^(n+1), through the cartesian frame the
     * step lands in: R is the in-plane displacement, RPhi the out-of-plane one. */
    const double R    = part.x(ip, 0) + v[0]*dt;
    const double RPhi = v[2]*dt;

    const double R_new = std::sqrt(R*R + RPhi*RPhi);
    part.x(ip, 0) = R_new;
    part.x(ip, 1) = part.x(ip, 1) + dt*v[1];
    part.x(ip, 2) = part.x(ip, 2) + std::asin(RPhi/R_new);

    /* R and phi velocities into the new reference frame */
    const double v_R   = ( R*v[0] + RPhi*v[2])/R_new;
    const double v_phi = (-RPhi*v[0] + R*v[2])/R_new;

    part.v(ip, 0) = v_R;
    part.v(ip, 1) = v[1];
    part.v(ip, 2) = v_phi;
} // boris_push_cylindrical

} // namespace boris

#endif // JOREK_BORIS_H
