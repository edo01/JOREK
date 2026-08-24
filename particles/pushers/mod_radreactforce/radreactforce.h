/* particles/pushers/mod_radreactforce/radreactforce.h -- the synchrotron
 * radiation-reaction force of mod_radreactforce.f90, next door.
 *
 * F. Andersson et al. / see https://arxiv.org/pdf/1412.1966.pdf.  As in the
 * Fortran, the test particle is assumed to be an electron.
 *
 * DUPLICATED FROM mod_radreactforce.f90 -- keep the two in step.  Only the
 * kinetic operator is here; the guiding-centre ones (radreactforce_gc,
 * radreactforce_gc_rhs) have no ported caller and stay Fortran-only.
 */
#ifndef JOREK_RADREACTFORCE_H
#define JOREK_RADREACTFORCE_H

#include <cmath>
#include <cstddef>

#include "models/constants/constants.h"
#include "tools/mod_coordinate_transforms/coordinate_transforms.h"
#include "jgx/macros.h"

namespace radreactforce {

/**
 * mod_radreactforce::radreactforce_chartime -- the characteristic time of the
 * radiation-reaction force.
 *
 * @param B       magnetic field strength [T]
 * @param gamma   particle Lorentz factor
 * @param mass    particle mass [kg]
 * @param charge  particle charge [e]
 */
JGX_HD inline double radreactforce_chartime(const double B, const double gamma,
                                            const double mass, const int charge) {
  const double mc = mass*jorek::SPEED_OF_LIGHT;
  const double q  = static_cast<double>(charge)*jorek::EL_CHG;
  return 6.0*jorek::PI*jorek::EPS_ZERO*gamma*(mc*mc*mc) / (q*q*q*q*B*B);
} // radreactforce_chartime

/**
 * mod_radreactforce::radreactforce_kinetic -- one forward-Euler step of the
 * radiation-reaction force on a particle's momentum.
 *
 * The term carrying the explicit time dependence of the magnetic field is
 * omitted, as it is in the Fortran.
 *
 * @param part, ip  the particle set and the particle in it; p is updated in
 *                  place, in the set's [AMU m/s] units
 * @param B         magnetic field at the particle, cylindrical (R,Z,phi) [T]
 * @param dt        time step [s]
 * @param mass      particle mass [AMU]
 */
template<class PS>
JGX_HD inline void radreactforce_kinetic(PS& part, const std::size_t ip,
                                         const double B[3], const double dt,
                                         const double mass) {
  // Convert to SI units
  const double m = mass*jorek::ATOMIC_MASS_UNIT;
  double p[3];
  for (int k = 0; k < 3; ++k) p[k] = part.p(ip, k)*jorek::ATOMIC_MASS_UNIT;

  double B_cart[3];
  coordinate_transforms::vector_cylindrical_to_cartesian(part.x(ip, 2), B, B_cart);
  const double B_norm2 = B[0]*B[0] + B[1]*B[1] + B[2]*B[2];
  const double B_norm  = std::sqrt(B_norm2);

  // Calculate the characteristic time and pperp
  const double mc = m*jorek::SPEED_OF_LIGHT;
  const double p2 = p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
  const double gamma = std::sqrt(1.0 + p2/(mc*mc));
  const double tau = radreactforce_chartime(B_norm, gamma, m,
                                            static_cast<int>(part.q(ip)));

  double p_perp[3];
  const double pdotB = B_cart[0]*p[0] + B_cart[1]*p[1] + B_cart[2]*p[2];
  for (int k = 0; k < 3; ++k) p_perp[k] = p[k] - B_cart[k]*pdotB/B_norm2;
  const double p_perp2 = p_perp[0]*p_perp[0] + p_perp[1]*p_perp[1]
                       + p_perp[2]*p_perp[2];

  // Apply RR-force and convert back to JOREK units
  for (int k = 0; k < 3; ++k) {
    p[k] -= dt*(p_perp[k] + p[k]*p_perp2/(mc*mc))/tau;
    part.p(ip, k) = p[k]/jorek::ATOMIC_MASS_UNIT;
  }
} // radreactforce_kinetic

} /* namespace radreactforce */

#endif /* JOREK_RADREACTFORCE_H */
