/* models/phys_module/phys.cpp -- storage for the pushed copy of phys_module's scalars.
 *
 * One object, written only by jgx_c_set_phys (from mod_jgx_phys.f90) and read
 * through jorek::phys(). Same shape as jgx_record_registry.cpp: Fortran fills
 * it, everything downstream treats it as read-only.
 */
#include "models/phys_module/phys.h"

#include "models/constants/constants.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

jorek::phys_state g_phys  = {};
bool              g_valid = false;

}

namespace jorek {

const phys_state& host_phys() {
  if (!g_valid) {
    std::fprintf(stderr, "phys_module: read before mod_jgx_phys pushed a copy "
                         "(call jgx_set_phys)\n");
    std::abort();
  }
  return g_phys;
}

} /* namespace jorek */

extern "C" {

void jgx_c_set_phys(double F0, double central_mass, double central_density,
                    double tstep, int find_RZ_nearby_iter,
                    double find_RZ_nearby_tol) {
  g_phys.F0              = F0;
  g_phys.central_mass    = central_mass;
  g_phys.central_density = central_density;
  g_phys.tstep           = tstep;

  g_phys.find_RZ_nearby_iter = find_RZ_nearby_iter;
  g_phys.find_RZ_nearby_tol  = find_RZ_nearby_tol;

  /* As phys_module's callers spell it, left to right:
   *   t_norm = sqrt(mu_zero * ATOMIC_MASS_UNIT * central_mass * central_density * 1.d20) */
  g_phys.t_norm  = std::sqrt(jorek::MU_ZERO * jorek::ATOMIC_MASS_UNIT
                             * central_mass * central_density * 1.0e20);
  g_phys.t_jorek = tstep * g_phys.t_norm;

  g_valid = true;

  /* Keep the device mirror in step with the host copy, so that a kernel and the
   * host path of the same routine cannot read different values. Guarded because
   * the push lives in a device translation unit -- there is no such symbol to
   * call when JGX_DEVICE=off. */
#ifdef JGX_HAS_DEVICE
  jorek::phys_push_to_device(g_phys);
#endif
}

} /* extern "C" */
