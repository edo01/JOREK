/* models/phys_module/phys.h -- the runtime physics scalars of phys_module.f90.
 *
 * On a device build there are two copies of the phys_state, one per address
 * space, and phys() picks by where it is compiled.
 */
#ifndef JOREK_PHYS_H
#define JOREK_PHYS_H

#include "jgx/macros.h"

namespace jorek {

struct phys_state {
  double F0              = 0;  /* fixed toroidal field factor, B_phi = F0/R */
  double central_mass    = 0;  /* average ion mass [amu] */
  double central_density = 0;  /* density on axis [10^20 m^-3] */
  double tstep           = 0;  /* fluid time step [JOREK units] */

  /* Namelist inputs rather than per-step values, but they reach ported kernels
   * the same way: find_RZ_nearby's loop bound and convergence threshold. */
  int    find_RZ_nearby_iter = 0;  /* max newton iterations */
  double find_RZ_nearby_tol  = 0;  /* squared element tolerance [element size] */

  /* Derived here rather than passed, so the seam carries state and not results.
   * Same expression and same association as phys_module's callers use. */
  double t_norm          = 0;  /* one JOREK time unit [s] */
  double t_jorek         = 0;  /* tstep * t_norm [s] */
};

/* The host copy of phys_state. */
const phys_state& host_phys();

#if defined(JGX_DEVICE_CUDA) || defined(JGX_DEVICE_HIP)
/* The device mirror of phys_state, defined in phys_device.*.cpp. 
 *
 * NOTE: It switches on the compilation *pass*, not the build! 
 */
extern JGX_DEVICE_VAR phys_state d_phys;
#endif

/* Returns the right copy of the phys_state object depending on the calling
 * site.
 * 
 * NOTE: It switches on the compilation *pass*, not the build! 
 */
JGX_HD inline const phys_state& phys() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return d_phys;
#else
  return host_phys();
#endif
}

/* Mirror the pushed copy onto the device.
 */
void phys_push_to_device(const phys_state& p);

/* phys_pull_from_device exists for test purposes.
 */
void phys_pull_from_device(phys_state& out);

} /* namespace jorek */

/* The only writer, called from mod_jgx_phys.f90. Declared here as well as in
 * that module's interface block so C++ callers. */
extern "C" void jgx_c_set_phys(double F0, double central_mass,
                               double central_density, double tstep,
                               int find_RZ_nearby_iter, double find_RZ_nearby_tol);

#endif /* JOREK_PHYS_H */
