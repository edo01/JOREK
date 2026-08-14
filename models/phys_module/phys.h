/* models/phys_module/phys.h -- the runtime physics scalars of phys_module.f90.
 *
 * The counterpart of mod_settings.h for values the build cannot know: F0 and
 * the central mass/density come from the namelist, tstep changes every fluid
 * step. They are not components of any derived type, so the record registry
 * (jgx_record_api.h) does not reach them; Fortran pushes a copy instead, with
 * mod_jgx_phys.f90 as the only writer.
 *
 * The copy is refreshed wherever the field set it belongs to is (re)built --
 * see mod_jgx_phys.f90 -- so a kernel cannot read values older than the step it
 * runs in. Reading before the first push aborts rather than returning zeros: a
 * zero t_jorek silently disables the time interpolation instead of failing.
 *
 * On a device build there are two copies of the same POD, one for the host 
 * address space and for the device, and phys() picks by where it is compiled.
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

/* The host copy. Aborts if mod_jgx_phys has not pushed one yet. */
const phys_state& host_phys();

#if defined(JGX_DEVICE_CUDA) || defined(JGX_DEVICE_HIP)
/* The device mirror, defined in phys_device.hip.cpp and written by every
 * jgx_c_set_phys. Declared here rather than in the backend because phys() below
 * has to return it, and phys() is what the kernels call. Guarded because there
 * is no such object in a host-only build. */
extern JGX_DEVICE_VAR phys_state d_phys;
#endif

/* The copy for wherever this is compiled: the host one on the host, the mirror
 * in device code.
 */
JGX_HD inline const phys_state& phys() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return d_phys;
#else
  return host_phys();
#endif
}

#ifdef JGX_HAS_DEVICE
/* Mirror the pushed copy onto the device. Called by jgx_c_set_phys after every
 * push -- six scalars, so unconditionally rather than on a dirty bit.
 */
void phys_push_to_device(const phys_state& p);

/* Read the mirror back, for the test. It checks the transfer only: that phys()
 * resolves to the mirror rather than to host_phys() is a compile error under
 * -Werror=cross-execution-space-call. */
void phys_pull_from_device(phys_state& out);
#endif

} /* namespace jorek */

/* The only writer, called from mod_jgx_phys.f90. Declared here as well as in
 * that module's interface block so C++ callers. */
extern "C" void jgx_c_set_phys(double F0, double central_mass,
                               double central_density, double tstep,
                               int find_RZ_nearby_iter, double find_RZ_nearby_tol);

#endif /* JOREK_PHYS_H */
