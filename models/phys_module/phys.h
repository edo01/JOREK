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
 */
#ifndef JOREK_PHYS_H
#define JOREK_PHYS_H

namespace jorek {

struct phys_state {
  double F0              = 0;  /* fixed toroidal field factor, B_phi = F0/R */
  double central_mass    = 0;  /* average ion mass [amu] */
  double central_density = 0;  /* density on axis [10^20 m^-3] */
  double tstep           = 0;  /* fluid time step [JOREK units] */

  /* Derived here rather than passed, so the seam carries state and not results.
   * Same expression and same association as phys_module's callers use. */
  double t_norm          = 0;  /* one JOREK time unit [s] */
  double t_jorek         = 0;  /* tstep * t_norm [s] */
};

/* The current copy. Aborts if mod_jgx_phys has not pushed one yet. */
const phys_state& phys();

} /* namespace jorek */

#endif /* JOREK_PHYS_H */
