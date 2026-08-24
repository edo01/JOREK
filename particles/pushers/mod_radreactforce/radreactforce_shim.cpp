/* The C entry point of radreactforce.h.
 *
 * The particle crosses as its record base and is rebuilt as a one-record set,
 * the same way the pusher's shim does it: the operator writes p in place, so it
 * needs the record and not a copy of three doubles. */
#include "particles/pushers/mod_radreactforce/radreactforce.h"
#include "particles/particle_types/particle_set.h"

#include <cstddef>
#include <cstdint>

extern "C" {

/* mod_radreactforce::radreactforce_kinetic */
void jgx_host_radreactforce_kinetic(void* part_base, const double* B,
                                    const double dt, const double mass) {
  auto part = jorek::particle_kin_rel_set_aos::from_aos(
      part_base, jorek::particle_kin_rel_set_aos::record(), 1);
  radreactforce::radreactforce_kinetic(part, 0, B, dt, mass);
}

} /* extern "C" */
