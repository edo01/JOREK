#include "particles/pushers/mod_boris/boris.h"
#include "particles/particle_types/particle_set.h"

#include <cstddef>

extern "C" {
    /* mod_boris::boris_push_cylindrical.
     */
    void jgx_host_boris_push_cylindrical(void* part_base, const double m,
                                         const double* E, const double* B,
                                         const double dt) {

        auto part = jorek::particle_kin_lf_set_aos::from_aos(
            part_base, jorek::particle_kin_lf_set_aos::record(), 1);

        boris::boris_push_cylindrical(part, 0, m, E, B, dt);
    }
}
