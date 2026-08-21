/* particles/mod_runaway_evolution/runaway_evolution_host.h -- the host launcher
 * of the runaway-electron evolution.
 */
#ifndef JOREK_RUNAWAY_EVOLUTION_HOST_H
#define JOREK_RUNAWAY_EVOLUTION_HOST_H

#include <cstddef>

#include "particles/mod_runaway_evolution/runaway_evolution.h"

namespace jorek {

/**
 * mod_runaway_evolution::evolve_REs
 *
 * @param rhs_data  first element of feedback_rhs
 * @param rhs_ext   its five extents, in Fortran declaration order
 * @see evolve_RE for the remaining parameters
 */
template<bool Debug, class PS, class FS>
inline void evolve_REs(PS& part, const FS& fields,
                       double* rhs_data, const std::size_t rhs_ext[5],
                       const re_projection_indices& idx,
                       const double mass, const double time,
                       const double timestep, const int nstep,
                       const double phi_search,
                       kinetic_relativistic::push_diagnostics& diag) {
    const std::size_t n_particles = part.n_records;
    const std::size_t rhs_size = rhs_ext[0]*rhs_ext[1]*rhs_ext[2]*rhs_ext[3]*rhs_ext[4];

    #pragma omp parallel reduction(+: rhs_data[0:rhs_size])
    {
        /*
         * rhs arrives as a bare pointer and its extents rather than as a view, because
         * the reduction clause redirects rhs_data at a per-thread copy and only a view
         * built inside the region points at it. Building one outside would have every
         * thread accumulate into the original, which is both a race and a double count.
         *
         * That private copy is also why evolve_RE's jgx::atomic_add costs nothing
         * here: there is no sharing left for an atomic to protect.
         */
        const re_rhs_view rhs(rhs_data, rhs_ext);

        /* Each thread keeps its own first occurrence and the merge below takes
         * whichever arrives first, so which one survives depends on the
         * schedule -- as it did in the Fortran, where the message came from
         * whichever thread reached the critical section first. */
        kinetic_relativistic::push_diagnostics my_diag;

        #pragma omp for schedule(runtime)
        for (std::size_t ip = 0; ip < n_particles; ++ip)
            evolve_RE<Debug>(part, ip, fields, rhs, idx, mass, time, timestep,
                             nstep, phi_search, my_diag);

        #pragma omp critical
        {
            if (my_diag.not_found != 0 && diag.not_found == 0) {
                diag.not_found = my_diag.not_found;
                diag.nf_R      = my_diag.nf_R;
                diag.nf_Z      = my_diag.nf_Z;
            }
            if (my_diag.bad_i_to != 0 && diag.bad_i_to == 0) {
                diag.bad_i_from = my_diag.bad_i_from;
                diag.bad_i_to   = my_diag.bad_i_to;
            }
        }
    }
} // evolve_REs

} // namespace jorek

#endif // JOREK_RUNAWAY_EVOLUTION_HOST_H
