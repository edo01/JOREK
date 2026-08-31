/* particles/mod_epf_evolution/epf_evolution_host.h -- the host launcher of the
 * energetic-particle evolution.
 */
#ifndef JOREK_EPF_EVOLUTION_HOST_H
#define JOREK_EPF_EVOLUTION_HOST_H

#include <cstddef>

#include "particles/mod_epf_evolution/epf_evolution.h"

namespace jorek {

/**
 * mod_epf_evolution::evolve_epf
 *
 * @param rhs_data  first element of feedback_rhs
 * @param rhs_ext   its five extents, in Fortran declaration order
 * @see evolve_epf_particle for the remaining parameters
 */
template<bool Debug, class PS, class FS>
inline void evolve_epf(PS& part, const FS& fields,
                       double* rhs_data, const std::size_t rhs_ext[5],
                       const epf_projection_indices& idx,
                       const double mass, const double time,
                       const double timestep, const int nstep,
                       const int proj_period, const double phi_search,
                       find_rz_nearby::diagnostics& diag) {
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
         * That private copy is also why epf_deposit's jgx::atomic_add costs nothing
         * here: there is no sharing left for an atomic to protect.
         */
        const epf_rhs_view rhs(rhs_data, rhs_ext);

        /* Each thread keeps its own first occurrence and the merge below takes
         * whichever arrives first, so which one survives depends on the
         * schedule -- as it did in the Fortran, where the message came from
         * whichever thread reached the critical section first. */
        find_rz_nearby::diagnostics my_diag;

        #pragma omp for schedule(runtime)
        for (std::size_t ip = 0; ip < n_particles; ++ip)
            evolve_epf_particle<Debug>(part, ip, fields, rhs, idx, mass, time,
                                       timestep, nstep, proj_period, phi_search,
                                       my_diag);

        #pragma omp critical
        {
            find_rz_nearby::merge_diagnostics(diag, my_diag.not_found,
                                              my_diag.nf_R, my_diag.nf_Z,
                                              my_diag.bad_i_from, my_diag.bad_i_to);
        }
    }
} // evolve_epf

} // namespace jorek

#endif // JOREK_EPF_EVOLUTION_HOST_H
