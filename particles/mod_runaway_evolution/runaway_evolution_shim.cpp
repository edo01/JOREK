#include "particles/mod_runaway_evolution/runaway_evolution.h"
#include "particles/mod_fields_linear/fields_linear.h"
#include "particles/particle_types/particle_set.h"

#include <cstddef>
#include <cstdint>

/* find_RZ_nearby underneath takes the DEBUG build as a template parameter
 * rather than reading the preprocessor itself; the shim is the place that has
 * to know. Same switch as kinetic_relativistic_shim.cpp. */
#ifdef DEBUG
static constexpr bool kFindRZNearbyDebug = true;
#else
static constexpr bool kFindRZNearbyDebug = false;
#endif


extern "C" {
    /* mod_runaway_evolution::evolve_REs.
     *
     * The whole particle group crosses at once -- one call for the loop, not
     * one per particle -- so the field set is built here rather than rebuilt
     * per push.
     *
     * The interpolator is jorek_fields_interp_linear; the facade picks it and
     * stops on anything else.
     */
    void jgx_host_runaway_evolution_evolve_REs(void* part_base, const int32_t n_particles,
                                                void* el_base, const int32_t n_elements,
                                                void* nd_base, const int32_t n_nodes,
                                                const void* interp_base,
                                                double* rhs_data, const int32_t* rhs_ext,
                                                const int32_t i_P_par,
                                                const int32_t i_P_perp,
                                                const int32_t i_j_Phi,
                                                const double mass, const double time,
                                                const double timestep,
                                                const int32_t nstep,
                                                const double phi_search,
                                                int32_t* not_found,
                                                double* nf_R, double* nf_Z,
                                                int32_t* bad_i_from, int32_t* bad_i_to) {

        auto part = jorek::particle_kin_rel_set_from_registry(
            part_base, static_cast<std::size_t>(n_particles));

        const auto fields = jorek::fields_linear_set_from_registry(
            el_base, static_cast<std::size_t>(n_elements),
            nd_base, static_cast<std::size_t>(n_nodes),
            interp_base);

        std::size_t ext[5];
        for (int d = 0; d < 5; ++d) ext[d] = static_cast<std::size_t>(rhs_ext[d]);

        const jorek::re_projection_indices idx = {
            static_cast<std::size_t>(i_P_par),
            static_cast<std::size_t>(i_P_perp),
            static_cast<std::size_t>(i_j_Phi)
        };

        kinetic_relativistic::push_diagnostics diag;

        jorek::evolve_REs<kFindRZNearbyDebug>(part, fields, rhs_data, ext, idx,
                                              mass, time, timestep,
                                              static_cast<int>(nstep), phi_search, diag);

        *not_found  = static_cast<int32_t>(diag.not_found);
        *nf_R       = diag.nf_R;
        *nf_Z       = diag.nf_Z;
        *bad_i_from = static_cast<int32_t>(diag.bad_i_from);
        *bad_i_to   = static_cast<int32_t>(diag.bad_i_to);
    }
}
