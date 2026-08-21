#include "particles/mod_runaway_evolution/runaway_evolution_host.h"
#include "particles/mod_fields_linear/fields_linear.h"
#include "particles/particle_types/particle_set.h"

#ifdef JGX_HAS_DEVICE
#include "particles/mod_runaway_evolution/runaway_evolution_device.h"
#endif

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

namespace { //util functions

/* What the entry points share: the same argument list arrives at each of them,
 * because the Fortran facade picks between them through one interface. */

void ext_from(const std::int32_t* rhs_ext, std::size_t ext[5]) {
    for (int d = 0; d < 5; ++d) ext[d] = static_cast<std::size_t>(rhs_ext[d]);
}

jorek::re_projection_indices idx_from(std::int32_t i_P_par, std::int32_t i_P_perp,
                                      std::int32_t i_j_Phi) {
    return { static_cast<std::size_t>(i_P_par),
             static_cast<std::size_t>(i_P_perp),
             static_cast<std::size_t>(i_j_Phi) };
}

void report(const kinetic_relativistic::push_diagnostics& diag,
            std::int32_t* not_found, double* nf_R, double* nf_Z,
            std::int32_t* bad_i_from, std::int32_t* bad_i_to) {
    *not_found  = static_cast<std::int32_t>(diag.not_found);
    *nf_R       = diag.nf_R;
    *nf_Z       = diag.nf_Z;
    *bad_i_from = static_cast<std::int32_t>(diag.bad_i_from);
    *bad_i_to   = static_cast<std::int32_t>(diag.bad_i_to);
}

} /* anonymous namespace */


extern "C" {
/**
 * @todo :  too many parameters. Some of them used to live in the sim object which 
 *          doesn't cross the seam.
 */
    /* mod_runaway_evolution::evolve_REs - host execution entry. */
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

        auto part = jorek::particle_kin_rel_set_aos::from_aos(
            part_base, jorek::particle_kin_rel_set_aos::record(),
            static_cast<std::size_t>(n_particles));

        const auto fields = jorek::fields_linear_set_from_registry(
            el_base, static_cast<std::size_t>(n_elements),
            nd_base, static_cast<std::size_t>(n_nodes),
            interp_base);

        std::size_t ext[5];
        ext_from(rhs_ext, ext);

        kinetic_relativistic::push_diagnostics diag;

        jorek::evolve_REs<kFindRZNearbyDebug>(
            part, fields, rhs_data, ext, idx_from(i_P_par, i_P_perp, i_j_Phi),
            mass, time, timestep, static_cast<int>(nstep), phi_search, diag);

        report(diag, not_found, nf_R, nf_Z, bad_i_from, bad_i_to);
    }

#ifdef JGX_HAS_DEVICE
     /* mod_runaway_evolution::evolve_REs - device execution entry. */
    void jgx_host_runaway_evolution_evolve_REs_device(void* part_base, const int32_t n_particles,
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

        std::size_t ext[5];
        ext_from(rhs_ext, ext);

        const std::size_t idx[3] = { static_cast<std::size_t>(i_P_par),
                                     static_cast<std::size_t>(i_P_perp),
                                     static_cast<std::size_t>(i_j_Phi) };

        jgx_device_runaway_evolution_evolve_REs(
            part_base,
            &jorek::particle_kin_rel_set_aos::record(),
            static_cast<std::size_t>(n_particles),
            el_base, &jorek::element_set_aos::record(), static_cast<std::size_t>(n_elements),
            nd_base, &jorek::node_set_aos::record(), static_cast<std::size_t>(n_nodes),
            interp_base,
            &jgx::data::registered_record(jorek::JGX_REC_FIELDS_INTERP_LINEAR, jorek::JGX_FIL_COUNT),
            rhs_data, ext, idx,
            mass, time, timestep, nstep, phi_search,
            not_found, nf_R, nf_Z, bad_i_from, bad_i_to);
    }
#endif /* JGX_HAS_DEVICE */
}
