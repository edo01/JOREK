#include "particles/mod_epf_evolution/epf_evolution_host.h"
#include "particles/mod_fields_linear/fields_linear.h"
#include "particles/particle_types/particle_set.h"

#ifdef JGX_HAS_DEVICE
#include "particles/mod_epf_evolution/epf_evolution_device.h"
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

/* The seven indices arrive as one array in epf_var order, so the facade names
 * them once and neither side re-orders them. */
void idx_from(const std::int32_t* proj_idx, std::size_t idx[jorek::EPF_N_VAR]) {
    for (int v = 0; v < jorek::EPF_N_VAR; ++v)
        idx[v] = static_cast<std::size_t>(proj_idx[v]);
}

void report(const find_rz_nearby::diagnostics& diag,
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
    /* mod_epf_evolution::evolve_epf - host execution entry. */
    void jgx_host_epf_evolution_evolve_epf(void* part_base, const int32_t n_particles,
                                           void* el_base, const int32_t n_elements,
                                           void* nd_base, const int32_t n_nodes,
                                           const void* interp_base,
                                           double* rhs_data, const int32_t* rhs_ext,
                                           const int32_t* proj_idx,
                                           const double mass, const double time,
                                           const double timestep,
                                           const int32_t nstep,
                                           const int32_t proj_period,
                                           const double phi_search,
                                           int32_t* not_found,
                                           double* nf_R, double* nf_Z,
                                           int32_t* bad_i_from, int32_t* bad_i_to) {

        auto part = jorek::particle_kin_lf_set_aos::from_aos(
            part_base, jorek::particle_kin_lf_set_aos::record(),
            static_cast<std::size_t>(n_particles));

        const auto fields = jorek::fields_linear_set_from_registry(
            el_base, static_cast<std::size_t>(n_elements),
            nd_base, static_cast<std::size_t>(n_nodes),
            interp_base);

        std::size_t ext[5];
        ext_from(rhs_ext, ext);

        jorek::epf_projection_indices idx;
        idx_from(proj_idx, idx.var);

        find_rz_nearby::diagnostics diag;

        jorek::evolve_epf<kFindRZNearbyDebug>(
            part, fields, rhs_data, ext, idx,
            mass, time, timestep, static_cast<int>(nstep),
            static_cast<int>(proj_period), phi_search, diag);

        report(diag, not_found, nf_R, nf_Z, bad_i_from, bad_i_to);
    }

#ifdef JGX_HAS_DEVICE
     /* mod_epf_evolution::evolve_epf - device execution entry. */
    void jgx_host_epf_evolution_evolve_epf_device(void* part_base, const int32_t n_particles,
                                                  void* el_base, const int32_t n_elements,
                                                  void* nd_base, const int32_t n_nodes,
                                                  const void* interp_base,
                                                  double* rhs_data, const int32_t* rhs_ext,
                                                  const int32_t* proj_idx,
                                                  const double mass, const double time,
                                                  const double timestep,
                                                  const int32_t nstep,
                                                  const int32_t proj_period,
                                                  const double phi_search,
                                                  int32_t* not_found,
                                                  double* nf_R, double* nf_Z,
                                                  int32_t* bad_i_from, int32_t* bad_i_to) {

        std::size_t ext[5];
        ext_from(rhs_ext, ext);

        std::size_t idx[jorek::EPF_N_VAR];
        idx_from(proj_idx, idx);

        jgx_device_epf_evolution_evolve_epf(
            part_base,
            &jorek::particle_kin_lf_set_aos::record(),
            static_cast<std::size_t>(n_particles),
            el_base, &jorek::element_set_aos::record(), static_cast<std::size_t>(n_elements),
            nd_base, &jorek::node_set_aos::record(), static_cast<std::size_t>(n_nodes),
            interp_base,
            &jgx::data::registered_record(jorek::JGX_REC_FIELDS_INTERP_LINEAR, jorek::JGX_FIL_COUNT),
            rhs_data, ext, idx,
            mass, time, timestep, nstep, proj_period, phi_search,
            not_found, nf_R, nf_Z, bad_i_from, bad_i_to);
    }
#endif /* JGX_HAS_DEVICE */
}
