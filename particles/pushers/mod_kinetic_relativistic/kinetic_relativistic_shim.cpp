#include "particles/pushers/mod_kinetic_relativistic/kinetic_relativistic.h"
#include "particles/mod_fields_linear/fields_linear.h"
#include "particles/particle_types/particle_set.h"

#include <cstddef>
#include <cstdint>

/* find_RZ_nearby underneath takes the DEBUG build as a template parameter
 * rather than reading the preprocessor itself; the shim is the place that has
 * to know. Same switch as find_rz_nearby_shim.cpp. */
#ifdef DEBUG
static constexpr bool kFindRZNearbyDebug = true;
#else
static constexpr bool kFindRZNearbyDebug = false;
#endif


extern "C" {
    /* mod_kinetic_relativistic::volume_preserving_push_jorek.
     *
     * The interpolator is jorek_fields_interp_linear; the facade picks it, and
     * sends every other strategy down the Fortran path.
     * 
     * The particle set is a particle_kin_rel_set.
     */
    void jgx_host_volume_preserving_push_jorek(void* part_base,
                                               void* el_base, const int32_t n_elements,
                                               void* nd_base, const int32_t n_nodes,
                                               const void* interp_base,
                                               const double mass, const double time,
                                               const double timestep,
                                               const double phi_search,
                                               int32_t* ifail,
                                               int32_t* not_found,
                                               double* nf_R, double* nf_Z,
                                               int32_t* bad_i_from, int32_t* bad_i_to) {

        auto part = jorek::particle_kin_rel_set_from_registry(part_base, 1);

        const auto fields = jorek::fields_linear_set_from_registry(
            el_base, static_cast<std::size_t>(n_elements),
            nd_base, static_cast<std::size_t>(n_nodes),
            interp_base);

        int fail = static_cast<int>(*ifail);
        kinetic_relativistic::push_diagnostics diag;

        kinetic_relativistic::volume_preserving_push_jorek<kFindRZNearbyDebug>(
            part, 0, fields, mass, time, timestep, phi_search, fail, diag);

        *ifail      = static_cast<int32_t>(fail);
        *not_found  = static_cast<int32_t>(diag.not_found);
        *nf_R       = diag.nf_R;
        *nf_Z       = diag.nf_Z;
        *bad_i_from = static_cast<int32_t>(diag.bad_i_from);
        *bad_i_to   = static_cast<int32_t>(diag.bad_i_to);
    }
}
