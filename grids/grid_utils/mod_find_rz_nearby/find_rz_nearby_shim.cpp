#include "grids/grid_utils/mod_find_rz_nearby/find_rz_nearby.h"
#include "datatypes/data_structure/element_set.h"
#include "datatypes/data_structure/node_set.h"

#include <cstddef>
#include <cstdint>

/* The kernel takes the DEBUG build as a template parameter rather than reading
 * the preprocessor itself, so the ported body stays free of #ifdef. This is the
 * one place that has to know. */
#ifdef DEBUG
static constexpr bool kFindRZNearbyDebug = true;
#else
static constexpr bool kFindRZNearbyDebug = false;
#endif


extern "C" {
    /* mod_find_rz_nearby::find_RZ_nearby.
     *
     * i_elm_old / i_elm_new are mesh ids (1-based) rather than indices: both
     * carry sentinels (0, and a negative i_elm_new for a lost particle), so
     * neither survives a shift. phi_search is the facade's business -- see
     * find_rz_nearby.h.
     */
    void jgx_host_find_RZ_nearby(void* el_base, const int32_t n_elements,
                                 void* nd_base, const int32_t n_nodes,
                                 const double R_old, const double Z_old,
                                 const double s_old, const double t_old,
                                 const int32_t i_elm_old,
                                 const double R_new, const double Z_new,
                                 const double p, const double phi_search,
                                 double* s_new, double* t_new,
                                 int32_t* i_elm_new, int32_t* ifail,
                                 int32_t* not_found,
                                 int32_t* bad_i_from, int32_t* bad_i_to) {

        const auto el = jorek::element_set_aos::from_registry(el_base, static_cast<std::size_t>(n_elements));
        const auto nd = jorek::node_set_aos::from_registry(nd_base, static_cast<std::size_t>(n_nodes));

        int ielm = 0, fail = 0, nf = 0, bfrom = 0, bto = 0;
        find_rz_nearby::find_RZ_nearby<kFindRZNearbyDebug>(
            el, nd, R_old, Z_old, s_old, t_old, static_cast<int>(i_elm_old),
            R_new, Z_new, p, phi_search, *s_new, *t_new, ielm, fail, nf, bfrom, bto);

        *i_elm_new  = static_cast<int32_t>(ielm);
        *ifail      = static_cast<int32_t>(fail);
        *not_found  = static_cast<int32_t>(nf);
        *bad_i_from = static_cast<int32_t>(bfrom);
        *bad_i_to   = static_cast<int32_t>(bto);
    }
}
