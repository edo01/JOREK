#include "grids/grid_utils/find_RZ/find_RZ.h"
#include "datatypes/data_structure/element_set.h"
#include "datatypes/data_structure/node_set.h"

#include <cstddef>
#include <cstdint>


extern "C" {
    /* find_RZ_general. ielm_out is a mesh id (1-based) or 0, not an index, so
     * it crosses unconverted -- 0 is a sentinel. Both find_RZ and find_RZP
     * reach this through the same shim; the phi they differ over is worked out
     * on the Fortran side, where n_plane / i_plane_rtree / n_coord_period live.
     */
    void jgx_host_find_RZ_general(void* el_base, const int32_t n_elements,
                                  void* nd_base, const int32_t n_nodes,
                                  const double R_find, const double Z_find,
                                  const double phi_find,
                                  double* R_out, double* Z_out, int32_t* ielm_out,
                                  double* s_out, double* t_out,
                                  int32_t* ifail, int32_t* checked_elms) {

        const auto el = jorek::element_set_aos::from_registry(el_base, static_cast<std::size_t>(n_elements));
        const auto nd = jorek::node_set_aos::from_registry(nd_base, static_cast<std::size_t>(n_nodes));

        int ielm = 0, fail = 0, checked = 0;
        find_rz::find_RZ_general(el, nd, R_find, Z_find, phi_find,
                                 *R_out, *Z_out, ielm, *s_out, *t_out, fail, checked);

        *ielm_out     = static_cast<int32_t>(ielm);
        *ifail        = static_cast<int32_t>(fail);
        *checked_elms = static_cast<int32_t>(checked);
    }
}
