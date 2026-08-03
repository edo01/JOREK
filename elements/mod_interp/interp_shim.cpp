#include "elements/mod_interp/interp.h"
#include "datatypes/data_structure/element_set.h"
#include "datatypes/data_structure/node_set.h"

#include <cstddef>
#include <cstdint>


extern "C" {
    void jgx_host_sincosperiod_moivre_explicit(const double phi,
                                               double* HZ, double* dHZ,
                                               const int n_tor_in, const int n_period_in) {

        // Preparation of the views: HZ/dHZ are rank-1, of extent n_tor_in
        const std::size_t ext[1] = { static_cast<std::size_t>(n_tor_in) };
        const jgx::view<double, 1> HZ_view (HZ,  ext);
        const jgx::view<double, 1> dHZ_view(dHZ, ext);

        interp::sincosperiod_moivre_explicit(phi, HZ_view, dHZ_view, n_tor_in, n_period_in);
    }

    /* mod_interp::interp_PRZ_1.
     *
     * The mesh crosses the seam as a base pointer plus a record count; the
     * layout comes from the registry (element_set.h, node_set.h). Workspace is
     * supplied by the caller -- Fortran already declares these as automatic
     * arrays, and on device the placement is the launcher's decision.
     *
     * i_elm0 and i_v0 are 0-based; the facade converts.
     */
    void jgx_host_interp_PRZ_1(void* el_base, const int32_t n_elements,
                               void* nd_base, const int32_t n_nodes,
                               const int32_t i_elm0, const int32_t* i_v0,
                               const int32_t n_v,
                               const double s, const double t, const double phi,
                               const int32_t n_period, const int32_t use_deltas,
                               double* P, double* P_s, double* P_t, double* P_phi,
                               double* R, double* R_s, double* R_t,
                               double* Z, double* Z_s, double* Z_t,
                               double* w_values, double* w_xR, double* w_xZ,
                               double* w_H, double* w_H_s, double* w_H_t,
                               double* w_HZ, double* w_dHZ) {

        const auto el = jorek::element_set_from_registry(el_base, static_cast<std::size_t>(n_elements));
        const auto nd = jorek::node_set_from_registry(nd_base, static_cast<std::size_t>(n_nodes));

        const std::size_t n_vertex_max = el.vertex.extent[1];
        const std::size_t n_degrees    = el.size.extent[2];
        const std::size_t n_tor        = nd.values.extent[1];
        const std::size_t nv           = static_cast<std::size_t>(n_v);

        const std::size_t ve[4] = { n_tor, n_degrees, nv, n_vertex_max };
        const std::size_t xe[2] = { n_degrees, n_vertex_max };
        const std::size_t ze[1] = { n_tor };
        const std::size_t pe[1] = { nv };

        interp::interp_PRZ_workspace<double> w;
        w.values = jgx::view<double, 4>(w_values, ve);
        w.xR     = jgx::view<double, 2>(w_xR,  xe);
        w.xZ     = jgx::view<double, 2>(w_xZ,  xe);
        w.H      = jgx::view<double, 2>(w_H,   xe);
        w.H_s    = jgx::view<double, 2>(w_H_s, xe);
        w.H_t    = jgx::view<double, 2>(w_H_t, xe);
        w.HZ     = jgx::view<double, 1>(w_HZ,  ze);
        w.dHZ    = jgx::view<double, 1>(w_dHZ, ze);

        const jgx::view<const int32_t, 1> i_v(i_v0, pe);
        jgx::view<double, 1> Pv(P, pe), Psv(P_s, pe), Ptv(P_t, pe), Ppv(P_phi, pe);

        interp::interp_PRZ_1(el, nd, w, static_cast<std::size_t>(i_elm0),
                             i_v, n_v, s, t, phi, n_period, use_deltas != 0,
                             Pv, Psv, Ptv, Ppv, *R, *R_s, *R_t, *Z, *Z_s, *Z_t);
    }
}
