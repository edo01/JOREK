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

    void jgx_host_sincosperiod_moivre_ncoord(const double phi,
                                             double* HZ_coord, double* dHZ_coord,
                                             const int n_coord_tor_in, const int n_coord_period_in) {

        const std::size_t ext[1] = { static_cast<std::size_t>(n_coord_tor_in) };
        const jgx::view<double, 1> HZ_view (HZ_coord,  ext);
        const jgx::view<double, 1> dHZ_view(dHZ_coord, ext);

        interp::sincosperiod_moivre_ncoord(phi, HZ_view, dHZ_view,
                                           n_coord_tor_in, n_coord_period_in);
    }

    /* mod_interp::interp_PRZ_1.
     *
     * The mesh crosses the seam as a base pointer plus a record count; the
     * layout comes from the registry (element_set.h, node_set.h).
     */
    void jgx_host_interp_PRZ_1(void* el_base, const int32_t n_elements,
                               void* nd_base, const int32_t n_nodes,
                               const int32_t i_elm0, const int32_t* i_v0,
                               const int32_t n_v,
                               const double s, const double t, const double phi,
                               const int32_t n_period, const int32_t use_deltas,
                               double* P, double* P_s, double* P_t, double* P_phi,
                               double* R, double* R_s, double* R_t,
                               double* Z, double* Z_s, double* Z_t) {

        // Extracting the AoS view from the registry
        const auto el = jorek::element_set_from_registry(el_base, static_cast<std::size_t>(n_elements));
        const auto nd = jorek::node_set_from_registry(nd_base, static_cast<std::size_t>(n_nodes));

        const std::size_t pe[1] = { static_cast<std::size_t>(n_v) };

        // Preparing the views for interp_PRZ_1
        const jgx::view<const int32_t, 1> i_v(i_v0, pe);
        jgx::view<double, 1> Pv(P, pe), Psv(P_s, pe), Ptv(P_t, pe), Ppv(P_phi, pe);

        interp::interp_PRZ_1(el, nd, static_cast<std::size_t>(i_elm0),
                             i_v, n_v, s, t, phi, n_period, use_deltas != 0,
                             Pv, Psv, Ptv, Ppv, *R, *R_s, *R_t, *Z, *Z_s, *Z_t);
    }

    /* mod_interp::interp_PRZP_1. As jgx_host_interp_PRZ_1, plus the toroidal
     * periodicity of the (R,Z) coordinates and their phi derivatives. */
    void jgx_host_interp_PRZP_1(void* el_base, const int32_t n_elements,
                                void* nd_base, const int32_t n_nodes,
                                const int32_t i_elm0, const int32_t* i_v0,
                                const int32_t n_v,
                                const double s, const double t, const double phi,
                                const int32_t n_period, const int32_t n_coord_period,
                                const int32_t use_deltas,
                                double* P, double* P_s, double* P_t, double* P_phi,
                                double* R, double* R_s, double* R_t, double* R_phi,
                                double* Z, double* Z_s, double* Z_t, double* Z_phi) {

        const auto el = jorek::element_set_from_registry(el_base, static_cast<std::size_t>(n_elements));
        const auto nd = jorek::node_set_from_registry(nd_base, static_cast<std::size_t>(n_nodes));

        const std::size_t pe[1] = { static_cast<std::size_t>(n_v) };

        const jgx::view<const int32_t, 1> i_v(i_v0, pe);
        jgx::view<double, 1> Pv(P, pe), Psv(P_s, pe), Ptv(P_t, pe), Ppv(P_phi, pe);

        interp::interp_PRZP_1(el, nd, static_cast<std::size_t>(i_elm0),
                              i_v, n_v, s, t, phi, n_period, n_coord_period,
                              use_deltas != 0, Pv, Psv, Ptv, Ppv,
                              *R, *R_s, *R_t, *R_phi, *Z, *Z_s, *Z_t, *Z_phi);
    }
}
