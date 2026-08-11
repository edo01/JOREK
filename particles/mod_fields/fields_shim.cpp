#include "particles/mod_fields_linear/fields_linear.h"

#include <cstddef>
#include <cstdint>


extern "C" {
    /* mod_fields::calc_EBpsiU_reduced.*/
    void jgx_host_fields_calc_EBpsiU_reduced(void* el_base, const int32_t n_elements,
                                             void* nd_base, const int32_t n_nodes,
                                             const void* interp_base,
                                             const int32_t i_elm0,
                                             const double s, const double t, const double phi,
                                             const double time,
                                             double* E, double* B,
                                             double* psi, double* U) {

        // the fields: the grid, and the strategy that interpolates it in time
        const auto f = jorek::fields_linear_set_from_registry(
            el_base, static_cast<std::size_t>(n_elements),
            nd_base, static_cast<std::size_t>(n_nodes),
            interp_base);

        const std::size_t ve[1] = { 3 };
        jgx::view<double, 1> Ev(E, ve), Bv(B, ve);

        f.calc_EBpsiU_reduced(time, static_cast<std::size_t>(i_elm0),
                              s, t, phi, Ev, Bv, *psi, *U);
    }

    /* mod_fields::grad_st_to_RZ. */
    void jgx_host_fields_grad_st_to_RZ(const int32_t n_v,
                                       const double* P_s, const double* P_t,
                                       const double R_s, const double R_t,
                                       const double Z_s, const double Z_t,
                                       double* P_R, double* P_Z) {

        const std::size_t pe[1] = { static_cast<std::size_t>(n_v) };

        const jgx::view<const double, 1> P_sv(P_s, pe), P_tv(P_t, pe);
        jgx::view<double, 1> P_Rv(P_R, pe), P_Zv(P_Z, pe);

        jorek::grad_st_to_RZ(n_v, P_sv, P_tv, R_s, R_t, Z_s, Z_t, P_Rv, P_Zv);
    }

    /* mod_fields::EB_from_psiU. */
    void jgx_host_fields_EB_from_psiU(const double R_inv, const double F0,
                                      const double t_norm,
                                      const double psi_R, const double psi_Z,
                                      const double U_R, const double U_Z,
                                      const double U_phi, const double psi_time,
                                      double* E, double* B) {

        const std::size_t ve[1] = { 3 };
        jgx::view<double, 1> Ev(E, ve), Bv(B, ve);

        jorek::EB_from_psiU(R_inv, F0, t_norm, psi_R, psi_Z, U_R, U_Z, U_phi,
                            psi_time, Ev, Bv);
    }
}
