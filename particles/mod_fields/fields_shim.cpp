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
}
