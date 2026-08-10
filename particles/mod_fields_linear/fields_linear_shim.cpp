#include "particles/mod_fields_linear/fields_linear.h"

#include <cstddef>
#include <cstdint>


extern "C" {
    /* mod_fields_linear::do_interp_PRZ_1.
     *
     * `this` is a class(fields_base) at the call site and a polymorphic dummy has
     * no interoperable form, so the facade stays Fortran and passes what the type
     * holds: the two mesh bases with their record counts, the flags of
     * fields_base and the two restart times of jorek_fields_interp_linear.
     *
     */
    void jgx_host_fields_linear_do_interp_PRZ_1(void* el_base, const int32_t n_elements,
                                                void* nd_base, const int32_t n_nodes,
                                                const int32_t is_static,
                                                const int32_t flag_zero_dpsidt,
                                                const double time_now, const double time_prev,
                                                const int32_t i_elm0, const int32_t* i_v0,
                                                const int32_t n_v,
                                                const double s, const double t, const double phi,
                                                const int32_t n_period,
                                                const double time, const double t_jorek,
                                                double* P, double* P_s, double* P_t,
                                                double* P_phi, double* P_time,
                                                double* R, double* R_s, double* R_t,
                                                double* Z, double* Z_s, double* Z_t) {

        // we get a linear field interpolator from the registry
        const auto f = jorek::fields_interp_linear_set_from_registry(
            el_base, static_cast<std::size_t>(n_elements),
            nd_base, static_cast<std::size_t>(n_nodes),
            is_static, flag_zero_dpsidt, time_now, time_prev);

        const std::size_t pe[1] = { static_cast<std::size_t>(n_v) };

        const jgx::view<const int32_t, 1> i_v(i_v0, pe);
        jgx::view<double, 1> Pv(P, pe), Psv(P_s, pe), Ptv(P_t, pe);
        jgx::view<double, 1> Ppv(P_phi, pe), Ptimev(P_time, pe);

        f.interp_PRZ(time, static_cast<std::size_t>(i_elm0),
                     i_v, n_v, s, t, phi, n_period, t_jorek,
                     Pv, Psv, Ptv, Ppv, Ptimev,
                     *R, *R_s, *R_t, *Z, *Z_s, *Z_t);
    }

    /* mod_fields_linear::do_interp_PRZP_1. As above, plus the toroidal
     * periodicity of the (R,Z) coordinates and their phi derivatives. */
    void jgx_host_fields_linear_do_interp_PRZP_1(void* el_base, const int32_t n_elements,
                                                 void* nd_base, const int32_t n_nodes,
                                                 const int32_t is_static,
                                                 const int32_t flag_zero_dpsidt,
                                                 const double time_now, const double time_prev,
                                                 const int32_t i_elm0, const int32_t* i_v0,
                                                 const int32_t n_v,
                                                 const double s, const double t, const double phi,
                                                 const int32_t n_period,
                                                 const int32_t n_coord_period,
                                                 const double time, const double t_jorek,
                                                 double* P, double* P_s, double* P_t,
                                                 double* P_phi, double* P_time,
                                                 double* R, double* R_s, double* R_t, double* R_phi,
                                                 double* Z, double* Z_s, double* Z_t, double* Z_phi) {

        // we get a linear field interpolator from the registry
        const auto f = jorek::fields_interp_linear_set_from_registry(
            el_base, static_cast<std::size_t>(n_elements),
            nd_base, static_cast<std::size_t>(n_nodes),
            is_static, flag_zero_dpsidt, time_now, time_prev);

        const std::size_t pe[1] = { static_cast<std::size_t>(n_v) };

        const jgx::view<const int32_t, 1> i_v(i_v0, pe);
        jgx::view<double, 1> Pv(P, pe), Psv(P_s, pe), Ptv(P_t, pe);
        jgx::view<double, 1> Ppv(P_phi, pe), Ptimev(P_time, pe);

        // the interpolation strategy is deferred exactly as it happens in fortran.
        f.interp_PRZP_1(time, static_cast<std::size_t>(i_elm0),
                        i_v, n_v, s, t, phi, n_period, n_coord_period,
                        t_jorek, Pv, Psv, Ptv, Ppv, Ptimev,
                        *R, *R_s, *R_t, *R_phi, *Z, *Z_s, *Z_t, *Z_phi);
    }
}
