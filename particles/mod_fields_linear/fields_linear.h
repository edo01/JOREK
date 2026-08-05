#ifndef FIELDS_LINEAR_H
#define FIELDS_LINEAR_H

#include <cmath>
#include <cstddef>

#include "elements/mod_interp/interp.h"
#include "jgx/jorek/jorek_settings.h"
#include "jgx/macros.h"
#include "jgx/view.h"

namespace fields_linear
{
    /**
     * mod_fields_linear::do_interp_PRZ_1 -- interpolate n_v variables and the
     * geometry at (s,t,phi) in element ie, then interpolate linearly in time
     * between the two restart files the field set holds.
     *
     * @param f         the field interpolator
     * @param time      time to interpolate at, SI
     * @param ie        element index into f.element_list
     * @param i_v       indices of the n_v variables to interpolate
     * @param n_v       number of variables, <= JGX_N_VALUES_MAX
     * @param s, t      element-local coordinates
     * @param phi       toroidal angle
     * @param n_period  toroidal periodicity
     * @param t_jorek   one JOREK time unit in seconds, times tstep; <= 0 disables
     *                  the time interpolation.
     * @param[out] P, P_s, P_t, P_phi, P_time  variables, their s/t/phi derivatives
     *                  and their time derivative (size n_v)
     * @param[out] R, R_s, R_t, Z, Z_s, Z_t    geometry and its derivatives
     */
    template<class FS, class IdxView, class OutView, class Real = double>
    JGX_HD inline void do_interp_PRZ_1(const FS& f, const double time,
                                       const std::size_t ie,
                                       const IdxView& i_v, const int n_v,
                                       const double s, const double t, const double phi,
                                       const int n_period, const double t_jorek,
                                       OutView P, OutView P_s, OutView P_t,
                                       OutView P_phi, OutView P_time,
                                       double& R, double& R_s, double& R_t,
                                       double& Z, double& Z_s, double& Z_t) {
        for (int i = 0; i < n_v; ++i) P_time(i) = 0.0;

        interp::interp_PRZ_1(f.element_list, f.node_list, ie, i_v, n_v, s, t, phi,
                             n_period, false, P, P_s, P_t, P_phi,
                             R, R_s, R_t, Z, Z_s, Z_t);

        if (t_jorek <= 0.0) return;

        // The deltas, in the scratch the Fortran gets from automatic arrays.
        const std::size_t pe[1] = { static_cast<std::size_t>(n_v) };
        Real Pd_[JGX_N_VALUES_MAX], Pd_s_[JGX_N_VALUES_MAX];
        Real Pd_t_[JGX_N_VALUES_MAX], Pd_phi_[JGX_N_VALUES_MAX];
        const jgx::view<Real, 1> Pd(Pd_, pe), Pd_s(Pd_s_, pe);
        const jgx::view<Real, 1> Pd_t(Pd_t_, pe), Pd_phi(Pd_phi_, pe);

        // R..Z_t come out identical: the geometry does not read values/deltas.
        interp::interp_PRZ_1(f.element_list, f.node_list, ie, i_v, n_v, s, t, phi,
                             n_period, true, Pd, Pd_s, Pd_t, Pd_phi,
                             R, R_s, R_t, Z, Z_s, Z_t);

        double dt;
        if (fabs(f.time_now - f.time_prev) > 1e-10 && !f.is_static) {
            dt = 1.0/(f.time_now - f.time_prev);
            const double df = (f.time_now - time)*dt;
            for (int i = 0; i < n_v; ++i) {
                P    (i) -= df*Pd    (i);
                P_s  (i) -= df*Pd_s  (i);
                P_t  (i) -= df*Pd_t  (i);
                P_phi(i) -= df*Pd_phi(i);
            }
        } else {
            dt = 1.0/t_jorek;
        }

        for (int i = 0; i < n_v; ++i) P_time(i) = Pd(i)*dt;
    } // do_interp_PRZ_1

    /**
     * mod_fields_linear::do_interp_PRZP_1 -- as do_interp_PRZ_1, for a
     * non-axisymmetric configuration: the geometry has phi derivatives of its own.
     *
     * @param f               the field interpolator
     * @param time            time to interpolate at, SI
     * @param ie              element index into f.element_list
     * @param i_v             indices of the n_v variables to interpolate
     * @param n_v             number of variables, <= JGX_N_VALUES_MAX
     * @param s, t            element-local coordinates
     * @param phi             toroidal angle
     * @param n_period        toroidal periodicity
     * @param n_coord_period  toroidal periodicity of the (R,Z) coordinates
     * @param t_jorek         one JOREK time unit in seconds, times tstep; <= 0 disables
     *                        the time interpolation.
     * @param[out] P, P_s, P_t, P_phi, P_time  variables, their s/t/phi derivatives
     *                        and their time derivative (size n_v)
     * @param[out] R, R_s, R_t, R_phi, Z, Z_s, Z_t, Z_phi  geometry and its derivatives
     */
    template<class FS, class IdxView, class OutView, class Real = double>
    JGX_HD inline void do_interp_PRZP_1(const FS& f, const double time,
                                        const std::size_t ie,
                                        const IdxView& i_v, const int n_v,
                                        const double s, const double t, const double phi,
                                        const int n_period, const int n_coord_period,
                                        const double t_jorek,
                                        OutView P, OutView P_s, OutView P_t,
                                        OutView P_phi, OutView P_time,
                                        double& R, double& R_s, double& R_t, double& R_phi,
                                        double& Z, double& Z_s, double& Z_t, double& Z_phi) {
        for (int i = 0; i < n_v; ++i) P_time(i) = 0.0;

        interp::interp_PRZP_1(f.element_list, f.node_list, ie, i_v, n_v, s, t, phi,
                              n_period, n_coord_period, false, P, P_s, P_t, P_phi,
                              R, R_s, R_t, R_phi, Z, Z_s, Z_t, Z_phi);

        if (t_jorek <= 0.0) return;

        const std::size_t pe[1] = { static_cast<std::size_t>(n_v) };
        Real Pd_[JGX_N_VALUES_MAX], Pd_s_[JGX_N_VALUES_MAX];
        Real Pd_t_[JGX_N_VALUES_MAX], Pd_phi_[JGX_N_VALUES_MAX];
        const jgx::view<Real, 1> Pd(Pd_, pe), Pd_s(Pd_s_, pe);
        const jgx::view<Real, 1> Pd_t(Pd_t_, pe), Pd_phi(Pd_phi_, pe);

        interp::interp_PRZP_1(f.element_list, f.node_list, ie, i_v, n_v, s, t, phi,
                              n_period, n_coord_period, true, Pd, Pd_s, Pd_t, Pd_phi,
                              R, R_s, R_t, R_phi, Z, Z_s, Z_t, Z_phi);

        double dt;
        if (fabs(f.time_now - f.time_prev) > 1e-10 && !f.is_static) {
            dt = 1.0/(f.time_now - f.time_prev);
            const double df = (f.time_now - time)*dt;
            for (int i = 0; i < n_v; ++i) {
                P    (i) -= df*Pd    (i);
                P_s  (i) -= df*Pd_s  (i);
                P_t  (i) -= df*Pd_t  (i);
                P_phi(i) -= df*Pd_phi(i);
            }
        } else {
            dt = 1.0/t_jorek;
        }

        for (int i = 0; i < n_v; ++i) P_time(i) = Pd(i)*dt;
    } // do_interp_PRZP_1
} // namespace fields_linear


#endif //FIELDS_LINEAR_H
