#ifndef INTERP_H
#define INTERP_H

#include <cmath>
#include <cstddef>

#include "elements/mod_basisfunctions/basisfunctions.h"
#include "jgx/jorek/jorek_settings.h"
#include "jgx/macros.h"
#include "jgx/view.h"

/**
 * @todo: Fix parameters definitions -- mirrors mod_settings.f90 (n_order = 3).
 */
#define JGX_N_DEGREES     4
#define JGX_N_VERTEX_MAX  4
#define JGX_N_TOR         3

namespace interp
{
    template<class BasisFunctionsView>
    JGX_HD inline void sincosperiod_moivre_explicit(const double phi, 
                                                    BasisFunctionsView HZ_view, BasisFunctionsView dHZ_view, 
                                                    const int n_tor_in, const int n_period_in) {
        const int n_mode = (n_tor_in - 1)/2; // number of modes excluding 0

        HZ_view (0) = 1.0;
        dHZ_view(0) = 0.0;

        for (int i = 1; i <= n_mode; ++i) {
            const double n     = static_cast<double>(n_period_in*i);
            const double phase = n*phi;
            // exp(i*phase) = cos(phase) + i*sin(phase)
            const double re    = cos(phase);
            const double im    = sin(phase);
            HZ_view (2*i - 1) =    re;
            HZ_view (2*i    ) =    im;
            dHZ_view(2*i - 1) = -n*im;
            dHZ_view(2*i    ) =  n*re;
        }
    } // sincosperiod_moivre_explicit

    /**
     * mod_interp::interp_PRZ_1 -- interpolate n_v variables and the geometry (R,Z)
     * at (s,t,phi) inside element ie, along with their s/t/phi first derivatives.
     *
     * @param el          element set
     * @param nd          node set
     * @param ie          element index into el
     * @param i_v         indices of the n_v variables to interpolate
     * @param n_v         number of variables
     * @param s, t        element-local coordinates
     * @param phi         toroidal angle
     * @param n_period    toroidal periodicity
     * @param use_deltas  interpolate nd.deltas instead of nd.values
     * @param[out] P, P_s, P_t, P_phi  interpolated variables and their derivatives (size n_v)
     * @param[out] R, R_s, R_t, Z, Z_s, Z_t  interpolated geometry and its derivatives
     */
    template<class ES, class NS, class IdxView, class OutView, class Real = double>
    JGX_HD inline void interp_PRZ_1(const ES& el, const NS& nd,
                                    const std::size_t ie,
                                    const IdxView& i_v, const int n_v,
                                    const double s, const double t, const double phi,
                                    const int n_period, const bool use_deltas,
                                    OutView P, OutView P_s, OutView P_t, OutView P_phi,
                                    double& R, double& R_s, double& R_t,
                                    double& Z, double& Z_s, double& Z_t) {
        constexpr std::size_t n_vertex_max = JGX_N_VERTEX_MAX;
        constexpr std::size_t n_degrees    = JGX_N_DEGREES;
        constexpr std::size_t n_tor        = JGX_N_TOR;

        // Preparing the views for basisfunctions_2D_1_T and sincosperiod_moivre_explicit
        const std::size_t he[2] = { n_degrees, n_vertex_max };
        const std::size_t ze[1] = { n_tor };
        Real H_[n_degrees*n_vertex_max], H_s_[n_degrees*n_vertex_max], H_t_[n_degrees*n_vertex_max];
        Real HZ_[n_tor], dHZ_[n_tor];
        const jgx::view<Real, 2> H(H_, he), H_s(H_s_, he), H_t(H_t_, he);
        const jgx::view<Real, 1> HZ(HZ_, ze), dHZ(dHZ_, ze);

        basisfunctions::basisfunctions_2D_1_T(s, t, H, H_s, H_t);

        for (int i = 0; i < n_v; ++i) {
            P(i) = 0.0; P_s(i) = 0.0; P_t(i) = 0.0; P_phi(i) = 0.0;
        }

        sincosperiod_moivre_explicit(phi, HZ, dHZ, static_cast<int>(n_tor), n_period);

        R = 0.0; R_s = 0.0; R_t = 0.0;
        Z = 0.0; Z_s = 0.0; Z_t = 0.0;

        for (std::size_t kv = 0; kv < n_vertex_max; ++kv) {
            const std::size_t iv =
                static_cast<std::size_t>(el.vertex(ie, kv)) - 1;  // 1-based in the mesh

            for (std::size_t kf = 0; kf < n_degrees; ++kf) {
                const Real sz = el.size(ie, kv, kf);
                const Real h  = H  (kf, kv);
                const Real hs = H_s(kf, kv);
                const Real ht = H_t(kf, kv);

                const Real xR = nd.x(iv, 0, kf, 0) * sz;
                const Real xZ = nd.x(iv, 0, kf, 1) * sz;
                R   += xR * h;   R_s += xR * hs;  R_t += xR * ht;
                Z   += xZ * h;   Z_s += xZ * hs;  Z_t += xZ * ht;

                for (int i = 0; i < n_v; ++i) {
                    const std::size_t i_var = static_cast<std::size_t>(i_v(i));
                    Real v = 0.0, vp = 0.0;
                    for (std::size_t kt = 0; kt < n_tor; ++kt) {
                        const Real d = use_deltas ? nd.deltas(iv, kt, kf, i_var)
                                                  : nd.values(iv, kt, kf, i_var);
                        v  += d * HZ (kt);
                        vp += d * dHZ(kt);
                    }
                    v *= sz; vp *= sz;
                    P    (i) += v  * h;
                    P_s  (i) += v  * hs;
                    P_t  (i) += v  * ht;
                    P_phi(i) += vp * h;
                }
            }
        }
    } // interp_PRZ_1
} // namespace interp


#endif //INTERP_H