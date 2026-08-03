#ifndef INTERP_H
#define INTERP_H

#include <cmath>
#include <cstddef>

#include "elements/mod_basisfunctions/basisfunctions.h"
#include "jgx/macros.h"
#include "jgx/view.h"

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

    /* Per-call temporaries. Fortran declares these as automatic arrays sized by
     * n_v; where they live on device (registers, shared, a scratch slab) is the
     * caller's decision, so they are passed in rather than declared here.
     */
    template<class Real = double>
    struct interp_PRZ_workspace {
        jgx::view<Real, 4> values;  // (kt, kf, i, kv)
        jgx::view<Real, 2> xR;      // (kf, kv)
        jgx::view<Real, 2> xZ;      // (kf, kv)
        jgx::view<Real, 2> H;       // (kf, kv)
        jgx::view<Real, 2> H_s;     // (kf, kv)
        jgx::view<Real, 2> H_t;     // (kf, kv)
        jgx::view<Real, 1> HZ;      // (kt)
        jgx::view<Real, 1> dHZ;     // (kt)
    };

    /* mod_interp::interp_PRZ_1 -- interpolate n_v variables and the geometry at
     * (s,t,phi) inside element ie. Names no layout: el and nd may be either the
     * AoS or the SoA set.
     *
     * ie and i_v are 0-based; the caller converts. Values read *out of* the mesh
     * are still Fortran-numbered, so el.vertex(...) is decremented here.
     */
    template<class ES, class NS, class IdxView, class OutView, class Real = double>
    JGX_HD inline void interp_PRZ_1(const ES& el, const NS& nd,
                                    interp_PRZ_workspace<Real>& w,
                                    const std::size_t ie,
                                    const IdxView& i_v, const int n_v,
                                    const double s, const double t, const double phi,
                                    const int n_period, const bool use_deltas,
                                    OutView P, OutView P_s, OutView P_t, OutView P_phi,
                                    double& R, double& R_s, double& R_t,
                                    double& Z, double& Z_s, double& Z_t) {
        const std::size_t n_vertex_max = el.vertex.extent[1];
        const std::size_t n_degrees    = el.size.extent[2];
        const std::size_t n_tor        = nd.values.extent[1];

        basisfunctions::basisfunctions_2D_1_T(s, t, w.H, w.H_s, w.H_t);

        for (int i = 0; i < n_v; ++i) {
            P(i) = 0.0; P_s(i) = 0.0; P_t(i) = 0.0; P_phi(i) = 0.0;
        }

        sincosperiod_moivre_explicit(phi, w.HZ, w.dHZ,
                                     static_cast<int>(n_tor), n_period);

        // Preload values and premultiply with sizes(:,kv)
        for (std::size_t kv = 0; kv < n_vertex_max; ++kv) {
            const std::size_t iv =
                static_cast<std::size_t>(el.vertex(ie, kv)) - 1;  // 1-based in the mesh

            for (int i = 0; i < n_v; ++i) {
                const std::size_t i_var = static_cast<std::size_t>(i_v(i));
                for (std::size_t kf = 0; kf < n_degrees; ++kf) {
                    const Real sz = el.size(ie, kv, kf);
                    for (std::size_t kt = 0; kt < n_tor; ++kt)
                        w.values(kt, kf, i, kv) =
                            (use_deltas ? nd.deltas(iv, kt, kf, i_var)
                                        : nd.values(iv, kt, kf, i_var)) * sz;
                }
            }
            for (std::size_t kf = 0; kf < n_degrees; ++kf) {
                const Real sz = el.size(ie, kv, kf);
                w.xR(kf, kv) = nd.x(iv, 0, kf, 0) * sz;
                w.xZ(kf, kv) = nd.x(iv, 0, kf, 1) * sz;
            }
        }

        R = 0.0; R_s = 0.0; R_t = 0.0;
        Z = 0.0; Z_s = 0.0; Z_t = 0.0;
        for (std::size_t kv = 0; kv < n_vertex_max; ++kv)
            for (std::size_t kf = 0; kf < n_degrees; ++kf) {
                R   += w.xR(kf, kv) * w.H  (kf, kv);
                R_s += w.xR(kf, kv) * w.H_s(kf, kv);
                R_t += w.xR(kf, kv) * w.H_t(kf, kv);
                Z   += w.xZ(kf, kv) * w.H  (kf, kv);
                Z_s += w.xZ(kf, kv) * w.H_s(kf, kv);
                Z_t += w.xZ(kf, kv) * w.H_t(kf, kv);
            }

        for (std::size_t kv = 0; kv < n_vertex_max; ++kv)
            for (int i = 0; i < n_v; ++i)
                for (std::size_t kf = 0; kf < n_degrees; ++kf) {
                    Real v = 0.0, vp = 0.0;
                    for (std::size_t kt = 0; kt < n_tor; ++kt) {
                        v  += w.values(kt, kf, i, kv) * w.HZ (kt);
                        vp += w.values(kt, kf, i, kv) * w.dHZ(kt);
                    }
                    P    (i) += v  * w.H  (kf, kv);
                    P_s  (i) += v  * w.H_s(kf, kv);
                    P_t  (i) += v  * w.H_t(kf, kv);
                    P_phi(i) += vp * w.H  (kf, kv);
                }
    } // interp_PRZ_1
} // namespace interp


#endif //INTERP_H