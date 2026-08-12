#ifndef FIND_RZ_H
#define FIND_RZ_H

#include <cmath>
#include <cstddef>

#include "elements/mod_interp/interp.h"
#include "jgx/macros.h"

namespace find_rz
{
    /**
     * find_RZ_single -- solve for the (s,t) inside element ie whose geometry is
     * (R_find, Z_find), by Newton's method on the 2x2 system, restarting from
     * five different seeds before giving up.
     *
     * @param el, nd            element and node sets
     * @param ie                0-based element index
     * @param R_find, Z_find    the point being located
     * @param phi_find          toroidal angle
     * @param[out] R_out, Z_out geometry at the converged (s,t)
     * @param[out] s_out, t_out the converged element-local coordinates
     * @param[out] ifail        0 on success, 999 if no seed converged
     * @return                  true on success, i.e. ifail == 0
     */
    template<class ES, class NS>
    JGX_HD inline bool find_RZ_single(const ES& el, const NS& nd, const std::size_t ie,
                                      const double R_find, const double Z_find,
                                      const double phi_find,
                                      double& R_out, double& Z_out,
                                      double& s_out, double& t_out, int& ifail) {
        constexpr int    ntrial = 20;
        constexpr double tolx   = 1.0e-8;
        constexpr double tolf   = 1.0e-15;

        // The five restart seeds, in the order the Fortran tried them
        constexpr double seed[5][2] = {
            {0.50, 0.50}, {0.75, 0.75}, {0.75, 0.25}, {0.25, 0.75}, {0.25, 0.25}
        };

        for (int istart = 0; istart < 5; ++istart) {
            double x[2] = { seed[istart][0], seed[istart][1] };

            ifail = 999;

            for (int i = 0; i < ntrial; ++i) {
                double RRg1, dRRg1_dr, dRRg1_ds, dummy;
                double ZZg1, dZZg1_dr, dZZg1_ds, dummy2;
                interp::interp_RZP_1(el, nd, ie, x[0], x[1], phi_find,
                                     RRg1, dRRg1_dr, dRRg1_ds, dummy,
                                     ZZg1, dZZg1_dr, dZZg1_ds, dummy2);

                const double fvec[2] = { RRg1 - R_find, ZZg1 - Z_find };
                const double errf    = std::abs(fvec[0]) + std::abs(fvec[1]);

                if (errf <= tolf) {
                    s_out = x[0]; t_out = x[1];
                    R_out = RRg1; Z_out = ZZg1;
                    ifail = 0;
                    return true;
                }

                double p[2] = { -fvec[0], -fvec[1] };

                const double temp = p[0];
                const double dis  = dZZg1_ds*dRRg1_dr - dRRg1_ds*dZZg1_dr;

                if (dis == 0.0) break;

                p[0] = (dZZg1_ds*p[0] - dRRg1_ds*p[1])/dis;
                p[1] = (dRRg1_dr*p[1] - dZZg1_dr*temp )/dis;

                const double errx = std::abs(p[0]) + std::abs(p[1]);

                for (int k = 0; k < 2; ++k) {
                    p[k] = p[k] >  0.25 ?  0.25 : p[k];
                    p[k] = p[k] < -0.25 ? -0.25 : p[k];
                    x[k] += p[k];
                    x[k] = x[k] < 0.0 ? 0.0 : x[k];
                    x[k] = x[k] > 1.0 ? 1.0 : x[k];
                }

                if (errx <= tolx) {
                    s_out = x[0]; t_out = x[1];
                    R_out = RRg1; Z_out = ZZg1;
                    ifail = 0;
                    return true;
                }
            }
        }
        return false;
    } // find_RZ_single

    /**
     * find_RZ_general -- locate (R_find, Z_find) anywhere in the mesh.
     *
     * @warning This is a linear scan over every element, standing in for the
     * element r-tree the Fortran used (elements_containing_point). It is
     * O(n_elements) with a full Newton solve per element and will be slow on a
     * real grid; it is here so the search can cross the seam at all, pending a
     * port of the tree itself. Because the scan visits elements in mesh order
     * rather than r-tree order, a point on an element boundary may come back
     * attributed to a different -- equally valid -- element than before.
     *
     * @param el, nd              element and node sets
     * @param R_find, Z_find      the point being located
     * @param phi_find            toroidal angle
     * @param[out] R_out, Z_out   geometry at the converged (s,t)
     * @param[out] ielm_out       1-based mesh id of the element found, 0 if none
     * @param[out] s_out, t_out   the converged element-local coordinates
     * @param[out] ifail          0 on success; 999 if elements were tried and
     *                            none matched, 99 if there were none to try.
     *                            The Fortran reached the same two codes by a
     *                            longer route -- 999 fell out of the last
     *                            find_RZ_single, and 99 only when the candidate
     *                            list came back empty.
     * @param[out] checked_elms   how many elements were tried
     */
    template<class ES, class NS>
    JGX_HD inline void find_RZ_general(const ES& el, const NS& nd,
                                       const double R_find, const double Z_find,
                                       const double phi_find,
                                       double& R_out, double& Z_out, int& ielm_out,
                                       double& s_out, double& t_out,
                                       int& ifail, int& checked_elms) {
        ielm_out = 0;
        ifail    = 99;

        for (std::size_t ie = 0; ie < el.n_elements; ++ie) {
            if (find_RZ_single(el, nd, ie, R_find, Z_find, phi_find,
                               R_out, Z_out, s_out, t_out, ifail)) {
                ielm_out     = static_cast<int>(ie) + 1;  // 1-based in the mesh
                checked_elms = static_cast<int>(ie) + 1;
                return;
            }
        }

        checked_elms = static_cast<int>(el.n_elements);
    } // find_RZ_general
} // namespace find_rz

#endif // FIND_RZ_H
