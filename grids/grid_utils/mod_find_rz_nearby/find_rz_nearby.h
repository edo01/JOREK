#ifndef FIND_RZ_NEARBY_H
#define FIND_RZ_NEARBY_H

#include <cmath>
#include <cstddef>

#include "elements/mod_interp/interp.h"
#include "grids/grid_utils/find_RZ/find_RZ.h"
#include "grids/mod_neighbours/neighbours.h"
#include "models/phys_module/phys.h"
#include "jgx/macros.h"

namespace find_rz_nearby
{
    /**
     * Auxiliary subroutine for find_RZ_nearby -- the geometry and the inverse
     * jacobian determinant at (st, p) in element i_elm.
     *
     * @param i_elm  mesh element id (1-based)
     * @param[out] x interpolated (R,Z)
     * @param[out] inv_st_jac_det  1/jac, clamped when jac is near zero. Note
     *             this is a different guard from mod_fields::jac -- a larger
     *             threshold, and it clamps the inverse rather than the
     *             determinant.
     */
    template<class ES, class NS>
    JGX_HD inline void try_interp(const ES& el, const NS& nd, const int i_elm,
                                  const double st[2], const double p,
                                  double x[2],
                                  double& R_s, double& R_t, double& Z_s, double& Z_t,
                                  double& inv_st_jac_det) {
        double R_p, Z_p;
        interp::interp_RZP_1(el, nd, static_cast<std::size_t>(i_elm) - 1,
                             st[0], st[1], p,
                             x[0], R_s, R_t, R_p,
                             x[1], Z_s, Z_t, Z_p);

        // Guard against the determinant being close to zero
        const double jac = R_s*Z_t - R_t*Z_s;
        if (std::abs(jac) < 1e-8) inv_st_jac_det = std::copysign(1e8, jac);
        else                      inv_st_jac_det = 1.0/jac;
    } // try_interp

    /**
     * mod_find_rz_nearby::find_RZ_nearby -- find the (s,t) of x_new = (R_new,
     * Z_new) starting from x_old and the element it was in, crossing into
     * neighbouring elements as the newton iteration leaves the current one.
     * See the Fortran header for the picture.
     *
     * This is one body for both the reduced and the stellarator configuration.
     * The Fortran differs between them only in which global search it falls
     * back to -- find_RZ, which builds phi from the plane index, or find_RZP,
     * which wraps the phi it was given. That is the *only* difference, so it
     * is lifted out as phi_search and worked out by the facade, and nothing
     * here is written twice.
     *
     * @param el, nd            element and node sets
     * @param R_old, Z_old      the old (R,Z) location
     * @param s_old, t_old      the old (s,t), used as the starting guess
     * @param i_elm_old         mesh id (1-based) of the element it was in
     * @param R_new, Z_new      the new (R,Z) location
     * @param p                 toroidal angle for the local interpolation
     * @param phi_search        toroidal angle for the global fallback search.
     *                          Not the same as p in the reduced configuration,
     *                          where find_RZ ignores p and uses the plane angle.
     * @param[out] s_new, t_new the found coordinates
     * @param[out] i_elm_new    mesh id of the element found; negative when the
     *                          particle is lost, carrying where it went
     * @param[out] ifail        0 found; -1 lost; 2 after a NaN; 3 after the
     *                          iteration ran out; otherwise the search's own
     * @param[out] not_found    the iteration ran out *and* the global search
     *                          failed. The Fortran printed three lines of
     *                          advice here; a kernel cannot, so the facade does.
     * @param[out] bad_i_from, bad_i_to  0 when all is well; otherwise the
     *                          element pair whose connectivity disagreed. This
     *                          is coord_in_neighbour's own diagnostic, which
     *                          its facade would have printed -- calling the
     *                          kernel directly bypasses that, so it is carried
     *                          out to this facade instead. Only the first
     *                          occurrence in a call survives; the Fortran
     *                          printed every one.
     *
     * @tparam Debug  compile in the edge-crossing consistency check that the
     *                Fortran guarded with #ifdef DEBUG. The shim decides.
     */
    template<bool Debug, class ES, class NS>
    JGX_HD inline void find_RZ_nearby(const ES& el, const NS& nd,
                                      const double R_old, const double Z_old,
                                      const double s_old, const double t_old,
                                      const int i_elm_old,
                                      const double R_new, const double Z_new,
                                      const double p, const double phi_search,
                                      double& s_new, double& t_new,
                                      int& i_elm_new, int& ifail, int& not_found,
                                      int& bad_i_from, int& bad_i_to) {
        const int    iter_max = jorek::phys().find_RZ_nearby_iter;
        const double tol      = jorek::phys().find_RZ_nearby_tol;

        double x_dump[2];   // the global search reports a position nobody reads
        int    checked;

        not_found = 0;
        bad_i_from = 0;
        bad_i_to   = 0;

        // Check if element is valid
        if (i_elm_old < 1 || i_elm_old > static_cast<int>(el.n_elements)) {
            find_rz::find_RZ_general(el, nd, R_new, Z_new, phi_search,
                                     x_dump[0], x_dump[1], i_elm_new,
                                     s_new, t_new, ifail, checked);
            return;
        }

        // Setup initial values
        i_elm_new = i_elm_old;                       // start in the current element
        double st_new[2] = { s_old, t_old };         // start at the old position
        const double x_new[2] = { R_new, Z_new };
        double x_step[2];                            // (R,Z) of the trial position

        double R_s, R_t, Z_s, Z_t, inv_st_jac_det;
        // Find the jacobian at the current s and t position. x_step is assigned
        // (R_old, Z_old) first in the Fortran, then immediately overwritten here.
        try_interp(el, nd, i_elm_new, st_new, p, x_step, R_s, R_t, Z_s, Z_t, inv_st_jac_det);

        double err2 = (x_step[0]-x_new[0])*(x_step[0]-x_new[0])
                    + (x_step[1]-x_new[1])*(x_step[1]-x_new[1]);
        ifail = 0;

        [[maybe_unused]] double x_tmp[2] = { 0.0, 0.0 };
        bool converged = false;

        // Newton iteration to find s and t in or out of this element
        for (int newton_iter_number = 1; newton_iter_number <= iter_max; ++newton_iter_number) {
            // Perform newton iteration by calculating the inverse of the jacobian matrix explicitly
            double st_step[2];
            st_step[0] = ( Z_t*(x_new[0]-x_step[0]) - R_t*(x_new[1]-x_step[1])) * inv_st_jac_det;
            st_step[1] = (-Z_s*(x_new[0]-x_step[0]) + R_s*(x_new[1]-x_step[1])) * inv_st_jac_det;

            // Limit this step if it goes outside of the element. fact is the
            // overshoot: how many times one st_step crosses the boundary.
            double fact = 0.0;
            for (int k = 0; k < 2; ++k) {
                // distance to 0 or 1, whichever the step heads for
                const double dist = st_step[k] > 0.0 ? 1.0 - st_new[k] : st_new[k];
                const double f    = std::abs(st_step[k]) / (dist > 1e-30 ? dist : 1e-30);
                if (f > fact) fact = f;
            }

            if (fact >= 1.0 - 1e-12) {   // we are on the boundary
                st_new[0] += st_step[0]/fact;
                st_new[1] += st_step[1]/fact;

                if constexpr (Debug) {
                    try_interp(el, nd, i_elm_new, st_new, p, x_tmp,
                               R_s, R_t, Z_s, Z_t, inv_st_jac_det);
                }

                const int i_elm_tmp = i_elm_new;
                int bad = 0;
                neighbours::coord_in_neighbour(el, static_cast<std::size_t>(i_elm_tmp) - 1,
                                               i_elm_new, st_new, bad);
                if (bad != 0 && bad_i_to == 0) { bad_i_from = i_elm_tmp; bad_i_to = bad; }

                if (i_elm_new < 0) {
                    find_rz::find_RZ_general(el, nd, x_new[0], x_new[1], phi_search,
                                             x_dump[0], x_dump[1], i_elm_new,
                                             s_new, t_new, ifail, checked);
                    if (ifail != 0) i_elm_new = 0;
                    /* @bug Faithful to the Fortran: st_new is not refreshed from
                     * s_new/t_new here, so the try_interp below runs in the newly
                     * found element with the *old* element's (s,t), and s_new/t_new
                     * are then overwritten from it -- discarding what the search
                     * found. Live at the magnetic axis, where update_neighbours
                     * writes -1 into neighbours. Reproduced, not fixed. */
                }

                if (i_elm_new == 0) {   // No element on that side, particle is lost
                    i_elm_new = -i_elm_tmp;  // Save position of particle
                    // The Fortran recomputes (R,Z) into x_new here; it is dead,
                    // nothing reads x_new after this point.
                    s_new = st_new[0];
                    t_new = st_new[1];
                    ifail = -1;
                    return;
                }

                try_interp(el, nd, i_elm_new, st_new, p, x_step,
                           R_s, R_t, Z_s, Z_t, inv_st_jac_det);

                if constexpr (Debug) {
                    const double dR = x_step[0]-x_tmp[0], dZ = x_step[1]-x_tmp[1];
                    if (std::sqrt(dR*dR + dZ*dZ) > 1e-8) {
                        i_elm_new = 0;
                        return;
                    }
                }
            } else {
                st_new[0] += st_step[0];
                st_new[1] += st_step[1];
                try_interp(el, nd, i_elm_new, st_new, p, x_step,
                           R_s, R_t, Z_s, Z_t, inv_st_jac_det);
            }

            err2 = (x_step[0]-x_new[0])*(x_step[0]-x_new[0])
                 + (x_step[1]-x_new[1])*(x_step[1]-x_new[1]);
            s_new = st_new[0];
            t_new = st_new[1];

            if (err2 < tol) { converged = true; break; }
        }

        if (std::isnan(err2)) {
            find_rz::find_RZ_general(el, nd, x_new[0], x_new[1], phi_search,
                                     x_dump[0], x_dump[1], i_elm_new,
                                     s_new, t_new, ifail, checked);
            if (ifail == 0) ifail = 2;
            return;
        }

        if (!converged) {   // the loop ran out rather than exiting early
            find_rz::find_RZ_general(el, nd, x_new[0], x_new[1], phi_search,
                                     x_dump[0], x_dump[1], i_elm_new,
                                     s_new, t_new, ifail, checked);
            if (ifail == 0) ifail = 3;
            else            not_found = 1;
            return;
        }
    } // find_RZ_nearby
} // namespace find_rz_nearby

#endif // FIND_RZ_NEARBY_H
