#ifndef NEIGHBOURS_H
#define NEIGHBOURS_H

#include <cstddef>

#include "models/mod_settings/mod_settings.h"
#include "jgx/macros.h"

namespace neighbours
{
    /**
     * mod_neighbours::neighbours_side_co_counter -- find the side of elm2 that
     * faces elm1, and whether the two elements run the same way round.
     *
     * Element ids are mesh ids (1-based, as stored in el.neighbours and
     * el.vertex); sides are 0-based, so side2 = -1 means elm2 has no side
     * pointing back at elm1.
     *
     * @param el                element set
     * @param elm1, elm2        mesh element ids
     * @param side1             side of elm1 facing elm2
     * @param[out] side2        side of elm2 facing elm1, -1 if there is none
     * @param[out] nb           the two really are neighbours
     * @param[out] co           they share an orientation (meaningless if !nb)
     */
    template<class ES>
    JGX_HD inline void neighbours_side_co_counter(const ES& el,
                                                  const int elm1, const int elm2,
                                                  const int side1,
                                                  int& side2, bool& nb, bool& co) {
        constexpr std::size_t n_vertex_max = JGX_N_VERTEX_MAX;

        const std::size_t ie1 = static_cast<std::size_t>(elm1) - 1;
        const std::size_t ie2 = static_cast<std::size_t>(elm2) - 1;

        co = false;  // does not mean anything if not neighbours
        nb = false;

        // Find the side in elm2 pointing to elm1. If elm2 has no neighbour ==
        // elm1 use the last one that is 0 (i.e. the one on the axis itself.)
        side2 = -1;
        for (std::size_t i = 0; i < n_vertex_max; ++i) {
            if (el.neighbours(ie2, i) == elm1) {
                side2 = static_cast<int>(i);
                break;
            }
        }
        if (side2 < 0) return;

        nb = true;

        // node numbers are related to sides as node1 = mod(side,4), node2 = mod(side+1,4)
        const int n1a = el.vertex(ie1, static_cast<std::size_t>( side1      % 4));
        const int n1b = el.vertex(ie1, static_cast<std::size_t>((side1 + 1) % 4));
        const int n2a = el.vertex(ie2, static_cast<std::size_t>( side2      % 4));
        const int n2b = el.vertex(ie2, static_cast<std::size_t>((side2 + 1) % 4));

        // Find if match cross or straight (i.e. 1->2/2->1 or 1->1/2->2). We do
        // not need to check the node position since there is only one option
        // and one of the node numbers must match.
        if      (n1a == n2b) co = true;
        else if (n1b == n2a) co = true;
        else if (n1a == n2a) co = false;
        else if (n1b == n2b) co = false;
    } // neighbours_side_co_counter

    /**
     * mod_neighbours::coord_in_neighbour -- given (s,t) that has left element
     * i_from, name the element it entered and rewrite (s,t) in that element's
     * local coordinates.
     *
     * On the boundary between elements the following is guaranteed:
     *  * one of the local coordinates (s,t) is either 0 or 1
     *    (side 0: t=0, 1: s=1, 2: t=1, 3: s=0)
     *  * the other coordinates x_i, x_j (elements i and j) are related:
     *    |dx_i/dx_j| = 1
     *
     * @param el          element set
     * @param ie_from     0-based index of the element being left
     * @param[out] i_to   the element entered, as a raw el.neighbours entry: a
     *                    mesh id (1-based) if one exists there, 0 if not, and
     *                    negative to ask the caller for a global search. Not
     *                    converted to 0-based, because 0 is a sentinel here.
     * @param[in,out] st  element-local coordinates, rewritten for i_to
     * @param[out] bad_i_to  0 when all is well; otherwise the connectivity of
     *                    the two elements disagrees, i_to comes back 0, and
     *                    this holds the element that disagreed. The Fortran
     *                    printed that id here and a device kernel cannot, so
     *                    the diagnostic is handed to the facade
     *                    (mod_neighbours.f90) rather than dropped.
     */
    template<class ES, class Real = double>
    JGX_HD inline void coord_in_neighbour(const ES& el, const std::size_t ie_from,
                                          int& i_to, Real st[2], int& bad_i_to) {
        bad_i_to = 0;

        int q_from;  // quadrant
        if (st[0] > st[1]) {
            q_from = (1.0 - st[0] > st[1]) ? 0 : 1;
        } else {
            q_from = (1.0 - st[0] <= st[1]) ? 2 : 3;
        }

        i_to = el.neighbours(ie_from, static_cast<std::size_t>(q_from));
        if (i_to <= 0) return;

        // Check once more that they are neighbours and determine the orientation
        const int i_from = static_cast<int>(ie_from) + 1;  // mesh ids from here on
        int  q_to;
        bool nb, co;
        neighbours_side_co_counter(el, i_from, i_to, q_from, q_to, nb, co);
        if (!nb || q_to < 0) {
            bad_i_to = i_to;
            i_to     = 0;
            return;
        }

        // x is the coordinate along the boundary from vertex i_side to i_side+1
        Real x = 0.0;
        switch (q_from) {
            case 0: x =       st[0]; break;
            case 1: x =       st[1]; break;
            case 2: x = 1.0 - st[0]; break;
            case 3: x = 1.0 - st[1]; break;
        }

        if (co) x = 1.0 - x;  // if the vectors along the boundary are antiparallel

        switch (q_to) {
            case 0: st[0] =       x; st[1] = 0.0;     break;
            case 1: st[0] = 1.0;     st[1] =       x; break;
            case 2: st[0] = 1.0 - x; st[1] = 1.0;     break;
            case 3: st[0] = 0.0;     st[1] = 1.0 - x; break;
        }
    } // coord_in_neighbour
} // namespace neighbours

#endif // NEIGHBOURS_H
