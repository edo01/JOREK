/* particles/mod_epf_evolution/epf_evolution.h -- Per-particle body only of 
 * the energetic-particle kernel.
 *
 * Reduced MHD with jorek_fields_interp_linear only.
 *
 * One step is three pieces: the push (a field evaluation, a Boris kick and a
 * search), epf_project (the seven velocity moments the particle then
 * contributes) and epf_deposit (spreading those over the element's degrees of
 * freedom). They are separate because the device runs them as separate kernels:
 * the deposit is atomic-free only when one thread owns a feedback cell and sums
 * the whole element's run into it, which is a different decomposition from one
 * thread per particle. The host launcher calls all three back to back on one
 * particle, which is what the Fortran did.
 *
 * The order inside a step is the Fortran's and is not the runaway kernel's: the
 * field is evaluated *before* the push and the moments are formed *after* it, so
 * a projection pairs the magnetic field at x^n with the velocity at v^(n+1/2)
 * and the position at x^(n+1). Sequencing it any other way would change the
 * physics that has been run with, so B is carried across the push rather than
 * re-evaluated.
 */
#ifndef JOREK_EPF_EVOLUTION_H
#define JOREK_EPF_EVOLUTION_H

#include <cmath>
#include <cstddef>

#include "elements/mod_basisfunctions/basisfunctions.h"
#include "elements/mod_interp/interp.h"
#include "grids/grid_utils/mod_find_rz_nearby/find_rz_nearby.h"
#include "models/constants/constants.h"
#include "models/mod_settings/mod_settings.h"
#include "particles/pushers/mod_boris/boris.h"
#include "jgx/atomic.h"
#include "jgx/macros.h"
#include "jgx/view.h"

namespace jorek {

/* feedback_rhs(n_degrees, n_vertex_max, n_elements, n_tor, n_proj), the layout
 * the Fortran array has. The host arm accumulates into that array in place, so
 * it has no other choice. */
using epf_rhs_view = jgx::view<double, 5, jgx::layout_left>;

/* The same five extents with `ie` fastest -- see re_rhs_view_device
 * (particles/mod_runaway_evolution/runaway_evolution.h) for why the device arm
 * is free to choose, and why it chooses this. */
using epf_rhs_view_device = jgx::view<double, 5, jgx::layout_perm<2, 0, 1, 3, 4>>;

/**
 * The seven projections this scheme feeds back, in the order the Fortran
 * deposits them.
 *
 * They are slots in epf_projection::m and in epf_projection_indices::var, not
 * positions in the feedback array -- where those sit is a runtime property of
 * the coupling setup, which is what the indices carry.
 */
enum epf_var {
    EPF_PI_RR = 0,
    EPF_PI_ZZ,
    EPF_PI_PHIPHI,
    EPF_PI_RZ,
    EPF_PI_RPHI,
    EPF_PI_ZPHI,
    EPF_RHO_EP,
    EPF_N_VAR
};

/**
 * Where the seven projections live along the last dimension of the feedback
 * array, 0-based, in epf_var order.
 *
 * They are runtime module variables (particles/coupling_variables.f90) set
 * during coupling setup, so unlike the array extents they cannot be -D defines,
 * and unlike F0 they are not physics state -- they cross as arguments.
 *
 * One array rather than seven named members: the device deposit reaches it with
 * a runtime `var`, and a kernel parameter can be indexed dynamically where a
 * local array would be spilled.
 */
struct epf_projection_indices {
    std::size_t var[EPF_N_VAR] = {};
};

/**
 * What one particle contributes to the feedback, before it is spread over the
 * element's degrees of freedom.
 *
 * The seven moments carry no dependence on (n, m, i_tor): the projection
 * factorises into one of them times a basis function times a toroidal harmonic,
 * which is what lets the deposit be done by a thread that owns a feedback cell
 * rather than by the particle's own thread.
 *
 * The weight is folded in here, and so is the mass*u*mu0 that the six pressures
 * carry and the density does not -- the Fortran's `base`, less the two factors
 * that do depend on (n, m, i_tor).
 *
 * A lost particle, or one on a step that collects no projection, comes back
 * with i_elm <= 0 and seven zeros, so a deposit over it adds nothing and needs
 * no branch of its own.
 */
struct epf_projection {
    double m[EPF_N_VAR] = {};
    double s      = 0.0;
    double t      = 0.0;
    double phi    = 0.0;
    int    i_elm  = 0;    /*< 1-based, as the particle stores it */
};

/**
 * The magnetic-field-dependent half of one particle's projection: the seven
 * velocity moments.
 *
 * @param part, ip  the particle set and the particle in it, *after* the push
 * @param B         the magnetic field at the position the particle was pushed
 *                  *from* -- see the file header
 * @param mass      particle mass [AMU]
 */
template<class PS>
JGX_HD inline epf_projection epf_project(const PS& part, const std::size_t ip,
                                         const double B[3], const double mass) {
    epf_projection pr;
    pr.i_elm = static_cast<int>(part.i_elm(ip));
    if (pr.i_elm <= 0) return pr;  /* the moments stay 0: deposits nothing */

    pr.s   = part.st(ip, 0);
    pr.t   = part.st(ip, 1);
    pr.phi = part.x(ip, 2);

    const double v[3] = { part.v(ip, 0), part.v(ip, 1), part.v(ip, 2) };

    /* normalised B and the orthonormal velocity components */
    const double B_mag = std::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
    double Bn[3];
    for (int c = 0; c < 3; ++c) Bn[c] = B[c]/B_mag;

    const double v_par = Bn[0]*v[0] + Bn[1]*v[1] + Bn[2]*v[2];

    /* The (R, phi) part of B is what the tilde frame is built on, so this
     * divisor goes to zero where B is purely vertical -- unguarded, as in the
     * Fortran. */
    const double B_rphi = std::sqrt(Bn[0]*Bn[0] + Bn[2]*Bn[2]);
    const double v_tilde_r = (-Bn[0]*v[2] + Bn[2]*v[0])/B_rphi;
    const double v_tilde_z = ( v[1] - Bn[1]*v_par)/B_rphi;

    /* parallel and perpendicular pressures */
    const double p_perp  = 0.5*(v_tilde_r*v_tilde_r + v_tilde_z*v_tilde_z);
    const double p_par   = v_par*v_par;
    const double p_atrop = p_par - p_perp;

    const double w  = part.weight(ip);
    const double wp = w*mass*ATOMIC_MASS_UNIT*MU_ZERO;

    pr.m[EPF_PI_RR]     = wp*(p_perp + Bn[0]*Bn[0]*p_atrop);
    pr.m[EPF_PI_ZZ]     = wp*(p_perp + Bn[1]*Bn[1]*p_atrop);
    pr.m[EPF_PI_PHIPHI] = wp*(p_perp + Bn[2]*Bn[2]*p_atrop);
    pr.m[EPF_PI_RZ]     = wp*(Bn[0]*Bn[1]*p_atrop);
    pr.m[EPF_PI_RPHI]   = wp*(Bn[0]*Bn[2]*p_atrop);
    pr.m[EPF_PI_ZPHI]   = wp*(Bn[1]*Bn[2]*p_atrop);
    pr.m[EPF_RHO_EP]    = w;
    return pr;
} // epf_project

/**
 * One particle's contribution spread over its element's degrees of freedom.
 *
 * The counterpart of the block-per-element accumulation on the device: same
 * outer product, opposite decomposition. Here the particle's own thread walks
 * (n, m, i_tor) and adds into a feedback array the launcher has already made
 * private, so the atomic_add costs nothing.
 *
 * @param pr      what epf_project worked out
 * @param fields  for element_list.size, the per-(n, m, ie) basis scaling
 * @param rhs     the feedback array to accumulate into
 * @param idx     which of rhs' last dimension the seven projections use
 */
template<class FS, class RhsView>
JGX_HD inline void epf_deposit(const epf_projection& pr, const FS& fields,
                               RhsView rhs, const epf_projection_indices& idx) {
    /* n_degrees, not the Fortran's own `n_order+1` loop bound: the two agree
     * for the cubic basis, which is the only one this scheme has been run with
     * -- the Fortran declares HH(4,4) and would overrun it on a quintic mesh,
     * where basisfunctions fills nine degrees. The feedback array's first
     * extent is n_order+1 there as well, so a quintic build needs both sides
     * settled before either is trusted. */
    constexpr std::size_t n_deg = JGX_N_DEGREES;
    constexpr std::size_t n_vtx = JGX_N_VERTEX_MAX;
    constexpr std::size_t n_tor = JGX_N_TOR;

    if (pr.i_elm <= 0) return;
    const std::size_t ie = static_cast<std::size_t>(pr.i_elm) - 1;

    /* The buffer is the Fortran HH(4,n_degrees); the body indexes it
     * (degree,vertex), so HH(n,m) here is the Fortran's HH(m,n).
     *
     * The Fortran calls basisfunctions, which fills the s and t derivatives as
     * well; nothing reads them, and the values here are the ones _2D_1_T would
     * have produced, since it obtains them from this same routine. */
    double HH_[4*n_deg], HZ_[n_tor];
    const std::size_t bf_ext[2] = { n_deg, 4 };
    const std::size_t hz_ext[1] = { n_tor };
    const jgx::view<double, 2, jgx::layout_right> HH(HH_, bf_ext);
    const jgx::view<double, 1> HZ(HZ_, hz_ext);

    basisfunctions::basisfunctions_2D_0_T(pr.s, pr.t, HH);
    interp::mode_moivre_explicit(pr.phi, HZ, JGX_N_TOR, JGX_N_PERIOD);

    for (std::size_t n = 0; n < n_deg; ++n) {
        for (std::size_t m = 0; m < n_vtx; ++m) {
            const double proj_factor = HH(n, m)*fields.element_list.size(ie, m, n);

            /* Every worker deposits into the same element, so the += is a
             * device atomic and a plain += on the host, where the launcher's
             * reduction has already made the array private. */
            for (std::size_t i_tor = 0; i_tor < n_tor; ++i_tor) {
                const double base = HZ(i_tor)*proj_factor;
                for (int v = 0; v < EPF_N_VAR; ++v)
                    jgx::atomic_add(&rhs(n, m, ie, i_tor, idx.var[v]),
                                    base*pr.m[v]);
            }
        }
    }
} // epf_deposit

/**
 * The push half of one step: the field at the particle, a Boris kick, and the
 * search that re-establishes (i_elm, s, t).
 *
 * The Fortran keeps the pre-push (R,Z) in rzp_old because boris_push_cylindrical
 * overwrites particle%x; (s,t) and i_elm it copies as well, but the push does
 * not touch those, so reading them off the particle is the same values.
 *
 * @param[out] E, B  the field the particle was pushed in, which the projection
 *                   needs afterwards
 * @param phi_search see kinetic_relativistic::find_particle_st
 * @param[out] ifail find_RZ_nearby's own code, as in the Fortran: written by
 *                   every step and read by none of them
 */
template<bool Debug, class PS, class FS>
JGX_HD inline void epf_push(PS& part, const std::size_t ip, const FS& fields,
                            const double mass, const double time,
                            const double timestep, const double phi_search,
                            double E_[3], double B_[3], int& ifail,
                            find_rz_nearby::diagnostics& diag) {
    const std::size_t ve[1] = { 3 };
    const jgx::view<double, 1> E(E_, ve), B(B_, ve);
    double psi, U;
    fields.calc_EBpsiU_reduced(time, static_cast<std::size_t>(part.i_elm(ip)) - 1,
                               part.st(ip, 0), part.st(ip, 1), part.x(ip, 2),
                               E, B, psi, U);

    const double R_old = part.x(ip, 0), Z_old = part.x(ip, 1);
    const double s_old = part.st(ip, 0), t_old = part.st(ip, 1);
    const int    i_elm_old = static_cast<int>(part.i_elm(ip));

    boris::boris_push_cylindrical(part, ip, mass, E_, B_, timestep);

    double s_new, t_new;
    int    i_elm_new, not_found, bad_i_from, bad_i_to;

    /* The Fortran omits the optional phi, so the local interpolation runs at
     * p = 0 -- a reduced-MHD assumption, as in the relativistic pusher. */
    find_rz_nearby::find_RZ_nearby<Debug>(fields.element_list, fields.node_list,
                                          R_old, Z_old, s_old, t_old, i_elm_old,
                                          part.x(ip, 0), part.x(ip, 1),
                                          0.0, phi_search,
                                          s_new, t_new, i_elm_new, ifail,
                                          not_found, bad_i_from, bad_i_to);

    part.st(ip, 0) = s_new;
    part.st(ip, 1) = t_new;
    part.i_elm(ip) = i_elm_new;

    find_rz_nearby::merge_diagnostics(diag, not_found, R_old, Z_old,
                                      bad_i_from, bad_i_to);
} // epf_push

/**
 * One energetic particle, advanced over nstep steps of the Boris pusher,
 * depositing its projection contribution into rhs every proj_period steps.
 *
 * The Fortran leaves the loop the moment the particle is lost, so a particle
 * that walks out of the mesh contributes nothing for the remaining steps -- and
 * one lost by the push itself contributes nothing for the step that lost it,
 * the projection sitting after the search.
 *
 * @param part, ip     the particle set and the particle in it
 * @param fields       the grid and the interpolation strategy
 * @param rhs          the feedback array this thread accumulates into
 * @param idx          which of rhs' last dimension the seven projections use
 * @param mass         particle mass [AMU]
 * @param time         current time [s]
 * @param timestep     time step [s]
 * @param nstep        number of pusher steps to take
 * @param proj_period  collect a projection on every step whose 1-based number
 *                     is a multiple of this. The facade has already clamped it
 *                     to nstep.
 * @param phi_search   see kinetic_relativistic::find_particle_st
 * @param[inout] diag  what the searches would have printed; first occurrence wins
 */
template<bool Debug, class PS, class FS, class RhsView>
JGX_HD inline void evolve_epf_particle(PS& part, const std::size_t ip,
                                       const FS& fields, RhsView rhs,
                                       const epf_projection_indices& idx,
                                       const double mass, const double time,
                                       const double timestep, const int nstep,
                                       const int proj_period,
                                       const double phi_search,
                                       find_rz_nearby::diagnostics& diag) {
    for (int k = 1; k <= nstep; ++k) {
        if (part.i_elm(ip) <= 0) break;

        /* ifail is written by every step and read by none of them, here as in
         * the Fortran -- a lost particle shows up as i_elm <= 0. */
        int ifail = 0;
        double E[3], B[3];
        epf_push<Debug>(part, ip, fields, mass, time, timestep, phi_search,
                        E, B, ifail, diag);

        /* the push may have taken it out of the domain */
        if (part.i_elm(ip) <= 0) break;

        if (k % proj_period != 0) continue;

        epf_deposit(epf_project(part, ip, B, mass), fields, rhs, idx);
    } // steps
} // evolve_epf_particle

} // namespace jorek

#endif // JOREK_EPF_EVOLUTION_H
