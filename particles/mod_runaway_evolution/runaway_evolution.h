/* particles/mod_runaway_evolution/runaway_evolution.h -- the runaway-electron
 * kernel behind mod_runaway_evolution.f90, next door.
 *
 * Reduced MHD with jorek_fields_interp_linear only.
 *
 * The per-particle body only. The launchers are one file out each way -- OpenMP
 * over the particles in runaway_evolution_host.h, one thread per particle in
 * runaway_evolution_device.hip.cpp -- so that nothing in here names a host
 * parallel construct and the device translation unit can include it.
 */
#ifndef JOREK_RUNAWAY_EVOLUTION_H
#define JOREK_RUNAWAY_EVOLUTION_H

#include <cmath>
#include <cstddef>

#include "elements/mod_basisfunctions/basisfunctions.h"
#include "elements/mod_interp/interp.h"
#include "models/constants/constants.h"
#include "models/mod_settings/mod_settings.h"
#include "particles/pushers/mod_kinetic_relativistic/kinetic_relativistic.h"
#include "tools/mod_coordinate_transforms/coordinate_transforms.h"
#include "jgx/atomic.h"
#include "jgx/macros.h"
#include "jgx/view.h"

namespace jorek {

/* feedback_rhs(n_degrees, n_vertex_max, n_elements, n_tor, n_proj). */
using re_rhs_view = jgx::view<double, 5, jgx::layout_left>;

/**
 * Where the three runaway-electron projections live along the last dimension of
 * the feedback array.
 *
 * They are runtime module variables (particles/coupling_variables.f90) set
 * during coupling setup, so unlike the array extents they cannot be -D defines,
 * and unlike F0 they are not physics state -- they cross as arguments, 0-based.
 */
struct re_projection_indices {
    std::size_t P_par  = 0;
    std::size_t P_perp = 0;
    std::size_t j_Phi  = 0;
};

/**
 * One runaway electron, advanced over nstep steps of the pusher, depositing its
 * projection contribution into rhs as it goes.
 *
 * The Fortran leaves the loop the moment the particle is lost, so a particle
 * that walks out of the mesh contributes nothing for the remaining steps.
 *
 * @param part, ip   the particle set and the particle in it
 * @param fields     the grid and the interpolation strategy
 * @param rhs        the feedback array this thread accumulates into
 * @param idx        which of rhs' last dimension the three projections use
 * @param mass       particle mass [AMU]
 * @param time       current time [s]
 * @param timestep   time step [s]
 * @param nstep      number of pusher steps to take
 * @param phi_search see kinetic_relativistic::find_particle_st
 * @param[inout] diag  what the searches would have printed; first occurrence wins
 */
template<bool Debug, class PS, class FS, class RhsView>
JGX_HD inline void evolve_RE(PS& part, const std::size_t ip, const FS& fields,
                             RhsView rhs, const re_projection_indices& idx,
                             const double mass, const double time,
                             const double timestep, const int nstep,
                             const double phi_search,
                             kinetic_relativistic::push_diagnostics& diag) {
    constexpr std::size_t n_deg = JGX_N_DEGREES;
    constexpr std::size_t n_vtx = JGX_N_VERTEX_MAX;
    constexpr std::size_t n_tor = JGX_N_TOR;

    /* The buffer is the Fortran HH(4,n_degrees); the body indexes it
     * (degree,vertex), so HH(n,m) here is the Fortran's HH(m,n). Same view the
     * non-transposed shim entries build -- see basisfunctions_shim.cpp.
     *
     * The Fortran calls basisfunctions, which fills the s and t derivatives as
     * well; nothing reads them, and the values here are the ones _2D_1_T would
     * have produced, since it obtains them from this same routine. */
    double HH_[4*n_deg], HZ_[n_tor];
    const std::size_t bf_ext[2] = { n_deg, 4 };
    const std::size_t hz_ext[1] = { n_tor };
    const jgx::view<double, 2, jgx::layout_right> HH(HH_, bf_ext);
    const jgx::view<double, 1> HZ(HZ_, hz_ext);

    for (int k = 0; k < nstep; ++k) {
        if (part.i_elm(ip) <= 0) break;

        basisfunctions::basisfunctions_2D_0_T(part.st(ip, 0), part.st(ip, 1), HH);
        interp::mode_moivre_explicit(part.x(ip, 2), HZ, JGX_N_TOR, JGX_N_PERIOD);

        // Determines velocity in cylindrical coordinates
        const double p_cart[3] = { part.p(ip, 0), part.p(ip, 1), part.p(ip, 2) };
        double cylindrical_momentum[3];
        coordinate_transforms::vector_cartesian_to_cylindrical(part.x(ip, 2), p_cart,
                                                               cylindrical_momentum);

        const double p2 = cylindrical_momentum[0]*cylindrical_momentum[0]
                        + cylindrical_momentum[1]*cylindrical_momentum[1]
                        + cylindrical_momentum[2]*cylindrical_momentum[2];
        const double v_scale = std::sqrt(p2/(SPEED_OF_LIGHT*SPEED_OF_LIGHT) + mass*mass);

        double cylindrical_velocity[3];
        for (int c = 0; c < 3; ++c) cylindrical_velocity[c] = cylindrical_momentum[c]/v_scale;

        double E_[3], B_[3], psi, U;
        const std::size_t ve[1] = { 3 };
        const jgx::view<double, 1> E(E_, ve), B(B_, ve);
        const std::size_t ie = static_cast<std::size_t>(part.i_elm(ip)) - 1;
        fields.calc_EBpsiU_reduced(time, ie, part.st(ip, 0), part.st(ip, 1),
                                   part.x(ip, 2), E, B, psi, U);

        const double B_mag = std::sqrt(B_[0]*B_[0] + B_[1]*B_[1] + B_[2]*B_[2]);
        double B_norm2[3];
        for (int c = 0; c < 3; ++c) B_norm2[c] = B_[c]/B_mag;

        const double v_par = cylindrical_velocity[0]*B_norm2[0]
                           + cylindrical_velocity[1]*B_norm2[1]
                           + cylindrical_velocity[2]*B_norm2[2];

        double v_perp2 = 0.0;
        for (int c = 0; c < 3; ++c) {
            const double d = cylindrical_velocity[c] - v_par*B_norm2[c];
            v_perp2 += d*d;
        }
        const double v_perp = std::sqrt(v_perp2);

        const double gamma_m = std::sqrt(MASS_ELECTRON*MASS_ELECTRON
                                       + p2*ATOMIC_MASS_UNIT*ATOMIC_MASS_UNIT
                                           /(SPEED_OF_LIGHT*SPEED_OF_LIGHT));

        for (std::size_t n = 0; n < n_deg; ++n) {
            for (std::size_t m = 0; m < n_vtx; ++m) {
                const double proj_factor = HH(n, m)*fields.element_list.size(ie, m, n)
                                         * part.weight(ip);

                // PCS for REs
                const double v_Ppar  = proj_factor*gamma_m*v_par*v_par*MU_ZERO;
                const double v_Pperp = proj_factor*gamma_m*v_perp*v_perp/2.0*MU_ZERO;

                const double v_jPhi = -proj_factor*static_cast<double>(part.q(ip))
                                    * EL_CHG*cylindrical_velocity[2]*part.x(ip, 0)*MU_ZERO;

                /* Every worker deposits into the same element, so the += is a
                 * device atomic and a plain += on the host, where the launcher's
                 * reduction has already made the array private. */
                for (std::size_t i_tor = 0; i_tor < n_tor; ++i_tor) {
                    jgx::atomic_add(&rhs(n, m, ie, i_tor, idx.P_par ), HZ(i_tor)*v_Ppar);
                    jgx::atomic_add(&rhs(n, m, ie, i_tor, idx.P_perp), HZ(i_tor)*v_Pperp);
                    jgx::atomic_add(&rhs(n, m, ie, i_tor, idx.j_Phi ), HZ(i_tor)*v_jPhi);
                }
            }
        }

        /* ifail is written by every push and read by none of them, here as in
         * the Fortran -- a lost particle shows up as i_elm <= 0. */
        int ifail = 0;
        kinetic_relativistic::push_diagnostics step_diag;
        kinetic_relativistic::volume_preserving_push_jorek<Debug>(
            part, ip, fields, mass, time, timestep, phi_search, ifail, step_diag);

        if (step_diag.not_found != 0 && diag.not_found == 0) {
            diag.not_found = step_diag.not_found;
            diag.nf_R      = step_diag.nf_R;
            diag.nf_Z      = step_diag.nf_Z;
        }
        if (step_diag.bad_i_to != 0 && diag.bad_i_to == 0) {
            diag.bad_i_from = step_diag.bad_i_from;
            diag.bad_i_to   = step_diag.bad_i_to;
        }
    } // steps
} // evolve_RE

} // namespace jorek

#endif // JOREK_RUNAWAY_EVOLUTION_H
