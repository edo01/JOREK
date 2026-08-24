/* particles/pushers/mod_kinetic_relativistic/kinetic_relativistic.h -- the
 * JOREK-specific Volume Preserving Algorithm of mod_kinetic_relativistic.f90,
 * next door.
 *
 * R. Zhang et al., Phys. Plasmas 22 (2015) 044501; see also C. Sommariva et
 * al., Nucl. Fusion 58 (2018) 016043.
 *
 * Reduced MHD only -- the fields come from fields_set::calc_EBpsiU_reduced,
 * which is the fullmhd-off, stellarator-off branch of calc_EBpsiU. The Fortran
 * facade sends the other configurations elsewhere; see mod_kinetic_relativistic.f90
 * next door.
 */
#ifndef JOREK_KINETIC_RELATIVISTIC_H
#define JOREK_KINETIC_RELATIVISTIC_H

#include <cmath>
#include <cstddef>

#include "grids/grid_utils/mod_find_rz_nearby/find_rz_nearby.h"
#include "models/constants/constants.h"
#include "particles/pushers/mod_radreactforce/radreactforce.h"
#include "particles/pushers/mod_pusher_tools/pusher_tools.h"
#include "tools/mod_coordinate_transforms/coordinate_transforms.h"
#include "jgx/macros.h"
#include "jgx/view.h"

namespace kinetic_relativistic {

/**
 * What the two find_RZ_nearby calls inside a push would have printed.
 *
 * The kernel cannot write, so it carries the diagnostics out and the facade
 * prints them -- the same arrangement find_RZ_nearby's own facade uses, one
 * level further up. Only the first occurrence in a push survives.
 */
struct push_diagnostics {
    int    not_found  = 0;   /*< the search ran out of iterations and the global
                              *  fallback failed too */
    double nf_R = 0.0;       /*< the (R,Z) it was searching from, for the message */
    double nf_Z = 0.0;
    int    bad_i_from = 0;   /*< element pair whose connectivity disagreed */
    int    bad_i_to   = 0;
};

/**
 * mod_kinetic_relativistic::volume_preserving_first_half_step_jorek.
 *
 * Leaves particle%p dimensionless -- the second half-step puts the dimension
 * back. Both are private to the module, so neither gets a facade of its own.
 *
 * @param half_position  (x,y,z), advanced by half a step in place
 * @param mass           particle mass [AMU]
 * @param dt             time step [s]
 * @param[out] scaling_factor  [s^2*C/(kg*m)], for the second half-step
 */
template<class PS>
JGX_HD inline void volume_preserving_first_half_step_jorek(PS& part, const std::size_t ip,
                                                           double half_position[3],
                                                           const double mass, const double dt,
                                                           double& scaling_factor) {
    scaling_factor = 5.0e-1*dt*static_cast<double>(part.q(ip))*jorek::EL_CHG
                   / (jorek::ATOMIC_MASS_UNIT*mass*jorek::SPEED_OF_LIGHT);

    // compute dimensionless momentum
    double p[3];
    for (int k = 0; k < 3; ++k) {
        p[k] = part.p(ip, k)/(mass*jorek::SPEED_OF_LIGHT);
        part.p(ip, k) = p[k];
    }

    // compute coordinates at half-step
    const double gamma = std::sqrt(1.0 + (p[0]*p[0] + p[1]*p[1] + p[2]*p[2]));
    for (int k = 0; k < 3; ++k)
        half_position[k] += (5.0e-1*dt*jorek::SPEED_OF_LIGHT*p[k])/gamma;
} // volume_preserving_first_half_step_jorek

/**
 * mod_kinetic_relativistic::volume_preserving_second_half_step_jorek.
 *
 * @param half_position  (x,y,z), advanced to t_(i+1) in place
 * @param E, B           electric and magnetic field, cartesian
 */
template<class PS>
JGX_HD inline void volume_preserving_second_half_step_jorek(PS& part, const std::size_t ip,
                                                            double half_position[3],
                                                            const double scaling_factor,
                                                            const double E[3], const double B[3],
                                                            const double mass, const double dt) {
    double p[3];

    // compute momentum at t_(i+1/2)
    for (int k = 0; k < 3; ++k) p[k] = part.p(ip, k) + scaling_factor*E[k];

    // rotate momentum with respect to the magnetic field
    double M[3][3], p_rot[3];
    pusher_tools::cayley_transform(
        jorek::SPEED_OF_LIGHT*scaling_factor
            / std::sqrt(1.0 + (p[0]*p[0] + p[1]*p[1] + p[2]*p[2])), B, M);
    pusher_tools::matvec3(M, p, p_rot);

    // compute momentum at t_(i+1)
    for (int k = 0; k < 3; ++k) p[k] = p_rot[k] + scaling_factor*E[k];

    // update position at t_(i+1)
    const double gamma = std::sqrt(1.0 + (p[0]*p[0] + p[1]*p[1] + p[2]*p[2]));
    for (int k = 0; k < 3; ++k)
        half_position[k] += (5.0e-1*dt*jorek::SPEED_OF_LIGHT*p[k])/gamma;

    // compute dimensional momentum
    for (int k = 0; k < 3; ++k) part.p(ip, k) = p[k]*mass*jorek::SPEED_OF_LIGHT;
} // volume_preserving_second_half_step_jorek

/**
 * One "find the (i_elm,s,t) coordinates" step of the push.
 *
 * The Fortran hands particle%st and particle%i_elm to find_RZ_nearby as both
 * the guess and the result; the old values are read before the new ones are
 * written, so copying them out first is the same thing without the aliasing.
 *
 * @param R_new, Z_new  where the particle has moved to
 * @param phi_search    the angle the global fallback searches at, worked out by
 *                      the facade -- see mod_find_rz_nearby.f90
 */
template<bool Debug, class PS, class FS>
JGX_HD inline void find_particle_st(PS& part, const std::size_t ip, const FS& fields,
                                    const double R_new, const double Z_new,
                                    const double phi_search,
                                    int& ifail, push_diagnostics& diag) {
    const double R_old = part.x(ip, 0), Z_old = part.x(ip, 1);
    const double s_old = part.st(ip, 0), t_old = part.st(ip, 1);
    const int    i_elm_old = static_cast<int>(part.i_elm(ip));

    double s_new, t_new;
    int    i_elm_new, not_found, bad_i_from, bad_i_to;

    /* The Fortran omits the optional phi, so the local interpolation runs at
     * p = 0. That is a reduced-MHD assumption: find_RZ_nearby stops on a
     * stellarator model when phi is absent. */
    find_rz_nearby::find_RZ_nearby<Debug>(fields.element_list, fields.node_list,
                                          R_old, Z_old, s_old, t_old, i_elm_old,
                                          R_new, Z_new, 0.0, phi_search,
                                          s_new, t_new, i_elm_new, ifail,
                                          not_found, bad_i_from, bad_i_to);

    part.st(ip, 0) = s_new;
    part.st(ip, 1) = t_new;
    part.i_elm(ip) = i_elm_new;

    if (not_found != 0 && diag.not_found == 0) {
        diag.not_found = not_found;
        diag.nf_R = R_old;
        diag.nf_Z = Z_old;
    }
    if (bad_i_to != 0 && diag.bad_i_to == 0) {
        diag.bad_i_from = bad_i_from;
        diag.bad_i_to   = bad_i_to;
    }
} // find_particle_st

/**
 * mod_kinetic_relativistic::volume_preserving_push_jorek -- integrate one
 * relativistic particle over one step of the VPA in JOREK fields.
 *
 * @param part, ip     the particle set and the particle in it
 * @param fields       the grid and the interpolation strategy
 * @param mass         particle mass [AMU]
 * @param time         current time [s]
 * @param timestep     time step [s]
 * @param phi_search   see find_particle_st
 * @param use_radreact whether to apply the radiation-reaction force after the
 *                     second half-step, which is what the Fortran spells as the
 *                     separate volume_preserving_radiation_push_jorek
 * @param[out] ifail   find_RZ_nearby's own code, from whichever of the two
 *                     searches ran last. Untouched when the particle was
 *                     already lost on entry, as in the Fortran.
 * @param[out] diag    what the searches would have printed
 *
 * @tparam Debug  find_RZ_nearby's #ifdef DEBUG consistency check; the shim decides.
 */
template<bool Debug, class PS, class FS>
JGX_HD inline void volume_preserving_push_jorek(PS& part, const std::size_t ip,
                                                const FS& fields,
                                                const double mass, const double time,
                                                const double timestep,
                                                const double phi_search,
                                                const bool use_radreact,
                                                int& ifail, push_diagnostics& diag) {
    diag = push_diagnostics{};

    // check if the particle is valid
    if (part.i_elm(ip) <= 0) return;

    // half_position: 0:x, 1:y, 2:z, 3:R, 4:Z, 5:phi
    double half_position[6];

    // transform the particle position from cylindrical to cartesian coordinates
    const double x_cyl[3] = { part.x(ip, 0), part.x(ip, 1), part.x(ip, 2) };
    coordinate_transforms::cylindrical_to_cartesian(x_cyl, half_position);

    // compute first half-step
    double scaling_factor;
    volume_preserving_first_half_step_jorek(part, ip, half_position, mass, timestep,
                                            scaling_factor);

    // calculate cylindrical coordinates from cartesian ones
    coordinate_transforms::cartesian_to_cylindrical(half_position, half_position + 3);

    // find the (i_elm,s,t) coordinates
    find_particle_st<Debug>(part, ip, fields, half_position[3], half_position[4],
                            phi_search, ifail, diag);

    // check if the particle is lost, exit if it is the case
    if (part.i_elm(ip) <= 0) return;

    // copy RZPHI coordinates in particles
    for (int k = 0; k < 3; ++k) part.x(ip, k) = half_position[3 + k];

    // compute magnetic and electric fields
    double E_[3], B_[3], psi, U;
    const std::size_t ve[1] = { 3 };
    const jgx::view<double, 1> E(E_, ve), B(B_, ve);
    fields.calc_EBpsiU_reduced(time + 5.0e-1*timestep,
                               static_cast<std::size_t>(part.i_elm(ip)) - 1,
                               part.st(ip, 0), part.st(ip, 1), part.x(ip, 2),
                               E, B, psi, U);

    // compute the second half-step
    double E_cart[3], B_cart[3];
    coordinate_transforms::vector_cylindrical_to_cartesian(part.x(ip, 2), E_, E_cart);
    coordinate_transforms::vector_cylindrical_to_cartesian(part.x(ip, 2), B_, B_cart);
    volume_preserving_second_half_step_jorek(part, ip, half_position, scaling_factor,
                                             E_cart, B_cart, mass, timestep);

    /* The one line that separates volume_preserving_radiation_push_jorek from
     * this routine in the Fortran: the radiation-reaction force, taking the
     * cylindrical B at the half-step. A runtime flag rather than a second
     * instantiation -- it is uniform over a launch, and the locals it needs are
     * live only inside the branch. */
    if (use_radreact)
        radreactforce::radreactforce_kinetic(part, ip, B_, timestep, mass);

    // transform back from cartesian to cylindrical coordinates
    coordinate_transforms::cartesian_to_cylindrical(half_position, half_position + 3);

    // find the (i_elm,s,t) coordinates
    find_particle_st<Debug>(part, ip, fields, half_position[3], half_position[4],
                            phi_search, ifail, diag);

    // copy new RZPHI position into particle
    for (int k = 0; k < 3; ++k) part.x(ip, k) = half_position[3 + k];
} // volume_preserving_push_jorek

} // namespace kinetic_relativistic

#endif // JOREK_KINETIC_RELATIVISTIC_H
