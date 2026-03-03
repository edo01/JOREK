#include <cmath>

extern "C"
void compute_re_kinematics(
    const double* cylindrical_velocity,
    const double* B_norm2,
    const double* cylindrical_momentum,
    const double* mass_electron,
    const double* atomic_mass_unit,
    const double* speed_of_light,
    double* v_par,
    double* v_perp,
    double* gamma_m)
{
    // Dot product v · B
    double vp = 0.0;
    for (int i = 0; i < 3; ++i)
        vp += cylindrical_velocity[i] * B_norm2[i];

    *v_par = vp;

    // Compute perpendicular velocity norm
    double diff[3];
    for (int i = 0; i < 3; ++i)
        diff[i] = cylindrical_velocity[i] - vp * B_norm2[i];

    double vperp2 = 0.0;
    for (int i = 0; i < 3; ++i)
        vperp2 += diff[i] * diff[i];

    *v_perp = std::sqrt(vperp2);

    // gamma_m
    double mom2 = 0.0;
    for (int i = 0; i < 3; ++i)
        mom2 += cylindrical_momentum[i] * cylindrical_momentum[i];

    *gamma_m = std::sqrt(
        (*mass_electron) * (*mass_electron) +
        mom2 * (*atomic_mass_unit) * (*atomic_mass_unit) /
        ((*speed_of_light) * (*speed_of_light))
    );
}