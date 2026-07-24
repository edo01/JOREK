#pragma once
// Device-side physics for RE small-angle Coulomb collisions plus the pcg32 RNG.
// GPU port of:
//   - tools/pcg_basic.c                       (pcg32 RNG)
//   - tools/mod_bessel.f90                    (scaled Bessel K0/K1/K2)
//   - particles/pushers/mod_ccoll_relativistic.f90
//     (standard kinetic operator only: electron + single main ion, Ti=Te; the
//      partial-screening / impurity operator is NOT ported)
//
// Included by kernel_re_evolution_common.hip.hpp after its constants section
// (uses PI_VAL, EL_CHG, SPEED_OF_LIGHT, MASS_ELECTRON, ATOMIC_MASS_UNIT).
// No extern "C" entry points here.

// ---------------------------------------------------------------------------
// Physical constants used only by the collision physics (models/constants.f90)
// ---------------------------------------------------------------------------
static constexpr double EPS_ZERO_VAL = 8.854187817e-12;   // vacuum permittivity [F/m]
static constexpr double K_BOLTZ_VAL  = 1.3806488e-23;     // Boltzmann constant [J/K]
static constexpr double HBAR_VAL     = 1.05457180e-34;    // reduced Planck constant [Js]

// ===========================================================================
//                 bind(C) struct mirrors (see mod_particle_types.f90)
// ===========================================================================

// Tabulated L0/L1 collision data + species parameters.
// Fortran: type ccoll_data_c
struct ccoll_data_c {
    int nu;               // number of u (p/mc) grid points
    int nth;              // number of theta (T/mc^2) grid points
    const double* log10_u;     // (nu)      log10 of u abscissa (host-precomputed)
    const double* log10_theta; // (nth)     log10 of theta abscissa
    const double* L0;          // (nu, nth) tabulated L0, column-major
    const double* L1;          // (nu, nth) tabulated L1, column-major
    double u_min, u_max;       // table domain edges (linear scale)
    double th_min, th_max;
    double ma;            // test particle mass [kg]
    double qa;            // test particle charge number
    double mb_ion;        // main ion mass [kg]
    double qb_ion;        // main ion charge number
};

// RE collision / radiation-reaction runtime parameters.
// Fortran: type re_gpu_params_c (doubles + int64 first, ints last)
struct re_gpu_params_c {
    double central_density;    // JOREK central density [1e20 m^-3]
    double T_norm;             // conversion factor: P_T (JOREK units) -> Te [K]
    long long rng_seed;        // base seed for the per-thread pcg32 streams
    int re_ccoll;              // 1 = apply small-angle collisions each substep
    int re_radreact;           // 1 = apply the radiation-reaction force in the push
};

// ===========================================================================
//                              pcg32 RNG (device)
// ===========================================================================
// Transcribed from tools/pcg_basic.c (O'Neill, Apache-2.0).  One stream per
// GPU worker thread, mirroring the CPU's one-pcg32-stream-per-OMP-thread setup.
// Bit-identical to the CPU C implementation so GPU and CPU runs share streams.

struct pcg32_state {
    unsigned long long state;
    unsigned long long inc;
};

__device__ __forceinline__
unsigned int pcg32_next_u32(pcg32_state& rng)
{
    unsigned long long oldstate = rng.state;
    rng.state = oldstate * 6364136223846793005ULL + rng.inc;
    unsigned int xorshifted = (unsigned int)(((oldstate >> 18u) ^ oldstate) >> 27u);
    unsigned int rot = (unsigned int)(oldstate >> 59u);
    // CPU C form is (xorshifted >> rot) | (xorshifted << ((-rot) & 31)); the
    // (32u - rot) & 31u below is identical for every rot in [0,31].
    return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

__device__ __forceinline__
void pcg32_srandom(pcg32_state& rng, unsigned long long initstate, unsigned long long initseq)
{
    rng.state = 0U;
    rng.inc = (initseq << 1u) | 1u;
    pcg32_next_u32(rng);
    rng.state += initstate;
    pcg32_next_u32(rng);
}

// Uniform double in [0,1) with 2^-32 precision (matches pcg32_random_double_r).
__device__ __forceinline__
double pcg32_next_double(pcg32_state& rng)
{
    return ldexp((double)pcg32_next_u32(rng), -32);
}

// +-1 Bernoulli matching the first two moments of N(0,1): the CPU wrappers use
// rnd = -1 + 2*floor(2*u), i.e. +1 iff u >= 0.5 iff the top bit of the u32 draw
// is set.  Consumes exactly one draw like pcg32_next_double.
__device__ __forceinline__
double pcg32_next_pm1(pcg32_state& rng)
{
    return (pcg32_next_u32(rng) & 0x80000000u) ? 1.0 : -1.0;
}

// ===========================================================================
//              Scaled modified Bessel functions (tools/mod_bessel.f90)
// ===========================================================================
// bessel_kNexp(x) = exp(x) * K_N(x); rational minimax approximations
// (Russon & Blair via Cody/Stoltz/Burkardt).  Only the scaled (jint=2)
// branches are transcribed — that is the only variant the collision code uses.

__device__
double bessel_k0exp_gpu(double x)
{
    constexpr double xsmall = 1.11e-16;
    constexpr double xinf   = 1.79e+308;

    const double p[6] = { 5.8599221412826100000e-04, 1.3166052564989571850e-01,
                          1.1999463724910714109e+01, 4.6850901201934832188e+02,
                          5.9169059852270512312e+03, 2.4708152720399552679e+03 };
    const double q[2] = {-2.4994418972832303646e+02, 2.1312714303849120380e+04 };
    const double f[4] = {-1.6414452837299064100e+00,-2.9601657892958843866e+02,
                         -1.7733784684952985886e+04,-4.0320340761145482298e+05 };
    const double g[3] = {-2.5064972445877992730e+02, 2.9865713163054025489e+04,
                         -1.6128136304458193998e+06 };
    const double pp[10] = { 1.1394980557384778174e+02, 3.6832589957340267940e+03,
                            3.1075408980684392399e+04, 1.0577068948034021957e+05,
                            1.7398867902565686251e+05, 1.5097646353289914539e+05,
                            7.1557062783764037541e+04, 1.8321525870183537725e+04,
                            2.3444738764199315021e+03, 1.1600249425076035558e+02 };
    const double qq[10] = { 2.0013443064949242491e+02, 4.4329628889746408858e+03,
                            3.1474655750295278825e+04, 9.7418829762268075784e+04,
                            1.5144644673520157801e+05, 1.2689839587977598727e+05,
                            5.8824616785857027752e+04, 1.4847228371802360957e+04,
                            1.8821890840982713696e+03, 9.2556599177304839811e+01 };

    if (x <= 0.0) return xinf;

    if (x <= 1.0) {
        double temp = log(x);
        if (x < xsmall) return p[5] / q[1] - temp;
        double xx = x * x;
        double sump = ((((p[0]*xx + p[1])*xx + p[2])*xx + p[3])*xx + p[4])*xx + p[5];
        double sumq = (xx + q[0]) * xx + q[1];
        double sumf = ((f[0]*xx + f[1])*xx + f[2])*xx + f[3];
        double sumg = ((xx + g[0]) * xx + g[1]) * xx + g[2];
        double result = sump / sumq - xx * sumf * temp / sumg - temp;
        return result * exp(x);
    }

    double xx = 1.0 / x;
    double sump = pp[0];
    for (int i = 1; i < 10; ++i) sump = sump * xx + pp[i];
    double sumq = xx;
    for (int i = 0; i < 9; ++i) sumq = (sumq + qq[i]) * xx;
    sumq += qq[9];
    return sump / sumq / sqrt(x);
}

__device__
double bessel_k1exp_gpu(double x)
{
    constexpr double xleast = 2.23e-308;
    constexpr double xsmall = 1.11e-16;
    constexpr double xinf   = 1.79e+308;

    const double p[5] = { 4.8127070456878442310e-1, 9.9991373567429309922e+1,
                          7.1885382604084798576e+3, 1.7733324035147015630e+5,
                          7.1938920065420586101e+5 };
    const double q[3] = {-2.8143915754538725829e+2, 3.7264298672067697862e+4,
                         -2.2149374878243304548e+6 };
    const double f[5] = {-2.2795590826955002390e-1,-5.3103913335180275253e+1,
                         -4.5051623763436087023e+3,-1.4758069205414222471e+5,
                         -1.3531161492785421328e+6 };
    const double g[3] = {-3.0507151578787595807e+2, 4.3117653211351080007e+4,
                         -2.7062322985570842656e+6 };
    const double pp[11] = { 6.4257745859173138767e-2, 7.5584584631176030810e+0,
                            1.3182609918569941308e+2, 8.1094256146537402173e+2,
                            2.3123742209168871550e+3, 3.4540675585544584407e+3,
                            2.8590657697910288226e+3, 1.3319486433183221990e+3,
                            3.4122953486801312910e+2, 4.4137176114230414036e+1,
                            2.2196792496874548962e+0 };
    const double qq[9] = { 3.6001069306861518855e+1, 3.3031020088765390854e+2,
                           1.2082692316002348638e+3, 2.1181000487171943810e+3,
                           1.9448440788918006154e+3, 9.6929165726802648634e+2,
                           2.5951223655579051357e+2, 3.4552228452758912848e+1,
                           1.7710478032601086579e+0 };

    if (x < xleast) return xinf;

    if (x <= 1.0) {
        if (x < xsmall) return 1.0 / x;
        double xx = x * x;
        double sump = ((((p[0]*xx + p[1])*xx + p[2])*xx + p[3])*xx + p[4])*xx + q[2];
        double sumq = ((xx + q[0])*xx + q[1])*xx + q[2];
        double sumf = (((f[0]*xx + f[1])*xx + f[2])*xx + f[3])*xx + f[4];
        double sumg = ((xx + g[0])*xx + g[1])*xx + g[2];
        double result = (xx * log(x) * sumf / sumg + sump / sumq) / x;
        return result * exp(x);
    }

    double xx = 1.0 / x;
    double sump = pp[0];
    for (int i = 1; i < 11; ++i) sump = sump * xx + pp[i];
    double sumq = xx;
    for (int i = 0; i < 8; ++i) sumq = (sumq + qq[i]) * xx;
    sumq += qq[8];
    return sump / sumq / sqrt(x);
}

__device__ __forceinline__
double bessel_k2exp_gpu(double x)
{
    if (x == 0.0) return bessel_k0exp_gpu(x);
    return bessel_k0exp_gpu(x) + 2.0 * bessel_k1exp_gpu(x) / x;
}

// ===========================================================================
//     L0/L1 interpolation + mu functions (mod_ccoll_relativistic.f90)
// ===========================================================================

// Bilinear interpolation on the equidistant log10 grid (interp_bilinear for
// x=log10(u), y=log10(theta)).  Unlike the CPU (which can index one past the
// table at the exact upper edge), indices are clamped to the last cell.
__device__ __forceinline__
double interp_bilinear_log_gpu(const double* __restrict__ x, int nx,
                               const double* __restrict__ y, int ny,
                               const double* __restrict__ fcm,   // (nx, ny) column-major
                               double xq, double yq)
{
    double dx = x[1] - x[0];
    double dy = y[1] - y[0];
    int ix = (int)floor((xq - x[0]) / dx);
    int iy = (int)floor((yq - y[0]) / dy);
    ix = max(0, min(ix, nx - 2));
    iy = max(0, min(iy, ny - 2));

    return ( fcm[ix     + nx * iy      ] * (x[ix+1] - xq) * (y[iy+1] - yq)
           + fcm[ix + 1 + nx * iy      ] * (xq - x[ix])   * (y[iy+1] - yq)
           + fcm[ix     + nx * (iy + 1)] * (x[ix+1] - xq) * (yq - y[iy])
           + fcm[ix + 1 + nx * (iy + 1)] * (xq - x[ix])   * (yq - y[iy]) ) / (dx * dy);
}

// interp_L0L1: bilinear inside the tabulated domain, with the CPU's
// out-of-domain approximations (K0/K1exp above the u range, erf formula below).
__device__
void interp_L0L1_gpu(const ccoll_data_c& dat, double u, double theta,
                     double& L0, double& L1)
{
    // Clamp temperature to the tabulated range (CPU prints a warning here).
    double th = (theta > dat.th_max) ? dat.th_max : theta;

    if (u > dat.u_max) {
        L0 = bessel_k0exp_gpu(1.0 / th);
        L1 = bessel_k1exp_gpu(1.0 / th);
    } else if (u > dat.u_min && th > dat.th_min) {
        double lu  = log10(u);
        double lth = log10(th);
        L0 = interp_bilinear_log_gpu(dat.log10_u, dat.nu, dat.log10_theta, dat.nth, dat.L0, lu, lth);
        L1 = interp_bilinear_log_gpu(dat.log10_u, dat.nu, dat.log10_theta, dat.nth, dat.L1, lu, lth);
    } else {
        // Below the tabulated domain (uses the unclamped theta, as on CPU).
        L0 = sqrt(PI_VAL * theta / 2.0) * erf(u / sqrt(2.0 * theta));
        L1 = L0;
    }
}

// ccoll_mufuncs (values only — the kinetic push needs no derivatives).
__device__
void ccoll_mufuncs_gpu(const ccoll_data_c& dat, double u, double th,
                       double& mu0, double& mu1, double& mu2)
{
    double u2      = u * u;
    double gammasq = 1.0 + u2;
    double gamma   = sqrt(gammasq);
    double expBessel2    = bessel_k2exp_gpu(1.0 / th);
    double expgammatheta = exp((1.0 - gamma) / th);
    double tg  = th * gamma;
    double th2 = th * th;

    double L0, L1;
    interp_L0L1_gpu(dat, u, th, L0, L1);

    mu0 = (gammasq * L0 - th * L1 + (th - gamma) * u * expgammatheta) / expBessel2;
    mu1 = (gammasq * L1 - th * L0 + (th * gamma - 1.0) * u * expgammatheta) / expBessel2;
    mu2 = (2.0 * tg * L1 + (1.0 + 2.0 * th2) * u * expgammatheta) / (th * expBessel2);
}

// ===========================================================================
//      Coulomb logarithm + collision coefficients (2 species, unrolled)
// ===========================================================================

// ccoll_clog for the electron background and ONE ion species.
// qa_C / qi_C are charges in Coulombs (the CPU passes qa*EL_CHG, Z0*EL_CHG).
__device__ __forceinline__
void ccoll_clog_gpu(double ma, double qa_C, double mi, double qi_C,
                    double ne, double the, double ni, double thi, double u,
                    double& cloge, double& clogi)
{
    double debyeLength = sqrt(EPS_ZERO_VAL * SPEED_OF_LIGHT * SPEED_OF_LIGHT
                              / ((ne * EL_CHG * EL_CHG) / (the * MASS_ELECTRON)));

    // Electron contribution.  NOTE: as on CPU, bcl = qa_C*EL_CHG (charge times
    // elementary charge), which is negative for electrons, so max(bcl,bqm)
    // effectively selects the quantum impact parameter.  Kept identical.
    double ubar = SPEED_OF_LIGHT * sqrt(u*u / (1.0 + u*u) + 3.0 * the);
    double mr   = ma * MASS_ELECTRON / (ma + MASS_ELECTRON);
    double bcl  = qa_C * EL_CHG / (4.0 * PI_VAL * EPS_ZERO_VAL * mr * ubar * ubar);
    double bqm  = HBAR_VAL / (2.0 * mr * ubar);
    cloge = log(debyeLength / fmax(bcl, bqm));

    // Ion contribution.
    ubar = SPEED_OF_LIGHT * sqrt(u*u / (1.0 + u*u) + 3.0 * thi);
    mr   = ma * mi / (ma + mi);
    bcl  = qa_C * qi_C / (4.0 * PI_VAL * EPS_ZERO_VAL * mr * ubar * ubar);
    bqm  = HBAR_VAL / (2.0 * mr * ubar);
    clogi = log(debyeLength / fmax(bcl, bqm));
}

// ccoll_coeffs for one background species: K, Dpar, Dperp only.
__device__ __forceinline__
void ccoll_coeffs_gpu(const ccoll_data_c& dat,
                      double ma, double qa_C, double clog,
                      double mb, double qb_C, double nb, double thb, double u,
                      double& K, double& Dpar, double& Dperp)
{
    double gamma = sqrt(1.0 + u*u);
    double Gab = nb * (qa_C * qb_C) * (qa_C * qb_C) * clog
               / (4.0 * PI_VAL * EPS_ZERO_VAL * EPS_ZERO_VAL * ma * ma
                  * SPEED_OF_LIGHT * SPEED_OF_LIGHT * SPEED_OF_LIGHT);
    double u2 = u * u;
    double u3 = u2 * u;

    double mu0, mu1, mu2;
    ccoll_mufuncs_gpu(dat, u, thb, mu0, mu1, mu2);

    K     = -Gab * (mu0 / gamma + (ma / mb) * mu1) / u2;
    Dpar  =  Gab * gamma * thb * mu1 / u3;
    Dperp =  Gab * (u2 * (mu0 + gamma * thb * mu2) - thb * mu1) / (2.0 * gamma * u3);
}

// ===========================================================================
//        Euler–Maruyama kinetic collision push (standard operator)
// ===========================================================================
// Mirrors ccoll_kinetic_relativistic_explicitpush for electron + one main ion.
// pm is the momentum in the particle SoA units [AMU m/s]; it is normalised to
// p/mc internally and updated in place.  Draws exactly 3 RNG values.
__device__
void ccoll_kinetic_push_gpu(const ccoll_data_c& dat,
                            double ne, double the, double ni, double thi,
                            double dt, double group_mass, double pm[3],
                            pcg32_state& rng,
                            // DEBUG (ccoll validation, MODE B): when non-null,
                            // write uin(3),uout(3) of this collision to
                            // du_dump[0..5]. Null in production -> no effect.
                            double* du_dump = nullptr)
{
    double mc = group_mass * SPEED_OF_LIGHT;
    double uin[3] = { pm[0] / mc, pm[1] / mc, pm[2] / mc };

    // Wiener process for this step: dW = sqrt(dt) * (+-1 Bernoulli draws).
    // The three draws are consumed unconditionally, exactly as the CPU
    // rng(i_rng)%next(rnd) call does, to keep the stream aligned with the CPU.
    double dW[3] = { sqrt(dt) * pcg32_next_pm1(rng),
                     sqrt(dt) * pcg32_next_pm1(rng),
                     sqrt(dt) * pcg32_next_pm1(rng) };
    // double dW[3] = { sqrt(dt), -sqrt(dt), sqrt(dt) };

    double u = sqrt(uin[0]*uin[0] + uin[1]*uin[1] + uin[2]*uin[2]);
    if (u <= 0.0) return;
    double uhat[3] = { uin[0]/u, uin[1]/u, uin[2]/u };

    double qa_C = dat.qa * EL_CHG;
    double qi_C = dat.qb_ion * EL_CHG;

    double cloge, clogi;
    ccoll_clog_gpu(dat.ma, qa_C, dat.mb_ion, qi_C, ne, the, ni, thi, u, cloge, clogi);

    // Electron + main ion contributions.
    double K, Dpar, Dperp, Kb, Dparb, Dperpb;
    ccoll_coeffs_gpu(dat, dat.ma, qa_C, cloge, MASS_ELECTRON, -EL_CHG, ne, the, u,
                     K, Dpar, Dperp);
    ccoll_coeffs_gpu(dat, dat.ma, qa_C, clogi, dat.mb_ion, qi_C, ni, thi, u,
                     Kb, Dparb, Dperpb);
    K     += Kb;
    Dpar  += Dparb;
    Dperp += Dperpb;

    // Euler–Maruyama update.
    double udotW = uhat[0]*dW[0] + uhat[1]*dW[1] + uhat[2]*dW[2];
    double cpar  = K * dt + sqrt(2.0 * Dpar) * udotW;
    double cperp = sqrt(2.0 * Dperp);
    double uout[3] = {
        uin[0] + cpar * uhat[0] + cperp * (dW[0] - udotW * uhat[0]),
        uin[1] + cpar * uhat[1] + cperp * (dW[1] - udotW * uhat[1]),
        uin[2] + cpar * uhat[2] + cperp * (dW[2] - udotW * uhat[2])
    };

    pm[0] = uout[0] * mc;
    pm[1] = uout[1] * mc;
    pm[2] = uout[2] * mc;

    if (du_dump) {
        du_dump[0] = uin[0];  du_dump[1] = uin[1];  du_dump[2] = uin[2];
        du_dump[3] = uout[0]; du_dump[4] = uout[1]; du_dump[5] = uout[2];
    }
}
