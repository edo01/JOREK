// =============================================================================
// HIP kernel implementation for evolving relativistic runaway electrons (REs)
// Equivalent of evolve_REs_gpu (OpenMP target) in mod_particle_evolution.f90
//
// All Fortran arrays are column-major. Indices in comments refer to 1-based
// Fortran conventions; C code uses 0-based offsets.
// =============================================================================
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include "models/mod_settings.h"
#include "optimization_defines.h"

// TODO: There are some parameters passed to different functions only for debugging purposes (e.g. debug_j, debug_k).
// They may be handled through precompiler guard GPU_DEBUG as well.

// TODO: Discuss if this method to take hard-coded compile-time parameters from mod_settings.h is ok

// ---------------------------------------------------------------------------
// Compile-time parameters, taken and renamed from models/mod_settings.h
// ---------------------------------------------------------------------------

static constexpr int N_TOR = n_tor;
static constexpr int N_COORD_TOR = n_coord_tor;
static constexpr int N_PERIOD = n_period;
static constexpr int N_COORD_PERIOD = n_coord_period;
static constexpr int N_ORDER = n_order;

static constexpr int NV       = n_vertex_max;
static constexpr int NDEG     = n_degrees;
static constexpr int NDIM     = n_dim;
static constexpr int NMODE    = (N_TOR - 1) / 2;

static constexpr int NVAR = 3;
static constexpr int P_PAR_IDX = 0;
static constexpr int P_PERP_IDX = 1;
static constexpr int J_PHI_IDX = 2;

// ---------------------------------------------------------------------------
// Physical constants (matching jorek/models/constants.f90)
// ---------------------------------------------------------------------------
static constexpr double PI_VAL           = 3.14159265358979323846;
static constexpr double MU_ZERO          = 4.0e-7 * PI_VAL;
static constexpr double EL_CHG           = 1.602176565e-19;
static constexpr double ATOMIC_MASS_UNIT = 1.660539040e-27;
static constexpr double MASS_ELECTRON    = 9.10938291e-31;
static constexpr double SPEED_OF_LIGHT   = 2.997924580105029e+8;

// ---------------------------------------------------------------------------
// Fortran column-major indexing helpers (0-based indices)
// ---------------------------------------------------------------------------
__device__ __forceinline__
int idx2(int i0, int i1, int d0)
{ return i0 + d0 * i1; }

__device__ __forceinline__
int idx3(int i0, int i1, int i2, int d0, int d1)
{ return i0 + d0 * (i1 + d1 * i2); }

__device__ __forceinline__
int idx4(int i0, int i1, int i2, int i3, int d0, int d1, int d2)
{ return i0 + d0 * (i1 + d1 * (i2 + d2 * i3)); }

int idx4_host(int i0, int i1, int i2, int i3, int d0, int d1, int d2)
{ return i0 + d0 * (i1 + d1 * (i2 + d2 * i3)); }


__device__ __forceinline__
int idx5(int i0, int i1, int i2, int i3, int i4,
         int d0, int d1, int d2, int d3)
{ return i0 + d0 * (i1 + d1 * (i2 + d2 * (i3 + d3 * i4))); }


int idx5_host(int i0, int i1, int i2, int i3, int i4,
         int d0, int d1, int d2, int d3)
{ return i0 + d0 * (i1 + d1 * (i2 + d2 * (i3 + d3 * i4))); }

__device__ __forceinline__
bool rz_dbg_enabled(int debug_j, int debug_k)
{ return (debug_j >= 0 && debug_j < 3 && debug_k == 100000); }

// ---------------------------------------------------------------------------
// HIP error check macro
// ---------------------------------------------------------------------------
#define HIP_CHECK(call)                                                       \
  do {                                                                        \
    hipError_t err = (call);                                                  \
    if (err != hipSuccess) {                                                  \
      fprintf(stderr, "HIP error %s at %s:%d\n",                             \
              hipGetErrorString(err), __FILE__, __LINE__);                    \
    }                                                                         \
  } while (0)

// ===========================================================================================
//                   DATA STRUCTURES  (Fortran bind(C) compatible)
// ===========================================================================================

// Particle SoA for relativistic kinetic particles.
// Fortran: type particle_SoA_kinetic_relativistic
struct particle_SoA_kinetic_relativistic {
    double* x;        // (num_particles, 3) position (cylindrical: R, Z, phi)
    double* p;        // (num_particles, 3) momentum (Cartesian)
    double* st;       // (num_particles, 2) element-local coordinates (s, t)
    double* weight;   // (num_particles)    macro-particle weight
    int*    i_elm;    // (num_particles)    element index (1-based; <=0 means lost)
    int*    i_life;   // (num_particles)    life-step counter
    int*    t_birth;  // (num_particles)    birth time-step
};

// Group of particles sharing the same species properties.
// Fortran: type particle_group
struct particle_group {
    double mass;                 // species mass in AMU
    double charge;               // charge number (e.g. -1.0 for electrons)
    int    num_particles;        // total number of particles allocated
    particle_SoA_kinetic_relativistic particles; // pointer to SoA data
};

// Node (DOF) list in Structure-of-Arrays layout.
// Fortran: type node_list_SoA
struct node_list_SoA {
    int     n_nodes;   // total number of nodes
    double* x;         // (n_nodes, N_COORD_TOR, NDEG, NDIM) grid coordinates
    double* values;    // (n_nodes, N_TOR, NDEG, NVAR) field values at current time
    double* deltas;    // (n_nodes, N_TOR, NDEG, NVAR) field increments (for time interp)
};

// Element list in Structure-of-Arrays layout.
// Fortran: type element_list_SoA
struct element_list_SoA {
    int     n_elements; // total number of elements
    int*    vertex;     // (n_elements, NV)       1-based node indices per vertex
    int*    neighbours; // (n_elements, NV)       1-based neighbour element indices (0=boundary)
    double* size;       // (n_elements, NV, NDEG) basis-function scale factors
};

// Linear time-interpolated field accessor.
// Fortran: type jorek_fields_interp_linear
struct jorek_fields_interp_linear {
    node_list_SoA    node_list;
    element_list_SoA element_list;
    double  time_now;            // current simulation time (normalised)
    double  time_prev;           // previous time-step time (normalised)
    int     flag_static;         // 1 = fields are static (no time interpolation)
    int     flag_zero_dpsidt;    // 1 = force dPsi/dt = 0 in E-field
    double  F0;                  // vacuum toroidal field function: F0 = R*B_phi
    double  t_norm;              // time normalisation sqrt(mu0*AMU*mass_ref*n_ref*1e20)
    int*    mode_coord;          // (N_COORD_TOR) toroidal mode numbers for grid harmonics
};

// Top-level simulation context passed from Fortran.
// Fortran: type particle_sim
struct particle_sim {
    jorek_fields_interp_linear fields;  // interpolated field data
    particle_group             group;   // particle group data
    double sim_time;                    // current simulation time
    int    my_id, n_mpi;                // MPI identifiers
};

// ===========================================================================================
//                            DEVICE HELPER FUNCTIONS
// ===========================================================================================

// TODO: in basisfunctions_2D_0 and basisfunctions_2D_1, we are assuming N_ORDER = 3
// Implement also for N_ORDER = 5 ??

// ---------------------------------------------------------------------------
// 2D cubic basis functions – H indexed as idx2(kf, kv, NDEG) = kf + NDEG * kv
// kv in [0, NV-1], kf in [0, NDEG-1]  (kf is the fast-varying dimension)
// ---------------------------------------------------------------------------
__device__ __forceinline__
void basisfunctions_2D_0(double s, double t,
                         double* __restrict__ H)
{
    double sm1  = s - 1.0;
    double tm1  = t - 1.0;
    double sm12 = sm1 * sm1;
    double s2   = s * s;
    double tm12 = tm1 * tm1;
    double t2   = t * t;

    // vertex 1 (kv=0)
    H[idx2(0, 0, NDEG)] = sm12 * (1.0 + 2.0*s) * tm12 * (1.0 + 2.0*t);
    H[idx2(1, 0, NDEG)] = 3.0 * sm12 * s * tm12 * (1.0 + 2.0*t);
    H[idx2(2, 0, NDEG)] = 3.0 * sm12 * (1.0 + 2.0*s) * tm12 * t;
    H[idx2(3, 0, NDEG)] = 9.0 * sm12 * s * tm12 * t;
    // vertex 2 (kv=1)
    H[idx2(0, 1, NDEG)] = -(s2 * (-3.0 + 2.0*s) * tm12 * (1.0 + 2.0*t));
    H[idx2(1, 1, NDEG)] = -3.0 * sm1 * s2 * tm12 * (1.0 + 2.0*t);
    H[idx2(2, 1, NDEG)] = -3.0 * s2 * (-3.0 + 2.0*s) * tm12 * t;
    H[idx2(3, 1, NDEG)] = -9.0 * sm1 * s2 * tm12 * t;
    // vertex 3 (kv=2)
    H[idx2(0, 2, NDEG)] = s2 * (-3.0 + 2.0*s) * t2 * (-3.0 + 2.0*t);
    H[idx2(1, 2, NDEG)] = 3.0 * sm1 * s2 * t2 * (-3.0 + 2.0*t);
    H[idx2(2, 2, NDEG)] = 3.0 * s2 * (-3.0 + 2.0*s) * tm1 * t2;
    H[idx2(3, 2, NDEG)] = 9.0 * sm1 * s2 * tm1 * t2;
    // vertex 4 (kv=3)
    H[idx2(0, 3, NDEG)] = -(sm12 * (1.0 + 2.0*s) * t2 * (-3.0 + 2.0*t));
    H[idx2(1, 3, NDEG)] = -3.0 * sm12 * s * t2 * (-3.0 + 2.0*t);
    H[idx2(2, 3, NDEG)] = -3.0 * sm12 * (1.0 + 2.0*s) * tm1 * t2;
    H[idx2(3, 3, NDEG)] = -9.0 * sm12 * s * tm1 * t2;
}

// ---------------------------------------------------------------------------
// 2D cubic basis functions with first derivatives.
// H, H_s, H_t all indexed as idx2(kf, kv, NDEG) = kf + NDEG * kv
// (kf is the fast-varying dimension, matching basisfunctions_2D_0).
// Used by interp_RZP_1_gpu and calc_EBpsiU.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void basisfunctions_2D_1(double s, double t,
                         double* __restrict__ H,
                         double* __restrict__ H_s,
                         double* __restrict__ H_t)
{
    basisfunctions_2D_0(s, t, H);

    double sm1  = s - 1.0;
    double tm1  = t - 1.0;
    double sm12 = sm1 * sm1;
    double s2   = s * s;
    double tm12 = tm1 * tm1;
    double t2   = t * t;

    // vertex 1
    H_s[idx2(0, 0, NDEG)] = 6.0*sm1*s * tm12*(1.0+2.0*t);
    H_t[idx2(0, 0, NDEG)] = 6.0*sm12*(1.0+2.0*s) * tm1*t;
    H_s[idx2(1, 0, NDEG)] = 3.0*sm1*(-1.0+3.0*s) * tm12*(1.0+2.0*t);
    H_t[idx2(1, 0, NDEG)] = 18.0*sm12*s * tm1*t;
    H_s[idx2(2, 0, NDEG)] = 18.0*sm1*s * tm12*t;
    H_t[idx2(2, 0, NDEG)] = 3.0*sm12*(1.0+2.0*s) * tm1*(-1.0+3.0*t);
    H_s[idx2(3, 0, NDEG)] = 9.0*sm1*(-1.0+3.0*s) * tm12*t;
    H_t[idx2(3, 0, NDEG)] = 9.0*sm12*s * tm1*(-1.0+3.0*t);
    // vertex 2
    H_s[idx2(0, 1, NDEG)] = -6.0*sm1*s * tm12*(1.0+2.0*t);
    H_t[idx2(0, 1, NDEG)] = -6.0*s2*(-3.0+2.0*s) * tm1*t;
    H_s[idx2(1, 1, NDEG)] = -3.0*s*(-2.0+3.0*s) * tm12*(1.0+2.0*t);
    H_t[idx2(1, 1, NDEG)] = -18.0*sm1*s2 * tm1*t;
    H_s[idx2(2, 1, NDEG)] = -18.0*sm1*s * tm12*t;
    H_t[idx2(2, 1, NDEG)] = 3.0*s2*(-3.0+2.0*s) * (1.0-3.0*t)*tm1;
    H_s[idx2(3, 1, NDEG)] = -9.0*s*(-2.0+3.0*s) * tm12*t;
    H_t[idx2(3, 1, NDEG)] = 9.0*sm1*s2 * (1.0-3.0*t)*tm1;
    // vertex 3
    H_s[idx2(0, 2, NDEG)] = 6.0*sm1*s * t2*(-3.0+2.0*t);
    H_t[idx2(0, 2, NDEG)] = 6.0*s2*(-3.0+2.0*s) * tm1*t;
    H_s[idx2(1, 2, NDEG)] = 3.0*s*(-2.0+3.0*s) * t2*(-3.0+2.0*t);
    H_t[idx2(1, 2, NDEG)] = 18.0*sm1*s2 * tm1*t;
    H_s[idx2(2, 2, NDEG)] = 18.0*sm1*s * tm1*t2;
    H_t[idx2(2, 2, NDEG)] = 3.0*s2*(-3.0+2.0*s) * t*(-2.0+3.0*t);
    H_s[idx2(3, 2, NDEG)] = 9.0*s*(-2.0+3.0*s) * tm1*t2;
    H_t[idx2(3, 2, NDEG)] = 9.0*sm1*s2 * t*(-2.0+3.0*t);
    // vertex 4
    H_s[idx2(0, 3, NDEG)] = -6.0*sm1*s * t2*(-3.0+2.0*t);
    H_t[idx2(0, 3, NDEG)] = -6.0*sm12*(1.0+2.0*s) * tm1*t;
    H_s[idx2(1, 3, NDEG)] = 3.0*(1.0-3.0*s)*sm1 * t2*(-3.0+2.0*t);
    H_t[idx2(1, 3, NDEG)] = -18.0*sm12*s * tm1*t;
    H_s[idx2(2, 3, NDEG)] = -18.0*sm1*s * tm1*t2;
    H_t[idx2(2, 3, NDEG)] = -3.0*sm12*(1.0+2.0*s) * t*(-2.0+3.0*t);
    H_s[idx2(3, 3, NDEG)] = 9.0*(1.0-3.0*s)*sm1 * tm1*t2;
    H_t[idx2(3, 3, NDEG)] = -9.0*sm12*s * t*(-2.0+3.0*t);
}

// ---------------------------------------------------------------------------
// sincosperiod_moivre: compute toroidal harmonics and derivatives
// HZ[N_TOR], dHZ[N_TOR]
// ---------------------------------------------------------------------------
__device__ __forceinline__
void sincosperiod_moivre(double phi, double* __restrict__ HZ, double* __restrict__ dHZ)
{
    HZ[0]  = 1.0;
    dHZ[0] = 0.0;
    for (int i = 1; i <= NMODE; ++i) {
        double phase = double(N_PERIOD * i) * phi;
        double c = cos(phase);
        double sn = sin(phase);
        HZ [2*i - 1] = c;
        HZ [2*i]     = sn;
        dHZ[2*i - 1] = sn * (-N_PERIOD * i);
        dHZ[2*i]     = c  * ( N_PERIOD * i);
    }
}

// ---------------------------------------------------------------------------
// mode_moivre: compute toroidal harmonics (without derivatives)
// ---------------------------------------------------------------------------
__device__ __forceinline__
void mode_moivre(double phi, double* __restrict__ HZ)
{
    HZ[0] = 1.0;
    for (int i = 1; i <= NMODE; ++i) {
        double phase = double(N_PERIOD * i) * phi;
        HZ[2*i - 1] = cos(phase);
        HZ[2*i]     = sin(phase);
    }
}

// ---------------------------------------------------------------------------
// Coordinate transforms
// ---------------------------------------------------------------------------

// Cylindrical (R,Z,phi) -> Cartesian (x,y,z)
__device__ __forceinline__
void cylindrical_to_cartesian(const double* __restrict__ cyl, double* __restrict__ xyz)
{
    double cp = cos(-cyl[2]);
    double sp = sin(-cyl[2]);
    xyz[0] = cyl[0] * cp;
    xyz[1] = cyl[0] * sp;
    xyz[2] = cyl[1];
}

// Cartesian (x,y,z) -> Cylindrical (R,Z,phi)
__device__ __forceinline__
void cartesian_to_cylindrical(const double* __restrict__ xyz, double* __restrict__ cyl)
{
    cyl[0] = sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1]);
    cyl[1] = xyz[2];
    cyl[2] = atan2(-xyz[1], xyz[0]);
}

// Vector in Cartesian -> Cylindrical  (eR, eZ, ephi)
__device__ __forceinline__
void vector_cartesian_to_cylindrical(double phi, const double* __restrict__ a, double* __restrict__ b)
{
    double sp = sin(phi), cp = cos(phi);
    b[0] =  a[0]*cp - a[1]*sp;
    b[1] =  a[2];
    b[2] = -(a[0]*sp + a[1]*cp);
}

// Vector in Cylindrical -> Cartesian
__device__ __forceinline__
void vector_cylindrical_to_cartesian(double phi, const double* __restrict__ a, double* __restrict__ b)
{
    double sp = sin(phi), cp = cos(phi);
    b[0] =  a[0]*cp - a[2]*sp;
    b[1] = -(a[0]*sp + a[2]*cp);
    b[2] =  a[1];
}

// ---------------------------------------------------------------------------
// Cayley transform rotation: rotates momentum `pm` in-place using magnetic
// field `B_cart` and time-step `scaling`. Implemented as a __device__ helper
// so the volume_preserving_push kernel code stays concise.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void cayley_transform_rotate(double pm[3], const double B_cart[3], double scaling)
{
    double alpha = SPEED_OF_LIGHT * scaling /
                   sqrt(1.0 + pm[0]*pm[0] + pm[1]*pm[1] + pm[2]*pm[2]);

    const double vx = B_cart[0];
    const double vy = B_cart[1];
    const double vz = B_cart[2];

    double A[3][3];
    double B[3][3];
    double cayley_mat[3][3];

    /* ---- compute B = I + alpha * skew(vec) ---- */
    B[0][0] = 1.0;        B[0][1] =  alpha*vz;   B[0][2] = -alpha*vy;
    B[1][0] = -alpha*vz;  B[1][1] = 1.0;         B[1][2] =  alpha*vx;
    B[2][0] =  alpha*vy;  B[2][1] = -alpha*vx;   B[2][2] = 1.0;

    /* ---- compute A = (I - alpha * skew(vec))^{-1} numerator ---- */
    A[0][0] = 1.0 + alpha*alpha*vx*vx;
    A[1][0] = alpha*(alpha*vx*vy - vz);
    A[2][0] = alpha*(alpha*vz*vx + vy);

    A[0][1] = alpha*(alpha*vy*vx + vz);
    A[1][1] = 1.0 + alpha*alpha*vy*vy;
    A[2][1] = alpha*(alpha*vz*vy - vx);

    A[0][2] = alpha*(alpha*vz*vx - vy);
    A[1][2] = alpha*(alpha*vy*vz + vx);
    A[2][2] = 1.0 + alpha*alpha*vz*vz;

    /* ---- matrix multiply cayley_mat = A * B ---- */
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            cayley_mat[i][j] =
                A[i][0]*B[0][j] +
                A[i][1]*B[1][j] +
                A[i][2]*B[2][j];
        }

    /* ---- normalization factor ---- */
    double denom = 1.0 + alpha*alpha*(vx*vx + vy*vy + vz*vz);

    for (int i=0;i<3;i++)
        for (int j=0;j<3;j++)
            cayley_mat[i][j] /= denom;

    /* ---- rotate momentum vector ---- */
    double p0 = pm[0];
    double p1 = pm[1];
    double p2 = pm[2];

    pm[0] = cayley_mat[0][0]*p0 + cayley_mat[0][1]*p1 + cayley_mat[0][2]*p2;
    pm[1] = cayley_mat[1][0]*p0 + cayley_mat[1][1]*p1 + cayley_mat[1][2]*p2;
    pm[2] = cayley_mat[2][0]*p0 + cayley_mat[2][1]*p1 + cayley_mat[2][2]*p2;
}

// ---------------------------------------------------------------------------
// interp_RZP_1_gpu: interpolate R, Z and first derivatives from the SoA
// grid data at element i_elm_f (1-based!), local coordinates (s,t,phi).
// ---------------------------------------------------------------------------
__device__
void interp_RZP_1_gpu(const double* __restrict__ nl_x,
                       const int*    __restrict__ el_vertex,
                       const double* __restrict__ el_size,
                       int n_elements, int n_nodes,
                       const int* __restrict__ mode_coord,
                       int i_elm_f,     // 1 based
                       double s, double t, double phi,
                       double &R, double &R_s, double &R_t, double &R_p,
                       double &Z, double &Z_s, double &Z_t, double &Z_p,
                       int debug_j, int debug_k)
{
#if GPU_DEBUG
    const bool dbg = rz_dbg_enabled(debug_j, debug_k);
    if (dbg) {
        printf("[GPU_DEBUG j=%d k=%d] INTERP_ENTER: i_elm=%d st=[%.17e,%.17e] phi=%.17e\n",
               debug_j, debug_k, i_elm_f, s, t, phi);
    }
#endif

    double G[NV * NDEG], G_s[NV * NDEG], G_t[NV * NDEG];
    basisfunctions_2D_1(s, t, G, G_s, G_t);

    // Toroidal coordinate harmonics (for N_COORD_TOR == 1 this is trivial)
    double HZ_coord[N_COORD_TOR], HZ_coord_p[N_COORD_TOR];
    HZ_coord[0]   = 1.0;
    HZ_coord_p[0] = 0.0;
    for (int it = 1; it <= (N_COORD_TOR - 1) / 2; ++it) {
        int mc_cos = mode_coord[2*it - 1];
        int mc_sin = mode_coord[2*it];
        HZ_coord  [2*it - 1] =  cos(double(mc_cos) * phi);
        HZ_coord_p[2*it - 1] = -double(mc_cos) * sin(double(mc_cos) * phi);
        HZ_coord  [2*it]     = -sin(double(mc_sin) * phi);
        HZ_coord_p[2*it]     = -double(mc_sin) * cos(double(mc_sin) * phi);
    }

    R = 0.0; R_s = 0.0; R_t = 0.0; R_p = 0.0;
    Z = 0.0; Z_s = 0.0; Z_t = 0.0; Z_p = 0.0;
    int ie = i_elm_f - 1;       // Element idx, 0-based

    for (int kv = 0; kv < NV; ++kv) {
        int iv = el_vertex[idx2(ie, kv, n_elements)] - 1;       // Node number, 0-based
        for (int kf = 0; kf < NDEG; ++kf) {
            double ss = el_size[idx3(ie, kv, kf, n_elements, NV)];
            double g   = G  [idx2(kf, kv, NDEG)];
            double gs  = G_s[idx2(kf, kv, NDEG)];
            double gt  = G_t[idx2(kf, kv, NDEG)];

            for (int it = 0; it < N_COORD_TOR; ++it) {
                // nl_x layout: (n_nodes, N_COORD_TOR, NDEG, NDIM)
                double xx1 = nl_x[idx4(iv, it, kf, 0, n_nodes, N_COORD_TOR, NDEG)];
                double xx2 = nl_x[idx4(iv, it, kf, 1, n_nodes, N_COORD_TOR, NDEG)];
                double hz  = HZ_coord[it];
                double dhz = HZ_coord_p[it];

                R   += xx1 * ss * g  * hz;
                R_s += xx1 * ss * gs * hz;
                R_t += xx1 * ss * gt * hz;
                R_p += xx1 * ss * g  * dhz;

                Z   += xx2 * ss * g  * hz;
                Z_s += xx2 * ss * gs * hz;
                Z_t += xx2 * ss * gt * hz;
                Z_p += xx2 * ss * g  * dhz;
            }
        }
    }

#if GPU_DEBUG
    if (dbg) {
        printf("[GPU_DEBUG j=%d k=%d] INTERP_EXIT: i_elm=%d R=%.17e Z=%.17e R_s=%.17e R_t=%.17e Z_s=%.17e Z_t=%.17e\n",
               debug_j, debug_k, i_elm_f, R, Z, R_s, R_t, Z_s, Z_t);
    }
#endif
}

// ---------------------------------------------------------------------------
// try_interp_gpu: wrapper used by find_RZ_nearby_gpu
// ---------------------------------------------------------------------------
__device__
void try_interp_gpu(const double* __restrict__ nl_x,
                    const int*    __restrict__ el_vertex,
                    const double* __restrict__ el_size,
                    int n_elements, int n_nodes,
                    const int* __restrict__ mode_coord,
                    int i_elm_f, const double st[2], double phi,
                    double x[2],
                    double &R_s, double &R_t, double &Z_s, double &Z_t,
                    double &inv_jac,
                    int debug_j, int debug_k)
{
    double R_p, Z_p;
    interp_RZP_1_gpu(nl_x, el_vertex, el_size, n_elements, n_nodes, mode_coord,
                     i_elm_f, st[0], st[1], phi,
                     x[0], R_s, R_t, R_p, x[1], Z_s, Z_t, Z_p, debug_j, debug_k);
    double jac = R_s * Z_t - R_t * Z_s;
    if (fabs(jac) < 1.0e-8)
        inv_jac = jac>0 ? 1.0e8 : -1.0e8;
    else
        inv_jac = 1.0 / jac;

#if GPU_DEBUG
    if (rz_dbg_enabled(debug_j, debug_k)) {
        printf("[GPU_DEBUG j=%d k=%d] TRY_INTERP: i_elm=%d st=[%.17e,%.17e] x=[%.17e,%.17e] jac=%.17e inv_jac=%.17e\n",
               debug_j, debug_k, i_elm_f, st[0], st[1], x[0], x[1], jac, inv_jac);
    }
#endif
}

// ---------------------------------------------------------------------------
// neighbours_side_co_counter_gpu
// Find if two elements (elm1, elm2 which is on side1 of elm1) have the same
// orientation. Element node numbering must always be consecutive when going
// co or counter-clockwise around the element.
// ---------------------------------------------------------------------------
__device__
void neighbours_side_co_counter_gpu(const int* __restrict__ el_vertex,
                                    const int* __restrict__ el_neighbours,
                                    int n_elements,
                                    int elm1, int elm2, int side1,
                                    int &side2, bool &is_nb, bool &co,      // elm1, elm2, side1, side2 are 1-based
                                    int debug_j, int debug_k)      
{
    co    = false;
    is_nb = false;
    side2 = 0;
    int ie1 = elm1 - 1, ie2 = elm2 - 1;

    // Find the side in elm2 pointing to elm1
    // If elm2 has no neighbour -> elm1 use the last one that is 0 (i.e. the one on the axis itself)
    for (int i = 0; i < NV; ++i) {
        if (el_neighbours[idx2(ie2, i, n_elements)] == elm1) {
            side2 = i + 1;
            break;
        }
    }
    if (side2 > 0) {
        is_nb = true;
        // Determine node numbers of the sides
        // Node numbers are related to sides as node1=(side-1)%4+1, node2=side%4+1
        int n1a = el_vertex[idx2(ie1, (side1 - 1) % 4, n_elements)];
        int n1b = el_vertex[idx2(ie1,  side1      % 4, n_elements)];
        int n2a = el_vertex[idx2(ie2, (side2 - 1) % 4, n_elements)];
        int n2b = el_vertex[idx2(ie2,  side2      % 4, n_elements)];
        co = (n1a == n2b) || (n1b == n2a);
    }

#if GPU_DEBUG
    if (rz_dbg_enabled(debug_j, debug_k)) {
        printf("[GPU_DEBUG j=%d k=%d] NB_SIDE: elm1=%d elm2=%d side1=%d side2=%d nb=%d co=%d\n",
               debug_j, debug_k, elm1, elm2, side1, side2, int(is_nb), int(co));
    }
#endif
}

// ---------------------------------------------------------------------------
// coord_in_neighbour_gpu: transform (i_from, st) to neighbour element
// i_from 1-based, returns i_to (>0 found, -1 search needed, 0 lost)
// ---------------------------------------------------------------------------
__device__
void coord_in_neighbour_gpu(const int* __restrict__ el_vertex,
                            const int* __restrict__ el_neighbours,
                            int n_elements,
                            int i_from, int &i_to, double st[2],
                            int debug_j, int debug_k)        // i_from, i_to are 1-based
{
    // q for "Quadrant"
    int q_from;
    if (st[0] > st[1]) {
        q_from = (1.0 - st[0] > st[1]) ? 1 : 2;
    } else {
        q_from = (1.0 - st[0] <= st[1]) ? 3 : 4;
    }

    i_to = el_neighbours[idx2(i_from - 1, q_from - 1, n_elements)];
#if GPU_DEBUG
    if (rz_dbg_enabled(debug_j, debug_k)) {
        printf("[GPU_DEBUG j=%d k=%d] COORD_NB_IN: i_from=%d q_from=%d i_to_raw=%d st_in=[%.17e,%.17e]\n",
               debug_j, debug_k, i_from, q_from, i_to, st[0], st[1]);
    }
#endif
    if (i_to <= 0) return;

    // Check once more that they are neighbours and determine the orientation
    int q_to; bool nb, co;
    neighbours_side_co_counter_gpu(el_vertex, el_neighbours, n_elements,
                                   i_from, i_to, q_from, q_to, nb, co, debug_j, debug_k);
    if (!nb || q_to == 0) { i_to = 0; return; }

    // x is the coordinate along the boundary from vertex i_side to i_side+1
    double x;
    switch (q_from) {
        case 1: x = st[0];       break;
        case 2: x = st[1];       break;
        case 3: x = 1.0 - st[0]; break;
        default:x = 1.0 - st[1]; break;
    }
    if (co) x = 1.0 - x;        // if the vectors along the boundary are antiparallel

    switch (q_to) {
        case 1: st[0] = x;       st[1] = 0.0;       break;
        case 2: st[0] = 1.0;     st[1] = x;         break;
        case 3: st[0] = 1.0 - x; st[1] = 1.0;       break;
        default:st[0] = 0.0;     st[1] = 1.0 - x;   break;
    }

#if GPU_DEBUG
    if (rz_dbg_enabled(debug_j, debug_k)) {
        printf("[GPU_DEBUG j=%d k=%d] COORD_NB_OUT: i_from=%d i_to=%d q_to=%d co=%d st_out=[%.17e,%.17e]\n",
               debug_j, debug_k, i_from, i_to, q_to, int(co), st[0], st[1]);
    }
#endif
}

// ---------------------------------------------------------------------------
// find_RZ_single_gpu: Newton search in a single element (5 starting points)
// i_elm_f is 1-based.  ifail=0 on success, 999 on failure.
// ---------------------------------------------------------------------------
__device__
void find_RZ_single_gpu(const double* __restrict__ nl_x,
                         const int*    __restrict__ el_vertex,
                         const double* __restrict__ el_size,
                         int n_elements, int n_nodes,
                         const int*    __restrict__ mode_coord,
                         int i_elm_f,
                         double R_find, double Z_find,
                         double &R_out, double &Z_out,
                         int &ielm_out, double &s_out, double &t_out,           // both i_elm_f, ielm_out are 1-based
                         int &ifail,
                         int debug_j, int debug_k)
{
    constexpr int ntrial = 20;
    constexpr double tolx = 1.0e-8;
    constexpr double tolf = 1.0e-15;
    double phi_loc = 0.0;

#if GPU_DEBUG
    const bool dbg = rz_dbg_enabled(debug_j, debug_k);
    if (dbg) {
        printf("[GPU_DEBUG j=%d k=%d] FIND_RZ_SINGLE_ENTER: i_elm=%d target=[%.17e,%.17e]\n",
               debug_j, debug_k, i_elm_f, R_find, Z_find);
    }
#endif

    ielm_out = i_elm_f;

    double starts[5][2] = {{0.5,0.5},{0.75,0.75},{0.75,0.25},{0.25,0.75},{0.25,0.25}};

    for (int ist = 0; ist < 5; ++ist) {
        double x[2] = {starts[ist][0], starts[ist][1]};
        ifail = 999;

#if GPU_DEBUG
        if (dbg) {
            printf("[GPU_DEBUG j=%d k=%d] FIND_RZ_SINGLE_START: i_elm=%d istart=%d x0=[%.17e,%.17e]\n",
                   debug_j, debug_k, i_elm_f, ist + 1, x[0], x[1]);
        }
#endif

        for (int i = 0; i < ntrial; ++i) {
            double R_s, R_t, R_p, Z_s, Z_t, Z_p, RRg1, ZZg1;
            interp_RZP_1_gpu(nl_x, el_vertex, el_size, n_elements, n_nodes,
                             mode_coord, i_elm_f, x[0], x[1], phi_loc,
                             RRg1, R_s, R_t, R_p, ZZg1, Z_s, Z_t, Z_p, debug_j, debug_k);

            double fvec[2] = {RRg1 - R_find, ZZg1 - Z_find};
            double errf = fabs(fvec[0]) + fabs(fvec[1]);

#if GPU_DEBUG
            if (dbg) {
                printf("[GPU_DEBUG j=%d k=%d] FIND_RZ_SINGLE_ITER: i_elm=%d istart=%d it=%d x=[%.17e,%.17e] f=[%.17e,%.17e] errf=%.17e\n",
                       debug_j, debug_k, i_elm_f, ist + 1, i + 1, x[0], x[1], fvec[0], fvec[1], errf);
            }
#endif

            if (errf <= tolf) {
                s_out = x[0]; t_out = x[1];
                ielm_out = i_elm_f;
                R_out = RRg1; Z_out = ZZg1;
                ifail = 0;
#if GPU_DEBUG
                if (dbg) {
                    printf("[GPU_DEBUG j=%d k=%d] FIND_RZ_SINGLE_OK_F: i_elm=%d it=%d st=[%.17e,%.17e] RZ=[%.17e,%.17e]\n",
                           debug_j, debug_k, i_elm_f, i + 1, s_out, t_out, R_out, Z_out);
                }
#endif
                return;
            }

            double p[2] = {-fvec[0], -fvec[1]};
            double dis = Z_t * R_s - R_t * Z_s;     // determinant of the jacobian
            if (dis == 0.0) {
#if GPU_DEBUG
                if (dbg) {
                    printf("[GPU_DEBUG j=%d k=%d] FIND_RZ_SINGLE_SINGULAR: i_elm=%d istart=%d it=%d\n",
                           debug_j, debug_k, i_elm_f, ist + 1, i + 1);
                }
#endif
                break;        // Jacobian is singular, cannot continue
            }
            double tmp = p[0];
            p[0] = ( Z_t * p[0] - R_t * p[1]) / dis;
            p[1] = ( R_s * p[1] - Z_s * tmp) / dis;

            double errx = fabs(p[0]) + fabs(p[1]);
            p[0] = fmin(p[0],  0.25); p[0] = fmax(p[0], -0.25);
            p[1] = fmin(p[1],  0.25); p[1] = fmax(p[1], -0.25);

            x[0] += p[0]; x[1] += p[1];
            x[0] = fmax(x[0], 0.0); x[0] = fmin(x[0], 1.0);
            x[1] = fmax(x[1], 0.0); x[1] = fmin(x[1], 1.0);

            if (errx <= tolx) {
                s_out = x[0]; t_out = x[1];
                ielm_out = i_elm_f;
                R_out = RRg1; Z_out = ZZg1;
                ifail = 0;
#if GPU_DEBUG
                if (dbg) {
                    printf("[GPU_DEBUG j=%d k=%d] FIND_RZ_SINGLE_OK_X: i_elm=%d it=%d st=[%.17e,%.17e] RZ=[%.17e,%.17e]\n",
                           debug_j, debug_k, i_elm_f, i + 1, s_out, t_out, R_out, Z_out);
                }
#endif
                return;
            }
        }
    }

#if GPU_DEBUG
    if (dbg) {
        printf("[GPU_DEBUG j=%d k=%d] FIND_RZ_SINGLE_FAIL: i_elm=%d ifail=%d\n",
               debug_j, debug_k, i_elm_f, ifail);
    }
#endif
}

// ---------------------------------------------------------------------------
// find_RZ_gpu: brute-force search over all elements
// ---------------------------------------------------------------------------
__device__
void find_RZ_gpu(const double* __restrict__ nl_x,
                 const int*    __restrict__ el_vertex,
                 const double* __restrict__ el_size,
                 const int*    __restrict__ el_neighbours,
                 int n_elements, int n_nodes,
                 const int*    __restrict__ mode_coord,
                 double R_find, double Z_find,
                 double &R_out, double &Z_out,
                 int &ielm_out, double &s_out, double &t_out,       // i_elm_out is 1-based
                 int &ifail,
                 int debug_j, int debug_k)
{
#if GPU_DEBUG
    const bool dbg = rz_dbg_enabled(debug_j, debug_k);
    if (dbg) {
        printf("[GPU_DEBUG j=%d k=%d] FIND_RZ_GPU_ENTER: target=[%.17e,%.17e]\n",
               debug_j, debug_k, R_find, Z_find);
    }
#endif

    ielm_out = 0;
    for (int k = 1; k <= n_elements; ++k) {
        find_RZ_single_gpu(nl_x, el_vertex, el_size, n_elements, n_nodes,
                            mode_coord, k, R_find, Z_find,
                            R_out, Z_out, ielm_out, s_out, t_out, ifail, debug_j, debug_k);
#if GPU_DEBUG
        if (dbg) {
            printf("[GPU_DEBUG j=%d k=%d] FIND_RZ_GPU_TRY: k=%d ifail=%d ielm_out=%d st=[%.17e,%.17e]\n",
                   debug_j, debug_k, k, ifail, ielm_out, s_out, t_out);
        }
#endif
        if (ifail == 0) return;
    }
    if (ielm_out == 0) ifail = 99;
    if (ifail == 999) ielm_out = 0;

#if GPU_DEBUG
    if (dbg) {
        printf("[GPU_DEBUG j=%d k=%d] FIND_RZ_GPU_EXIT: ifail=%d ielm_out=%d RZ=[%.17e,%.17e] st=[%.17e,%.17e]\n",
               debug_j, debug_k, ifail, ielm_out, R_out, Z_out, s_out, t_out);
    }
#endif
}

// ---------------------------------------------------------------------------
// find_RZ_nearby_gpu: Newton iteration with neighbour-hopping
// All element indices are 1-based (matching Fortran convention).
// On exit: i_elm_new <=0 means particle lost.
// ---------------------------------------------------------------------------
__device__
void find_RZ_nearby_gpu(const double* __restrict__ nl_x,
                        const int*    __restrict__ el_vertex,
                        const double* __restrict__ el_size,
                        const int*    __restrict__ el_neighbours,
                        int n_elements, int n_nodes,
                        const int*    __restrict__ mode_coord,
                        double R_old, double Z_old,
                        double s_old, double t_old, int i_elm_old,
                        double R_new, double Z_new,
                        double &s_new, double &t_new, int &i_elm_new,
                        int &ifail,
                        int debug_j, int debug_k)     // both i_elm_old, i_elm_new are 1-based
{
#if GPU_DEBUG
    const bool dbg = rz_dbg_enabled(debug_j, debug_k);
    if (dbg) {
        printf("[GPU_DEBUG j=%d k=%d] FRZN_ENTER: old=[%.17e,%.17e] new=[%.17e,%.17e] st_old=[%.17e,%.17e] i_elm_old=%d\n",
               debug_j, debug_k, R_old, Z_old, R_new, Z_new, s_old, t_old, i_elm_old);
    }
#endif

    // Accuracy defaults
    // Note: tolerances are squared!, units of element size
    constexpr double element_tolerance = 1.0e-24;   // tolerance for finding a position inside an element
    constexpr int    newton_iter_max   = 200;       // Number of iterations to try

    // Check if element is valied
    if (i_elm_old < 1 || i_elm_old > n_elements) {
        i_elm_new = 0;
#if GPU_DEBUG
        if (dbg) {
            printf("[GPU_DEBUG j=%d k=%d] FRZN_INVALID_ELM: i_elm_old=%d n_elements=%d\n",
                   debug_j, debug_k, i_elm_old, n_elements);
        }
#endif
        return;
    }
    double p_loc = 0.0;  // phi coordinate is not used in the interpolation, so set to arbitrary value

    double x_step[2] = {R_old, Z_old};
    i_elm_new = i_elm_old;
    double st[2] = {s_old, t_old};
    double x_target[2] = {R_new, Z_new};

    // Find the jacobian at the current s and t position
    double R_s, R_t, Z_s, Z_t, inv_jac;
    try_interp_gpu(nl_x, el_vertex, el_size, n_elements, n_nodes, mode_coord,
                   i_elm_new, st, p_loc, x_step, R_s, R_t, Z_s, Z_t, inv_jac, debug_j, debug_k);

    double dx0 = x_step[0] - x_target[0];
    double dx1 = x_step[1] - x_target[1];
    double err2 = dx0*dx0 + dx1*dx1;
    ifail = 0;
    int iter;

#if GPU_DEBUG
    if (dbg) printf("[GPU_DEBUG j=%d k=%d] FRZN_INIT: i_elm=%d x_step=[%.17e,%.17e] st=[%.17e,%.17e] inv_jac=%.17e err2=%.17e\n",
               debug_j, debug_k, i_elm_new, x_step[0], x_step[1], st[0], st[1], inv_jac, err2);
#endif

    // Newton iteration to find s and t in or out of this element
    for (iter = 1; iter <= newton_iter_max; ++iter) {
        // Compute trial newton step
        double st_step0 = ( Z_t * (x_target[0] - x_step[0]) - R_t * (x_target[1] - x_step[1])) * inv_jac;
        double st_step1 = (-Z_s * (x_target[0] - x_step[0]) + R_s * (x_target[1] - x_step[1])) * inv_jac;

        // Limit this tep if it goes outside of the element
        double dist0 = (st_step0 > 0.0) ? (1.0 - st[0]) : st[0];
        double dist1 = (st_step1 > 0.0) ? (1.0 - st[1]) : st[1];
        double fact0  = fabs(st_step0) / fmax(dist0, 1.0e-30);
        double fact1  = fabs(st_step1) / fmax(dist1, 1.0e-30);
        double fact   = fmax(fact0, fact1);

    #if GPU_DEBUG
        if (dbg) {
            printf("[GPU_DEBUG j=%d k=%d] FRZN_ITER: it=%d i_elm=%d st=[%.17e,%.17e] st_step=[%.17e,%.17e] fact=%.17e err2=%.17e\n",
               debug_j, debug_k, iter, i_elm_new, st[0], st[1], st_step0, st_step1, fact, err2);
            printf("[GPU_DEBUG j=%d k=%d] FRZN_ITER: it=%d i_elm=%d Z_t=%.17e R_t=%.17e Z_s=%.17e R_s=%.17e inv_jac=%.17e\n",
               debug_j, debug_k, iter, i_elm_new, Z_t, R_t, Z_s, R_s, inv_jac);
            printf("[GPU_DEBUG j=%d k=%d] FRZN_ITER: it=%d i_elm=%d x_target=[%.17e,%.17e] x_step=[%.17e,%.17e]\n",
               debug_j, debug_k, iter, i_elm_new, x_target[0], x_target[1], x_step[0], x_step[1]);
        }
    #endif

        // if fact >= 1, we are on the boundary
        // That is it is the overshoot: i.e. how many times we overshoot boudnary with one st_step
        if (fact >= 1.0 - 1.0e-12) {

            st[0] += st_step0 / fact;
            st[1] += st_step1 / fact;
            int i_elm_tmp = i_elm_new;
            coord_in_neighbour_gpu(el_vertex, el_neighbours, n_elements,
                                   i_elm_tmp, i_elm_new, st, debug_j, debug_k);
#if GPU_DEBUG
            if (dbg) printf("[GPU_DEBUG j=%d k=%d] FRZN_HOP: from=%d to=%d st_on_edge=[%.17e,%.17e]\n",
                       debug_j, debug_k, i_elm_tmp, i_elm_new, st[0], st[1]);
#endif
            if (i_elm_new < 0) {
#if GPU_DEBUG
                printf("[GPU_DEBUG] find_RZ_nearby: coord_in_neighbour returned i_elm_new<0, fallback. iter=%d i_elm_tmp=%d R_new=%.17e Z_new=%.17e st=[%.17e,%.17e]\n",
                       iter, i_elm_tmp, R_new, Z_new, st[0], st[1]);
#endif
                double R_out, Z_out;
                find_RZ_gpu(nl_x, el_vertex, el_size, el_neighbours,
                            n_elements, n_nodes, mode_coord,
                            R_new, Z_new, R_out, Z_out,
                            i_elm_new, s_new, t_new, ifail, debug_j, debug_k);
                return;
            }
            if (i_elm_new == 0) {       // No element on that side, particle is lost
                i_elm_new = -i_elm_tmp; // Save position of particle
#if GPU_DEBUG
                printf("[GPU_DEBUG] find_RZ_nearby: PARTICLE LOST (i_elm_new=0, ifail=-1): iter=%d i_elm_old=%d i_elm_tmp=%d R_old=%.17e Z_old=%.17e R_new=%.17e Z_new=%.17e st=[%.17e,%.17e] err2=%.17e\n",
                       iter, i_elm_old, i_elm_tmp, R_old, Z_old, R_new, Z_new, st[0], st[1], err2);
#endif
                // Compute new R and Z in x_tmp
                double x_tmp[2]; double dummy;
                try_interp_gpu(nl_x, el_vertex, el_size, n_elements, n_nodes, mode_coord,
                               i_elm_tmp, st, p_loc, x_tmp, R_s, R_t, Z_s, Z_t, dummy, debug_j, debug_k);
                s_new = st[0]; t_new = st[1];
                ifail = -1;
                return;
            }
            try_interp_gpu(nl_x, el_vertex, el_size, n_elements, n_nodes, mode_coord,
                           i_elm_new, st, p_loc, x_step, R_s, R_t, Z_s, Z_t, inv_jac, debug_j, debug_k);
        } else {
            st[0] += st_step0;
            st[1] += st_step1;
            try_interp_gpu(nl_x, el_vertex, el_size, n_elements, n_nodes, mode_coord,
                           i_elm_new, st, p_loc, x_step, R_s, R_t, Z_s, Z_t, inv_jac, debug_j, debug_k);
        }

        dx0 = x_step[0] - x_target[0];
        dx1 = x_step[1] - x_target[1];
        err2 = dx0*dx0 + dx1*dx1;
        s_new = st[0];
        t_new = st[1];
        if (err2 < element_tolerance) {
#if GPU_DEBUG
            if (dbg) printf("[GPU_DEBUG j=%d k=%d] FRZN_CONVERGED: it=%d i_elm_new=%d st=[%.17e,%.17e] err2=%.17e\n",
                       debug_j, debug_k, iter, i_elm_new, s_new, t_new, err2);
#endif
            return;
        }
    }

    if (isnan(err2)) {
#if GPU_DEBUG
        printf("[GPU_DEBUG] find_RZ_nearby: NaN in err2, setting i_elm=-2. iter=%d i_elm_new=%d R_new=%.17e Z_new=%.17e\n",
               iter, i_elm_new, R_new, Z_new);
#endif
        i_elm_new = -2;
        return;
    }
    if (iter > newton_iter_max) {
#if GPU_DEBUG
        printf("[GPU_DEBUG] find_RZ_nearby: exceeded newton_iter_max, fallback. i_elm_new=%d err2=%.17e R_new=%.17e Z_new=%.17e\n",
               i_elm_new, err2, R_new, Z_new);
#endif
        double R_out, Z_out;
        find_RZ_gpu(nl_x, el_vertex, el_size, el_neighbours,
                    n_elements, n_nodes, mode_coord,
                    R_new, Z_new, R_out, Z_out,
                    i_elm_new, s_new, t_new, ifail, debug_j, debug_k);
    }
}

// ---------------------------------------------------------------------------
// calc_EBpsiU: compute E, B, psi, U at a point using linear time interpolation
// ---------------------------------------------------------------------------
__device__
void calc_EBpsiU(const double* __restrict__ nl_values,
                 const double* __restrict__ nl_deltas,
                 const double* __restrict__ nl_x,
                 const int*    __restrict__ el_vertex,
                 const double* __restrict__ el_size,
                 int n_elements, int n_nodes,
                 double time_now, double time_prev,
                 int flag_static, int flag_zero_dpsidt,
                 double F0, double t_norm,
                 int i_elm_f, const double st[2], double phi,       // i_elm_f is 1-based
                 double time,
                 double E[3], double B[3], double &psi, double &U,
                 int debug_j, int debug_k)
{
    double HT[NDEG * NV], HT_s[NDEG * NV], HT_t[NDEG * NV];
    basisfunctions_2D_1(st[0], st[1], HT, HT_s, HT_t);

    double HZ[N_TOR], dHZ[N_TOR];
    sincosperiod_moivre(phi, HZ, dHZ);

    int ie = i_elm_f - 1;
#if GPU_DEBUG
    const bool dbg = rz_dbg_enabled(debug_j, debug_k);
    if(dbg) printf("[GPU_DEBUG j=%d k=%d] calc_EBpsiU START: i_elm=%d st=[%.17e,%.17e] phi=%.17e time=%.17e fields%%time_now=%.17e fields%%time_prev=%.17e F0=%.17e t_norm=%.17e\n",
           debug_j, debug_k, i_elm_f, st[0], st[1], phi, time, time_now, time_prev, F0, t_norm);
#endif

    double P[2] = {0.0, 0.0};
    double P_s[2] = {0.0, 0.0};
    double P_t[2] = {0.0, 0.0};
    double P_phi[2] = {0.0, 0.0};
    double P_time[2] = {0.0, 0.0};

    // First interpolation of values
    double R = 0.0, R_s = 0.0, R_t = 0.0;
    double Zc = 0.0, Z_s = 0.0, Z_t = 0.0;
    for (int kv = 0; kv < NV; ++kv) {
        int iv = el_vertex[idx2(ie, kv, n_elements)] - 1;
        for (int kf = 0; kf < NDEG; ++kf) {
            double sz = el_size[idx3(ie, kv, kf, n_elements, NV)];
            int ifv = idx2(kf, kv, NDEG);
            double h  = HT [ifv];
            double hs = HT_s[ifv];
            double ht = HT_t[ifv];

            for (int ivar = 0; ivar < 2; ++ivar) {
                double v = 0.0, vp = 0.0;
                for (int it = 0; it < N_TOR; ++it) {
                    double val = nl_values[idx4(iv, it, kf, ivar, n_nodes, N_TOR, NDEG)] * sz;
                    v  += val * HZ[it];         // v = dot_product(values(1:n_tor,kf,1,kv),HZ(1:n_tor))
                    vp += val * dHZ[it];        // vp = dot_product(values(1:n_tor,kf,1,kv),dHZ(1:n_tor))
                }
                P[ivar]     += v  * h;
                P_s[ivar]   += v  * hs;
                P_t[ivar]   += v  * ht;
                P_phi[ivar] += vp * h;
            }

            // Reuse already-loaded h/hs/ht — avoids 6 redundant HT array reads
            double xR = nl_x[idx4(iv, 0, kf, 0, n_nodes, N_COORD_TOR, NDEG)] * sz;
            double xZ = nl_x[idx4(iv, 0, kf, 1, n_nodes, N_COORD_TOR, NDEG)] * sz;
            R   += xR * h;
            R_s += xR * hs;
            R_t += xR * ht;
            Zc  += xZ * h;
            Z_s += xZ * hs;
            Z_t += xZ * ht;
        }
    }

    // Second interpolation of differentials (deltas)
    // Pd[] declared here to limit live register range to this block only
    double Pd[2] = {0.0, 0.0};
    double Pd_s[2] = {0.0, 0.0};
    double Pd_t[2] = {0.0, 0.0};
    double Pd_phi[2] = {0.0, 0.0};

    if (t_norm > 0.0) {

#if GPU_DEBUG
        if(dbg) {
            printf("[GPU_DEBUG j=%d k=%d] calc_EBpsiU INTERP DIFFERENTIALS START: i_elm=%d st=[%.17e,%.17e] phi=%.17e\n",
                   debug_j, debug_k, i_elm_f, st[0], st[1], phi);
            printf("[GPU_DEBUG j=%d k=%d i_elm=%d] calc_EBpsiU INTERP DIFFERENTIALS START): HT=[%.17e, %.17e, %.17e, %.17e, %.17e, %.17e, %.17e, %.17e, %.17e, %.17e, %.17e, %.17e, %.17e, %.17e, %.17e, %.17e]\n", debug_j, debug_k, i_elm_f,
                HT[idx2(0, 0, NDEG)], HT[idx2(1, 0, NDEG)], HT[idx2(2, 0, NDEG)], HT[idx2(3, 0, NDEG)],
                HT[idx2(0, 1, NDEG)], HT[idx2(1, 1, NDEG)], HT[idx2(2, 1, NDEG)], HT[idx2(3, 1, NDEG)],
                HT[idx2(0, 2, NDEG)], HT[idx2(1, 2, NDEG)], HT[idx2(2, 2, NDEG)], HT[idx2(3, 2, NDEG)],
                HT[idx2(0, 3, NDEG)], HT[idx2(1, 3, NDEG)], HT[idx2(2, 3, NDEG)], HT[idx2(3, 3, NDEG)]);
            printf("[GPU_DEBUG j=%d k=%d i_elm=%d] calc_EBpsiU INTERP DIFFERENTIALS START: sizes=[%d, %d,%d, %d, %d, %d,%d, %d, %d, %d,%d, %d, %d, %d,%d, %d]\n", debug_j, debug_k, i_elm_f,
                el_size[idx3(ie, 0, 0, n_elements, NV)], el_size[idx3(ie, 0, 1, n_elements, NV)], el_size[idx3(ie, 0, 2, n_elements, NV)], el_size[idx3(ie, 0, 3, n_elements, NV)],
                el_size[idx3(ie, 1, 0, n_elements, NV)], el_size[idx3(ie, 1, 1, n_elements, NV)], el_size[idx3(ie, 1, 2, n_elements, NV)], el_size[idx3(ie, 1, 3, n_elements, NV)],
                el_size[idx3(ie, 2, 0, n_elements, NV)], el_size[idx3(ie, 2, 1, n_elements, NV)], el_size[idx3(ie, 2, 2, n_elements, NV)], el_size[idx3(ie, 2, 3, n_elements, NV)],
                el_size[idx3(ie, 3, 0, n_elements, NV)], el_size[idx3(ie, 3, 1, n_elements, NV)], el_size[idx3(ie, 3, 2, n_elements, NV)], el_size[idx3(ie, 3, 3, n_elements, NV)]);
        }
#endif

        for (int kv = 0; kv < NV; ++kv) {
            int iv = el_vertex[idx2(ie, kv, n_elements)] - 1;

            for (int kf = 0; kf < NDEG; ++kf) {
                double sz = el_size[idx3(ie, kv, kf, n_elements, NV)];
                int ifv = idx2(kf, kv, NDEG);
                double h  = HT [ifv];
                double hs = HT_s[ifv];
                double ht = HT_t[ifv];

                for (int ivar = 0; ivar < 2; ++ivar) {
                    double v = 0.0, vp = 0.0;
                    for (int it = 0; it < N_TOR; ++it) {
                        double d = nl_deltas[idx4(iv, it, kf, ivar, n_nodes, N_TOR, NDEG)] * sz;
                        v  += d * HZ[it];       // v = dot_product(deltas(1:n_tor,kf,1,kv),HZ(1:n_tor))
                        vp += d * dHZ[it];      // vp = dot_product(deltas(1:n_tor,kf,1,kv),dHZ(1:n_tor))
                    }
                    Pd[ivar]     += v  * h;
                    Pd_s[ivar]   += v  * hs;
                    Pd_t[ivar]   += v  * ht;
                    Pd_phi[ivar] += vp * h;
                }
            }
        }

        double dt;
        if (fabs(time_now - time_prev) > 1.0e-10 && !flag_static) {
            dt = 1.0 / (time_now - time_prev);
            double df = (time_now - time) * dt;
            for (int i = 0; i < 2; ++i) {
                P[i]     -= Pd[i]     * df;
                P_s[i]   -= Pd_s[i]   * df;
                P_t[i]   -= Pd_t[i]   * df;
                P_phi[i] -= Pd_phi[i] * df;
            }
        } else {
            dt = 1.0 / t_norm;
        }
        P_time[0] = Pd[0] * dt;
        P_time[1] = Pd[1] * dt;
    }

#if GPU_DEBUG
    if(dbg) {
        printf("[GPU_DEBUG j=%d k=%d] calc_EBpsiU INTERP DIFFERENTIALS END: i_elm=%d R=%.17e Z=%.17e R_s=%.17e R_t=%.17e Z_s=%.17e Z_t=%.17e\n",
               debug_j, debug_k, i_elm_f, R, Zc, R_s, R_t, Z_s, Z_t);
        printf("[GPU_DEBUG j=%d k=%d] calc_EBpsiU INTERP DIFFERENTIALS END: Pd=[%.17e,%.17e] Pd_s=[%.17e,%.17e] Pd_t=[%.17e,%.17e] Pd_phi=[%.17e,%.17e]\n",
               debug_j, debug_k, Pd[0], Pd[1], Pd_s[0], Pd_s[1], Pd_t[0], Pd_t[1], Pd_phi[0], Pd_phi[1]);
        printf("[GPU_DEBUG j=%d k=%d] calc_EBpsiU INTERP DIFFERENTIALS END: P=[%.17e,%.17e] P_s=[%.17e,%.17e] P_t=[%.17e,%.17e] P_phi=[%.17e,%.17e] P_time=[%.17e,%.17e]\n",
               debug_j, debug_k, P[0], P[1], P_s[0], P_s[1], P_t[0], P_t[1], P_phi[0], P_phi[1], P_time[0], P_time[1]);
    }
#endif

    double R_inv      = 1.0 / R;
    double st_jac_inv = 1.0 / (R_s * Z_t - R_t * Z_s);
    double t_norm_inv = 1.0 / t_norm;

    double psi_R = ( P_s[0] * Z_t - P_t[0] * Z_s) * st_jac_inv;
    double psi_Z = (-P_s[0] * R_t + P_t[0] * R_s) * st_jac_inv;
    double U_R   = ( P_s[1] * Z_t - P_t[1] * Z_s) * st_jac_inv;
    double U_Z   = (-P_s[1] * R_t + P_t[1] * R_s) * st_jac_inv;
    double U_phi = P_phi[1];

    psi = P[0];
    U   = P[1] * t_norm_inv;

    if (flag_zero_dpsidt) P_time[0] = 0.0;

    // Magnetic field (cylindrical)
    B[0] =  psi_Z * R_inv;
    B[1] = -psi_R * R_inv;
    B[2] =  F0    * R_inv;

    // Electric field (cylindrical)
    double neg_F0_tnorm_inv = -F0 * t_norm_inv;
    E[0] = neg_F0_tnorm_inv * U_R;
    E[1] = neg_F0_tnorm_inv * U_Z;
    E[2] = (neg_F0_tnorm_inv * U_phi - P_time[0]) * R_inv;

    // Projection: E = E - E * B / |B| (element-wise, matching Fortran)
    double Bnorm_inv = 1.0 / sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
    E[0] -= E[0] * B[0] * Bnorm_inv;
    E[1] -= E[1] * B[1] * Bnorm_inv;
    E[2] -= E[2] * B[2] * Bnorm_inv;

#if GPU_DEBUG
    if(dbg) printf("[GPU_DEBUG j=%d k=%d] calc_EBpsiU OUTPUT: psi=%.17e U=%.17e E=[%.17e,%.17e,%.17e] B=[%.17e,%.17e,%.17e]\n",
           debug_j, debug_k, psi, U, E[0], E[1], E[2], B[0], B[1], B[2]);
#endif
}

// ---------------------------------------------------------------------------
// volume_preserving_push: VPA integrator for a relativistic particle
// All position/element data is modified in-place.
// ---------------------------------------------------------------------------
__device__
void volume_preserving_push(double x[3], double p_mom[3], double st[2],
                            int &i_elm_f, double charge,                // i_elm_f is 1-based
                            const double* __restrict__ nl_values,
                            const double* __restrict__ nl_deltas,
                            const double* __restrict__ nl_x,
                            const int*    __restrict__ el_vertex,
                            const double* __restrict__ el_size,
                            const int*    __restrict__ el_neighbours,
                            int n_elements, int n_nodes,
                            const int*    __restrict__ mode_coord,
                            double time_now, double time_prev,
                            int flag_static, int flag_zero_dpsidt,
                            double F0, double t_norm,
                            double mass, double time, double timestep,
                            int &ifail,
                            int debug_j, int debug_k, int my_id)
{
    const bool dbg = rz_dbg_enabled(debug_j, debug_k);

    // No need to check if particle is valid, it is already done in the caller
    // Turn particle position from cylindrical to cartesian coordinates
    double cur_xyz[3];
    cylindrical_to_cartesian(x, cur_xyz);

    // --- First half-step (position advance) ---

    double scaling = 0.5 * timestep * charge * EL_CHG / (ATOMIC_MASS_UNIT * mass * SPEED_OF_LIGHT);
    double mc = mass * SPEED_OF_LIGHT;
#if GPU_DEBUG
    if (dbg) {
        printf("[GPU_DEBUG j=%d k=%d] VPA_INPUT: i_elm=%d x=[%.17e,%.17e,%.17e] p=[%.17e,%.17e,%.17e]\n",
               debug_j, debug_k, i_elm_f, x[0], x[1], x[2], p_mom[0], p_mom[1], p_mom[2]);
        printf("[GPU_DEBUG j=%d k=%d] VPA_INPUT: st=[%.17e,%.17e] charge=%.17e mass=%.17e time=%.17e tstep=%.17e\n",
               debug_j, debug_k, st[0], st[1], charge, mass, time, timestep);
        printf("[GPU_DEBUG j=%d k=%d] VPA_INPUT: time_now=%.17e time_prev=%.17e F0=%.17e t_norm=%.17e\n",
               debug_j, debug_k, time_now, time_prev, F0, t_norm);
        printf("[GPU_DEBUG j=%d k=%d] VPA_STEP1: scaling=%.17e mc=%.17e cur_xyz=[%.17e,%.17e,%.17e]\n",
               debug_j, debug_k, scaling, mc, cur_xyz[0], cur_xyz[1], cur_xyz[2]);
    }
#endif
    // Compute dimensionless momentum
    // i.e. Normalise momentum: p -> p / (mass * c)
    double pm[3] = {p_mom[0] / mc, p_mom[1] / mc, p_mom[2] / mc};

    // Compute coordinates at half-step
    double pdot = pm[0]*pm[0] + pm[1]*pm[1] + pm[2]*pm[2];
    double gamma = sqrt(1.0 + pdot);
    double dt_half_c = 0.5 * timestep * SPEED_OF_LIGHT;
    double half_xyz[3] = {
        cur_xyz[0] + dt_half_c * pm[0] / gamma,
        cur_xyz[1] + dt_half_c * pm[1] / gamma,
        cur_xyz[2] + dt_half_c * pm[2] / gamma
    };

    // Compute cylindrical coordinates from cartesian ones
    double half_cyl[3];
    cartesian_to_cylindrical(half_xyz, half_cyl);
#if GPU_DEBUG
    if (dbg) {
        printf("[GPU_DEBUG j=%d k=%d] VPA_STEP1: pm_normed=[%.17e,%.17e,%.17e] gamma=%.17e\n",
               debug_j, debug_k, pm[0], pm[1], pm[2], gamma);
        printf("[GPU_DEBUG j=%d k=%d] VPA_STEP1: half_xyz=[%.17e,%.17e,%.17e] half_cyl=[%.17e,%.17e,%.17e]\n",
               debug_j, debug_k, half_xyz[0], half_xyz[1], half_xyz[2], half_cyl[0], half_cyl[1], half_cyl[2]);
    }
#endif

    // Find element for half-step position  (that is (i_elm, s, t) coordinates)
    double s_new, t_new; int i_elm_new;
    find_RZ_nearby_gpu(nl_x, el_vertex, el_size, el_neighbours,
                       n_elements, n_nodes, mode_coord,
                       x[0], x[1], st[0], st[1], i_elm_f,
                       half_cyl[0], half_cyl[1],
                       s_new, t_new, i_elm_new, ifail, debug_j, debug_k);

#if GPU_DEBUG
    if (dbg || ifail != 0)
        printf("[GPU_DEBUG j=%d k=%d] VPA_FIND1: s_new=%.17e t_new=%.17e i_elm_new=%d ifail=%d\n",
               debug_j, debug_k, s_new, t_new, i_elm_new, ifail);
#endif

    // If the particle is lost, exit
    if (i_elm_new <= 0) { i_elm_f = i_elm_new; return; }
    // Otherwise, copy new coordinates and element index to the particle data
    x[0] = half_cyl[0]; x[1] = half_cyl[1]; x[2] = half_cyl[2];
    st[0] = s_new; st[1] = t_new;
    i_elm_f = i_elm_new;

    // --- Compute E, B at half-step ---
    double E[3], B_field[3], psi_loc, U_loc;
    calc_EBpsiU(nl_values, nl_deltas, nl_x, el_vertex, el_size,
                n_elements, n_nodes,
                time_now, time_prev, flag_static, flag_zero_dpsidt,
                F0, t_norm,
                i_elm_f, st, x[2], time + 0.5 * timestep,
                E, B_field, psi_loc, U_loc,
                debug_j, debug_k);
#if GPU_DEBUG
    if (dbg) {
        printf("[GPU_DEBUG j=%d k=%d] VPA_STEP2: E_cyl=[%.17e,%.17e,%.17e] B_cyl=[%.17e,%.17e,%.17e]\n",
               debug_j, debug_k, E[0], E[1], E[2], B_field[0], B_field[1], B_field[2]);
        printf("[GPU_DEBUG j=%d k=%d] VPA_STEP2: psi=%.17e U=%.17e\n",
               debug_j, debug_k, psi_loc, U_loc);
    }
#endif

    // --- Second half-step ---

    // Convert E, B to Cartesian for VPA rotation
    double E_cart[3], B_cart[3];
    vector_cylindrical_to_cartesian(x[2], E, E_cart);
    vector_cylindrical_to_cartesian(x[2], B_field, B_cart);
#if GPU_DEBUG
    if (dbg) printf("[GPU_DEBUG j=%d k=%d] VPA_STEP2: E_cart=[%.17e,%.17e,%.17e] B_cart=[%.17e,%.17e,%.17e]\n",
               debug_j, debug_k, E_cart[0], E_cart[1], E_cart[2], B_cart[0], B_cart[1], B_cart[2]);
#endif

    // --- Momentum update: E-kick + Cayley rotation + E-kick ---
    // First electric kick
    pm[0] += scaling * E_cart[0];
    pm[1] += scaling * E_cart[1];
    pm[2] += scaling * E_cart[2];
#if GPU_DEBUG
    if (dbg) printf("[GPU_DEBUG j=%d k=%d] VPA_STEP2: pm_afterE1=[%.17e,%.17e,%.17e]\n",
               debug_j, debug_k, pm[0], pm[1], pm[2]);
#endif

    // Cayley transform rotation
    cayley_transform_rotate(pm, B_cart, scaling);
#if GPU_DEBUG
    if (dbg) printf("[GPU_DEBUG j=%d k=%d] VPA_STEP2: pm_afterCayley=[%.17e,%.17e,%.17e]\n",
               debug_j, debug_k, pm[0], pm[1], pm[2]);
#endif

    // Second electric kick
    pm[0] += scaling * E_cart[0];
    pm[1] += scaling * E_cart[1];
    pm[2] += scaling * E_cart[2];
#if GPU_DEBUG
    if (dbg) printf("[GPU_DEBUG j=%d k=%d] VPA_STEP2: pm_afterE2=[%.17e,%.17e,%.17e]\n",
               debug_j, debug_k, pm[0], pm[1], pm[2]);
#endif

    // --- Second half position update ---
    pdot = pm[0]*pm[0] + pm[1]*pm[1] + pm[2]*pm[2];
    gamma = sqrt(1.0 + pdot);
    half_xyz[0] += dt_half_c * pm[0] / gamma;
    half_xyz[1] += dt_half_c * pm[1] / gamma;
    half_xyz[2] += dt_half_c * pm[2] / gamma;

    // Restore dimensional momentum
    p_mom[0] = pm[0] * mc;
    p_mom[1] = pm[1] * mc;
    p_mom[2] = pm[2] * mc;

    // Turn back from cartesian to cylindrical coordinates
    cartesian_to_cylindrical(half_xyz, half_cyl);
#if GPU_DEBUG
    if (dbg) printf("[GPU_DEBUG j=%d k=%d] VPA_STEP2: final_xyz=[%.17e,%.17e,%.17e] final_cyl=[%.17e,%.17e,%.17e]\n",
               debug_j, debug_k, half_xyz[0], half_xyz[1], half_xyz[2], half_cyl[0], half_cyl[1], half_cyl[2]);
#endif

    // Find new (i_elm, s, t) coordinates
    find_RZ_nearby_gpu(nl_x, el_vertex, el_size, el_neighbours,
                       n_elements, n_nodes, mode_coord,
                       x[0], x[1], st[0], st[1], i_elm_f,
                       half_cyl[0], half_cyl[1],
                       s_new, t_new, i_elm_new, ifail, debug_j, debug_k);
#if GPU_DEBUG
    if (dbg || ifail != 0) {
        printf("[GPU_DEBUG j=%d k=%d] VPA_FIND2: s_new=%.17e t_new=%.17e i_elm_new=%d ifail=%d\n",
               debug_j, debug_k, s_new, t_new, i_elm_new, ifail);
        printf("[GPU_DEBUG j=%d k=%d] VPA_FIND2: p_mom=[%.17e,%.17e,%.17e]\n",
               debug_j, debug_k, p_mom[0], p_mom[1], p_mom[2]);
    }
#endif

    // Copy new R-Z-Phi position into particle                       
    x[0] = half_cyl[0]; x[1] = half_cyl[1]; x[2] = half_cyl[2];
    st[0] = s_new; st[1] = t_new;
    i_elm_f = i_elm_new;
}


// ===========================================================================================
//                                  MAIN HIP KERNEL
// ===========================================================================================

// ---------------------------------------------------------------------------
// evolve_REs_kernel: each thread evolves one particle through all time steps.
// feedback_rhs accumulation uses atomicAdd.
//
// Particle arrays layout (Fortran column-major):
//   p_x[j + num_particles*dim], p_p[j + num_particles*dim], p_st[j + num_particles*dim]  (0-based j, dim)
//   p_i_elm[j], p_weight[j], p_q[j]
//
// feedback_rhs layout (column-major, 0-based):
//   (n_elements, NDEG, NV, N_TOR, NVAR)  -- n_elements first for GPU coalescing
//   feedback_rhs[ie + n_elements*(n + NDEG*(m + NV*(it + N_TOR*var)))]
// ---------------------------------------------------------------------------
__global__
void evolve_REs_kernel(
    // Particle SoA
    double* __restrict__ p_x,            // (num_particles, 3)
    double* __restrict__ p_p,            // (num_particles, 3)
    double* __restrict__ p_st,           // (num_particles, 2)
    int*    __restrict__ p_i_elm,        // (num_particles)
    const double* __restrict__ p_weight, // (num_particles)
    double charge,                       // group charge number (uniform per group)
    // Field node list SoA
    const double* __restrict__ nl_values, // (n_nodes, N_TOR, NDEG, NVAR)
    const double* __restrict__ nl_deltas,
    const double* __restrict__ nl_x,      // (n_nodes, N_COORD_TOR, NDEG, NDIM)
    int n_nodes,
    // Field element list SoA
    const int*    __restrict__ el_vertex,     // (n_elements, NV)
    const int*    __restrict__ el_neighbours, // (n_elements, NV)
    const double* __restrict__ el_size,       // (n_elements, NV, NDEG)
    int n_elements,
    // Field time parameters
    double time_now, double time_prev,
    int flag_static, int flag_zero_dpsidt,
    // Physics parameters
    double F0, double t_norm,
    // Simulation parameters
    double sim_time, double group_mass, double tstep_part_adj, 
    int nstep_particles, int num_particles,
    // Feedback RHS (atomically updated)
    double* __restrict__ feedback_rhs, // column-major (n_elements, NDEG, NV, N_TOR, NVAR) -- n_elements first for GPU coalescing
    // mode_coord for interp_RZP_1_gpu
    const int* __restrict__ mode_coord,
    int my_id)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_particles) return;

    // Load particle data into registers
    double x[3]  = {p_x[idx2(j, 0, num_particles)], p_x[idx2(j, 1, num_particles)], p_x[idx2(j, 2, num_particles)]};
    double pm[3] = {p_p[idx2(j, 0, num_particles)], p_p[idx2(j, 1, num_particles)], p_p[idx2(j, 2, num_particles)]};
    double st[2] = {p_st[idx2(j, 0, num_particles)], p_st[idx2(j, 1, num_particles)]};
    int    i_elm = p_i_elm[j];
    double w = p_weight[j];

#if GPU_DEBUG
    if (j < 3) {
        printf("[GPU_DEBUG j=%d] START: my_id=%d i_elm=%d x=[%.17e,%.17e,%.17e] p=[%.17e,%.17e,%.17e] st=[%.17e,%.17e] w=%.17e q=%.17e\n",
               j, my_id, i_elm, x[0], x[1], x[2], pm[0], pm[1], pm[2], st[0], st[1], w, charge);
        printf("[GPU_DEBUG j=%d] PARAMS: sim_time=%.17e tstep=%.17e nstep=%d group_mass=%.17e\n",
               j, sim_time, tstep_part_adj, nstep_particles, group_mass);
        printf("[GPU_DEBUG j=%d] PARAMS: time_now=%.17e time_prev=%.17e flag_static=%d, flag_zero_dpsidt=%d F0=%.17e t_norm=%.17e\n",
               j, time_now, time_prev, flag_static, flag_zero_dpsidt, F0, t_norm);
        printf("[GPU_DEBUG j=%d] PARAMS: flag_static=%d flag_zero_dpsidt=%d n_el=%d n_nodes=%d NVAR=%d\n",
               j, flag_static, flag_zero_dpsidt, n_elements, n_nodes, NVAR);
    }
#endif

    for (int k = 0; k < nstep_particles; ++k) {

        if (i_elm <= 0) break;

        // ===========================================================
        // 1. Projection: compute feedback_rhs contribution
        // ===========================================================

        // Basis functions for projection (non-transposed)
        double HH[NV * NDEG];
        basisfunctions_2D_0(st[0], st[1], HH);

        // Toroidal harmonics
        double HZ_proj[N_TOR];
        mode_moivre(x[2], HZ_proj);

        // Cylindrical momentum and velocity
        double cyl_mom[3];
        vector_cartesian_to_cylindrical(x[2], pm, cyl_mom);
        double pdot_cyl = cyl_mom[0]*cyl_mom[0] + cyl_mom[1]*cyl_mom[1] + cyl_mom[2]*cyl_mom[2];
        double denom_v = sqrt(pdot_cyl / (SPEED_OF_LIGHT*SPEED_OF_LIGHT) + group_mass*group_mass);
        double cyl_vel[3] = {cyl_mom[0] / denom_v, cyl_mom[1] / denom_v, cyl_mom[2] / denom_v};

        // Compute E, B at current position
        double E_loc[3], B_loc[3], psi_loc, U_loc;
        calc_EBpsiU(nl_values, nl_deltas, nl_x, el_vertex, el_size,
                    n_elements, n_nodes,
                    time_now, time_prev, flag_static, flag_zero_dpsidt,
                    F0, t_norm,
                    i_elm, st, x[2], sim_time,
                    E_loc, B_loc, psi_loc, U_loc,
                    j, k); 

        double Bnorm = sqrt(B_loc[0]*B_loc[0] + B_loc[1]*B_loc[1] + B_loc[2]*B_loc[2]);
        double B_hat[3] = {B_loc[0]/Bnorm, B_loc[1]/Bnorm, B_loc[2]/Bnorm};

        double v_par = cyl_vel[0]*B_hat[0] + cyl_vel[1]*B_hat[1] + cyl_vel[2]*B_hat[2];
        double v_perp_diff[3] = {cyl_vel[0] - v_par*B_hat[0],
                                 cyl_vel[1] - v_par*B_hat[1],
                                 cyl_vel[2] - v_par*B_hat[2]};
        double v_perp = sqrt(v_perp_diff[0]*v_perp_diff[0] + v_perp_diff[1]*v_perp_diff[1] + v_perp_diff[2]*v_perp_diff[2]);

        double gamma_m = sqrt(MASS_ELECTRON*MASS_ELECTRON
                            + pdot_cyl * ATOMIC_MASS_UNIT*ATOMIC_MASS_UNIT
                              / (SPEED_OF_LIGHT*SPEED_OF_LIGHT));

        double v_Ppar  = gamma_m * v_par * v_par * MU_ZERO;
        double v_Pperp = gamma_m * v_perp * v_perp * 0.5 * MU_ZERO;
        double v_jPhi  = -double(charge) * EL_CHG * cyl_vel[2] * x[0] * MU_ZERO;

        // Accumulate to feedback_rhs with atomicAdd.
        // Layout (column-major): (n_elements, NDEG, NV, N_TOR, NVAR)
        //   index = ie + n_elements*(n + NDEG*(m + NV*(it + N_TOR*var)))  (all 0-based)
        // n_elements is first so warp threads (differing in ie) access adjacent addresses.
        int ie = i_elm - 1;
#if GPU_DEBUG
        if (rz_dbg_enabled(j, k)) {
            printf("[GPU_DEBUG j=%d k=%d] PROJ: cyl_mom=[%.17e,%.17e,%.17e] cyl_vel=[%.17e,%.17e,%.17e]\n",
                   j, k, cyl_mom[0], cyl_mom[1], cyl_mom[2], cyl_vel[0], cyl_vel[1], cyl_vel[2]);
            printf("[GPU_DEBUG j=%d k=%d] PROJ: E=[%.17e,%.17e,%.17e] B=[%.17e,%.17e,%.17e]\n",
                   j, k, E_loc[0], E_loc[1], E_loc[2], B_loc[0], B_loc[1], B_loc[2]);
            printf("[GPU_DEBUG j=%d k=%d] PROJ: psi=%.17e U=%.17e Bnorm=%.17e\n",
                   j, k, psi_loc, U_loc, Bnorm);
            printf("[GPU_DEBUG j=%d k=%d] PROJ: v_par=%.17e v_perp=%.17e gamma_m=%.17e\n",
                   j, k, v_par, v_perp, gamma_m);
            printf("[GPU_DEBUG j=%d k=%d] PROJ: v_Ppar=%.17e v_Pperp=%.17e v_jPhi=%.17e\n",
                   j, k, v_Ppar, v_Pperp, v_jPhi);
        }
#endif
        for (int n = 0; n < NDEG; ++n) {
            for (int m = 0; m < NV; ++m) {
                double proj_factor = HH[idx2(n, m, NDEG)]
                                   * el_size[idx3(ie, m, n, n_elements, NV)]
                                   * w;
            
#if GPU_DEBUG
                if (rz_dbg_enabled(j, k)) {
                    printf("[GPU_DEBUG j=%d k=%d i_elm=%d] PROJ_FACTOR: deg=%d vert=%d HH=%.17e el_size=%.17e w=%.17e proj_factor=%.17e\n",
                           j, k, ie+1, n+1, m+1, HH[idx2(n, m, NDEG)], el_size[idx3(ie, m, n, n_elements, NV)], w, proj_factor);
                }
#endif

                for (int it = 0; it < N_TOR; ++it) {
                    double hz = HZ_proj[it];

#if GPU_DEBUG
                if (rz_dbg_enabled(j, k)) {
                    printf("[MYDEBUG j=%d k=%d i_elm=%d] FEEDBACK UPDATE: deg=%d vert=%d itor=%d, var_idx(incr)=%d -> add_value=%.17e\n",
                           j, k, ie+1, n+1, m+1, it+1, P_PAR_IDX+1, hz * v_Ppar * proj_factor);
                }
#endif

                    atomicAdd(&feedback_rhs[idx5(ie, n, m, it, P_PAR_IDX, n_elements, NDEG, NV, N_TOR)],
                              hz * v_Ppar * proj_factor);
                    atomicAdd(&feedback_rhs[idx5(ie, n, m, it, P_PERP_IDX, n_elements, NDEG, NV, N_TOR)],
                              hz * v_Pperp * proj_factor);
                    atomicAdd(&feedback_rhs[idx5(ie, n, m, it, J_PHI_IDX, n_elements, NDEG, NV, N_TOR)],
                              hz * v_jPhi * proj_factor);
                }
            }
        }

        // ===========================================================
        // 2. Push particle (VPA)
        // ===========================================================
        int ifail = 0;
        volume_preserving_push(x, pm, st, i_elm, charge,
                               nl_values, nl_deltas, nl_x,
                               el_vertex, el_size, el_neighbours,
                               n_elements, n_nodes, mode_coord,
                               time_now, time_prev,
                               flag_static, flag_zero_dpsidt,
                               F0, t_norm,
                               group_mass, sim_time, tstep_part_adj,
                               ifail,
                               j, k, my_id);
#if GPU_DEBUG
        if (ifail != 0) {
            printf("[GPU_DEBUG j=%d k=%d] VPA push failed: ifail=%d i_elm=%d x=[%.17e,%.17e,%.17e]\n",
                   j, k, ifail, i_elm, x[0], x[1], x[2]);
        }
#endif

    } // end time-step loop

#if GPU_DEBUG
    if (j < 3) {
        printf("[GPU_DEBUG j=%d] END: i_elm=%d x=[%.17e,%.17e,%.17e] p=[%.17e,%.17e,%.17e] st=[%.17e,%.17e]\n",
               j, i_elm, x[0], x[1], x[2], pm[0], pm[1], pm[2], st[0], st[1]);
    }
#endif

    // Store particle data back to global memory
    p_x[idx2(j, 0, num_particles)] = x[0]; p_x[idx2(j, 1, num_particles)] = x[1]; p_x[idx2(j, 2, num_particles)] = x[2];
    p_p[idx2(j, 0, num_particles)] = pm[0]; p_p[idx2(j, 1, num_particles)] = pm[1]; p_p[idx2(j, 2, num_particles)] = pm[2];
    p_st[idx2(j, 0, num_particles)] = st[0]; p_st[idx2(j, 1, num_particles)] = st[1];
    p_i_elm[j] = i_elm;

#if GPU_DEBUG
    if(j < 3) {
        printf("[GPU_DEBUG j=%d] WRITEBACK: i_elm=%d x=[%.17e,%.17e,%.17e] p=[%.17e,%.17e,%.17e] st=[%.17e,%.17e]\n",
               j, i_elm, x[0], x[1], x[2], pm[0], pm[1], pm[2], st[0], st[1]);
    }
#endif
}


// ===========================================================================================
//                       HOST LAUNCH FUNCTION (Fortran-callable via bind(C))
// ===========================================================================================

// Called from Fortran as:  call launch_evolve_REs(sim, feedback_rhs, tstep_part_adj, nstep_part_adj)
// All fields of particle_sim are already filled on the host by Fortran.
extern "C"
void launch_evolve_REs(particle_sim sim, double* h_feedback_rhs,
                       double tstep_part_adj, int nstep_part_adj)
{
    // --- Unpack sim ---
    // Fields
    const node_list_SoA&    nl  = sim.fields.node_list;
    const element_list_SoA& el  = sim.fields.element_list;
    const int    n_nodes        = nl.n_nodes;
    const int    n_elements     = el.n_elements;
    const double time_now       = sim.fields.time_now;
    const double time_prev      = sim.fields.time_prev;
    const int    flag_static    = sim.fields.flag_static;
    const int    flag_zero_dp   = sim.fields.flag_zero_dpsidt;
    const double F0             = sim.fields.F0;
    const double t_norm         = sim.fields.t_norm;

    // Group
    const particle_group& grp   = sim.group;
    const int    num_particles  = grp.num_particles;
    const double group_mass     = grp.mass;
    const double charge         = grp.charge;

    const particle_SoA_kinetic_relativistic* part = &grp.particles;

    // Simulation
    const double sim_time       = sim.sim_time;
    const int    nstep_particles= nstep_part_adj;

    // --- Compute buffer sizes ---
    const size_t sz_x       = 3 * num_particles * sizeof(double);
    const size_t sz_p       = 3 * num_particles * sizeof(double);
    const size_t sz_st      = 2 * num_particles * sizeof(double);
    const size_t sz_i_elm   = num_particles * sizeof(int);
    const size_t sz_weight  = num_particles * sizeof(double);

    const size_t sz_nl_x      = (size_t)N_COORD_TOR * NDEG * NDIM * n_nodes * sizeof(double);
    const size_t sz_nl_values = (size_t)N_TOR  * NDEG * NVAR * n_nodes * sizeof(double);
    const size_t sz_nl_deltas = (size_t)N_TOR  * NDEG * NVAR * n_nodes * sizeof(double);

    const size_t sz_el_vertex = (size_t)n_elements * NV   * sizeof(int);
    const size_t sz_el_neigh  = (size_t)n_elements * NV   * sizeof(int);
    const size_t sz_el_size   = (size_t)n_elements * NV   * NDEG * sizeof(double);

    const size_t sz_feedback   = (size_t)NDEG * NV * n_elements * N_TOR * NVAR * sizeof(double);
    const size_t sz_mode_coord = N_COORD_TOR * sizeof(int);

    // --- Allocate device memory ---
    double *d_x, *d_p, *d_st, *d_weight;
    int    *d_i_elm;
    double *d_nl_x, *d_nl_values, *d_nl_deltas;
    int    *d_el_vertex, *d_el_neighbours;
    double *d_el_size;
    double *d_feedback_rhs;
    int    *d_mode_coord;

    
    int n_devices;
    HIP_CHECK(hipGetDeviceCount(&n_devices));
    if(sim.my_id == 0)
        printf("[launch_evolve_REs] HIP Device count: %d\n", n_devices);
    HIP_CHECK(hipSetDevice(sim.my_id % n_devices)); // Ensure we are on the correct GPU before allocating memory
    int curr_dev;
    HIP_CHECK(hipGetDevice(&curr_dev));
    printf("[launch_evolve_REs] MPI process %d (global rank) using device=%d\n", sim.my_id, curr_dev);


    HIP_CHECK(hipMalloc(&d_x,       sz_x));
    HIP_CHECK(hipMalloc(&d_p,       sz_p));
    HIP_CHECK(hipMalloc(&d_st,      sz_st));
    HIP_CHECK(hipMalloc(&d_i_elm,   sz_i_elm));
    HIP_CHECK(hipMalloc(&d_weight,  sz_weight));

    HIP_CHECK(hipMalloc(&d_nl_x,      sz_nl_x));
    HIP_CHECK(hipMalloc(&d_nl_values, sz_nl_values));
    HIP_CHECK(hipMalloc(&d_nl_deltas, sz_nl_deltas));

    HIP_CHECK(hipMalloc(&d_el_vertex,     sz_el_vertex));
    HIP_CHECK(hipMalloc(&d_el_neighbours, sz_el_neigh));
    HIP_CHECK(hipMalloc(&d_el_size,       sz_el_size));

    HIP_CHECK(hipMalloc(&d_feedback_rhs, sz_feedback));
    HIP_CHECK(hipMalloc(&d_mode_coord,   sz_mode_coord));

    // --- Copy host -> device ---
    HIP_CHECK(hipMemcpy(d_x,      part->x,      sz_x,      hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_p,      part->p,      sz_p,      hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_st,     part->st,     sz_st,     hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_i_elm,  part->i_elm,  sz_i_elm,  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight, part->weight, sz_weight, hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpy(d_nl_x,      nl.x,      sz_nl_x,      hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_nl_values, nl.values,  sz_nl_values, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_nl_deltas, nl.deltas,  sz_nl_deltas, hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpy(d_el_vertex,     el.vertex,     sz_el_vertex, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_el_neighbours, el.neighbours, sz_el_neigh,  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_el_size,       el.size,       sz_el_size,   hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpy(d_feedback_rhs, h_feedback_rhs,       sz_feedback,   hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_mode_coord,   sim.fields.mode_coord, sz_mode_coord, hipMemcpyHostToDevice));

    // --- Launch kernel ---
    constexpr int BLOCK_SIZE = 256;
    int grid_size = (num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;

#if GPU_DEBUG
    fprintf(stderr, "[GPU_DEBUG host] launch_evolve_REs: num=%d n_el=%d n_nodes=%d NVAR=%d\n",
            num_particles, n_elements, n_nodes, NVAR);
    fprintf(stderr, "[GPU_DEBUG host]   sim_time=%.17e  tstep=%.17e  nstep=%d\n",
            sim_time, tstep_part_adj, nstep_particles);
    fprintf(stderr, "[GPU_DEBUG host]   P_par=%d P_perp=%d j_phi=%d (0-based)\n",
            P_PAR_IDX, P_PERP_IDX, J_PHI_IDX);
    fprintf(stderr, "[GPU_DEBUG host]   feedback: %zu bytes  grid=%d  block=%d\n",
            sz_feedback, grid_size, BLOCK_SIZE);
    if (num_particles > 0) {
        fprintf(stderr, "[GPU_DEBUG host]   p[0]: i_elm=%d x=[%.17e,%.17e,%.17e] p=[%.17e,%.17e,%.17e] w=%.17e\n",
                part->i_elm[0], part->x[0], part->x[num_particles], part->x[2*num_particles],
                part->p[0],     part->p[num_particles], part->p[2*num_particles], part->weight[0]);
    }
#endif

    hipLaunchKernelGGL(evolve_REs_kernel,
        dim3(grid_size), dim3(BLOCK_SIZE), 0, 0,
        // Particle SoA
        d_x, d_p, d_st, d_i_elm, d_weight, charge,
        // Field node list SoA
        d_nl_values, d_nl_deltas, d_nl_x, n_nodes,
        // Field element list SoA
        d_el_vertex, d_el_neighbours, d_el_size, n_elements,
        // Field time parameters
        time_now, time_prev, flag_static, flag_zero_dp,
        // Physics parameters
        F0, t_norm,
        // Simulation parameters
        sim_time, group_mass, tstep_part_adj,
        nstep_particles, num_particles,
        // Feedback RHS
        d_feedback_rhs,
        // mode_coord
        d_mode_coord,
        sim.my_id);

    HIP_CHECK(hipGetLastError());
#if GPU_DEBUG
    // Synchronise before reading results so any kernel printf output is flushed
    // and device-side errors are caught immediately rather than at the next API call.
    // Oss: without it, there is anyway implicit synchronisation at first hipMemcpy after kernel launch
    HIP_CHECK(hipDeviceSynchronize());
    fprintf(stderr, "[GPU_DEBUG host] kernel completed, copying results back.\n");
#endif

    // --- Copy results back: device -> host ---
    HIP_CHECK(hipMemcpy(part->x,       d_x,            sz_x,        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(part->p,       d_p,            sz_p,        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(part->st,      d_st,           sz_st,       hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(part->i_elm,   d_i_elm,        sz_i_elm,    hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_feedback_rhs,d_feedback_rhs, sz_feedback, hipMemcpyDeviceToHost));

    // --- Free device memory ---
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_p));
    HIP_CHECK(hipFree(d_st));
    HIP_CHECK(hipFree(d_i_elm));
    HIP_CHECK(hipFree(d_weight));
    HIP_CHECK(hipFree(d_nl_x));
    HIP_CHECK(hipFree(d_nl_values));
    HIP_CHECK(hipFree(d_nl_deltas));
    HIP_CHECK(hipFree(d_el_vertex));
    HIP_CHECK(hipFree(d_el_neighbours));
    HIP_CHECK(hipFree(d_el_size));
    HIP_CHECK(hipFree(d_feedback_rhs));
    HIP_CHECK(hipFree(d_mode_coord));
}