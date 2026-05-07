// =============================================================================
// HIP kernel implementation for evolving relativistic runaway electrons (REs)
// Equivalent of evolve_REs_gpu (OpenMP target) in mod_particle_evolution.f90
//
// All Fortran arrays are column-major. Indices in comments refer to 1-based
// Fortran conventions; C code uses 0-based offsets.
// =============================================================================
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include "models/mod_settings.h"
#include "optimization_defines.h"

// TODO: Discuss if this method to take hard-coded compile-time parameters from mod_settings.h is ok

// ---------------------------------------------------------------------------
// Compile-time parameters, taken and renamed from models/mod_settings.h
// ---------------------------------------------------------------------------

static constexpr int N_TOR = n_tor;
static constexpr int N_COORD_TOR = n_coord_tor;
static constexpr int N_PERIOD = n_period;

static constexpr int NV       = n_vertex_max;
static constexpr int NDEG     = n_degrees;
static constexpr int NDIM     = n_dim;
static constexpr int NMODE    = (N_TOR - 1) / 2;

static constexpr int NVAR = 3;
// nl_values / nl_deltas only carry P_par and P_perp (not j_Phi), so their
// first (fastest) dimension is 2, not NVAR. NVAR is for feedback_rhs only.
static constexpr int N_FIELD_VARS = 2;
static constexpr int P_PAR_IDX = 0;
static constexpr int P_PERP_IDX = 1;
static constexpr int J_PHI_IDX = 2;

// ---------------------------------------------------------------------------
// Particle sorting parameters
// ---------------------------------------------------------------------------
static constexpr int I_ELM_MAX = 11000;
static constexpr int I_ELM_BINS = I_ELM_MAX + 1; // extra bin for invalid i_elm
static constexpr int HIST_SCAN_CHUNK = 1024;
static constexpr int HIST_SCAN_THREADS = 256;

#if LUT_VALUES_DELTAS
// Shared-memory LUT for nl_values / nl_deltas in calc_EBpsiU.
// SLOT_SIZE: number of doubles cached per element (NV * NDEG * 2 * N_TOR).
// Tuning knobs (LUT_N_SLOTS, LUT_REFRESH_INTERVAL, LUT_MIN_OCCUPANCY) come from optimization_defines.h.
static constexpr int LUT_SLOT_SIZE = NV * NDEG * 2 * N_TOR;  // = 32 * N_TOR
#endif

// ---------------------------------------------------------------------------
// Physical constants (matching jorek/models/constants.f90)
// ---------------------------------------------------------------------------
static constexpr double PI_VAL           = 3.14159265358979323846;
static constexpr double MU_ZERO          = 4.0e-7 * PI_VAL;
static constexpr double EL_CHG           = 1.602176565e-19;
static constexpr double ATOMIC_MASS_UNIT = 1.660539040e-27;
static constexpr double MASS_ELECTRON    = 9.10938291e-31;
static constexpr double SPEED_OF_LIGHT   = 2.997924580105029e+8;

#define TO_KB(x) ((double)(x) / 1024.0)

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

// ---------------------------------------------------------------------------
// Sorting helpers: map i_elm to a histogram bin
// ---------------------------------------------------------------------------
__device__ __forceinline__
int i_elm_to_bin(int i_elm)
{
    if (i_elm >= 1 && i_elm <= I_ELM_MAX) return i_elm - 1;
    return I_ELM_MAX;
}

// ---------------------------------------------------------------------------
// Counting sort kernels for particle SoA
// ---------------------------------------------------------------------------
__global__
void count_i_elm_histogram(const int* __restrict__ i_elm,
                           int num_particles,
                           int* __restrict__ global_hist)
{
    extern __shared__ int sh_hist[];
    for (int idx = threadIdx.x; idx < I_ELM_BINS; idx += blockDim.x)
        sh_hist[idx] = 0;
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int j = tid; j < num_particles; j += stride) {
        int key = i_elm_to_bin(i_elm[j]);
        atomicAdd(&sh_hist[key], 1);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < I_ELM_BINS; idx += blockDim.x) {
        int val = sh_hist[idx];
        if (val > 0) atomicAdd(&global_hist[idx], val);
    }
}

__global__
void exclusive_scan_blocks(const int* __restrict__ in,
                           int* __restrict__ out,
                           int* __restrict__ block_sums,
                           int n)
{
    __shared__ int sh_data[HIST_SCAN_CHUNK];
    int base = blockIdx.x * HIST_SCAN_CHUNK;

    for (int i = threadIdx.x; i < HIST_SCAN_CHUNK; i += blockDim.x) {
        int idx = base + i;
        sh_data[i] = (idx < n) ? in[idx] : 0;
    }
    __syncthreads();

    for (int offset = 1; offset < HIST_SCAN_CHUNK; offset <<= 1) {
        for (int idx = (threadIdx.x + 1) * offset * 2 - 1;
             idx < HIST_SCAN_CHUNK;
             idx += blockDim.x * offset * 2) {
            sh_data[idx] += sh_data[idx - offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (block_sums) block_sums[blockIdx.x] = sh_data[HIST_SCAN_CHUNK - 1];
        sh_data[HIST_SCAN_CHUNK - 1] = 0;
    }
    __syncthreads();

    for (int offset = HIST_SCAN_CHUNK >> 1; offset > 0; offset >>= 1) {
        for (int idx = (threadIdx.x + 1) * offset * 2 - 1;
             idx < HIST_SCAN_CHUNK;
             idx += blockDim.x * offset * 2) {
            int t = sh_data[idx - offset];
            sh_data[idx - offset] = sh_data[idx];
            sh_data[idx] += t;
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < HIST_SCAN_CHUNK; i += blockDim.x) {
        int idx = base + i;
        if (idx < n) out[idx] = sh_data[i];
    }
}

__global__
void add_block_offsets(int* __restrict__ data,
                       const int* __restrict__ block_offsets,
                       int n)
{
    int base = blockIdx.x * HIST_SCAN_CHUNK;
    int add = block_offsets[blockIdx.x];
    for (int i = threadIdx.x; i < HIST_SCAN_CHUNK; i += blockDim.x) {
        int idx = base + i;
        if (idx < n) data[idx] += add;
    }
}

__global__
void scatter_particles_by_i_elm(
    const double* __restrict__ x_in,
    const double* __restrict__ p_in,
    const double* __restrict__ st_in,
    const int*    __restrict__ i_elm_in,
    const double* __restrict__ weight_in,
    int num_particles,
    int* __restrict__ cursors,
    double* __restrict__ x_out,
    double* __restrict__ p_out,
    double* __restrict__ st_out,
    int*    __restrict__ i_elm_out,
    double* __restrict__ weight_out)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_particles) return;

    int i_elm = i_elm_in[j];
    int key = i_elm_to_bin(i_elm);
    int pos = atomicAdd(&cursors[key], 1);

    x_out[idx2(pos, 0, num_particles)] = x_in[idx2(j, 0, num_particles)];
    x_out[idx2(pos, 1, num_particles)] = x_in[idx2(j, 1, num_particles)];
    x_out[idx2(pos, 2, num_particles)] = x_in[idx2(j, 2, num_particles)];

    p_out[idx2(pos, 0, num_particles)] = p_in[idx2(j, 0, num_particles)];
    p_out[idx2(pos, 1, num_particles)] = p_in[idx2(j, 1, num_particles)];
    p_out[idx2(pos, 2, num_particles)] = p_in[idx2(j, 2, num_particles)];

    st_out[idx2(pos, 0, num_particles)] = st_in[idx2(j, 0, num_particles)];
    st_out[idx2(pos, 1, num_particles)] = st_in[idx2(j, 1, num_particles)];

    weight_out[pos] = weight_in[j];
    i_elm_out[pos] = i_elm;
}

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
    double* x;         // (NDIM, NDEG, N_COORD_TOR, n_nodes) grid coordinates       — n_nodes slowest for stride-1 it-loop
    double* values;    // (N_FIELD_VARS=2, NDEG, N_TOR, n_nodes)  field values     — n_nodes slowest for stride-1 it-loop
    double* deltas;    // (N_FIELD_VARS=2, NDEG, N_TOR, n_nodes)  field increments — n_nodes slowest for stride-1 it-loop
};

// Element list in Structure-of-Arrays layout.
// Fortran: type element_list_SoA
struct element_list_SoA {
    int     n_elements; // total number of elements
    int*    vertex;     // (NV, n_elements)       1-based node indices — n_elements fastest for coalesced warp access
    int*    neighbours; // (NV, n_elements)       1-based neighbour element indices (0=boundary)
    double* size;       // (NDEG, NV, n_elements) basis-function scale factors — n_elements fastest
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

// TODO: in bf2D_0_scalar and bf2D_1_scalar we are assuming N_ORDER = 3
// Implement also for N_ORDER = 5 ??

// ---------------------------------------------------------------------------
// Scalar on-the-fly basis function evaluation (N_ORDER = 3, NV = NDEG = 4).
// Returns the single H value for a given (kf, kv) pair without allocating
// a full NV*NDEG array.  All threads in a warp share the same kf/kv at any
// loop iteration, so the if-else chain is warp-uniform — no divergence.
// ---------------------------------------------------------------------------
__device__ __forceinline__
double bf2D_0_scalar(double s, double t, int kf, int kv)
{
    double sm1  = s - 1.0;
    double tm1  = t - 1.0;
    double sm12 = sm1 * sm1;
    double s2   = s * s;
    double tm12 = tm1 * tm1;
    double t2   = t * t;

    if (kv == 0) {
        if      (kf == 0) return  sm12 * (1.0 + 2.0*s) * tm12 * (1.0 + 2.0*t);
        else if (kf == 1) return  3.0 * sm12 * s * tm12 * (1.0 + 2.0*t);
        else if (kf == 2) return  3.0 * sm12 * (1.0 + 2.0*s) * tm12 * t;
        else              return  9.0 * sm12 * s * tm12 * t;
    } else if (kv == 1) {
        if      (kf == 0) return -(s2 * (-3.0 + 2.0*s) * tm12 * (1.0 + 2.0*t));
        else if (kf == 1) return -3.0 * sm1 * s2 * tm12 * (1.0 + 2.0*t);
        else if (kf == 2) return -3.0 * s2 * (-3.0 + 2.0*s) * tm12 * t;
        else              return -9.0 * sm1 * s2 * tm12 * t;
    } else if (kv == 2) {
        if      (kf == 0) return  s2 * (-3.0 + 2.0*s) * t2 * (-3.0 + 2.0*t);
        else if (kf == 1) return  3.0 * sm1 * s2 * t2 * (-3.0 + 2.0*t);
        else if (kf == 2) return  3.0 * s2 * (-3.0 + 2.0*s) * tm1 * t2;
        else              return  9.0 * sm1 * s2 * tm1 * t2;
    } else {
        if      (kf == 0) return -(sm12 * (1.0 + 2.0*s) * t2 * (-3.0 + 2.0*t));
        else if (kf == 1) return -3.0 * sm12 * s * t2 * (-3.0 + 2.0*t);
        else if (kf == 2) return -3.0 * sm12 * (1.0 + 2.0*s) * tm1 * t2;
        else              return -9.0 * sm12 * s * tm1 * t2;
    }
}

// ---------------------------------------------------------------------------
// Scalar on-the-fly basis function evaluation with first derivatives.
// Fills h, hs, ht for the given (kf, kv) pair without allocating arrays.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void bf2D_1_scalar(double s, double t, int kf, int kv,
                   double &h, double &hs, double &ht)
{
    h = bf2D_0_scalar(s, t, kf, kv);

    double sm1  = s - 1.0;
    double tm1  = t - 1.0;
    double sm12 = sm1 * sm1;
    double s2   = s * s;
    double tm12 = tm1 * tm1;
    double t2   = t * t;

    if (kv == 0) {
        if (kf == 0) {
            hs = 6.0*sm1*s * tm12*(1.0+2.0*t);
            ht = 6.0*sm12*(1.0+2.0*s) * tm1*t;
        } else if (kf == 1) {
            hs = 3.0*sm1*(-1.0+3.0*s) * tm12*(1.0+2.0*t);
            ht = 18.0*sm12*s * tm1*t;
        } else if (kf == 2) {
            hs = 18.0*sm1*s * tm12*t;
            ht = 3.0*sm12*(1.0+2.0*s) * tm1*(-1.0+3.0*t);
        } else {
            hs = 9.0*sm1*(-1.0+3.0*s) * tm12*t;
            ht = 9.0*sm12*s * tm1*(-1.0+3.0*t);
        }
    } else if (kv == 1) {
        if (kf == 0) {
            hs = -6.0*sm1*s * tm12*(1.0+2.0*t);
            ht = -6.0*s2*(-3.0+2.0*s) * tm1*t;
        } else if (kf == 1) {
            hs = -3.0*s*(-2.0+3.0*s) * tm12*(1.0+2.0*t);
            ht = -18.0*sm1*s2 * tm1*t;
        } else if (kf == 2) {
            hs = -18.0*sm1*s * tm12*t;
            ht = 3.0*s2*(-3.0+2.0*s) * (1.0-3.0*t)*tm1;
        } else {
            hs = -9.0*s*(-2.0+3.0*s) * tm12*t;
            ht = 9.0*sm1*s2 * (1.0-3.0*t)*tm1;
        }
    } else if (kv == 2) {
        if (kf == 0) {
            hs = 6.0*sm1*s * t2*(-3.0+2.0*t);
            ht = 6.0*s2*(-3.0+2.0*s) * tm1*t;
        } else if (kf == 1) {
            hs = 3.0*s*(-2.0+3.0*s) * t2*(-3.0 +2.0*t);
            ht = 18.0*sm1*s2 * tm1*t;
        } else if (kf == 2) {
            hs = 18.0*sm1*s * tm1*t2;
            ht = 3.0*s2*(-3.0+2.0*s) * t*(-2.0+3.0*t);
        } else {
            hs = 9.0*s*(-2.0+3.0*s) * tm1*t2;
            ht = 9.0*sm1*s2 * t*(-2.0+3.0*t);
        }
    } else {
        if (kf == 0) {
            hs = -6.0*sm1*s * t2*(-3.0+2.0*t);
            ht = -6.0*sm12*(1.0+2.0*s) * tm1*t;
        } else if (kf == 1) {
            hs = 3.0*(1.0-3.0*s)*sm1 * t2*(-3.0+2.0*t);
            ht = -18.0*sm12*s * tm1*t;
        } else if (kf == 2) {
            hs = -18.0*sm1*s * tm1*t2;
            ht = -3.0*sm12*(1.0+2.0*s) * t*(-2.0+3.0*t);
        } else {
            hs = 9.0*(1.0-3.0*s)*sm1 * tm1*t2;
            ht = -9.0*sm12*s * t*(-2.0+3.0*t);
        }
    }
}

// ---------------------------------------------------------------------------
// sincosperiod_moivre: compute toroidal harmonics and derivatives
// HZ[N_TOR], dHZ[N_TOR]
// Modes are built on the fly via De Moivre recurrence to avoid storing
// intermediate cos/sin arrays and save registers.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void sincosperiod_moivre(double phi, double* __restrict__ HZ, double* __restrict__ dHZ)
{
    HZ[0]  = 1.0;
    dHZ[0] = 0.0;
    // Base angle for one period step; higher modes built via De Moivre recurrence.
    double base_c, base_s;
    sincos(double(N_PERIOD) * phi, &base_s, &base_c);
    double c_prev = 1.0, s_prev = 0.0;
    for (int i = 1; i <= NMODE; ++i) {
        double c = c_prev * base_c - s_prev * base_s;
        double sn = c_prev * base_s + s_prev * base_c;
        c_prev = c; s_prev = sn;
        HZ [2*i - 1] = c;
        HZ [2*i]     = sn;
        dHZ[2*i - 1] = sn * (-N_PERIOD * i);
        dHZ[2*i]     = c  * ( N_PERIOD * i);
    }
}

// ---------------------------------------------------------------------------
// mode_moivre: compute toroidal harmonics (without derivatives)
// Modes built on the fly via De Moivre recurrence (no intermediate storage).
// ---------------------------------------------------------------------------
__device__ __forceinline__
void mode_moivre(double phi, double* __restrict__ HZ)
{
    HZ[0] = 1.0;
    double base_c, base_s;
    sincos(double(N_PERIOD) * phi, &base_s, &base_c);
    double c_prev = 1.0, s_prev = 0.0;
    for (int i = 1; i <= NMODE; ++i) {
        double c = c_prev * base_c - s_prev * base_s;
        double sn = c_prev * base_s + s_prev * base_c;
        c_prev = c; s_prev = sn;
        HZ[2*i - 1] = c;
        HZ[2*i]     = sn;
    }
}

// ---------------------------------------------------------------------------
// Coordinate transforms
// ---------------------------------------------------------------------------

// Cylindrical (R,Z,phi) -> Cartesian (x,y,z)
__device__ __forceinline__
void cylindrical_to_cartesian(const double* __restrict__ cyl, double* __restrict__ xyz)
{
    double cp, sp;
    sincos(-cyl[2], &sp, &cp);
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
    double sp, cp;
    sincos(phi, &sp, &cp);
    b[0] =  a[0]*cp - a[1]*sp;
    b[1] =  a[2];
    b[2] = -(a[0]*sp + a[1]*cp);
}

// Vector in Cylindrical -> Cartesian
__device__ __forceinline__
void vector_cylindrical_to_cartesian(double phi, const double* __restrict__ a, double* __restrict__ b)
{
    double sp, cp;
    sincos(phi, &sp, &cp);
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
    double alpha = SPEED_OF_LIGHT * scaling *
                   rsqrt(1.0 + pm[0]*pm[0] + pm[1]*pm[1] + pm[2]*pm[2]);

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
                       double &Z, double &Z_s, double &Z_t, double &Z_p)
{
    // Toroidal coordinate harmonics (for N_COORD_TOR == 1 this is trivial)
    double HZ_coord[N_COORD_TOR], HZ_coord_p[N_COORD_TOR];
    HZ_coord[0]   = 1.0;
    HZ_coord_p[0] = 0.0;
    for (int it = 1; it <= (N_COORD_TOR - 1) / 2; ++it) {
        int mc_cos = mode_coord[2*it - 1];
        int mc_sin = mode_coord[2*it];
        double hzc_s, hzc_c, hzs_s, hzs_c;
        sincos(double(mc_cos) * phi, &hzc_s, &hzc_c);
        sincos(double(mc_sin) * phi, &hzs_s, &hzs_c);
        HZ_coord  [2*it - 1] =  hzc_c;
        HZ_coord_p[2*it - 1] = -double(mc_cos) * hzc_s;
        HZ_coord  [2*it]     = -hzs_s;
        HZ_coord_p[2*it]     = -double(mc_sin) * hzs_c;
    }

    R = 0.0; R_s = 0.0; R_t = 0.0; R_p = 0.0;
    Z = 0.0; Z_s = 0.0; Z_t = 0.0; Z_p = 0.0;
    int ie = i_elm_f - 1;       // Element idx, 0-based

    for (int kv = 0; kv < NV; ++kv) {
        int iv = el_vertex[idx2(kv, ie, NV)] - 1;       // Node number, 0-based
        for (int kf = 0; kf < NDEG; ++kf) {
            double ss = el_size[idx3(kf, kv, ie, NDEG, NV)];
            double g, gs, gt;
            bf2D_1_scalar(s, t, kf, kv, g, gs, gt);

            for (int it = 0; it < N_COORD_TOR; ++it) {
                // nl_x layout: (NDIM, NDEG, N_COORD_TOR, n_nodes)
                double xx1 = nl_x[idx4(0, kf, it, iv, NDIM, NDEG, N_COORD_TOR)];
                double xx2 = nl_x[idx4(1, kf, it, iv, NDIM, NDEG, N_COORD_TOR)];
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
                    double &inv_jac)
{
    double R_p, Z_p;
    interp_RZP_1_gpu(nl_x, el_vertex, el_size, n_elements, n_nodes, mode_coord,
                     i_elm_f, st[0], st[1], phi,
                     x[0], R_s, R_t, R_p, x[1], Z_s, Z_t, Z_p);
    double jac = R_s * Z_t - R_t * Z_s;
    if (fabs(jac) < 1.0e-8)
        inv_jac = jac>0 ? 1.0e8 : -1.0e8;
    else
        inv_jac = 1.0 / jac;
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
                                    int &side2, bool &is_nb, bool &co)      // elm1, elm2, side1, side2 are 1-based
{
    co    = false;
    is_nb = false;
    side2 = 0;
    int ie1 = elm1 - 1, ie2 = elm2 - 1;

    // Find the side in elm2 pointing to elm1
    // If elm2 has no neighbour -> elm1 use the last one that is 0 (i.e. the one on the axis itself)
    for (int i = 0; i < NV; ++i) {
        if (el_neighbours[idx2(i, ie2, NV)] == elm1) {
            side2 = i + 1;
            break;
        }
    }
    if (side2 > 0) {
        is_nb = true;
        // Determine node numbers of the sides
        // Node numbers are related to sides as node1=(side-1)%4+1, node2=side%4+1
        int n1a = el_vertex[idx2((side1 - 1) % 4, ie1, NV)];
        int n1b = el_vertex[idx2( side1      % 4, ie1, NV)];
        int n2a = el_vertex[idx2((side2 - 1) % 4, ie2, NV)];
        int n2b = el_vertex[idx2( side2      % 4, ie2, NV)];
        co = (n1a == n2b) || (n1b == n2a);
    }
}

// ---------------------------------------------------------------------------
// coord_in_neighbour_gpu: transform (i_from, st) to neighbour element
// i_from 1-based, returns i_to (>0 found, -1 search needed, 0 lost)
// ---------------------------------------------------------------------------
__device__
void coord_in_neighbour_gpu(const int* __restrict__ el_vertex,
                            const int* __restrict__ el_neighbours,
                            int n_elements,
                            int i_from, int &i_to, double st[2])        // i_from, i_to are 1-based
{
    // q for "Quadrant"
    int q_from;
    if (st[0] > st[1]) {
        q_from = (1.0 - st[0] > st[1]) ? 1 : 2;
    } else {
        q_from = (1.0 - st[0] <= st[1]) ? 3 : 4;
    }

    i_to = el_neighbours[idx2(q_from - 1, i_from - 1, NV)];
    if (i_to <= 0) return;

    // Check once more that they are neighbours and determine the orientation
    int q_to; bool nb, co;
    neighbours_side_co_counter_gpu(el_vertex, el_neighbours, n_elements,
                                   i_from, i_to, q_from, q_to, nb, co);
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
}

// ---------------------------------------------------------------------------
// find_RZ_single_gpu: Newton search in a single element (5 starting points)
// i_elm_f is 1-based.  ifail=0 on success, 999 on failure.
// ---------------------------------------------------------------------------
__device__ __noinline__
void find_RZ_single_gpu(const double* __restrict__ nl_x,
                         const int*    __restrict__ el_vertex,
                         const double* __restrict__ el_size,
                         int n_elements, int n_nodes,
                         const int*    __restrict__ mode_coord,
                         int i_elm_f,
                         double R_find, double Z_find,
                         double &R_out, double &Z_out,
                         int &ielm_out, double &s_out, double &t_out,           // both i_elm_f, ielm_out are 1-based
                         int &ifail)
{
    constexpr int ntrial = 20;
    constexpr double tolx = 1.0e-8;
    constexpr double tolf = 1.0e-15;
    double phi_loc = 0.0;

    ielm_out = i_elm_f;

    double starts[5][2] = {{0.5,0.5},{0.75,0.75},{0.75,0.25},{0.25,0.75},{0.25,0.25}};

    for (int ist = 0; ist < 5; ++ist) {
        double x[2] = {starts[ist][0], starts[ist][1]};
        ifail = 999;

        for (int i = 0; i < ntrial; ++i) {
            double R_s, R_t, R_p, Z_s, Z_t, Z_p, RRg1, ZZg1;
            interp_RZP_1_gpu(nl_x, el_vertex, el_size, n_elements, n_nodes,
                             mode_coord, i_elm_f, x[0], x[1], phi_loc,
                             RRg1, R_s, R_t, R_p, ZZg1, Z_s, Z_t, Z_p);

            double fvec[2] = {RRg1 - R_find, ZZg1 - Z_find};
            double errf = fabs(fvec[0]) + fabs(fvec[1]);

            if (errf <= tolf) {
                s_out = x[0]; t_out = x[1];
                ielm_out = i_elm_f;
                R_out = RRg1; Z_out = ZZg1;
                ifail = 0;
                return;
            }

            double p[2] = {-fvec[0], -fvec[1]};
            double dis = Z_t * R_s - R_t * Z_s;     // determinant of the jacobian
            if (dis == 0.0) {
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
                return;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// find_RZ_gpu: brute-force search over all elements
// ---------------------------------------------------------------------------
__device__ __noinline__
void find_RZ_gpu(const double* __restrict__ nl_x,
                 const int*    __restrict__ el_vertex,
                 const double* __restrict__ el_size,
                 const int*    __restrict__ el_neighbours,
                 int n_elements, int n_nodes,
                 const int*    __restrict__ mode_coord,
                 double R_find, double Z_find,
                 double &R_out, double &Z_out,
                 int &ielm_out, double &s_out, double &t_out,       // i_elm_out is 1-based
                 int &ifail)
{
    ielm_out = 0;
    for (int k = 1; k <= n_elements; ++k) {
        find_RZ_single_gpu(nl_x, el_vertex, el_size, n_elements, n_nodes,
                            mode_coord, k, R_find, Z_find,
                            R_out, Z_out, ielm_out, s_out, t_out, ifail);
        if (ifail == 0) return;
    }
    if (ielm_out == 0) ifail = 99;
    if (ifail == 999) ielm_out = 0;
}

// ---------------------------------------------------------------------------
// find_RZ_nearby_gpu: Newton iteration with neighbour-hopping
// All element indices are 1-based (matching Fortran convention).
// On exit: i_elm_new <=0 means particle lost.
// ---------------------------------------------------------------------------
__device__ __noinline__
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
                        int &ifail)     // both i_elm_old, i_elm_new are 1-based
{
    // Accuracy defaults
    // Note: tolerances are squared!, units of element size
    constexpr double element_tolerance = 1.0e-24;   // tolerance for finding a position inside an element
    constexpr int    newton_iter_max   = 200;       // Number of iterations to try

    // Check if element is valied
    if (i_elm_old < 1 || i_elm_old > n_elements) {
        i_elm_new = 0;
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
                   i_elm_new, st, p_loc, x_step, R_s, R_t, Z_s, Z_t, inv_jac);

    double dx0 = x_step[0] - x_target[0];
    double dx1 = x_step[1] - x_target[1];
    double err2 = dx0*dx0 + dx1*dx1;
    ifail = 0;
    int iter;

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

        // if fact >= 1, we are on the boundary
        // That is it is the overshoot: i.e. how many times we overshoot boudnary with one st_step
        if (fact >= 1.0 - 1.0e-12) {

            st[0] += st_step0 / fact;
            st[1] += st_step1 / fact;
            int i_elm_tmp = i_elm_new;
            coord_in_neighbour_gpu(el_vertex, el_neighbours, n_elements,
                                   i_elm_tmp, i_elm_new, st);
            if (i_elm_new < 0) {
                double R_out, Z_out;
                find_RZ_gpu(nl_x, el_vertex, el_size, el_neighbours,
                            n_elements, n_nodes, mode_coord,
                            R_new, Z_new, R_out, Z_out,
                            i_elm_new, s_new, t_new, ifail);
                return;
            }
            if (i_elm_new == 0) {       // No element on that side, particle is lost
                i_elm_new = -i_elm_tmp; // Save position of particle
                // Compute new R and Z in x_tmp
                double x_tmp[2]; double dummy;
                try_interp_gpu(nl_x, el_vertex, el_size, n_elements, n_nodes, mode_coord,
                               i_elm_tmp, st, p_loc, x_tmp, R_s, R_t, Z_s, Z_t, dummy);
                s_new = st[0]; t_new = st[1];
                ifail = -1;
                return;
            }
            try_interp_gpu(nl_x, el_vertex, el_size, n_elements, n_nodes, mode_coord,
                           i_elm_new, st, p_loc, x_step, R_s, R_t, Z_s, Z_t, inv_jac);
        } else {
            st[0] += st_step0;
            st[1] += st_step1;
            try_interp_gpu(nl_x, el_vertex, el_size, n_elements, n_nodes, mode_coord,
                           i_elm_new, st, p_loc, x_step, R_s, R_t, Z_s, Z_t, inv_jac);
        }

        dx0 = x_step[0] - x_target[0];
        dx1 = x_step[1] - x_target[1];
        err2 = dx0*dx0 + dx1*dx1;
        s_new = st[0];
        t_new = st[1];
        if (err2 < element_tolerance) {
            return;
        }
    }

    if (isnan(err2)) {
        i_elm_new = -2;
        return;
    }
    if (iter > newton_iter_max) {
        double R_out, Z_out;
        find_RZ_gpu(nl_x, el_vertex, el_size, el_neighbours,
                    n_elements, n_nodes, mode_coord,
                    R_new, Z_new, R_out, Z_out,
                    i_elm_new, s_new, t_new, ifail);
    }
}

// ---------------------------------------------------------------------------
// calc_EBpsiU: compute E, B, psi, U at a point using linear time interpolation
// ---------------------------------------------------------------------------
__device__ __noinline__
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
                 double E[3], double B[3]
#if LUT_VALUES_DELTAS
                 , const int*    sh_lut_keys
                 , const double* sh_cache_v_flat
                 , const double* sh_cache_d_flat
#endif
                 )
{
    int ie = i_elm_f - 1;

    double R = 0.0, R_s = 0.0, R_t = 0.0;
    double Zc = 0.0, Z_s = 0.0, Z_t = 0.0;

    double P[2]      = {0.0, 0.0};
    double P_s[2]    = {0.0, 0.0};
    double P_t[2]    = {0.0, 0.0};
    double P_phi[2]  = {0.0, 0.0};
    double Pd[2]     = {0.0, 0.0};
    double Pd_s[2]   = {0.0, 0.0};
    double Pd_t[2]   = {0.0, 0.0};
    double Pd_phi[2] = {0.0, 0.0};

#if LUT_VALUES_DELTAS
    int lut_cache_slot = -1;
    for (int s = 0; s < LUT_N_SLOTS; ++s)
        if (sh_lut_keys[s] == i_elm_f) { lut_cache_slot = s; break; }
#endif

    // Fused loop: geometry and field values share el_vertex, el_size, bf2D loads.
    // Toroidal harmonics computed on the fly via De Moivre recurrence inside the
    // it loop (one sincos call total, then complex multiply for higher modes).
    double base_c, base_s;
    sincos(double(N_PERIOD) * phi, &base_s, &base_c);

    for (int kv = 0; kv < NV; ++kv) {
        int iv = el_vertex[idx2(kv, ie, NV)] - 1;
        for (int kf = 0; kf < NDEG; ++kf) {
            double sz = el_size[idx3(kf, kv, ie, NDEG, NV)];
            double h, hs, ht;
            bf2D_1_scalar(st[0], st[1], kf, kv, h, hs, ht);

            double xR = nl_x[idx4(0, kf, 0, iv, NDIM, NDEG, N_COORD_TOR)] * sz;
            double xZ = nl_x[idx4(1, kf, 0, iv, NDIM, NDEG, N_COORD_TOR)] * sz;
            R   += xR * h;
            R_s += xR * hs;
            R_t += xR * ht;
            Zc  += xZ * h;
            Z_s += xZ * hs;
            Z_t += xZ * ht;

            double c = 1.0, s = 0.0;  // cos/sin of current mode i, reset each kv/kf

            double v0 = 0.0, vp0 = 0.0, vd0 = 0.0, vpd0 = 0.0;
            double v1 = 0.0, vp1 = 0.0, vd1 = 0.0, vpd1 = 0.0;

            for (int it = 0; it < N_TOR; ++it) {
                double hz_it, dhz_it;
                if (it == 0) {
                    hz_it = 1.0; dhz_it = 0.0;
                } else if (it == 1) {
                    c = base_c; s = base_s;
                    hz_it = c; dhz_it = -double(N_PERIOD) * s;
                } else if (it & 1) {
                    double cn = c * base_c - s * base_s;
                    double sn = c * base_s + s * base_c;
                    c = cn; s = sn;
                    int i = (it + 1) / 2;
                    hz_it = c; dhz_it = -double(N_PERIOD * i) * s;
                } else {
                    int i = it / 2;
                    hz_it = s; dhz_it = double(N_PERIOD * i) * c;
                }

#if LUT_VALUES_DELTAS
                double raw_v0 = (lut_cache_slot >= 0)
                    ? sh_cache_v_flat[lut_cache_slot * LUT_SLOT_SIZE + kv + NV*(it + N_TOR*(kf + NDEG*0))]
                    : nl_values[idx4(0, kf, it, iv, N_FIELD_VARS, NDEG, N_TOR)];
                double raw_d0 = (lut_cache_slot >= 0)
                    ? sh_cache_d_flat[lut_cache_slot * LUT_SLOT_SIZE + kv + NV*(it + N_TOR*(kf + NDEG*0))]
                    : nl_deltas[idx4(0, kf, it, iv, N_FIELD_VARS, NDEG, N_TOR)];
                double raw_v1 = (lut_cache_slot >= 0)
                    ? sh_cache_v_flat[lut_cache_slot * LUT_SLOT_SIZE + kv + NV*(it + N_TOR*(kf + NDEG*1))]
                    : nl_values[idx4(1, kf, it, iv, N_FIELD_VARS, NDEG, N_TOR)];
                double raw_d1 = (lut_cache_slot >= 0)
                    ? sh_cache_d_flat[lut_cache_slot * LUT_SLOT_SIZE + kv + NV*(it + N_TOR*(kf + NDEG*1))]
                    : nl_deltas[idx4(1, kf, it, iv, N_FIELD_VARS, NDEG, N_TOR)];
                double val_v0 = raw_v0 * sz, val_d0 = raw_d0 * sz;
                double val_v1 = raw_v1 * sz, val_d1 = raw_d1 * sz;
#else
                double val_v0 = nl_values[idx4(0, kf, it, iv, N_FIELD_VARS, NDEG, N_TOR)] * sz;
                double val_d0 = nl_deltas[idx4(0, kf, it, iv, N_FIELD_VARS, NDEG, N_TOR)] * sz;
                double val_v1 = nl_values[idx4(1, kf, it, iv, N_FIELD_VARS, NDEG, N_TOR)] * sz;
                double val_d1 = nl_deltas[idx4(1, kf, it, iv, N_FIELD_VARS, NDEG, N_TOR)] * sz;
#endif
                v0  += val_v0 * hz_it;  vp0 += val_v0 * dhz_it;
                vd0 += val_d0 * hz_it;  vpd0+= val_d0 * dhz_it;
                v1  += val_v1 * hz_it;  vp1 += val_v1 * dhz_it;
                vd1 += val_d1 * hz_it;  vpd1+= val_d1 * dhz_it;
            }

            P[0]      += v0  * h;   P_s[0]   += v0  * hs;
            P_t[0]    += v0  * ht;  P_phi[0] += vp0 * h;
            Pd[0]     += vd0 * h;   Pd_s[0]  += vd0 * hs;
            Pd_t[0]   += vd0 * ht;  Pd_phi[0]+= vpd0* h;

            P[1]      += v1  * h;   P_s[1]   += v1  * hs;
            P_t[1]    += v1  * ht;  P_phi[1] += vp1 * h;
            Pd[1]     += vd1 * h;   Pd_s[1]  += vd1 * hs;
            Pd_t[1]   += vd1 * ht;  Pd_phi[1]+= vpd1* h;
        }
    }

    double dt;
    if (fabs(time_now - time_prev) > 1.0e-10 && !flag_static) {
        dt = 1.0 / (time_now - time_prev);
        double df = (time_now - time) * dt;
        P[0]   -= Pd[0]   * df;  P_s[0]   -= Pd_s[0]   * df;
        P_t[0] -= Pd_t[0] * df;  P_phi[0] -= Pd_phi[0] * df;
        P[1]   -= Pd[1]   * df;  P_s[1]   -= Pd_s[1]   * df;
        P_t[1] -= Pd_t[1] * df;  P_phi[1] -= Pd_phi[1] * df;
    } else {
        dt = 1.0 / t_norm;
    }
    // Pd[0]*dt = dPsi/dt; only component 0 enters E_phi. Component 1 unused.
    double dpsidt = flag_zero_dpsidt ? 0.0 : Pd[0] * dt;

    double R_inv      = 1.0 / R;
    double st_jac_inv = 1.0 / (R_s * Z_t - R_t * Z_s);
    double t_norm_inv = 1.0 / t_norm;

    double psi_R = ( P_s[0] * Z_t - P_t[0] * Z_s) * st_jac_inv;
    double psi_Z = (-P_s[0] * R_t + P_t[0] * R_s) * st_jac_inv;
    double U_R   = ( P_s[1] * Z_t - P_t[1] * Z_s) * st_jac_inv;
    double U_Z   = (-P_s[1] * R_t + P_t[1] * R_s) * st_jac_inv;
    double U_phi = P_phi[1];

    // Magnetic field (cylindrical)
    B[0] =  psi_Z * R_inv;
    B[1] = -psi_R * R_inv;
    B[2] =  F0    * R_inv;

    // Electric field (cylindrical)
    double neg_F0_tnorm_inv = -F0 * t_norm_inv;
    E[0] = neg_F0_tnorm_inv * U_R;
    E[1] = neg_F0_tnorm_inv * U_Z;
    E[2] = (neg_F0_tnorm_inv * U_phi - dpsidt) * R_inv;

    // Projection: E = E - E * B / |B| (element-wise, matching Fortran)
    double Bnorm_inv = rsqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
    E[0] -= E[0] * B[0] * Bnorm_inv;
    E[1] -= E[1] * B[1] * Bnorm_inv;
    E[2] -= E[2] * B[2] * Bnorm_inv;
}

// ---------------------------------------------------------------------------
// volume_preserving_push: VPA integrator for a relativistic particle
// All position/element data is modified in-place.
// ---------------------------------------------------------------------------
__device__ __noinline__
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
                            int &ifail
#if LUT_VALUES_DELTAS
                            , const int*    sh_lut_keys
                            , const double* sh_cache_v_flat
                            , const double* sh_cache_d_flat
#endif
                            )
{
    // No need to check if particle is valid, it is already done in the caller
    // Turn particle position from cylindrical to cartesian coordinates
    double cur_xyz[3];
    cylindrical_to_cartesian(x, cur_xyz);

    // --- First half-step (position advance) ---

    double scaling = 0.5 * timestep * charge * EL_CHG / (ATOMIC_MASS_UNIT * mass * SPEED_OF_LIGHT);
    double mc = mass * SPEED_OF_LIGHT;
    // Compute dimensionless momentum
    // i.e. Normalise momentum: p -> p / (mass * c)
    double pm[3] = {p_mom[0] / mc, p_mom[1] / mc, p_mom[2] / mc};

    // Compute coordinates at half-step
    double pdot = pm[0]*pm[0] + pm[1]*pm[1] + pm[2]*pm[2];
    double gamma_inv = rsqrt(1.0 + pdot);
    double dt_half_c = 0.5 * timestep * SPEED_OF_LIGHT;
    double half_xyz[3] = {
        cur_xyz[0] + dt_half_c * pm[0] * gamma_inv,
        cur_xyz[1] + dt_half_c * pm[1] * gamma_inv,
        cur_xyz[2] + dt_half_c * pm[2] * gamma_inv
    };

    // Compute cylindrical coordinates from cartesian ones
    double half_cyl[3];
    cartesian_to_cylindrical(half_xyz, half_cyl);

    // Find element for half-step position  (that is (i_elm, s, t) coordinates)
    double s_new, t_new; int i_elm_new;
    find_RZ_nearby_gpu(nl_x, el_vertex, el_size, el_neighbours,
                       n_elements, n_nodes, mode_coord,
                       x[0], x[1], st[0], st[1], i_elm_f,
                       half_cyl[0], half_cyl[1],
                       s_new, t_new, i_elm_new, ifail);

    // If the particle is lost, exit
    if (i_elm_new <= 0) { i_elm_f = i_elm_new; return; }
    // Otherwise, copy new coordinates and element index to the particle data
    x[0] = half_cyl[0]; x[1] = half_cyl[1]; x[2] = half_cyl[2];
    st[0] = s_new; st[1] = t_new;
    i_elm_f = i_elm_new;

    // --- Compute E, B at half-step ---
    double E[3], B_field[3];
    calc_EBpsiU(nl_values, nl_deltas, nl_x, el_vertex, el_size,
                n_elements, n_nodes,
                time_now, time_prev, flag_static, flag_zero_dpsidt,
                F0, t_norm,
                i_elm_f, st, x[2], time + 0.5 * timestep,
                E, B_field
#if LUT_VALUES_DELTAS
                , sh_lut_keys, sh_cache_v_flat, sh_cache_d_flat
#endif
                );

    // --- Second half-step ---

    // Convert E, B to Cartesian for VPA rotation
    double E_cart[3], B_cart[3];
    vector_cylindrical_to_cartesian(x[2], E, E_cart);
    vector_cylindrical_to_cartesian(x[2], B_field, B_cart);

    // --- Momentum update: E-kick + Cayley rotation + E-kick ---
    // First electric kick
    pm[0] += scaling * E_cart[0];
    pm[1] += scaling * E_cart[1];
    pm[2] += scaling * E_cart[2];

    // Cayley transform rotation
    cayley_transform_rotate(pm, B_cart, scaling);

    // Second electric kick
    pm[0] += scaling * E_cart[0];
    pm[1] += scaling * E_cart[1];
    pm[2] += scaling * E_cart[2];

    // --- Second half position update ---
    pdot = pm[0]*pm[0] + pm[1]*pm[1] + pm[2]*pm[2];
    gamma_inv = rsqrt(1.0 + pdot);
    half_xyz[0] += dt_half_c * pm[0] * gamma_inv;
    half_xyz[1] += dt_half_c * pm[1] * gamma_inv;
    half_xyz[2] += dt_half_c * pm[2] * gamma_inv;

    // Restore dimensional momentum
    p_mom[0] = pm[0] * mc;
    p_mom[1] = pm[1] * mc;
    p_mom[2] = pm[2] * mc;

    // Turn back from cartesian to cylindrical coordinates
    cartesian_to_cylindrical(half_xyz, half_cyl);

    // Find new (i_elm, s, t) coordinates
    find_RZ_nearby_gpu(nl_x, el_vertex, el_size, el_neighbours,
                       n_elements, n_nodes, mode_coord,
                       x[0], x[1], st[0], st[1], i_elm_f,
                       half_cyl[0], half_cyl[1],
                       s_new, t_new, i_elm_new, ifail);

    // Copy new R-Z-Phi position into particle                       
    x[0] = half_cyl[0]; x[1] = half_cyl[1]; x[2] = half_cyl[2];
    st[0] = s_new; st[1] = t_new;
    i_elm_f = i_elm_new;
}


// ===========================================================================================
//                                  MAIN HIP KERNEL
// ===========================================================================================

// ---------------------------------------------------------------------------
// lut_build_cooperative: cooperatively build the shared-memory LUT.
// Called by all threads in a block; must NOT be called with divergent control flow.
// Phase 1 (thread 0): finds the LUT_N_SLOTS most-populated elements.
// Phase 2 (all threads): loads nl_values / nl_deltas fragments into shared memory.
// Two __syncthreads() barriers are embedded; the caller must not hold any
// pending sync before calling and must not rely on per-thread state that crosses
// those barriers (e.g., local variables captured in a lambda — use function params).
// ---------------------------------------------------------------------------
#if LUT_VALUES_DELTAS
__device__
void lut_build_cooperative(int i_elm_thread,
                            const int*    __restrict__ el_vertex,
                            int n_elements, int n_nodes,
                            const double* __restrict__ nl_values,
                            const double* __restrict__ nl_deltas,
                            int*    sh_lut_keys,
                            double* sh_cache_v,
                            double* sh_cache_d,
                            int*    sh_scratch)
{
    // Phase 1 — publish current element, thread 0 finds top-N_SLOTS.
    sh_scratch[threadIdx.x] = (i_elm_thread > 0) ? i_elm_thread : -1;
    __syncthreads();

    if (threadIdx.x == 0) {
        int top_elm[LUT_N_SLOTS], top_cnt[LUT_N_SLOTS];
        for (int s = 0; s < LUT_N_SLOTS; ++s) { top_elm[s] = -1; top_cnt[s] = 0; }

        for (int tid = 0; tid < BLOCK_SIZE; ++tid) {
            int elm = sh_scratch[tid];
            if (elm <= 0) continue;
            int found = -1;
            for (int s = 0; s < LUT_N_SLOTS; ++s)
                if (top_elm[s] == elm) { found = s; break; }
            if (found >= 0) {
                top_cnt[found]++;
            } else {
                int ms = 0;
                for (int s = 1; s < LUT_N_SLOTS; ++s)
                    if (top_cnt[s] < top_cnt[ms]) ms = s;
                if (top_cnt[ms] == 0) { top_elm[ms] = elm; top_cnt[ms] = 1; }
            }
        }
        for (int s = 0; s < LUT_N_SLOTS; ++s)
            sh_lut_keys[s] = (top_cnt[s] >= LUT_MIN_OCCUPANCY) ? top_elm[s] : -1;
    }
    __syncthreads();

    // Phase 2 — all threads cooperatively load field data.
    for (int s = 0; s < LUT_N_SLOTS; ++s) {
        int elm = sh_lut_keys[s];
        if (elm <= 0) continue;
        int ie_s = elm - 1;

        int ivs[NV];
        for (int kv = 0; kv < NV; ++kv)
            ivs[kv] = el_vertex[idx2(kv, ie_s, NV)] - 1;

        for (int pass = 0; pass * BLOCK_SIZE < LUT_SLOT_SIZE; ++pass) {
            int lid = pass * BLOCK_SIZE + threadIdx.x;
            if (lid < LUT_SLOT_SIZE) {
                int tmp    = lid;
                int kv_l   = tmp % NV;    tmp /= NV;
                int it_l   = tmp % N_TOR; tmp /= N_TOR;
                int kf_l   = tmp % NDEG;  tmp /= NDEG;
                int ivar_l = tmp;
                int node   = ivs[kv_l];
                // nl_values/nl_deltas layout: (N_FIELD_VARS=2, NDEG, N_TOR, n_nodes)
                int gi     = idx4(ivar_l, kf_l, it_l, node, N_FIELD_VARS, NDEG, N_TOR);
                sh_cache_v[s * LUT_SLOT_SIZE + lid] = nl_values[gi];
                sh_cache_d[s * LUT_SLOT_SIZE + lid] = nl_deltas[gi];
            }
        }
    }
    __syncthreads();
}
#endif

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
// ---------------------------------------------------------------------------
// evolve_proj_kernel: projection phase only.
// Reads particle state from current buffers (const), accumulates to feedback_rhs.
// Does NOT modify any particle array.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(BLOCK_SIZE, 2)
void evolve_proj_kernel(
    // Particle SoA — read-only
    const double* __restrict__ p_x,         // (num_particles, 3)
    const double* __restrict__ p_p,         // (num_particles, 3)
    const double* __restrict__ p_st,        // (num_particles, 2)
    const int*    __restrict__ p_i_elm,     // (num_particles)
    const double* __restrict__ p_weight,    // (num_particles)
    double charge,
    // Field node list SoA
    const double* __restrict__ nl_values,
    const double* __restrict__ nl_deltas,
    const double* __restrict__ nl_x,
    int n_nodes,
    // Field element list SoA
    const int*    __restrict__ el_vertex,
    const double* __restrict__ el_size,
    int n_elements,
    // Field time parameters
    double time_now, double time_prev,
    int flag_static, int flag_zero_dpsidt,
    // Physics parameters
    double F0, double t_norm,
    // Simulation parameters
    double sim_time, double group_mass,
    int num_particles,
    // Feedback RHS (atomically updated)
    double* __restrict__ feedback_rhs,
    const int* __restrict__ mode_coord)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_particles) return;

    double x[3]  = {p_x[idx2(j, 0, num_particles)], p_x[idx2(j, 1, num_particles)], p_x[idx2(j, 2, num_particles)]};
    double pm[3] = {p_p[idx2(j, 0, num_particles)], p_p[idx2(j, 1, num_particles)], p_p[idx2(j, 2, num_particles)]};
    double st[2] = {p_st[idx2(j, 0, num_particles)], p_st[idx2(j, 1, num_particles)]};
    int    i_elm = p_i_elm[j];
    double w = p_weight[j];

#if LUT_VALUES_DELTAS
    __shared__ int    sh_lut_keys[LUT_N_SLOTS];
    __shared__ double sh_cache_v[LUT_N_SLOTS * LUT_SLOT_SIZE];
    __shared__ double sh_cache_d[LUT_N_SLOTS * LUT_SLOT_SIZE];
    __shared__ int    sh_scratch[BLOCK_SIZE];

    lut_build_cooperative(i_elm, el_vertex, n_elements, n_nodes,
                          nl_values, nl_deltas,
                          sh_lut_keys, sh_cache_v, sh_cache_d, sh_scratch);
#endif

    if (i_elm > 0) {
        double HZ_proj[N_TOR];
        mode_moivre(x[2], HZ_proj);

        double cyl_mom[3];
        vector_cartesian_to_cylindrical(x[2], pm, cyl_mom);
        double pdot_cyl = cyl_mom[0]*cyl_mom[0] + cyl_mom[1]*cyl_mom[1] + cyl_mom[2]*cyl_mom[2];
        double denom_v_inv = rsqrt(pdot_cyl / (SPEED_OF_LIGHT*SPEED_OF_LIGHT) + group_mass*group_mass);
        double cyl_vel[3] = {cyl_mom[0] * denom_v_inv, cyl_mom[1] * denom_v_inv, cyl_mom[2] * denom_v_inv};

        double E_loc[3], B_loc[3];
        calc_EBpsiU(nl_values, nl_deltas, nl_x, el_vertex, el_size,
                    n_elements, n_nodes,
                    time_now, time_prev, flag_static, flag_zero_dpsidt,
                    F0, t_norm,
                    i_elm, st, x[2], sim_time,
                    E_loc, B_loc
#if LUT_VALUES_DELTAS
                    , sh_lut_keys, sh_cache_v, sh_cache_d
#endif
                    );

        double Bnorm_inv = rsqrt(B_loc[0]*B_loc[0] + B_loc[1]*B_loc[1] + B_loc[2]*B_loc[2]);
        double B_hat[3] = {B_loc[0]*Bnorm_inv, B_loc[1]*Bnorm_inv, B_loc[2]*Bnorm_inv};

        double v_par = cyl_vel[0]*B_hat[0] + cyl_vel[1]*B_hat[1] + cyl_vel[2]*B_hat[2];
        double v_perp_diff[3] = {cyl_vel[0] - v_par*B_hat[0],
                                 cyl_vel[1] - v_par*B_hat[1],
                                 cyl_vel[2] - v_par*B_hat[2]};
        double v_perp_sq = v_perp_diff[0]*v_perp_diff[0] + v_perp_diff[1]*v_perp_diff[1] + v_perp_diff[2]*v_perp_diff[2];

        double gamma_m = sqrt(MASS_ELECTRON*MASS_ELECTRON
                            + pdot_cyl * ATOMIC_MASS_UNIT*ATOMIC_MASS_UNIT
                              / (SPEED_OF_LIGHT*SPEED_OF_LIGHT));

        double v_Ppar  = gamma_m * v_par * v_par * MU_ZERO;
        double v_Pperp = gamma_m * v_perp_sq * 0.5 * MU_ZERO;
        double v_jPhi  = -double(charge) * EL_CHG * cyl_vel[2] * x[0] * MU_ZERO;

        int ie = i_elm - 1;
        for (int n = 0; n < NDEG; ++n) {
            for (int m = 0; m < NV; ++m) {
                double proj_factor = bf2D_0_scalar(st[0], st[1], n, m)
                                   * el_size[idx3(n, m, ie, NDEG, NV)]
                                   * w;

                for (int it = 0; it < N_TOR; ++it) {
                    double hz = HZ_proj[it];

                    atomicAdd(&feedback_rhs[idx5(ie, n, m, it, P_PAR_IDX, n_elements, NDEG, NV, N_TOR)],
                              hz * v_Ppar * proj_factor);
                    atomicAdd(&feedback_rhs[idx5(ie, n, m, it, P_PERP_IDX, n_elements, NDEG, NV, N_TOR)],
                              hz * v_Pperp * proj_factor);
                    atomicAdd(&feedback_rhs[idx5(ie, n, m, it, J_PHI_IDX, n_elements, NDEG, NV, N_TOR)],
                              hz * v_jPhi * proj_factor);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// evolve_push_kernel: push phase only.
// Reads particle state from _in (current) buffers, writes updated state to
// _out (alternate) buffers. Does NOT touch feedback_rhs.
// Writes to _out unconditionally so the alt buffer is always a complete copy.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(BLOCK_SIZE, 2)
void evolve_push_kernel(
    // Particle SoA — INPUT (current buffers, read-only)
    const double* __restrict__ p_x_in,
    const double* __restrict__ p_p_in,
    const double* __restrict__ p_st_in,
    const int*    __restrict__ p_i_elm_in,
    double charge,
    // Particle SoA — OUTPUT (alternate buffers, write-only)
    double* __restrict__ p_x_out,
    double* __restrict__ p_p_out,
    double* __restrict__ p_st_out,
    int*    __restrict__ p_i_elm_out,
    // Field node list SoA
    const double* __restrict__ nl_values,
    const double* __restrict__ nl_deltas,
    const double* __restrict__ nl_x,
    int n_nodes,
    // Field element list SoA
    const int*    __restrict__ el_vertex,
    const int*    __restrict__ el_neighbours,
    const double* __restrict__ el_size,
    int n_elements,
    // Field time parameters
    double time_now, double time_prev,
    int flag_static, int flag_zero_dpsidt,
    // Physics parameters
    double F0, double t_norm,
    // Simulation parameters
    double sim_time, double group_mass, double tstep_part_adj,
    int num_particles,
    const int* __restrict__ mode_coord)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_particles) return;

    double x[3]  = {p_x_in[idx2(j, 0, num_particles)], p_x_in[idx2(j, 1, num_particles)], p_x_in[idx2(j, 2, num_particles)]};
    double pm[3] = {p_p_in[idx2(j, 0, num_particles)], p_p_in[idx2(j, 1, num_particles)], p_p_in[idx2(j, 2, num_particles)]};
    double st[2] = {p_st_in[idx2(j, 0, num_particles)], p_st_in[idx2(j, 1, num_particles)]};
    int    i_elm = p_i_elm_in[j];

#if LUT_VALUES_DELTAS
    __shared__ int    sh_lut_keys[LUT_N_SLOTS];
    __shared__ double sh_cache_v[LUT_N_SLOTS * LUT_SLOT_SIZE];
    __shared__ double sh_cache_d[LUT_N_SLOTS * LUT_SLOT_SIZE];
    __shared__ int    sh_scratch[BLOCK_SIZE];

    lut_build_cooperative(i_elm, el_vertex, n_elements, n_nodes,
                          nl_values, nl_deltas,
                          sh_lut_keys, sh_cache_v, sh_cache_d, sh_scratch);
#endif

    if (i_elm > 0) {
        int ifail = 0;
        volume_preserving_push(x, pm, st, i_elm, charge,
                               nl_values, nl_deltas, nl_x,
                               el_vertex, el_size, el_neighbours,
                               n_elements, n_nodes, mode_coord,
                               time_now, time_prev,
                               flag_static, flag_zero_dpsidt,
                               F0, t_norm,
                               group_mass, sim_time, tstep_part_adj,
                               ifail
#if LUT_VALUES_DELTAS
                               , sh_lut_keys, sh_cache_v, sh_cache_d
#endif
                               );
    }

    // Always write to output buffers (captures lost-particle state too)
    p_x_out[idx2(j, 0, num_particles)] = x[0]; p_x_out[idx2(j, 1, num_particles)] = x[1]; p_x_out[idx2(j, 2, num_particles)] = x[2];
    p_p_out[idx2(j, 0, num_particles)] = pm[0]; p_p_out[idx2(j, 1, num_particles)] = pm[1]; p_p_out[idx2(j, 2, num_particles)] = pm[2];
    p_st_out[idx2(j, 0, num_particles)] = st[0]; p_st_out[idx2(j, 1, num_particles)] = st[1];
    p_i_elm_out[j] = i_elm;
}

// ---------------------------------------------------------------------------
// sort_particles_by_i_elm_gpu: GPU counting sort for particle SoA
// ---------------------------------------------------------------------------
static void sort_particles_by_i_elm_gpu(
    double*& d_x, double*& d_p, double*& d_st, int*& d_i_elm, double*& d_weight,
    double*& d_x_alt, double*& d_p_alt, double*& d_st_alt, int*& d_i_elm_alt, double*& d_weight_alt,
    int num_particles,
    int* d_hist, int* d_offsets, int* d_cursors,
    int* d_block_sums, int* d_block_offsets)
{
    if (num_particles <= 0) return;

    const size_t hist_bytes = I_ELM_BINS * sizeof(int);
    HIP_CHECK(hipMemset(d_hist, 0, hist_bytes));

    int grid_size = (num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t shared_bytes = I_ELM_BINS * sizeof(int);

    hipLaunchKernelGGL(count_i_elm_histogram,
        dim3(grid_size), dim3(BLOCK_SIZE), shared_bytes, 0,
        d_i_elm, num_particles, d_hist);
    HIP_CHECK(hipGetLastError());

    int scan_blocks = (I_ELM_BINS + HIST_SCAN_CHUNK - 1) / HIST_SCAN_CHUNK;
    hipLaunchKernelGGL(exclusive_scan_blocks,
        dim3(scan_blocks), dim3(HIST_SCAN_THREADS), 0, 0,
        d_hist, d_offsets, d_block_sums, I_ELM_BINS);
    HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL(exclusive_scan_blocks,
        dim3(1), dim3(HIST_SCAN_THREADS), 0, 0,
        d_block_sums, d_block_offsets, nullptr, scan_blocks);
    HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL(add_block_offsets,
        dim3(scan_blocks), dim3(HIST_SCAN_THREADS), 0, 0,
        d_offsets, d_block_offsets, I_ELM_BINS);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(d_cursors, d_offsets, hist_bytes, hipMemcpyDeviceToDevice));

    hipLaunchKernelGGL(scatter_particles_by_i_elm,
        dim3(grid_size), dim3(BLOCK_SIZE), 0, 0,
        d_x, d_p, d_st, d_i_elm, d_weight,
        num_particles, d_cursors,
        d_x_alt, d_p_alt, d_st_alt, d_i_elm_alt, d_weight_alt);
    HIP_CHECK(hipGetLastError());

    std::swap(d_x, d_x_alt);
    std::swap(d_p, d_p_alt);
    std::swap(d_st, d_st_alt);
    std::swap(d_i_elm, d_i_elm_alt);
    std::swap(d_weight, d_weight_alt);
}


// ===========================================================================================
//                       HOST LAUNCH FUNCTION (Fortran-callable via bind(C))
// ===========================================================================================

// ---------------------------------------------------------------------------
// Debug helpers: write p_i_elm and p_pol snapshots to binary files (rank-0 only).
//
// i_elm file  re_sort_debug_step<NNNNN>_<tag>_ielm.bin
//   Format: int32 num_particles, then num_particles x int32 i_elm values
//
// p_pol file  re_sort_debug_step<NNNNN>_<tag>_ppol.bin
//   Format: int32 num_particles, then num_particles x float64 p_pol values
//
// p_pol is the poloidal momentum magnitude: sqrt(p_R^2 + p_Z^2)
//   where p_R, p_Z are obtained by converting the stored Cartesian momentum
//   (p_x, p_y, p_z) to cylindrical using the particle's phi angle:
//     p_R   =  p_x*cos(phi) - p_y*sin(phi)
//     p_Z   =  p_z
//     p_pol = sqrt(p_R^2 + p_Z^2)
// ---------------------------------------------------------------------------
static void write_i_elm_snapshot(const int* h_i_elm, int num_particles,
                                 int fluid_step, const char* tag)
{
    char fname[256];
    snprintf(fname, sizeof(fname), "re_sort_debug_step%05d_%s_ielm.bin", fluid_step, tag);
    FILE* f = fopen(fname, "wb");
    if (!f) {
        fprintf(stderr, "[RE_SORT_DBG] Could not open %s for writing\n", fname);
        return;
    }
    fwrite(&num_particles, sizeof(int), 1, f);
    fwrite(h_i_elm, sizeof(int), num_particles, f);
    fclose(f);
    fprintf(stderr, "[RE_SORT_DBG] Wrote %s (%d particles)\n", fname, num_particles);
}



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

    // Sanity checks
    if(t_norm <= 0.0) {
        if(sim.my_id == 0)
            fprintf(stderr, "[launch_evolve_REs] Error: t_norm must be positive.\n");
        exit(1);
    }

    // --- Compute buffer sizes ---
    const size_t sz_x       = 3 * num_particles * sizeof(double);
    const size_t sz_p       = 3 * num_particles * sizeof(double);
    const size_t sz_st      = 2 * num_particles * sizeof(double);
    const size_t sz_i_elm   = num_particles * sizeof(int);
    const size_t sz_weight  = num_particles * sizeof(double);

    const size_t sz_nl_x      = (size_t)N_COORD_TOR * NDEG * NDIM * n_nodes * sizeof(double);
    const size_t sz_nl_values = (size_t)N_TOR  * NDEG * N_FIELD_VARS * n_nodes * sizeof(double);
    const size_t sz_nl_deltas = (size_t)N_TOR  * NDEG * N_FIELD_VARS * n_nodes * sizeof(double);

    const size_t sz_el_vertex = (size_t)n_elements * NV   * sizeof(int);
    const size_t sz_el_neigh  = (size_t)n_elements * NV   * sizeof(int);
    const size_t sz_el_size   = (size_t)n_elements * NV   * NDEG * sizeof(double);

    const size_t sz_feedback   = (size_t)NDEG * NV * n_elements * N_TOR * NVAR * sizeof(double);
    const size_t sz_mode_coord = N_COORD_TOR * sizeof(int);


    // print dimension of arrays used by kernel
    if(sim.my_id == 0) {
        printf("[Array Dimensions] x: %.2f KB (3 * %zu * %zu)\n", TO_KB(sz_x), (size_t)num_particles, sizeof(double));
        printf("[Array Dimensions] p: %.2f KB (3 * %zu * %zu)\n", TO_KB(sz_p), (size_t)num_particles, sizeof(double));
        printf("[Array Dimensions] st: %.2f KB (2 * %zu * %zu)\n", TO_KB(sz_st), (size_t)num_particles, sizeof(double));
        printf("[Array Dimensions] i_elm: %.2f KB (%zu * %zu)\n", TO_KB(sz_i_elm), (size_t)num_particles, sizeof(int));
        printf("[Array Dimensions] weight: %.2f KB (%zu * %zu)\n", TO_KB(sz_weight), (size_t)num_particles, sizeof(double));
        printf("[Array Dimensions] nl_x: %.2f KB (%d * %d * %d * %zu * %zu)\n", TO_KB(sz_nl_x), N_COORD_TOR, NDEG, NDIM, (size_t)n_nodes, sizeof(double));
        printf("[Array Dimensions] nl_values: %.2f KB (%d * %d * %d * %zu * %zu)\n", TO_KB(sz_nl_values), N_TOR, NDEG, N_FIELD_VARS, (size_t)n_nodes, sizeof(double));
        printf("[Array Dimensions] nl_deltas: %.2f KB (%d * %d * %d * %zu * %zu)\n", TO_KB(sz_nl_deltas), N_TOR, NDEG, N_FIELD_VARS, (size_t)n_nodes, sizeof(double));
        printf("[Array Dimensions] el_vertex: %.2f KB (%zu * %d * %zu)\n", TO_KB(sz_el_vertex), (size_t)n_elements, NV, sizeof(int));
        printf("[Array Dimensions] el_neigh: %.2f KB (%zu * %d * %zu)\n", TO_KB(sz_el_neigh), (size_t)n_elements, NV, sizeof(int));
        printf("[Array Dimensions] el_size: %.2f KB (%zu * %d * %d * %zu)\n", TO_KB(sz_el_size), (size_t)n_elements, NV, NDEG, sizeof(double));
        printf("[Array Dimensions] feedback: %.2f KB (%d * %d * %zu * %d * %d * %zu)\n", TO_KB(sz_feedback), NDEG, NV, (size_t)n_elements, N_TOR, NVAR, sizeof(double));
        printf("[Array Dimensions] mode_coord: %.2f KB (%d * %zu)\n", TO_KB(sz_mode_coord), N_COORD_TOR, sizeof(int));
    }


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

    // Save original allocation handles — d_x_curr/d_weight_curr may drift via swaps
    double *const d_x_orig      = d_x;
    double *const d_p_orig      = d_p;
    double *const d_st_orig     = d_st;
    int    *const d_i_elm_orig  = d_i_elm;
    double *const d_weight_orig = d_weight;

    HIP_CHECK(hipMalloc(&d_nl_x,      sz_nl_x));
    HIP_CHECK(hipMalloc(&d_nl_values, sz_nl_values));
    HIP_CHECK(hipMalloc(&d_nl_deltas, sz_nl_deltas));

    HIP_CHECK(hipMalloc(&d_el_vertex,     sz_el_vertex));
    HIP_CHECK(hipMalloc(&d_el_neighbours, sz_el_neigh));
    HIP_CHECK(hipMalloc(&d_el_size,       sz_el_size));

    HIP_CHECK(hipMalloc(&d_feedback_rhs, sz_feedback));
    HIP_CHECK(hipMalloc(&d_mode_coord,   sz_mode_coord));

    // --- Push double-buffer alt arrays (push reads curr, writes alt; then swap) ---
    double *d_x_push_alt, *d_p_push_alt, *d_st_push_alt;
    int    *d_i_elm_push_alt;
    HIP_CHECK(hipMalloc(&d_x_push_alt,     sz_x));
    HIP_CHECK(hipMalloc(&d_p_push_alt,     sz_p));
    HIP_CHECK(hipMalloc(&d_st_push_alt,    sz_st));
    HIP_CHECK(hipMalloc(&d_i_elm_push_alt, sz_i_elm));
    // Save original handles for these too (they swap with d_x_curr after each step)
    double *const d_x_push_alt_orig     = d_x_push_alt;
    double *const d_p_push_alt_orig     = d_p_push_alt;
    double *const d_st_push_alt_orig    = d_st_push_alt;
    int    *const d_i_elm_push_alt_orig = d_i_elm_push_alt;

    // --- Two streams for concurrent proj/push ---
    hipStream_t stream_proj, stream_push;
    HIP_CHECK(hipStreamCreate(&stream_proj));
    HIP_CHECK(hipStreamCreate(&stream_push));

    // --- HIP graph state (two alternating graphs per sort interval) ---
    hipGraph_t     graph_even = nullptr, graph_odd = nullptr;
    hipGraphExec_t graph_exec_even = nullptr, graph_exec_odd = nullptr;
    bool graph_valid  = false;
    int  interval_step = 0;  // step counter within current sort interval

    // --- Allocate sorting buffers ---
    double *d_x_sorted = nullptr, *d_p_sorted = nullptr, *d_st_sorted = nullptr, *d_weight_sorted = nullptr;
    int    *d_i_elm_sorted = nullptr;
    int    *d_hist = nullptr, *d_offsets = nullptr, *d_cursors = nullptr;
    int    *d_block_sums = nullptr, *d_block_offsets = nullptr;

#if N_SORTING > 0
    HIP_CHECK(hipMalloc(&d_x_sorted,      sz_x));
    HIP_CHECK(hipMalloc(&d_p_sorted,      sz_p));
    HIP_CHECK(hipMalloc(&d_st_sorted,     sz_st));
    HIP_CHECK(hipMalloc(&d_weight_sorted, sz_weight));
    HIP_CHECK(hipMalloc(&d_i_elm_sorted,  sz_i_elm));
    // Save original handles — sort swaps these pointers with d_x_curr etc.
    // just like the primary and push-alt buffers, so free the originals at cleanup.
    double *const d_x_sorted_orig      = d_x_sorted;
    double *const d_p_sorted_orig      = d_p_sorted;
    double *const d_st_sorted_orig     = d_st_sorted;
    double *const d_weight_sorted_orig = d_weight_sorted;
    int    *const d_i_elm_sorted_orig  = d_i_elm_sorted;

    HIP_CHECK(hipMalloc(&d_hist,    I_ELM_BINS * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_offsets, I_ELM_BINS * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cursors, I_ELM_BINS * sizeof(int)));

    int scan_blocks = (I_ELM_BINS + HIST_SCAN_CHUNK - 1) / HIST_SCAN_CHUNK;
    HIP_CHECK(hipMalloc(&d_block_sums, scan_blocks * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_block_offsets, scan_blocks * sizeof(int)));
#endif

    // --- Timing events ---
    hipEvent_t t_start, t_stop;
    HIP_CHECK(hipEventCreate(&t_start));
    HIP_CHECK(hipEventCreate(&t_stop));
    float elapsed_ms = 0.0f;

    // --- Copy host -> device ---
    HIP_CHECK(hipEventRecord(t_start, 0));
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
    HIP_CHECK(hipEventRecord(t_stop, 0));
    HIP_CHECK(hipEventSynchronize(t_stop));
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, t_start, t_stop));
    printf("[launch_evolve_REs rank %d] H2D transfers: %.3f ms\n", sim.my_id, elapsed_ms);

    // --- Launch one kernel per kinetic iteration ---
    int grid_size = (num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double *d_x_curr = d_x, *d_p_curr = d_p, *d_st_curr = d_st, *d_weight_curr = d_weight;
    int    *d_i_elm_curr = d_i_elm;
    int sort_call_count = 0;

    // Helper lambda: capture one HIP graph with proj on stream_proj and push on
    // stream_push, using the given input/output particle pointer pairs.
    // Both kernels are forked from stream_proj and joined back before EndCapture.
    auto capture_graph = [&](
        double* x_in,  double* p_in,  double* st_in,  int* i_elm_in,
        double* x_out, double* p_out, double* st_out, int* i_elm_out,
        hipGraph_t& g, hipGraphExec_t& ge)
    {
        hipEvent_t fev, jev;
        HIP_CHECK(hipEventCreateWithFlags(&fev, hipEventDisableTiming));
        HIP_CHECK(hipEventCreateWithFlags(&jev, hipEventDisableTiming));

        HIP_CHECK(hipStreamBeginCapture(stream_proj, hipStreamCaptureModeGlobal));

        // Fork: stream_push starts from the same graph node as stream_proj
        HIP_CHECK(hipEventRecord(fev, stream_proj));
        HIP_CHECK(hipStreamWaitEvent(stream_push, fev, 0));

        // Proj kernel on stream_proj
        hipLaunchKernelGGL(evolve_proj_kernel,
            dim3(grid_size), dim3(BLOCK_SIZE), 0, stream_proj,
            x_in, p_in, st_in, i_elm_in, d_weight_curr, charge,
            d_nl_values, d_nl_deltas, d_nl_x, n_nodes,
            d_el_vertex, d_el_size, n_elements,
            time_now, time_prev, flag_static, flag_zero_dp,
            F0, t_norm, sim_time, group_mass,
            num_particles,
            d_feedback_rhs, d_mode_coord);

        // Push kernel on stream_push
        hipLaunchKernelGGL(evolve_push_kernel,
            dim3(grid_size), dim3(BLOCK_SIZE), 0, stream_push,
            x_in, p_in, st_in, i_elm_in, charge,
            x_out, p_out, st_out, i_elm_out,
            d_nl_values, d_nl_deltas, d_nl_x, n_nodes,
            d_el_vertex, d_el_neighbours, d_el_size, n_elements,
            time_now, time_prev, flag_static, flag_zero_dp,
            F0, t_norm, sim_time, group_mass, tstep_part_adj,
            num_particles, d_mode_coord);

        // Join: stream_proj waits for stream_push (single exit node for the graph)
        HIP_CHECK(hipEventRecord(jev, stream_push));
        HIP_CHECK(hipStreamWaitEvent(stream_proj, jev, 0));

        HIP_CHECK(hipStreamEndCapture(stream_proj, &g));
        HIP_CHECK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));

        HIP_CHECK(hipEventDestroy(fev));
        HIP_CHECK(hipEventDestroy(jev));
    };

    HIP_CHECK(hipEventRecord(t_start, 0));
    for (int k = 0; k < nstep_particles; ++k) {
#if N_SORTING > 0
        if ((k % N_SORTING) == 0) {
            // Drain both streams before sort so it sees the final push state
            HIP_CHECK(hipStreamSynchronize(stream_push));
            HIP_CHECK(hipStreamSynchronize(stream_proj));

            // Invalidate graphs — pointers change after sort swap
            if (graph_valid) {
                HIP_CHECK(hipGraphExecDestroy(graph_exec_even));
                HIP_CHECK(hipGraphDestroy(graph_even));
                HIP_CHECK(hipGraphExecDestroy(graph_exec_odd));
                HIP_CHECK(hipGraphDestroy(graph_odd));
                graph_exec_even = nullptr; graph_even = nullptr;
                graph_exec_odd  = nullptr; graph_odd  = nullptr;
                graph_valid = false;
            }

            // if(sim.my_id == 0) {
            //     HIP_CHECK(hipMemcpy(part->i_elm,   d_i_elm_curr,   sz_i_elm,    hipMemcpyDeviceToHost));
            //     write_i_elm_snapshot(part->i_elm, num_particles, k/N_SORTING+1, "before_sort");
            // }

            hipEvent_t t_sort_start, t_sort_stop;
            HIP_CHECK(hipEventCreate(&t_sort_start));
            HIP_CHECK(hipEventCreate(&t_sort_stop));
            HIP_CHECK(hipEventRecord(t_sort_start, 0));
            sort_particles_by_i_elm_gpu(
                d_x_curr, d_p_curr, d_st_curr, d_i_elm_curr, d_weight_curr,
                d_x_sorted, d_p_sorted, d_st_sorted, d_i_elm_sorted, d_weight_sorted,
                num_particles, d_hist, d_offsets, d_cursors,
                d_block_sums, d_block_offsets);
            HIP_CHECK(hipEventRecord(t_sort_stop, 0));
            HIP_CHECK(hipEventSynchronize(t_sort_stop));
            float sort_ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&sort_ms, t_sort_start, t_sort_stop));
            ++sort_call_count;
            HIP_CHECK(hipEventDestroy(t_sort_start));
            HIP_CHECK(hipEventDestroy(t_sort_stop));

            // if(sim.my_id == 0) {
            //     HIP_CHECK(hipMemcpy(part->i_elm,   d_i_elm_curr,   sz_i_elm,    hipMemcpyDeviceToHost));
            //     write_i_elm_snapshot(part->i_elm, num_particles, k/N_SORTING+1, "after_sort");
            // }

            interval_step = 0;  // reset parity counter for new sort interval
        }
#endif

        // --- (Re)capture two alternating graphs at the start of each sort interval ---
        // graph_even: curr(P) → alt(Q); graph_odd: curr(Q) → alt(P)
        // Within an interval, d_x_curr and d_x_push_alt alternate P/Q each step.
        if (!graph_valid) {
            if (sim.my_id == 0)
                printf("[launch_evolve_REs rank %d] recapturing graphs at k=%d\n", sim.my_id, k);

            // even graph: input=curr(P), output=alt(Q)
            capture_graph(d_x_curr, d_p_curr, d_st_curr, d_i_elm_curr,
                          d_x_push_alt, d_p_push_alt, d_st_push_alt, d_i_elm_push_alt,
                          graph_even, graph_exec_even);

            // odd graph: input=alt(Q), output=curr(P)
            capture_graph(d_x_push_alt, d_p_push_alt, d_st_push_alt, d_i_elm_push_alt,
                          d_x_curr, d_p_curr, d_st_curr, d_i_elm_curr,
                          graph_odd, graph_exec_odd);

            graph_valid = true;
        }

        // --- Replay the correct graph for this step's parity ---
        hipGraphExec_t cur_exec = (interval_step % 2 == 0) ? graph_exec_even : graph_exec_odd;
        HIP_CHECK(hipGraphLaunch(cur_exec, stream_proj));
        // stream_proj's join_event waits for stream_push, so syncing stream_proj
        // guarantees both proj and push are complete before the pointer swap.
        HIP_CHECK(hipStreamSynchronize(stream_proj));
        HIP_CHECK(hipGetLastError());

        // --- Swap tracking pointers: next step's curr is this step's alt ---
        std::swap(d_x_curr,     d_x_push_alt);
        std::swap(d_p_curr,     d_p_push_alt);
        std::swap(d_st_curr,    d_st_push_alt);
        std::swap(d_i_elm_curr, d_i_elm_push_alt);
        ++interval_step;
    }
    HIP_CHECK(hipEventRecord(t_stop, 0));
    HIP_CHECK(hipEventSynchronize(t_stop));
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, t_start, t_stop));
    printf("[launch_evolve_REs rank %d] nstep_particles loop (%d steps): %.3f ms\n",
           sim.my_id, nstep_particles, elapsed_ms);

    // --- Drain streams before D2H ---
    HIP_CHECK(hipStreamSynchronize(stream_proj));
    HIP_CHECK(hipStreamSynchronize(stream_push));

    // --- Copy results back: device -> host ---
    HIP_CHECK(hipEventRecord(t_start, 0));
    HIP_CHECK(hipMemcpy(part->x,       d_x_curr,       sz_x,        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(part->p,       d_p_curr,       sz_p,        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(part->st,      d_st_curr,      sz_st,       hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(part->i_elm,   d_i_elm_curr,   sz_i_elm,    hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(part->weight,  d_weight_curr,  sz_weight,   hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_feedback_rhs,d_feedback_rhs, sz_feedback, hipMemcpyDeviceToHost));
    HIP_CHECK(hipEventRecord(t_stop, 0));
    HIP_CHECK(hipEventSynchronize(t_stop));
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, t_start, t_stop));
    printf("[launch_evolve_REs rank %d] D2H transfers: %.3f ms\n", sim.my_id, elapsed_ms);

    HIP_CHECK(hipEventDestroy(t_start));
    HIP_CHECK(hipEventDestroy(t_stop));

    // --- Destroy graphs and streams ---
    if (graph_valid) {
        HIP_CHECK(hipGraphExecDestroy(graph_exec_even));
        HIP_CHECK(hipGraphDestroy(graph_even));
        HIP_CHECK(hipGraphExecDestroy(graph_exec_odd));
        HIP_CHECK(hipGraphDestroy(graph_odd));
    }
    HIP_CHECK(hipStreamDestroy(stream_proj));
    HIP_CHECK(hipStreamDestroy(stream_push));

    // --- Free device memory ---
    // Three particle buffer allocations exist:
    //   A = d_x_orig       (primary hipMalloc)
    //   B = d_x_sorted     (sort scratch, #if N_SORTING > 0)
    //   C = d_x_push_alt_orig  (push alt hipMalloc)
    // d_x_curr and d_x_push_alt may point to any of A/B/C after swaps.
    // Always free by the original allocation handles to avoid double-free.
    HIP_CHECK(hipFree(d_x_orig));
    HIP_CHECK(hipFree(d_p_orig));
    HIP_CHECK(hipFree(d_st_orig));
    HIP_CHECK(hipFree(d_i_elm_orig));
    HIP_CHECK(hipFree(d_weight_orig));
    HIP_CHECK(hipFree(d_x_push_alt_orig));
    HIP_CHECK(hipFree(d_p_push_alt_orig));
    HIP_CHECK(hipFree(d_st_push_alt_orig));
    HIP_CHECK(hipFree(d_i_elm_push_alt_orig));
#if N_SORTING > 0
    HIP_CHECK(hipFree(d_x_sorted_orig));
    HIP_CHECK(hipFree(d_p_sorted_orig));
    HIP_CHECK(hipFree(d_st_sorted_orig));
    HIP_CHECK(hipFree(d_i_elm_sorted_orig));
    HIP_CHECK(hipFree(d_weight_sorted_orig));
    HIP_CHECK(hipFree(d_hist));
    HIP_CHECK(hipFree(d_offsets));
    HIP_CHECK(hipFree(d_cursors));
    HIP_CHECK(hipFree(d_block_sums));
    HIP_CHECK(hipFree(d_block_offsets));
#endif
    HIP_CHECK(hipFree(d_nl_x));
    HIP_CHECK(hipFree(d_nl_values));
    HIP_CHECK(hipFree(d_nl_deltas));
    HIP_CHECK(hipFree(d_el_vertex));
    HIP_CHECK(hipFree(d_el_neighbours));
    HIP_CHECK(hipFree(d_el_size));
    HIP_CHECK(hipFree(d_feedback_rhs));
    HIP_CHECK(hipFree(d_mode_coord));
}
