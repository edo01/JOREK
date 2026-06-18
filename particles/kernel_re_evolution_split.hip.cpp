// Split-kernel strategy: proj_stage + proj_accumulate (atomic-free) + evolve_push,
// overlapped on two HIP streams.  Shared device helpers are in
// kernel_re_evolution_common.hip.hpp.
// Active when USE_BATCH_KERNEL is NOT defined (default).
#include "optimization_defines.h"
#ifndef USE_BATCH_KERNEL
#include "particles/kernel_re_evolution_common.hip.hpp"

// ===========================================================================================
//                                  MAIN HIP KERNEL
// ===========================================================================================

// ---------------------------------------------------------------------------
// Per-step particle evolution: projection (proj) + push, one kinetic step each.
//
// Projection is done atomic-free in two kernels:
//   * proj_stage_kernel  (thread-per-particle): compute v_Ppar/v_Pperp/v_jPhi and
//     keep st, phi, weight; write them to global staging arrays.  No feedback writes.
//   * proj_accumulate_kernel (thread-per-(element,cell)): each thread owns one
//     feedback cell and sums the contributions of all particles in that element's
//     contiguous run (bounds from the sort's d_offsets/d_hist), writing the cell ONCE.
// This eliminates every atomicAdd from the projection: there is exactly one writer
// per output cell, and the per-particle contention is replaced by a parallel sum.
//
// Particle arrays layout (Fortran column-major):
//   p_x[j + num_particles*dim], p_p[j + num_particles*dim], p_st[j + num_particles*dim]
//   p_i_elm[j], p_weight[j]
// feedback_rhs is the single compact layout (see fb_idx_compact).
// ---------------------------------------------------------------------------

// proj_stage_kernel (PROJECTION PHASE 1, thread-per-particle): compute the
// per-particle velocity-moment contributions v_Ppar/v_Pperp/v_jPhi (needs the
// B-field interp) and stage them — plus st, phi, weight — to global scratch arrays
// indexed by particle.  Writes NO feedback (that is phase 2).  Lost particles
// (i_elm <= 0) write zero weight so phase 2 can sum them harmlessly.
// Launched before evolve_push_kernel on the same stream, so it reads pre-push state.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(BLOCK_SIZE, 2)
void proj_stage_kernel(
    // Particle SoA — read-only
    const double* __restrict__ p_x,
    const double* __restrict__ p_p,
    const double* __restrict__ p_st,
    const int*    __restrict__ p_i_elm,
    const double* __restrict__ p_weight,
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
    int flag_static,
    // Physics parameters
    double F0, double t_jorek,
    // Simulation parameters
    double sim_time, double group_mass,
    int num_particles,
    // Per-particle staging output (each array length num_particles)
    double* __restrict__ stg_vPpar,
    double* __restrict__ stg_vPperp,
    double* __restrict__ stg_vjPhi,
    double* __restrict__ stg_s,
    double* __restrict__ stg_t,
    double* __restrict__ stg_phi,
    double* __restrict__ stg_w)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_particles) return;

    int i_elm = p_i_elm[j];
    if (i_elm <= 0) {
        // Lost particle: zero weight so phase 2 adds nothing for it.
        stg_w[j] = 0.0;
        return;
    }

    double phi  = p_x[idx2(j, 2, num_particles)];
    double pm[3] = {p_p[idx2(j, 0, num_particles)],  p_p[idx2(j, 1, num_particles)],  p_p[idx2(j, 2, num_particles)]};
    double st[2] = {p_st[idx2(j, 0, num_particles)], p_st[idx2(j, 1, num_particles)]};
    double Rcyl  = p_x[idx2(j, 0, num_particles)];
    double w     = p_weight[j];

    double cyl_mom[3];
    vector_cartesian_to_cylindrical(phi, pm, cyl_mom);
    double pdot_cyl = cyl_mom[0]*cyl_mom[0] + cyl_mom[1]*cyl_mom[1] + cyl_mom[2]*cyl_mom[2];
    double inv_denom_v = 1.0 / sqrt(pdot_cyl / (SPEED_OF_LIGHT*SPEED_OF_LIGHT) + group_mass*group_mass);
    double cyl_vel[3] = {cyl_mom[0] * inv_denom_v, cyl_mom[1] * inv_denom_v, cyl_mom[2] * inv_denom_v};

    double B_loc[3];
    calc_B_only(nl_values, nl_deltas, nl_x, el_vertex, el_size,
                n_elements, n_nodes,
                F0,
                time_now, time_prev, t_jorek, flag_static,
                i_elm, st, phi, sim_time,
                B_loc);

    double Bnorm_inv = 1.0 / sqrt(B_loc[0]*B_loc[0] + B_loc[1]*B_loc[1] + B_loc[2]*B_loc[2]);
    double B_hat[3] = {B_loc[0]*Bnorm_inv, B_loc[1]*Bnorm_inv, B_loc[2]*Bnorm_inv};

    double v_par = cyl_vel[0]*B_hat[0] + cyl_vel[1]*B_hat[1] + cyl_vel[2]*B_hat[2];
    double v_perp_diff[3] = {cyl_vel[0] - v_par*B_hat[0],
                             cyl_vel[1] - v_par*B_hat[1],
                             cyl_vel[2] - v_par*B_hat[2]};
    double v_perp_sq = v_perp_diff[0]*v_perp_diff[0] + v_perp_diff[1]*v_perp_diff[1] + v_perp_diff[2]*v_perp_diff[2];

    double gamma_m = sqrt(MASS_ELECTRON*MASS_ELECTRON
                        + pdot_cyl * ATOMIC_MASS_UNIT*ATOMIC_MASS_UNIT
                          / (SPEED_OF_LIGHT*SPEED_OF_LIGHT));

    stg_vPpar[j]  = gamma_m * v_par * v_par * MU_ZERO;
    stg_vPperp[j] = gamma_m * v_perp_sq * 0.5 * MU_ZERO;
    stg_vjPhi[j]  = -double(charge) * EL_CHG * cyl_vel[2] * Rcyl * MU_ZERO;
    stg_s[j]      = st[0];
    stg_t[j]      = st[1];
    stg_phi[j]    = phi;
    stg_w[j]      = w;
}

// Tile size for proj_accumulate_kernel: number of particles whose per-particle
// factor vectors are cached in shared memory at once.  Shared cost per tile =
// PROJ_TILE * (NDEG*NV + N_TOR + NVAR) doubles (e.g. 64 * 26 * 8 = 13 KB at N_TOR=7).
// PROJ_TILE, ACCUM_BLOCK_SIZE and ACCUM_MIN_BLOCKS_PER_SM are sweepable knobs
// defined (with defaults) in optimization_defines.h.

// __launch_bounds__ for proj_accumulate: include the min-blocks-per-SM hint only
// when ACCUM_MIN_BLOCKS_PER_SM > 0 (0 = let ptxas choose registers freely).
#if ACCUM_MIN_BLOCKS_PER_SM > 0
#define ACCUM_LAUNCH_BOUNDS __launch_bounds__(ACCUM_BLOCK_SIZE, ACCUM_MIN_BLOCKS_PER_SM)
#else
#define ACCUM_LAUNCH_BOUNDS __launch_bounds__(ACCUM_BLOCK_SIZE)
#endif

// proj_accumulate_kernel (PROJECTION PHASE 2, BLOCK-PER-ELEMENT):
// One block owns one element ie and computes all PROJ_CELLS_PER_ELM feedback cells
// for it, atomic-free.  The projection factorizes per particle into an outer product
//   F[n,m,it,var] = sz(n,m) * Σ_p  bf2D(s_p,t_p)[n,m] * hz(phi_p)[it] * (v_var(p) w_p)
// so we precompute each particle's small factor vectors ONCE (bf2D: NDEG*NV, hz: N_TOR,
// vw: NVAR) into shared memory, then each thread (owning a few cells) sums the cheap
// FMA `bf[n,m]*hz[it]*vw[var]` over the run reading only shared memory — no bf2D/sincos
// in the inner loop, no re-reading global per cell.  The run is tiled by PROJ_TILE.
//
// Each cell is written by exactly one thread => no atomics.  feedback_rhs is
// accumulated across steps, so we add this step's contribution to the existing value.
// ---------------------------------------------------------------------------
__global__ ACCUM_LAUNCH_BOUNDS
void proj_accumulate_kernel(
    const int*    __restrict__ elm_offset,   // d_offsets: run start per element bin
    const int*    __restrict__ elm_count,    // d_hist:    run length per element bin
    const double* __restrict__ el_size,
    int n_elements,
    int num_particles,
    const double* __restrict__ stg_vPpar,
    const double* __restrict__ stg_vPperp,
    const double* __restrict__ stg_vjPhi,
    const double* __restrict__ stg_s,
    const double* __restrict__ stg_t,
    const double* __restrict__ stg_phi,
    const double* __restrict__ stg_w,
    double* __restrict__ feedback_rhs)
{
    const int ie = blockIdx.x;                 // one block per element
    if (ie >= n_elements) return;

    const int start = elm_offset[ie];
    const int cnt   = elm_count[ie];

    // Shared factor cache for the current particle tile.
    //
    // Bank-conflict avoidance (phase-a cooperative stores): in phase (a) each
    // tile particle is owned by one lane (consecutive q across a warp) and that
    // lane writes its whole row.  For a fixed column the per-lane address stride
    // equals the row length, so a row length that shares a factor with the 32
    // banks serializes the store (NDEG*NV = 16 → 16-way conflict on every bf
    // store).  Padding each row's leading dimension to an *odd* width (coprime
    // with 32) makes the store stride coprime to the bank count, so the 32 lanes
    // hit 32 distinct banks → conflict-free.  The phase-(b) reads are at a fixed
    // q (loop-uniform row) with the column varying per lane, which is a
    // broadcast pattern and stays conflict-free under either width.
    static constexpr int SH_BF_W = (NDEG * NV) | 1;  // 17: odd ⇒ coprime to 32
    static constexpr int SH_HZ_W = (N_TOR) | 1;      // always odd
    static constexpr int SH_VW_W = (NVAR) | 1;       // 3 already odd
    __shared__ double sh_bf[PROJ_TILE][SH_BF_W];  // bf2D_0_scalar(s,t,n,m)
    __shared__ double sh_hz[PROJ_TILE][SH_HZ_W];  // toroidal harmonics hz(phi,it)
    __shared__ double sh_vw[PROJ_TILE][SH_VW_W];  // {vPpar,vPperp,vjPhi} * weight

    // Each thread owns a fixed subset of the PROJ_CELLS_PER_ELM cells (grid-stride),
    // carrying a register accumulator per owned cell across all tiles.
    // PROJ_CELLS_PER_ELM (=336 at N_TOR=7); cells/thread = ceil(it / ACCUM_BLOCK_SIZE).
    constexpr int CELLS_PER_THREAD = (PROJ_CELLS_PER_ELM + ACCUM_BLOCK_SIZE - 1) / ACCUM_BLOCK_SIZE;
    double acc[CELLS_PER_THREAD];
    #pragma unroll
    for (int r = 0; r < CELLS_PER_THREAD; ++r) acc[r] = 0.0;

    if (cnt > 0) {
        for (int base = 0; base < cnt; base += PROJ_TILE) {
            int tile = min(PROJ_TILE, cnt - base);

            // Phase (a): cooperatively precompute the tile's per-particle factors.
            // One thread per (tile particle) does the bf2D/sincos work ONCE.
            for (int q = threadIdx.x; q < tile; q += blockDim.x) {
                int p = start + base + q;
                double w = stg_w[p];
                double s = stg_s[p], tt = stg_t[p], phi = stg_phi[p];
                // bf2D for all (n,m): layout index n*NV + m.
                #pragma unroll
                for (int n = 0; n < NDEG; ++n)
                    #pragma unroll
                    for (int m = 0; m < NV; ++m)
                        sh_bf[q][n * NV + m] = bf2D_0_scalar(s, tt, n, m);
                // hz for all it via mode_moivre recurrence.
                sh_hz[q][0] = 1.0;
                #pragma unroll
                for (int i = 1; i <= NMODE; ++i) {
                    double c, sn; sincos(double(N_PERIOD * i) * phi, &sn, &c);
                    sh_hz[q][2*i - 1] = c;
                    sh_hz[q][2*i]     = sn;
                }
                sh_vw[q][P_PAR_IDX]  = stg_vPpar[p]  * w;   // w==0 for lost -> contributes 0
                sh_vw[q][P_PERP_IDX] = stg_vPperp[p] * w;
                sh_vw[q][J_PHI_IDX]  = stg_vjPhi[p]  * w;
            }
            __syncthreads();

            // Phase (b): each cell-thread sums this tile's contribution from shared.
            #pragma unroll
            for (int r = 0; r < CELLS_PER_THREAD; ++r) {
                int c = threadIdx.x + r * ACCUM_BLOCK_SIZE;
                if (c >= PROJ_CELLS_PER_ELM) break;
                // Decode c = var + NVAR*(it + N_TOR*(m + NV*n)).
                int t = c;
                int var = t % NVAR;  t /= NVAR;
                int it  = t % N_TOR; t /= N_TOR;
                int m   = t % NV;    t /= NV;
                int n   = t;
                int nm  = n * NV + m;
                double s = 0.0;
                for (int q = 0; q < tile; ++q)
                    s += sh_bf[q][nm] * sh_hz[q][it] * sh_vw[q][var];
                acc[r] += s;
            }
            __syncthreads();  // tile factors consumed; safe to overwrite next tile
        }
    }

    // Write each owned cell once, scaling by the per-(n,m,ie) el_size factor.
    #pragma unroll
    for (int r = 0; r < CELLS_PER_THREAD; ++r) {
        int c = threadIdx.x + r * ACCUM_BLOCK_SIZE;
        if (c >= PROJ_CELLS_PER_ELM) break;
        int t = c;
        int var = t % NVAR;  t /= NVAR;
        int it  = t % N_TOR; t /= N_TOR;
        int m   = t % NV;    t /= NV;
        int n   = t;
        double sz = __ldg(&el_size[el_size_idx(n, m, ie, n_elements)]);
        feedback_rhs[fb_idx_compact(ie, n, m, it, var, n_elements)] += acc[r] * sz;
    }
}

// ---------------------------------------------------------------------------
// evolve_push_kernel: push phase only, one kinetic step.
// Reads particle state from the _in (current) buffers and writes the advanced
// state to the _out (alternate) buffers; does NOT touch feedback_rhs.  Writes
// _out unconditionally so the alternate buffer is always a complete copy
// (including lost particles).  Launched on stream_push so it overlaps the two
// projection kernels (proj_stage/proj_accumulate), which read the same _in
// state on stream_proj — the separate output buffer removes the read/write
// hazard that an in-place push would create against proj_stage's reads.
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
    double F0, double t_norm, double t_jorek,
    // Simulation parameters
    double sim_time, double group_mass, double tstep_part_adj,
    int num_particles,
    const int* __restrict__ mode_coord)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_particles) return;

    double x[3]  = {p_x_in[idx2(j, 0, num_particles)],  p_x_in[idx2(j, 1, num_particles)],  p_x_in[idx2(j, 2, num_particles)]};
    double pm[3] = {p_p_in[idx2(j, 0, num_particles)],  p_p_in[idx2(j, 1, num_particles)],  p_p_in[idx2(j, 2, num_particles)]};
    double st[2] = {p_st_in[idx2(j, 0, num_particles)], p_st_in[idx2(j, 1, num_particles)]};
    int    i_elm = p_i_elm_in[j];

    if (i_elm > 0) {
        int ifail = 0;
        volume_preserving_push(x, pm, st, i_elm, charge,
                               nl_values, nl_deltas, nl_x,
                               el_vertex, el_size, el_neighbours,
                               n_elements, n_nodes, mode_coord,
                               time_now, time_prev,
                               flag_static, flag_zero_dpsidt,
                               F0, t_norm, t_jorek,
                               group_mass, sim_time, tstep_part_adj,
                               ifail);
    }

    // Always write to output buffers (captures lost-particle state too).
    p_x_out[idx2(j, 0, num_particles)]  = x[0];  p_x_out[idx2(j, 1, num_particles)]  = x[1];  p_x_out[idx2(j, 2, num_particles)]  = x[2];
    p_p_out[idx2(j, 0, num_particles)]  = pm[0]; p_p_out[idx2(j, 1, num_particles)]  = pm[1]; p_p_out[idx2(j, 2, num_particles)]  = pm[2];
    p_st_out[idx2(j, 0, num_particles)] = st[0]; p_st_out[idx2(j, 1, num_particles)] = st[1];
    p_i_elm_out[j] = i_elm;
}


// ---------------------------------------------------------------------------

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
    const double t_jorek        = sim.fields.t_jorek;

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

    // Single (non-replicated) feedback buffer; proj phase 2 writes each cell once.
    const size_t sz_feedback   = (size_t)NDEG * NV * n_elements * N_TOR * NVAR * sizeof(double);
    const size_t sz_mode_coord = N_COORD_TOR * sizeof(int);

    // Per-particle projection staging arrays (filled by proj_stage_kernel, summed by
    // proj_accumulate_kernel): v_Ppar, v_Pperp, v_jPhi, s, t, phi, weight.
    const size_t sz_stage = (size_t)num_particles * sizeof(double);

#if GPU_DEBUG == 1
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
        printf("[Array Dimensions] feedback: %.2f KB ; proj staging (7x): %.2f KB\n",
               TO_KB(sz_feedback), TO_KB(7 * sz_stage));
#if NODES_FIRST == 1
        printf("[Layout] NODES_FIRST=1   : nl_x/values/deltas have n_nodes as fastest dim\n");
#else
        printf("[Layout] NODES_FIRST=0   : nl_x/values/deltas have n_nodes as slowest dim\n");
#endif
#if ELEMENTS_FIRST == 1
        printf("[Layout] ELEMENTS_FIRST=1: el_vertex/neigh/size have n_elements as fastest dim\n");
#else
        printf("[Layout] ELEMENTS_FIRST=0: el_vertex/neigh/size have n_elements as slowest dim\n");
#endif
#if FB_ELEMENTS_FIRST == 1
        printf("[Layout] FB_ELEMENTS_FIRST=1: feedback_rhs has n_elements as fastest dim\n");
#else
        printf("[Layout] FB_ELEMENTS_FIRST=0: feedback_rhs has n_elements as slowest dim (Fortran column-major)\n");
#endif
        printf("[Array Dimensions] mode_coord: %.2f KB (%d * %zu)\n", TO_KB(sz_mode_coord), N_COORD_TOR, sizeof(int));
    }
#endif

    // --- Allocate device memory ---
    double *d_x, *d_p, *d_st, *d_weight;
    int    *d_i_elm;
    double *d_nl_x, *d_nl_values, *d_nl_deltas;
    int    *d_el_vertex, *d_el_neighbours;
    double *d_el_size;
    double *d_feedback_rhs;       // single accumulation buffer returned to host
    int    *d_mode_coord;
    // Projection staging arrays (per particle)
    double *d_stg_vPpar, *d_stg_vPperp, *d_stg_vjPhi, *d_stg_s, *d_stg_t, *d_stg_phi, *d_stg_w;

    
    int n_devices;
    HIP_CHECK(hipGetDeviceCount(&n_devices));
    if(sim.my_id == 0)
        printf("[launch_evolve_REs] HIP Device count: %d\n", n_devices);
    HIP_CHECK(hipSetDevice(sim.my_id % n_devices)); // Ensure we are on the correct GPU before allocating memory
    int curr_dev;
    HIP_CHECK(hipGetDevice(&curr_dev));
    printf("[launch_evolve_REs] MPI process %d (global rank) using device=%d\n", sim.my_id, curr_dev);


    // Prefer L1 cache over shared memory for kernels that do not use shared memory.
    // count_i_elm_histogram and exclusive_scan_blocks use shared memory and are excluded.
    static bool cache_config_set = false;
    if (!cache_config_set) {
        HIP_CHECK(hipFuncSetCacheConfig(reinterpret_cast<const void*>(proj_stage_kernel), hipFuncCachePreferL1));
        HIP_CHECK(hipFuncSetCacheConfig(reinterpret_cast<const void*>(proj_accumulate_kernel), hipFuncCachePreferL1));
        HIP_CHECK(hipFuncSetCacheConfig(reinterpret_cast<const void*>(evolve_push_kernel), hipFuncCachePreferL1));
        cache_config_set = true;
    }

    HIP_CHECK(hipMalloc(&d_x,       sz_x));
    HIP_CHECK(hipMalloc(&d_p,       sz_p));
    HIP_CHECK(hipMalloc(&d_st,      sz_st));
    HIP_CHECK(hipMalloc(&d_i_elm,   sz_i_elm));
    HIP_CHECK(hipMalloc(&d_weight,  sz_weight));

    // Save original allocation handles — sort may swap d_x/d_weight with sorted buffers
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

    HIP_CHECK(hipMalloc(&d_feedback_rhs,     sz_feedback));
    HIP_CHECK(hipMalloc(&d_mode_coord,       sz_mode_coord));

    HIP_CHECK(hipMalloc(&d_stg_vPpar,  sz_stage));
    HIP_CHECK(hipMalloc(&d_stg_vPperp, sz_stage));
    HIP_CHECK(hipMalloc(&d_stg_vjPhi,  sz_stage));
    HIP_CHECK(hipMalloc(&d_stg_s,      sz_stage));
    HIP_CHECK(hipMalloc(&d_stg_t,      sz_stage));
    HIP_CHECK(hipMalloc(&d_stg_phi,    sz_stage));
    HIP_CHECK(hipMalloc(&d_stg_w,      sz_stage));


    // --- Allocate the alternate ("sorted") particle buffer set ---
    // This single alternate set is reused for two roles, never simultaneously:
    //   1. the counting sort's scatter destination (sort then swaps it with curr), and
    //   2. the push kernel's write-only _out double buffer (push then swaps with curr).
    // We do NOT allocate a third particle-buffer set: the sort only runs while both
    // streams are drained, so push and sort never contend for this set.
    double *d_x_sorted = nullptr, *d_p_sorted = nullptr, *d_st_sorted = nullptr, *d_weight_sorted = nullptr;
    int    *d_i_elm_sorted = nullptr;
    int    *d_hist = nullptr, *d_offsets = nullptr, *d_cursors = nullptr;
    int    *d_block_sums = nullptr, *d_block_offsets = nullptr;

    HIP_CHECK(hipMalloc(&d_x_sorted,      sz_x));
    HIP_CHECK(hipMalloc(&d_p_sorted,      sz_p));
    HIP_CHECK(hipMalloc(&d_st_sorted,     sz_st));
    HIP_CHECK(hipMalloc(&d_weight_sorted, sz_weight));
    HIP_CHECK(hipMalloc(&d_i_elm_sorted,  sz_i_elm));
    // Save original handles — the sort swaps these pointers with d_x etc., so free
    // the originals at cleanup to avoid double-free regardless of swap state.
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

    // --- Two streams + fork/join events for concurrent proj/push ---
    // The two projection kernels (proj_stage -> proj_accumulate) run on stream_proj
    // and the push runs on stream_push.  Both branches read the same post-sort `curr`
    // state and write disjoint outputs (proj -> feedback_rhs / staging, push -> alt),
    // so they may run concurrently.  The fork/join event pair serialises consecutive
    // steps on the GPU without a per-step host sync.
    hipStream_t stream_proj, stream_push;
    HIP_CHECK(hipStreamCreate(&stream_proj));
    HIP_CHECK(hipStreamCreate(&stream_push));
    hipEvent_t fork_event, join_event;
    HIP_CHECK(hipEventCreateWithFlags(&fork_event, hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&join_event, hipEventDisableTiming));

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

    // feedback_rhs is accumulated over the kinetic steps, so zero it once up front.
    HIP_CHECK(hipMemset(d_feedback_rhs, 0, sz_feedback));
    HIP_CHECK(hipMemcpy(d_mode_coord,   sim.fields.mode_coord, sz_mode_coord, hipMemcpyHostToDevice));
    HIP_CHECK(hipEventRecord(t_stop, 0));
    HIP_CHECK(hipEventSynchronize(t_stop));
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, t_start, t_stop));
    const float h2d_ms = elapsed_ms;
#if GPU_DEBUG == 1
    if (sim.my_id == 0)
        printf("[launch_evolve_REs rank %d] H2D transfers: %.3f ms\n", sim.my_id, h2d_ms);
#endif

    // --- Step loop: one kinetic step per iteration, proj + push overlapped ---
    // proj (stream_proj): proj_stage reads `curr` and writes the staging arrays,
    //   then proj_accumulate sums them into feedback_rhs (atomic-free).
    // push (stream_push): reads `curr` and writes the advanced state to `alt`.
    // Both branches read the same post-sort input and write disjoint outputs, so they
    // run concurrently; the fork/join event pair serialises consecutive steps on the
    // GPU without per-step host sync.  After each step `alt` becomes `curr` via a
    // host-side pointer swap.
    //
    // The counting sort runs at the start of every step; it scatters `curr` into `alt`
    // and swaps, so afterwards `curr` is sorted and `alt` is free scratch — exactly what
    // push needs as its write-only output.  Sorting every step keeps the element-run
    // layout (d_offsets/d_hist) that proj_accumulate consumes always fresh.  The sort
    // runs on the default (legacy) stream while both streams are drained, so sort and
    // push never contend for `alt`.
    int grid_size = (num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int sort_call_count = 0;

    // Running buffer pointers: `curr` holds the latest state, `alt` is the spare set
    // (push output / sort scratch).  Both follow the pointer swaps below.
    double *d_x_curr = d_x, *d_p_curr = d_p, *d_st_curr = d_st, *d_weight_curr = d_weight;
    int    *d_i_elm_curr = d_i_elm;
    double *d_x_alt = d_x_sorted, *d_p_alt = d_p_sorted, *d_st_alt = d_st_sorted, *d_weight_alt = d_weight_sorted;
    int    *d_i_elm_alt = d_i_elm_sorted;

    // Phase-2 (accumulate) grid: one block (ACCUM_BLOCK_SIZE threads) per element.
    int accum_grid = n_elements;

#if GPU_DEBUG == 1
    float total_sort_ms = 0.0f;
#endif

    HIP_CHECK(hipEventRecord(t_start, 0));
    for (int k = 0; k < nstep_particles; ++k) {
        // Drain both streams so the sort sees the final push output from step k-1.
        HIP_CHECK(hipStreamSynchronize(stream_push));
        HIP_CHECK(hipStreamSynchronize(stream_proj));
#if GPU_DEBUG == 1
        hipEvent_t t_sort_start, t_sort_stop;
        HIP_CHECK(hipEventCreate(&t_sort_start));
        HIP_CHECK(hipEventCreate(&t_sort_stop));
        HIP_CHECK(hipEventRecord(t_sort_start, 0));
#endif
        // Sort scatters curr -> spare set, then swaps the pointers internally, so on
        // return d_*_curr is sorted and the spare set is free scratch again.  It also
        // leaves d_offsets[ie] (run start) and d_hist[ie] (run length) for phase 2.
        sort_particles_by_i_elm_gpu(
            d_x_curr, d_p_curr, d_st_curr, d_i_elm_curr, d_weight_curr,
            d_x_alt, d_p_alt, d_st_alt, d_i_elm_alt, d_weight_alt,
            num_particles, d_hist, d_offsets, d_cursors,
            d_block_sums, d_block_offsets);
        ++sort_call_count;
#if GPU_DEBUG == 1
        HIP_CHECK(hipEventRecord(t_sort_stop, 0));
        HIP_CHECK(hipEventSynchronize(t_sort_stop));
        float sort_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&sort_ms, t_sort_start, t_sort_stop));
        total_sort_ms += sort_ms;
        HIP_CHECK(hipEventDestroy(t_sort_start));
        HIP_CHECK(hipEventDestroy(t_sort_stop));
#endif

        // Fork: stream_push waits until stream_proj reaches this point.  stream_proj
        // carries the join_event from step k-1, so stream_push cannot start step k
        // until both branches of step k-1 have finished.
        HIP_CHECK(hipEventRecord(fork_event, stream_proj));
        HIP_CHECK(hipStreamWaitEvent(stream_push, fork_event, 0));

        // Projection phase 1 (thread-per-particle): compute v_* moments + stage st/phi/w.
        // Reads curr (pre-push state); writes only the staging arrays.  On stream_proj.
        hipLaunchKernelGGL(proj_stage_kernel,
            dim3(grid_size), dim3(BLOCK_SIZE), 0, stream_proj,
            d_x_curr, d_p_curr, d_st_curr, d_i_elm_curr, d_weight_curr, charge,
            d_nl_values, d_nl_deltas, d_nl_x, n_nodes,
            d_el_vertex, d_el_size, n_elements,
            time_now, time_prev, flag_static,
            F0, t_jorek, sim_time, group_mass,
            num_particles,
            d_stg_vPpar, d_stg_vPperp, d_stg_vjPhi, d_stg_s, d_stg_t, d_stg_phi, d_stg_w);

        // Projection phase 2 (block-per-element): precompute per-particle factors in
        // shared, sum each element's run into feedback_rhs once per cell — no atomics.
        // On stream_proj, after proj_stage (consumes its staging output).
        hipLaunchKernelGGL(proj_accumulate_kernel,
            dim3(accum_grid), dim3(ACCUM_BLOCK_SIZE), 0, stream_proj,
            d_offsets, d_hist, d_el_size, n_elements, num_particles,
            d_stg_vPpar, d_stg_vPperp, d_stg_vjPhi, d_stg_s, d_stg_t, d_stg_phi, d_stg_w,
            d_feedback_rhs);
        HIP_CHECK(hipGetLastError());

        // Push kernel: reads curr, writes advanced state to alt (concurrent with proj).
        hipLaunchKernelGGL(evolve_push_kernel,
            dim3(grid_size), dim3(BLOCK_SIZE), 0, stream_push,
            d_x_curr, d_p_curr, d_st_curr, d_i_elm_curr, charge,
            d_x_alt, d_p_alt, d_st_alt, d_i_elm_alt,
            d_nl_values, d_nl_deltas, d_nl_x, n_nodes,
            d_el_vertex, d_el_neighbours, d_el_size, n_elements,
            time_now, time_prev, flag_static, flag_zero_dp,
            F0, t_norm, t_jorek, sim_time, group_mass, tstep_part_adj,
            num_particles, d_mode_coord);
        HIP_CHECK(hipGetLastError());

        // Join: stream_proj waits for push to finish before the next step's fork.
        HIP_CHECK(hipEventRecord(join_event, stream_push));
        HIP_CHECK(hipStreamWaitEvent(stream_proj, join_event, 0));

        // Swap: alt (the just-written push output) becomes curr for the next step.
        // weight is read-only and not pushed, so it is NOT swapped here — it stays with
        // curr and is re-permuted (and re-paired with the other arrays) by next step's sort.
        std::swap(d_x_curr,     d_x_alt);
        std::swap(d_p_curr,     d_p_alt);
        std::swap(d_st_curr,    d_st_alt);
        std::swap(d_i_elm_curr, d_i_elm_alt);
    }
    // Drain both streams before the timing stop / reduction / D2H copies.
    HIP_CHECK(hipStreamSynchronize(stream_proj));
    HIP_CHECK(hipStreamSynchronize(stream_push));
    HIP_CHECK(hipEventRecord(t_stop, 0));
    HIP_CHECK(hipEventSynchronize(t_stop));
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, t_start, t_stop));
#if GPU_DEBUG == 1
    if (sim.my_id == 0) {
        const float loop_ms = elapsed_ms;
        printf("[launch_evolve_REs rank %d] nstep_particles loop (%d steps, %d sorts): %.3f ms\n",
               sim.my_id, nstep_particles, sort_call_count, loop_ms);
        printf("[launch_evolve_REs rank %d] === PHASE BREAKDOWN (step loop) ===\n", sim.my_id);
        printf("[launch_evolve_REs rank %d]   proj+push (overlapped): %9.3f ms  (%.1f%%)\n",
               sim.my_id, loop_ms - total_sort_ms, 100.0 * (loop_ms - total_sort_ms) / loop_ms);
        printf("[launch_evolve_REs rank %d]   sort_particles        : %9.3f ms  (%.1f%%)\n",
               sim.my_id, total_sort_ms, 100.0 * total_sort_ms / loop_ms);
    }
#endif

    // (Projection writes feedback_rhs directly — no lane reduction needed.)

    // --- Copy results back: device -> host ---
#if GPU_DEBUG == 1
    HIP_CHECK(hipEventRecord(t_start, 0));
#endif
    // Copy from the final `curr` pointers (they may have drifted via swaps).
    HIP_CHECK(hipMemcpy(part->x,       d_x_curr,       sz_x,        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(part->p,       d_p_curr,       sz_p,        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(part->st,      d_st_curr,      sz_st,       hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(part->i_elm,   d_i_elm_curr,   sz_i_elm,    hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(part->weight,  d_weight_curr,  sz_weight,   hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_feedback_rhs,d_feedback_rhs,    sz_feedback, hipMemcpyDeviceToHost));
#if GPU_DEBUG == 1
    HIP_CHECK(hipEventRecord(t_stop, 0));
    HIP_CHECK(hipEventSynchronize(t_stop));
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, t_start, t_stop));
    const float d2h_ms = elapsed_ms;
    if (sim.my_id == 0)
        printf("[launch_evolve_REs rank %d] D2H transfers: %.3f ms\n", sim.my_id, d2h_ms);
#endif

    HIP_CHECK(hipEventDestroy(t_start));
    HIP_CHECK(hipEventDestroy(t_stop));

    // --- Destroy proj/push streams and fork/join events ---
    HIP_CHECK(hipStreamDestroy(stream_proj));
    HIP_CHECK(hipStreamDestroy(stream_push));
    HIP_CHECK(hipEventDestroy(fork_event));
    HIP_CHECK(hipEventDestroy(join_event));

#if GPU_DEBUG == 1
    // --- Consolidated whole-call phase breakdown ---
    if (sim.my_id == 0) {
        const float loop_ms     = elapsed_ms;  // whole step loop (stage+accumulate+push + sort)
        const float evolve_ms   = loop_ms - total_sort_ms;
        const float total_ms    = h2d_ms + loop_ms + d2h_ms;
        printf("[launch_evolve_REs rank %d] ===== WHOLE-CALL PHASE BREAKDOWN =====\n", sim.my_id);
        printf("[launch_evolve_REs rank %d]   H2D transfers         : %9.3f ms  (%.1f%%)\n",
               sim.my_id, h2d_ms,        100.0 * h2d_ms        / total_ms);
        printf("[launch_evolve_REs rank %d]   proj+push             : %9.3f ms  (%.1f%%)\n",
               sim.my_id, evolve_ms,     100.0 * evolve_ms     / total_ms);
        printf("[launch_evolve_REs rank %d]   sort_particles        : %9.3f ms  (%.1f%%)\n",
               sim.my_id, total_sort_ms, 100.0 * total_sort_ms / total_ms);
        printf("[launch_evolve_REs rank %d]   D2H transfers         : %9.3f ms  (%.1f%%)\n",
               sim.my_id, d2h_ms,        100.0 * d2h_ms        / total_ms);
        printf("[launch_evolve_REs rank %d]   ---------------------------------------\n", sim.my_id);
        printf("[launch_evolve_REs rank %d]   TOTAL (timed phases)  : %9.3f ms\n", sim.my_id, total_ms);
    }
#endif


    // --- Free device memory ---
    // Free the original allocation handles, not the swapped curr/alt pointers.
    HIP_CHECK(hipFree(d_x_orig));
    HIP_CHECK(hipFree(d_p_orig));
    HIP_CHECK(hipFree(d_st_orig));
    HIP_CHECK(hipFree(d_i_elm_orig));
    HIP_CHECK(hipFree(d_weight_orig));
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
    HIP_CHECK(hipFree(d_nl_x));
    HIP_CHECK(hipFree(d_nl_values));
    HIP_CHECK(hipFree(d_nl_deltas));
    HIP_CHECK(hipFree(d_el_vertex));
    HIP_CHECK(hipFree(d_el_neighbours));
    HIP_CHECK(hipFree(d_el_size));
    HIP_CHECK(hipFree(d_feedback_rhs));
    HIP_CHECK(hipFree(d_mode_coord));
    HIP_CHECK(hipFree(d_stg_vPpar));
    HIP_CHECK(hipFree(d_stg_vPperp));
    HIP_CHECK(hipFree(d_stg_vjPhi));
    HIP_CHECK(hipFree(d_stg_s));
    HIP_CHECK(hipFree(d_stg_t));
    HIP_CHECK(hipFree(d_stg_phi));
    HIP_CHECK(hipFree(d_stg_w));
}

#endif /* !USE_BATCH_KERNEL */
