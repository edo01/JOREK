// Batch-kernel strategy: fused evolve_batch_kernel runs nsteps in one launch,
// particle state held in registers across steps; optional shared-memory LUT for
// nl_values/nl_deltas.  Shared device helpers are in
// kernel_re_evolution_common.hip.hpp.
// Active when USE_BATCH_KERNEL is defined in optimization_defines.h.
#include "optimization_defines.h"
#ifdef USE_BATCH_KERNEL
#include "particles/kernel_re_evolution_common.hip.hpp"

#if LUT_VALUES_DELTAS == 1
// Shared-memory LUT for nl_values / nl_deltas in calc_EBpsiU.
// SLOT_SIZE: number of doubles cached per element (NV * NDEG * 2 * N_TOR).
// Tuning knobs (LUT_N_SLOTS, LUT_NEIGHBOR_PRELOAD) come from optimization_defines.h.
static constexpr int LUT_SLOT_SIZE = NV * NDEG * 2 * N_TOR;  // = 32 * N_TOR
// Transposed layout: sh_cache[lid * LUT_N_SLOTS + slot] instead of sh_cache[slot * LUT_SLOT_SIZE + lid].
// At the read site all threads in a warp share the same lid but have different slots,
// so the transposed layout makes them access consecutive addresses → no bank conflicts.

// Occupancy guard: the two caches (sh_cache_v + sh_cache_d) cost LUT_SLOT_SIZE*2*8 =
// 512*N_TOR bytes per slot.  The kernel targets 2 blocks/CU (__launch_bounds__(,2)),
// so per-block LDS must stay within ~half the CU's 64 KB.  Budget the cache at ~28 KB
// to leave room for sh_scratch and compiler-reserved LDS.  If this fires, reduce
// LUT_N_SLOTS or N_TOR (the caches scale linearly with both).
// The "* 2" is the two separate caches (sh_cache_v for nl_values, sh_cache_d for
// nl_deltas), each LUT_SLOT_SIZE doubles per slot — NOT the N_FIELD_VARS factor that
// already lives inside LUT_SLOT_SIZE.
static constexpr int LUT_CACHE_BYTES_PER_SLOT = 2 /*v+d caches*/ * LUT_SLOT_SIZE * (int)sizeof(double);
static_assert(LUT_N_SLOTS * LUT_CACHE_BYTES_PER_SLOT <= 32 * 1024,
              "LUT cache too large for 2 blocks/CU: reduce LUT_N_SLOTS or N_TOR");

// LUT_DEBUG: set to 1 in optimization_defines.h to instrument hit/miss counters.
#endif

static_assert(FB_LANE_FANOUT >= 1 && (FB_LANE_FANOUT & (FB_LANE_FANOUT - 1)) == 0,
              "FB_LANE_FANOUT must be a power of two");
static_assert(32 % FB_LANE_FANOUT == 0, "FB_LANE_FANOUT must divide warp size 32");

// feedback_rhs index (atomic-target layout, lane fan-out): signature (lane, ie, n, m, it, var)
// FB_ELEMENTS_FIRST=0: layout (FB_LANE_FANOUT, NVAR, N_TOR, NV, NDEG, n_elements) — lane fastest, n_elements slowest
// FB_ELEMENTS_FIRST=1: layout (n_elements, FB_LANE_FANOUT, NVAR, N_TOR, NV, NDEG) — n_elements fastest, lane second
__device__ __host__ __forceinline__
int fb_idx(int lane, int ie, int n, int m, int it, int var, int n_elements)
{
#if FB_ELEMENTS_FIRST == 1
    // (n_elements, FB_LANE_FANOUT, NVAR, N_TOR, NV, NDEG) — flatten manually
    int s = ie;
    s += n_elements * (lane
       + FB_LANE_FANOUT * (var
       + NVAR * (it
       + N_TOR * (m
       + NV * n))));
    return s;
#else
    // (FB_LANE_FANOUT, NVAR, N_TOR, NV, NDEG, n_elements)
    int s = lane;
    s += FB_LANE_FANOUT * (var
       + NVAR * (it
       + N_TOR * (m
       + NV * (n
       + NDEG * ie))));
    return s;
#endif
}

// ===========================================================================================
//                                  MAIN HIP KERNEL
// ===========================================================================================

// ---------------------------------------------------------------------------
// lut_build_cooperative: cooperatively build the shared-memory LUT.
// Called by all threads in a block; must NOT be called with divergent control flow.
//
// Phase 1 — find the elements to cache (the "keys"):
//   1a) Each thread publishes its element; a parallel run-length scan of the
//       i_elm-sorted block extracts the DISTINCT elements (the "base" set).  This
//       relies on particles being counting-sorted by i_elm before each batch
//       (always true in the batch loop) so equal elements are contiguous.  At step 0
//       of a batch there are very few distinct elements per block (~1.75 measured),
//       so the base set is small.
//   1b) Thread 0 writes all base elements into sh_lut_keys (capped at LUT_N_SLOTS),
//       then — if LUT_NEIGHBOR_PRELOAD — best-effort fills any LEFTOVER slots with
//       the (deduped) mesh neighbours of the base elements, to capture step-1 drift
//       (particles move mostly into neighbouring elements; ~5.82 distinct at step 1).
//       Base is inserted first and never evicted.
// Phase 2 (all threads): loads nl_values / nl_deltas fragments into shared memory.
//
// Two __syncthreads() barriers are embedded; the caller must not hold any
// pending sync before calling and must not rely on per-thread state that crosses
// those barriers (e.g., local variables captured in a lambda — use function params).
// ---------------------------------------------------------------------------
#if LUT_VALUES_DELTAS == 1
__device__
void lut_build_cooperative(int i_elm_thread,
                            const int*    __restrict__ el_vertex,
                            const int*    __restrict__ el_neighbours,
                            int n_elements, int n_nodes,
                            const double* __restrict__ nl_values,
                            const double* __restrict__ nl_deltas,
                            int*    sh_lut_keys,
                            double* sh_cache_v,
                            double* sh_cache_d,
                            int*    sh_scratch,
                            int*    sh_base_elms,
                            int*    sh_n_base)
{
    // Phase 1a — publish current element; parallel run-length scan extracts the
    // distinct (base) elements from the sorted block.
    sh_scratch[threadIdx.x] = (i_elm_thread > 0) ? i_elm_thread : -1;
    if (threadIdx.x == 0) *sh_n_base = 0;
    __syncthreads();

    {
        // A thread is a run boundary iff its element is valid and differs from the
        // element of the preceding thread (or it is thread 0).  Each boundary emits
        // one distinct base element.  Because the block is i_elm-sorted, this yields
        // exactly the set of distinct elements present.
        int elm = sh_scratch[threadIdx.x];
        bool boundary = (elm > 0) &&
                        (threadIdx.x == 0 || sh_scratch[threadIdx.x - 1] != elm);
        if (boundary) {
            int pos = atomicAdd(sh_n_base, 1);
            if (pos < LUT_N_SLOTS) sh_base_elms[pos] = elm;
        }
    }
    __syncthreads();

    // Phase 1b — thread 0 assembles sh_lut_keys: all base elements first, then
    // (optionally) best-effort neighbour preload into leftover slots, deduped.
    if (threadIdx.x == 0) {
        int n_base = *sh_n_base;
        if (n_base > LUT_N_SLOTS) n_base = LUT_N_SLOTS;
        int n = 0;
        for (int b = 0; b < n_base; ++b)
            sh_lut_keys[n++] = sh_base_elms[b];
#if LUT_NEIGHBOR_PRELOAD == 1
        // Fill remaining slots with the mesh neighbours of the base elements.
        // 0 = boundary (no neighbour) → skip; dedup against already-chosen keys so
        // each element occupies at most one slot.
        for (int b = 0; b < n_base && n < LUT_N_SLOTS; ++b) {
            int base = sh_base_elms[b];
            for (int kv = 0; kv < NV && n < LUT_N_SLOTS; ++kv) {
                int nb = __ldg(&el_neighbours[el_vert_idx(kv, base - 1, n_elements)]);
                if (nb <= 0) continue;
                bool dup = false;
                for (int s = 0; s < n; ++s)
                    if (sh_lut_keys[s] == nb) { dup = true; break; }
                if (!dup) sh_lut_keys[n++] = nb;
            }
        }
#endif
        for (int s = n; s < LUT_N_SLOTS; ++s) sh_lut_keys[s] = -1;
    }
    __syncthreads();

    // Phase 2 — all threads cooperatively load field data.
    // Layout: sh_cache[lid * LUT_N_SLOTS + s]  (transposed: slot is fast dim at read time).
    // Fill strategy: iterate over (pass, s) with lid = pass*BLOCK_SIZE + threadIdx.x fixed
    // per warp, and loop s in the inner dim. This way each thread writes to addresses
    // lid*LUT_N_SLOTS+0, lid*LUT_N_SLOTS+1, ... which are consecutive — but across
    // threads within a warp, lid differs by 1, so addresses differ by LUT_N_SLOTS (= 4)
    // doubles = 32 bytes, hitting only 4 banks out of 32 → 8-way conflict.
    //
    // Fix: reorganise so the warp's 32 threads cover 32 consecutive flat indices.
    // We iterate over flat index f = pass*BLOCK_SIZE + threadIdx.x and decompose
    // f = lid * LUT_N_SLOTS + s, so thread t in pass p writes to flat index
    // f = p*BLOCK_SIZE + t, i.e. lid = f / LUT_N_SLOTS, s = f % LUT_N_SLOTS.
    // Consecutive threads hit consecutive flat indices → stride-1 shared writes → no conflict.
    // The read-side index (lid * LUT_N_SLOTS + s) is identical, so reads are unchanged.
    {
        // Precompute node indices for all slots (needed for arbitrary s per flat index).
        int ivs[LUT_N_SLOTS][NV];
        for (int s = 0; s < LUT_N_SLOTS; ++s) {
            int elm = sh_lut_keys[s];
            for (int kv = 0; kv < NV; ++kv)
                ivs[s][kv] = (elm > 0) ? (__ldg(&el_vertex[el_vert_idx(kv, elm - 1, n_elements)]) - 1) : -1;
        }

        int total = LUT_SLOT_SIZE * LUT_N_SLOTS;
        for (int pass = 0; pass * BLOCK_SIZE < total; ++pass) {
            int f      = pass * BLOCK_SIZE + threadIdx.x;
            if (f >= total) continue;
            int lid    = f / LUT_N_SLOTS;
            int s      = f % LUT_N_SLOTS;
            int elm    = sh_lut_keys[s];
            if (elm <= 0) continue;
            int tmp    = lid;
            int kv_l   = tmp % NV;    tmp /= NV;
            int it_l   = tmp % N_TOR; tmp /= N_TOR;
            int kf_l   = tmp % NDEG;  tmp /= NDEG;
            int ivar_l = tmp;
            int node   = ivs[s][kv_l];
            int gi     = nl_val_idx(ivar_l, kf_l, it_l, node, n_nodes);
            sh_cache_v[f] = __ldg(&nl_values[gi]);
            sh_cache_d[f] = __ldg(&nl_deltas[gi]);
        }
    }
    __syncthreads();
}
#endif

// ---------------------------------------------------------------------------
// reduce_feedback_lanes: sum the FB_LANE_FANOUT replicas of each
// (ie, n, m, it, var) cell into a single value, writing to the compact
// (lane-free) output buffer that the host expects.
//
// One thread per output cell. The thread linear id `tid` is decoded so that
// consecutive `tid` maps to consecutive *output* memory:
//   FB_ELEMENTS_FIRST=1: ie varies fastest -> 32-wide coalesced stores.
//   FB_ELEMENTS_FIRST=0: var (NVAR) varies fastest -> matches host column-major.
// The inner loop over `lane` reads FB_LANE_FANOUT contiguous doubles per
// thread (lane is the fastest axis on the fat input side), so input traffic
// is also tight.
// ---------------------------------------------------------------------------
__global__
void reduce_feedback_lanes(const double* __restrict__ fb_fat,
                           double* __restrict__ fb_out,
                           int n_elements)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n_elements * NDEG * NV * N_TOR * NVAR;
    if (tid >= total) return;

    int ie, n, m, it, var;
#if FB_ELEMENTS_FIRST == 1
    // Output: (n_elements, NVAR, N_TOR, NV, NDEG), ie fastest.
    long long t = tid;
    ie  = (int)(t % n_elements);  t /= n_elements;
    var = (int)(t % NVAR);        t /= NVAR;
    it  = (int)(t % N_TOR);       t /= N_TOR;
    m   = (int)(t % NV);          t /= NV;
    n   = (int)t;
#else
    // Output: (NVAR, N_TOR, NV, NDEG, n_elements), var fastest.
    long long t = tid;
    var = (int)(t % NVAR);        t /= NVAR;
    it  = (int)(t % N_TOR);       t /= N_TOR;
    m   = (int)(t % NV);          t /= NV;
    n   = (int)(t % NDEG);        t /= NDEG;
    ie  = (int)t;
#endif

    double sum = 0.0;
    #pragma unroll
    for (int lane = 0; lane < FB_LANE_FANOUT; ++lane) {
        sum += fb_fat[fb_idx(lane, ie, n, m, it, var, n_elements)];
    }
    fb_out[fb_idx_compact(ie, n, m, it, var, n_elements)] = sum;
}

// ---------------------------------------------------------------------------
// evolve_REs_kernel: each thread evolves one particle through all time steps.
// feedback_rhs accumulation uses atomicAdd.
//
// Particle arrays layout (Fortran column-major):
//   p_x[j + num_particles*dim], p_p[j + num_particles*dim], p_st[j + num_particles*dim]  (0-based j, dim)
//   p_i_elm[j], p_weight[j], p_q[j]
//
// feedback_rhs layout (column-major, 0-based) — see fb_idx() for the lane-fanout aware form:
//   FB_ELEMENTS_FIRST=1: (n_elements, FB_LANE_FANOUT, NVAR, N_TOR, NV, NDEG)
//   FB_ELEMENTS_FIRST=0: (FB_LANE_FANOUT, NVAR, N_TOR, NV, NDEG, n_elements)
// The host-visible (compact) layout drops the lane axis and is restored by reduce_feedback_lanes.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// evolve_batch_kernel: fused proj+push kernel that runs nsteps kinetic steps
// in a single launch.  Each thread holds its particle state in registers for
// the entire batch (one global-memory read at entry, one write at exit).
// When LUT is enabled the shared-memory cache is built once and reused for
// all nsteps — this is the primary motivation for batching.
// nsteps is normally STEPS_PER_BATCH; the last batch may be smaller.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(BLOCK_SIZE, 2)
void evolve_batch_kernel(
    // Particle SoA — in/out (read once at start, written once at end)
    double* __restrict__ p_x,
    double* __restrict__ p_p,
    double* __restrict__ p_st,
    int*    __restrict__ p_i_elm,
    const double* __restrict__ p_weight,
    double charge,
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
    // Feedback RHS (atomically updated)
    double* __restrict__ feedback_rhs,
    const int* __restrict__ mode_coord,
    int nsteps
#if LUT_VALUES_DELTAS == 1 && LUT_DEBUG == 1
    , unsigned long long* __restrict__ g_lut_hits
    , unsigned long long* __restrict__ g_lut_misses
#endif
    )
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

#if LUT_VALUES_DELTAS == 1
    __shared__ int    sh_lut_keys[LUT_N_SLOTS];
    __shared__ double sh_cache_v[LUT_SLOT_SIZE * LUT_N_SLOTS];
    __shared__ double sh_cache_d[LUT_SLOT_SIZE * LUT_N_SLOTS];
    __shared__ int    sh_scratch[BLOCK_SIZE];
    __shared__ int    sh_base_elms[LUT_N_SLOTS];   // distinct step-0 elements (base set)
    __shared__ int    sh_n_base;                   // count of distinct elements found
#if LUT_DEBUG == 1
    // Per-step-class hit/miss counters, to validate neighbour preload:
    //   *_s0    = first step of the batch (particles freshly i_elm-sorted; the base
    //             elements are all cached, so this should be ~100% hit).
    //   *_drift = later steps (particles have drifted, mostly into neighbours; this
    //             rate measures whether LUT_NEIGHBOR_PRELOAD captured that drift).
    __shared__ unsigned long long sh_hits_s0,    sh_misses_s0;
    __shared__ unsigned long long sh_hits_drift, sh_misses_drift;
    if (threadIdx.x == 0) {
        sh_hits_s0 = 0ULL; sh_misses_s0 = 0ULL;
        sh_hits_drift = 0ULL; sh_misses_drift = 0ULL;
    }
    __syncthreads();
#endif

    // lut_build_cooperative requires all threads in the block to participate,
    // including out-of-range threads (they contribute i_elm = 0 / -1).
    int i_elm_lut = (j < num_particles) ? p_i_elm[j] : 0;
    lut_build_cooperative(i_elm_lut, el_vertex, el_neighbours, n_elements, n_nodes,
                          nl_values, nl_deltas,
                          sh_lut_keys, sh_cache_v, sh_cache_d, sh_scratch,
                          sh_base_elms, &sh_n_base);
#endif

    // Out-of-range threads must participate in lut_build_cooperative's
    // __syncthreads() barriers (the LUT is built once, above, before this point).
    // They do no work (i_elm = -1 keeps them in the skip branches).
    const bool active = (j < num_particles);

    double x[3]  = {0.0, 0.0, 0.0};
    double pm[3] = {0.0, 0.0, 0.0};
    double st[2] = {0.0, 0.0};
    int    i_elm = -1;
    double w     = 0.0;

    if (active) {
        x[0]  = p_x[idx2(j, 0, num_particles)];  x[1]  = p_x[idx2(j, 1, num_particles)];  x[2]  = p_x[idx2(j, 2, num_particles)];
        pm[0] = p_p[idx2(j, 0, num_particles)];  pm[1] = p_p[idx2(j, 1, num_particles)];  pm[2] = p_p[idx2(j, 2, num_particles)];
        st[0] = p_st[idx2(j, 0, num_particles)]; st[1] = p_st[idx2(j, 1, num_particles)];
        i_elm = p_i_elm[j];
        w     = p_weight[j];
    }

    for (int s = 0; s < nsteps; ++s) {
#if LUT_VALUES_DELTAS == 1 && LUT_DEBUG == 1
        // Route this step's LUT hits/misses to the step-0 or drift counter pair.
        unsigned long long* lut_hits_ptr   = (s == 0) ? &sh_hits_s0   : &sh_hits_drift;
        unsigned long long* lut_misses_ptr = (s == 0) ? &sh_misses_s0 : &sh_misses_drift;
#endif
        // --- PROJ phase: accumulate feedback_rhs from current register state ---
        if (i_elm > 0) {
            double HZ_proj[N_TOR];
            mode_moivre(x[2], HZ_proj);

            double cyl_mom[3];
            vector_cartesian_to_cylindrical(x[2], pm, cyl_mom);
            double pdot_cyl = cyl_mom[0]*cyl_mom[0] + cyl_mom[1]*cyl_mom[1] + cyl_mom[2]*cyl_mom[2];
            double inv_denom_v = 1.0 / sqrt(pdot_cyl / (SPEED_OF_LIGHT*SPEED_OF_LIGHT) + group_mass*group_mass);
            double cyl_vel[3] = {cyl_mom[0] * inv_denom_v, cyl_mom[1] * inv_denom_v, cyl_mom[2] * inv_denom_v};

            double B_loc[3];
            calc_B_only(nl_values, nl_deltas, nl_x, el_vertex, el_size,
                        n_elements, n_nodes,
                        F0,
                        time_now, time_prev, t_jorek, flag_static,
                        i_elm, st, x[2], sim_time,
                        B_loc
#if LUT_VALUES_DELTAS == 1
                        , sh_lut_keys, sh_cache_v, sh_cache_d
#endif
#if LUT_VALUES_DELTAS == 1 && LUT_DEBUG == 1
                        , lut_hits_ptr, lut_misses_ptr
#endif
                        );

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

            double v_Ppar  = gamma_m * v_par * v_par * MU_ZERO;
            double v_Pperp = gamma_m * v_perp_sq * 0.5 * MU_ZERO;
            double v_jPhi  = -double(charge) * EL_CHG * cyl_vel[2] * x[0] * MU_ZERO;

            int ie = i_elm - 1;
            // Lane-fanout: spread the FB_LANE_FANOUT replicas of each (ie,n,m,it,var)
            // across warp lanes so warps that share an `ie` (very common at step 0 of
            // each batch, where particles are freshly sorted) hit distinct addresses.
            const int fb_lane = threadIdx.x & (FB_LANE_FANOUT - 1);
            for (int n = 0; n < NDEG; ++n) {
                for (int m = 0; m < NV; ++m) {
                    double proj_factor = bf2D_0_scalar(st[0], st[1], n, m)
                                       * __ldg(&el_size[el_size_idx(n, m, ie, n_elements)])
                                       * w;

                    for (int it = 0; it < N_TOR; ++it) {
                        double hz = HZ_proj[it];

                        atomicAdd(&feedback_rhs[fb_idx(fb_lane, ie, n, m, it, P_PAR_IDX,  n_elements)], hz * v_Ppar  * proj_factor);
                        atomicAdd(&feedback_rhs[fb_idx(fb_lane, ie, n, m, it, P_PERP_IDX, n_elements)], hz * v_Pperp * proj_factor);
                        atomicAdd(&feedback_rhs[fb_idx(fb_lane, ie, n, m, it, J_PHI_IDX,  n_elements)], hz * v_jPhi  * proj_factor);
                    }
                }
            }
        }

        // --- PUSH phase: advance particle in registers ---
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
                                   ifail
#if LUT_VALUES_DELTAS == 1
                                   , sh_lut_keys, sh_cache_v, sh_cache_d
#endif
#if LUT_VALUES_DELTAS == 1 && LUT_DEBUG == 1
                                   , lut_hits_ptr, lut_misses_ptr
#endif
                                   );
        }
    }

#if LUT_VALUES_DELTAS == 1 && LUT_DEBUG == 1
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(&g_lut_hits[0],   sh_hits_s0);
        atomicAdd(&g_lut_misses[0], sh_misses_s0);
        atomicAdd(&g_lut_hits[1],   sh_hits_drift);
        atomicAdd(&g_lut_misses[1], sh_misses_drift);
    }
#endif

    // Write final state back to global memory — one write per particle per batch.
    if (active) {
        p_x[idx2(j, 0, num_particles)]  = x[0];  p_x[idx2(j, 1, num_particles)]  = x[1];  p_x[idx2(j, 2, num_particles)]  = x[2];
        p_p[idx2(j, 0, num_particles)]  = pm[0]; p_p[idx2(j, 1, num_particles)]  = pm[1]; p_p[idx2(j, 2, num_particles)]  = pm[2];
        p_st[idx2(j, 0, num_particles)] = st[0]; p_st[idx2(j, 1, num_particles)] = st[1];
        p_i_elm[j] = i_elm;
    }
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

    const size_t sz_feedback_compact = (size_t)NDEG * NV * n_elements * N_TOR * NVAR * sizeof(double);
    const size_t sz_feedback   = sz_feedback_compact * FB_LANE_FANOUT;
    const size_t sz_mode_coord = N_COORD_TOR * sizeof(int);

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
        printf("[Array Dimensions] feedback (fat, x FB_LANE_FANOUT=%d): %.2f KB ; compact: %.2f KB\n",
               FB_LANE_FANOUT, TO_KB(sz_feedback), TO_KB(sz_feedback_compact));
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
    double *d_feedback_rhs;       // fat accumulation buffer (with FB_LANE_FANOUT replicas)
    double *d_feedback_rhs_out;   // compact buffer returned to host
    int    *d_mode_coord;

    
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
#if LUT_VALUES_DELTAS == 0
        HIP_CHECK(hipFuncSetCacheConfig(reinterpret_cast<const void*>(evolve_batch_kernel), hipFuncCachePreferL1));
#endif
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
    HIP_CHECK(hipMalloc(&d_feedback_rhs_out, sz_feedback_compact));
    HIP_CHECK(hipMalloc(&d_mode_coord,       sz_mode_coord));


    // --- Allocate sorting buffers ---
    double *d_x_sorted = nullptr, *d_p_sorted = nullptr, *d_st_sorted = nullptr, *d_weight_sorted = nullptr;
    int    *d_i_elm_sorted = nullptr;
    int    *d_hist = nullptr, *d_offsets = nullptr, *d_cursors = nullptr;
    int    *d_block_sums = nullptr, *d_block_offsets = nullptr;

#if STEPS_PER_BATCH > 0
    HIP_CHECK(hipMalloc(&d_x_sorted,      sz_x));
    HIP_CHECK(hipMalloc(&d_p_sorted,      sz_p));
    HIP_CHECK(hipMalloc(&d_st_sorted,     sz_st));
    HIP_CHECK(hipMalloc(&d_weight_sorted, sz_weight));
    HIP_CHECK(hipMalloc(&d_i_elm_sorted,  sz_i_elm));
    // Save original handles — sort swaps these pointers with d_x etc., so free
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

    // Fortran caller (mod_particle_evolution.f90) always zero-inits fb_c before this call,
    // so the fat accumulation buffer just gets zeroed; we never have to fan the host data
    // out into per-lane replicas.
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

#if LUT_VALUES_DELTAS == 1 && LUT_DEBUG == 1
    // [0] = first batch step, [1] = drift (later steps).
    unsigned long long *d_lut_hits = nullptr, *d_lut_misses = nullptr;
    HIP_CHECK(hipMalloc(&d_lut_hits,   2 * sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc(&d_lut_misses, 2 * sizeof(unsigned long long)));
    HIP_CHECK(hipMemset(d_lut_hits,   0, 2 * sizeof(unsigned long long)));
    HIP_CHECK(hipMemset(d_lut_misses, 0, 2 * sizeof(unsigned long long)));
#endif

    // --- Batch loop: one kernel launch per sort interval ---
    // Each launch runs nsteps=STEPS_PER_BATCH kinetic steps internally (remainder on the
    // last batch).  The fused evolve_batch_kernel does proj+push in a single kernel,
    // holding particle state in registers across all steps; the LUT is built once per
    // launch so it is reused for all steps in the batch.
    // d_x / d_p / d_st / d_i_elm are updated in-place — no double-buffer swap needed.
    int grid_size = (num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int sort_call_count = 0;

#if GPU_DEBUG == 1
    // --- Phase-breakdown accumulators (summed over all batches) ---
    // Each batch is: sort -> evolve_batch_kernel.  We time both with dedicated
    // events so the per-phase totals are clean (no stale carry-over).
    float total_sort_ms   = 0.0f;
    float total_evolve_ms = 0.0f;
    hipEvent_t t_evolve_start, t_evolve_stop;
    HIP_CHECK(hipEventCreate(&t_evolve_start));
    HIP_CHECK(hipEventCreate(&t_evolve_stop));
#endif

    HIP_CHECK(hipEventRecord(t_start, 0));
    for (int k = 0; k < nstep_particles; k += STEPS_PER_BATCH) {
#if GPU_DEBUG == 1
        if(sim.my_id == 0)
            printf("[launch_evolve_REs rank %d] Starting batch %d / %d (particle steps %d to %d)\n",
               sim.my_id, sort_call_count+1, (nstep_particles + STEPS_PER_BATCH - 1) / STEPS_PER_BATCH,
               k, std::min(k + STEPS_PER_BATCH, nstep_particles));
#endif
        int batch = std::min(STEPS_PER_BATCH, nstep_particles - k);

#if GPU_DEBUG == 1
        hipEvent_t t_sort_start, t_sort_stop;
        HIP_CHECK(hipEventCreate(&t_sort_start));
        HIP_CHECK(hipEventCreate(&t_sort_stop));
        HIP_CHECK(hipEventRecord(t_sort_start, 0));
#endif
        sort_particles_by_i_elm_gpu(
            d_x, d_p, d_st, d_i_elm, d_weight,
            d_x_sorted, d_p_sorted, d_st_sorted, d_i_elm_sorted, d_weight_sorted,
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

        // Fused batch kernel: runs batch steps, reads+writes d_x/d_p/d_st/d_i_elm in-place.
#if GPU_DEBUG == 1
        HIP_CHECK(hipEventRecord(t_evolve_start, 0));
#endif
        hipLaunchKernelGGL(evolve_batch_kernel,
            dim3(grid_size), dim3(BLOCK_SIZE), 0, 0,
            d_x, d_p, d_st, d_i_elm, d_weight, charge,
            d_nl_values, d_nl_deltas, d_nl_x, n_nodes,
            d_el_vertex, d_el_neighbours, d_el_size, n_elements,
            time_now, time_prev, flag_static, flag_zero_dp,
            F0, t_norm, t_jorek, sim_time, group_mass, tstep_part_adj,
            num_particles, d_feedback_rhs, d_mode_coord, batch
#if LUT_VALUES_DELTAS == 1 && LUT_DEBUG == 1
            , d_lut_hits, d_lut_misses
#endif
            );
        HIP_CHECK(hipGetLastError());
#if GPU_DEBUG == 1
        HIP_CHECK(hipEventRecord(t_evolve_stop, 0));
        HIP_CHECK(hipEventSynchronize(t_evolve_stop));
        float evolve_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&evolve_ms, t_evolve_start, t_evolve_stop));
        total_evolve_ms += evolve_ms;
#else
        HIP_CHECK(hipDeviceSynchronize());
#endif

#if LUT_VALUES_DELTAS == 1 && LUT_DEBUG == 1
        {
            // [0] = first batch step, [1] = drift (later steps).
            unsigned long long h_hits[2] = {0, 0}, h_misses[2] = {0, 0};
            HIP_CHECK(hipMemcpy(h_hits,   d_lut_hits,   2 * sizeof(unsigned long long), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(h_misses, d_lut_misses, 2 * sizeof(unsigned long long), hipMemcpyDeviceToHost));
            unsigned long long tot0 = h_hits[0] + h_misses[0];
            unsigned long long tot1 = h_hits[1] + h_misses[1];
            unsigned long long tot  = tot0 + tot1;
            double rate0 = (tot0 > 0) ? 100.0 * (double)h_hits[0] / (double)tot0 : 0.0;
            double rate1 = (tot1 > 0) ? 100.0 * (double)h_hits[1] / (double)tot1 : 0.0;
            double rate  = (tot  > 0) ? 100.0 * (double)(h_hits[0] + h_hits[1]) / (double)tot : 0.0;
            printf("[LUT_DEBUG rank %d] batch %d: step0 hits=%llu misses=%llu rate=%.2f%% | "
                   "drift hits=%llu misses=%llu rate=%.2f%% | overall rate=%.2f%%\n",
                   sim.my_id, sort_call_count,
                   h_hits[0], h_misses[0], rate0,
                   h_hits[1], h_misses[1], rate1, rate);
            // Reset for next batch
            HIP_CHECK(hipMemset(d_lut_hits,   0, 2 * sizeof(unsigned long long)));
            HIP_CHECK(hipMemset(d_lut_misses, 0, 2 * sizeof(unsigned long long)));
        }
#endif

#if GPU_DEBUG == 1
        if(sim.my_id == 0)
            printf("[launch_evolve_REs rank %d] Finished batch %d / %d (particle steps %d to %d): evolve %.3f ms, sort %.3f ms\n",
               sim.my_id, sort_call_count, (nstep_particles + STEPS_PER_BATCH - 1) / STEPS_PER_BATCH,
               k, std::min(k + STEPS_PER_BATCH, nstep_particles), evolve_ms, sort_ms);
#endif
    }
    HIP_CHECK(hipEventRecord(t_stop, 0));
    HIP_CHECK(hipEventSynchronize(t_stop));
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, t_start, t_stop));
#if GPU_DEBUG == 1
    HIP_CHECK(hipEventDestroy(t_evolve_start));
    HIP_CHECK(hipEventDestroy(t_evolve_stop));
    if (sim.my_id == 0) {
        const float loop_ms  = elapsed_ms;
        const float other_ms = loop_ms - total_sort_ms - total_evolve_ms;
        printf("[launch_evolve_REs rank %d] nstep_particles loop (%d steps, %d sorts): %.3f ms\n",
               sim.my_id, nstep_particles, sort_call_count, loop_ms);
        printf("[launch_evolve_REs rank %d] === PHASE BREAKDOWN (batch loop) ===\n", sim.my_id);
        printf("[launch_evolve_REs rank %d]   evolve_batch_kernel : %9.3f ms  (%.1f%%)\n",
               sim.my_id, total_evolve_ms, 100.0 * total_evolve_ms / loop_ms);
        printf("[launch_evolve_REs rank %d]   sort_particles      : %9.3f ms  (%.1f%%)\n",
               sim.my_id, total_sort_ms,   100.0 * total_sort_ms   / loop_ms);
        printf("[launch_evolve_REs rank %d]   launch/sync overhead : %9.3f ms  (%.1f%%)\n",
               sim.my_id, other_ms,        100.0 * other_ms        / loop_ms);
    }
#endif

    // --- Reduce the FB_LANE_FANOUT replicas into the compact output buffer ---
    float reduce_ms = 0.0f;
    {
#if GPU_DEBUG == 1
        hipEvent_t r_start, r_stop;
        HIP_CHECK(hipEventCreate(&r_start));
        HIP_CHECK(hipEventCreate(&r_stop));
        HIP_CHECK(hipEventRecord(r_start, 0));
#endif

        long long total = (long long)n_elements * NDEG * NV * N_TOR * NVAR;
        int reduce_block = 256;
        long long reduce_grid_ll = (total + reduce_block - 1) / reduce_block;
        // Guard against oversize grid (very unlikely given typical sizes).
        int reduce_grid = (reduce_grid_ll > (long long)INT_MAX) ? INT_MAX : (int)reduce_grid_ll;
        hipLaunchKernelGGL(reduce_feedback_lanes,
                           dim3(reduce_grid), dim3(reduce_block), 0, 0,
                           d_feedback_rhs, d_feedback_rhs_out, n_elements);
#if GPU_DEBUG == 1
        HIP_CHECK(hipEventRecord(r_stop, 0));
        HIP_CHECK(hipEventSynchronize(r_stop));
        HIP_CHECK(hipEventElapsedTime(&reduce_ms, r_start, r_stop));
        if (sim.my_id == 0)
            printf("[launch_evolve_REs rank %d] reduce_feedback_lanes (FB_LANE_FANOUT=%d): %.3f ms\n",
                   sim.my_id, FB_LANE_FANOUT, reduce_ms);
        HIP_CHECK(hipEventDestroy(r_start));
        HIP_CHECK(hipEventDestroy(r_stop));
#endif
    }

    // --- Copy results back: device -> host ---
#if GPU_DEBUG == 1
    HIP_CHECK(hipEventRecord(t_start, 0));
#endif
    HIP_CHECK(hipMemcpy(part->x,       d_x,            sz_x,        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(part->p,       d_p,            sz_p,        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(part->st,      d_st,           sz_st,       hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(part->i_elm,   d_i_elm,        sz_i_elm,    hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(part->weight,  d_weight,       sz_weight,   hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_feedback_rhs,d_feedback_rhs_out, sz_feedback_compact, hipMemcpyDeviceToHost));
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

#if GPU_DEBUG == 1
    // --- Consolidated whole-call phase breakdown ---
    if (sim.my_id == 0) {
        const float total_ms = h2d_ms + total_evolve_ms + total_sort_ms + reduce_ms + d2h_ms;
        printf("[launch_evolve_REs rank %d] ===== WHOLE-CALL PHASE BREAKDOWN =====\n", sim.my_id);
        printf("[launch_evolve_REs rank %d]   H2D transfers       : %9.3f ms  (%.1f%%)\n",
               sim.my_id, h2d_ms,          100.0 * h2d_ms          / total_ms);
        printf("[launch_evolve_REs rank %d]   evolve_batch_kernel : %9.3f ms  (%.1f%%)\n",
               sim.my_id, total_evolve_ms, 100.0 * total_evolve_ms / total_ms);
        printf("[launch_evolve_REs rank %d]   sort_particles      : %9.3f ms  (%.1f%%)\n",
               sim.my_id, total_sort_ms,   100.0 * total_sort_ms   / total_ms);
        printf("[launch_evolve_REs rank %d]   reduce_feedback     : %9.3f ms  (%.1f%%)\n",
               sim.my_id, reduce_ms,       100.0 * reduce_ms       / total_ms);
        printf("[launch_evolve_REs rank %d]   D2H transfers       : %9.3f ms  (%.1f%%)\n",
               sim.my_id, d2h_ms,          100.0 * d2h_ms          / total_ms);
        printf("[launch_evolve_REs rank %d]   ---------------------------------------\n", sim.my_id);
        printf("[launch_evolve_REs rank %d]   TOTAL (timed phases): %9.3f ms\n", sim.my_id, total_ms);
    }
#endif


    // --- Free device memory ---
    HIP_CHECK(hipFree(d_x_orig));
    HIP_CHECK(hipFree(d_p_orig));
    HIP_CHECK(hipFree(d_st_orig));
    HIP_CHECK(hipFree(d_i_elm_orig));
    HIP_CHECK(hipFree(d_weight_orig));
#if STEPS_PER_BATCH > 0
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
    HIP_CHECK(hipFree(d_feedback_rhs_out));
    HIP_CHECK(hipFree(d_mode_coord));
#if LUT_VALUES_DELTAS == 1 && LUT_DEBUG == 1
    HIP_CHECK(hipFree(d_lut_hits));
    HIP_CHECK(hipFree(d_lut_misses));
#endif
}

#endif /* USE_BATCH_KERNEL */
