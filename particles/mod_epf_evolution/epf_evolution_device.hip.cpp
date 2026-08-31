/* particles/mod_epf_evolution/epf_evolution_device.hip.cpp -- the device
 * launcher of the energetic-particle kernel.
 *
 * The per-particle physics is not duplicated here: epf_push, epf_project and
 * epf_deposit come from the header the host arm uses. What is here is the
 * decomposition -- which thread does which piece of one step.
 *
 * One step is one launch, plus four more on the steps that collect a projection:
 *
 *   1. push_stage_kernel                      -- thread per particle: the field
 *      evaluation, the Boris kick, the search, and (when collecting) the seven
 *      velocity moments, staged;
 *   2. count_bins / scan_bins / assign_slots  -- a counting sort of the
 *      particles by the element the push left them in;
 *   3. jgx_c_scatter, scatter_index, scatter_rows -- the move itself: the
 *      record, the original index and the staged moments;
 *   4. proj_accumulate_kernel                 -- block per element, thread per
 *      feedback cell: the outer product summed over the element's run.
 *
 * Steps 2 and 3 are jgx/common/device_sort.h, shared with the runaway kernel;
 * why the sort is there at all is written up in that file.
 *
 * The sort runs only on the steps that project, unlike the runaway kernel where
 * every step does. Not an optimisation of the same code: the Fortran collects a
 * projection once every proj_collection_period steps, so on the steps between
 * there is no accumulation for the element runs to serve, and the push is the
 * whole of the step.
 *
 * The order inside a step is the Fortran's: field, push, search, *then* moments.
 * A projection therefore pairs the magnetic field at x^n with v^(n+1/2) and
 * x^(n+1) -- see the header. That is also why the sort follows the push rather
 * than preceding it: the element a particle deposits into is the one the search
 * left it in, not the one it started the step in.
 *
 * The particle order is restored before the download: the sort permutes the
 * device copy, and the host arm does not, so an `orig` index rides along and the
 * last scatter puts every record back where Fortran had it. A device build must
 * not be observable in the array that comes home.
 */
#include "particles/mod_epf_evolution/epf_evolution_device.h"
#include "particles/mod_epf_evolution/epf_evolution.h"
#include "particles/mod_fields_linear/fields_linear.h"
#include "particles/particle_types/particle_set.h"

#include "jgx/data/device_pack.h"
#include "jgx/data/device_buffer.h"
#include "jgx/common/device_sort.h"
#include "jgx/jgx_c_api.h"
#include "jgx/macros.h"
#include "jgx/device.h"

#include <cstddef>
#include <cstdint>
#include <utility>

/* Same switch as the host shim: find_RZ_nearby takes the DEBUG build as a
 * template parameter rather than reading the preprocessor itself. */
#ifdef DEBUG
static constexpr bool kFindRZNearbyDebug = true;
#else
static constexpr bool kFindRZNearbyDebug = false;
#endif

namespace {

using part_set    = jorek::particle_kin_lf_set_soa;
using fields_set  = jorek::fields_linear_set_soa;
using diagnostics = find_rz_nearby::diagnostics;

constexpr int kNDeg = JGX_N_DEGREES;
constexpr int kNVtx = JGX_N_VERTEX_MAX;
constexpr int kNTor = JGX_N_TOR;
constexpr int kNVar = jorek::EPF_N_VAR;      /* the six pressures and the density */

/* One element's share of the feedback array. */
constexpr int kCells = kNDeg * kNVtx * kNTor * kNVar;

constexpr unsigned kBlock = 256;             /* the per-particle kernel */

/* The accumulate block owns one element and one cell per thread, rounded up to
 * a whole warp. At n_tor = 1 that is 128 threads for 112 cells; at n_tor = 3,
 * 384 for 336. */
constexpr unsigned kAccumBlock = ((kCells + 63) / 64) * 64;
static_assert(kAccumBlock >= static_cast<unsigned>(kCells), "one cell per thread");
static_assert(kAccumBlock <= 1024, "a block is at most 1024 threads; n_tor is too large");

/* Particles whose factor vectors are held in shared memory at once. The runs are
 * n_particles/n_elements long, a few tens per rank at production sizes, so one
 * tile usually covers a whole run. */
constexpr int kProjTile = 64;

/* Row widths padded to an odd number of doubles -- see the runaway kernel's
 * note on the bank conflict this avoids in the cooperative store. */
constexpr int kShBF = (kNDeg * kNVtx) | 1;
constexpr int kShHZ = (kNTor) | 1;
constexpr int kShVW = (kNVar) | 1;

/* The counting sort, the warp primitives and the three riders are
 * jgx/common/device_sort.h; brought into this namespace so the launches below
 * read as they would if the definitions were here. */
using jgx::assign_slots;
using jgx::count_bins;
using jgx::iota_index;
using jgx::kScanThreads;
using jgx::scan_bins;
using jgx::scatter_index;
using jgx::scatter_rows;

using jgx::device_buffer;
using jgx::grid_for;

/* ------------------------------------------------------------------------- */
/*  the two physics kernels                                                   */
/* ------------------------------------------------------------------------- */

/* One step of the push, thread per particle, in place.
 *
 * In place rather than into a second buffer: nothing has read this step's
 * positions yet -- the projection is formed from what the push produces, not
 * from what it consumed -- and the sort's own destination block is the only
 * spare copy needed.
 *
 * `stage` says whether this step's projection is collected. The moments are
 * formed here rather than in a kernel of their own because they need the
 * magnetic field at the position the particle was pushed *from*, which only
 * this kernel has; re-evaluating it after the push would be a different field
 * and a second interpolation.
 *
 * A particle that is lost, on entry or by this push, stages seven zeros. The
 * accumulate never reads them -- a lost particle sits past the last element bin
 * -- but they cost one store and leave no uninitialised memory behind.
 */
__global__ void push_stage_kernel(part_set part, fields_set fields,
                                  double mass, double time, double timestep,
                                  double phi_search, bool stage,
                                  double* __restrict__ stg,
                                  diagnostics* __restrict__ diag) {
  const std::size_t ip = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  if (ip >= part.n_records) return;

  const std::size_t n = part.n_records;
  double E[3], B[3];
  diagnostics my_diag;
  bool alive = (part.i_elm(ip) > 0);

  if (alive) {
    /* ifail is written by every step and read by none of them, here as in the
     * Fortran -- a lost particle shows up as i_elm <= 0. */
    int ifail = 0;
    jorek::epf_push<kFindRZNearbyDebug>(part, ip, fields, mass, time, timestep,
                                        phi_search, E, B, ifail, my_diag);
    alive = (part.i_elm(ip) > 0);
  }

  if (stage) {
    if (alive) {
      const jorek::epf_projection pr = jorek::epf_project(part, ip, B, mass);
      for (int v = 0; v < kNVar; ++v) stg[v*n + ip] = pr.m[v];
    } else {
      for (int v = 0; v < kNVar; ++v) stg[v*n + ip] = 0.0;
    }
  }

  /* First occurrence wins, as it does under the host launcher's critical
   * section -- and which occurrence that is depends on the schedule there too.
   * The compare-and-swap is what elects the writer: whoever flips the flag from
   * zero owns the fields that go with it. */
  if (my_diag.not_found != 0 &&
      atomicCAS(&diag->not_found, 0, my_diag.not_found) == 0) {
    diag->nf_R = my_diag.nf_R;
    diag->nf_Z = my_diag.nf_Z;
  }
  if (my_diag.bad_i_to != 0 &&
      atomicCAS(&diag->bad_i_to, 0, my_diag.bad_i_to) == 0) {
    diag->bad_i_from = my_diag.bad_i_from;
  }
}

/* The projection, one block per element, one thread per feedback cell.
 *
 * The projection factorises per particle,
 *
 *   F[n,m,it,var] = size(ie,m,n) * SUM_p HH(s_p,t_p)[n,m] * HZ(phi_p)[it] * m_var(p)
 *
 * so each particle's three small factor vectors are worked out once, into shared
 * memory, and every cell-thread then sums cheap products out of shared memory.
 * The basis functions and the toroidal harmonics are evaluated once per particle
 * rather than once per particle per cell, and no cell is written by more than one
 * thread -- which is what makes the whole accumulation atomic-free.
 *
 * The running sum spans the whole run, which can be thousands of particles, while
 * one tile contributes a small slice of it -- so the tile partials go in with
 * Kahan compensation rather than a plain +=.
 */
__global__ void proj_accumulate_kernel(part_set part, fields_set fields,
                                       const int* __restrict__ run_start,
                                       const int* __restrict__ run_len,
                                       const double* __restrict__ stg,
                                       jorek::epf_rhs_view_device rhs,
                                       jorek::epf_projection_indices idx) {
  const std::size_t ie = blockIdx.x;
  const std::size_t n  = part.n_records;

  __shared__ double sh_bf[kProjTile][kShBF];
  __shared__ double sh_hz[kProjTile][kShHZ];
  __shared__ double sh_vw[kProjTile][kShVW];

  const int start = run_start[ie];
  const int cnt   = run_len[ie];

  const int c = static_cast<int>(threadIdx.x);
  const bool mine = (c < kCells);

  /* c = var + kNVar*(it + kNTor*(m + kNVtx*n)) */
  int d = c;
  const int var = d % kNVar; d /= kNVar;
  const int it  = d % kNTor; d /= kNTor;
  const int m   = d % kNVtx; d /= kNVtx;
  const int deg = d;
  const int nm  = deg * kNVtx + m;

  double acc = 0.0, cmp = 0.0;

  for (int base = 0; base < cnt; base += kProjTile) {
    const int tile = min(kProjTile, cnt - base);

    /* (a) one lane per tile particle works out its factor vectors, once. */
    for (int q = static_cast<int>(threadIdx.x); q < tile;
         q += static_cast<int>(blockDim.x)) {
      const std::size_t p = static_cast<std::size_t>(start + base + q);

      const std::size_t bf_ext[2] = { kNDeg, 4 };
      const std::size_t hz_ext[1] = { kNTor };
      const jgx::view<double, 2, jgx::layout_right> HH(&sh_bf[q][0], bf_ext);
      const jgx::view<double, 1> HZ(&sh_hz[q][0], hz_ext);

      /* HH is (degree, vertex) with a vertex stride of 4 = n_vertex_max, which
       * is the nm ordering phase (b) indexes, so the row is filled in place. */
      basisfunctions::basisfunctions_2D_0_T(part.st(p, 0), part.st(p, 1), HH);
      interp::mode_moivre_explicit(part.x(p, 2), HZ, JGX_N_TOR, JGX_N_PERIOD);

      for (int v = 0; v < kNVar; ++v) sh_vw[q][v] = stg[v*n + p];
    }
    __syncthreads();

    /* (b) each cell-thread sums this tile into its own accumulator. */
    if (mine) {
      double s = 0.0;
      for (int q = 0; q < tile; ++q)
        s += sh_bf[q][nm] * sh_hz[q][it] * sh_vw[q][var];
      const double y  = s - cmp;
      const double tt = acc + y;
      cmp = (tt - acc) - y;
      acc = tt;
    }
    __syncthreads();  /* the tile's rows are read; the next tile may overwrite */
  }

  if (!mine) return;

  rhs(deg, m, ie, it, idx.var[var]) +=
      acc * fields.element_list.size(ie, static_cast<std::size_t>(m),
                                     static_cast<std::size_t>(deg));
}

} /* anonymous namespace */


extern "C" void jgx_device_epf_evolution_evolve_epf(
    void* part_base, const jgx_record_desc* part_desc, size_t n_particles,
    void* el_base,   const jgx_record_desc* el_desc,   size_t n_elements,
    void* nd_base,   const jgx_record_desc* nd_desc,   size_t n_nodes,
    const void* interp_base, const jgx_record_desc* interp_desc,
    double* rhs_data, const size_t* rhs_ext, const size_t* idx,
    double mass, double time, double timestep, int32_t nstep,
    int32_t proj_period, double phi_search,
    int32_t* not_found, double* nf_R, double* nf_Z,
    int32_t* bad_i_from, int32_t* bad_i_to) {

  if (n_particles == 0 || n_elements == 0) return;

  /* --- the packs: one contiguous H2D each, then a transpose on the device --- */
  const jgx::data::device_pack part_pk(*part_desc, n_particles);
  const jgx::data::device_pack el_pk(*el_desc, n_elements);
  const jgx::data::device_pack nd_pk(*nd_desc, n_nodes);
  part_pk.upload(part_base);
  el_pk.upload(el_base);
  nd_pk.upload(nd_base);

  /* The sort's destination. Only the packed block, not a second staging buffer:
   * nothing uploads or downloads through this one, the two blocks just take
   * turns being the live copy. */
  const device_buffer alt_blk(jgx::data::soa_block_bytes(*part_desc, n_particles));

  void* curr_base = part_pk.base();
  void* alt_base  = alt_blk.get();

  /* The interpolator is a record of count 1 -- six scalars read where they lie,
   * on the host, and carried into the kernel inside the set. */
  const fields_set fields = jorek::make_fields_set(
      jorek::element_set_soa::from_soa(el_pk.base(), *el_desc, n_elements),
      jorek::node_set_soa::from_soa(nd_pk.base(), *nd_desc, n_nodes),
      jorek::make_fields_interp_linear_set<double>(interp_base, *interp_desc));

  /* --- the feedback array ------------------------------------------------- */
  /*
   * Zeroed rather than uploaded: the Fortran array is accumulated into across
   * groups, so what comes back is this call's contribution and the host adds it.
   * That is also what frees the layout -- see epf_rhs_view_device.
   */
  std::size_t rhs_size = 1;
  for (int d = 0; d < 5; ++d) rhs_size *= rhs_ext[d];

  const device_buffer rhs_buf(rhs_size * sizeof(double));
  const jorek::epf_rhs_view_device rhs(rhs_buf.as<double>(), rhs_ext);

  jorek::epf_projection_indices proj;
  for (int v = 0; v < kNVar; ++v) proj.var[v] = idx[v];

  const device_buffer diag_buf(sizeof(diagnostics));

  /* --- the sort's working set --------------------------------------------- */
  const int n_bins = static_cast<int>(n_elements) + 1;
  const device_buffer hist(std::size_t(n_bins) * sizeof(int));     /* zeroed here, and by every scan */
  const device_buffer offsets(std::size_t(n_bins) * sizeof(int));
  const device_buffer cursors(std::size_t(n_bins) * sizeof(int));
  const device_buffer run_len(std::size_t(n_bins) * sizeof(int));
  const device_buffer slot(n_particles * sizeof(std::int32_t));

  const device_buffer orig_a(n_particles * sizeof(std::int32_t));
  const device_buffer orig_b(n_particles * sizeof(std::int32_t));
  std::int32_t* orig_curr = orig_a.as<std::int32_t>();
  std::int32_t* orig_alt  = orig_b.as<std::int32_t>();

  /* --- the projection staging --------------------------------------------- */
  /* One block of kNVar rows of n_particles, not kNVar allocations: the sort
   * moves all of them in one scatter_rows. (s, t, phi) are not staged -- they
   * are read straight from the sorted particle set, which the accumulate holds
   * anyway. */
  const device_buffer stg_a(std::size_t(kNVar) * n_particles * sizeof(double));
  const device_buffer stg_b(std::size_t(kNVar) * n_particles * sizeof(double));
  double* stg_curr = stg_a.as<double>();
  double* stg_alt  = stg_b.as<double>();

  const unsigned grid_p = grid_for(n_particles, kBlock);
  const unsigned grid_e = static_cast<unsigned>(n_elements);

  iota_index<<<grid_p, kBlock>>>(orig_curr, n_particles);

  /* --- the step loop ------------------------------------------------------ */
  for (int k = 1; k <= nstep; ++k) {
    part_set curr = part_set::from_soa(curr_base, *part_desc, n_particles);

    /* The Fortran's `mod(k, proj_collection_period) /= 0` cycle, over a 1-based
     * step counter. Nothing here is synchronous: the whole loop is queued and
     * the host does not wait on it until the jgx_c_synchronize below. */
    const bool collect = (k % proj_period == 0);

    /* (1) field, push, search -- and the moments, if this step projects */
    push_stage_kernel<<<grid_p, kBlock>>>(
        curr, fields, mass, time, timestep, phi_search, collect, stg_curr,
        static_cast<diagnostics*>(diag_buf.get()));

    if (!collect) continue;

    /* (2) sort by the element the push left the particle in */
    count_bins<<<grid_p, kBlock>>>(curr.i_elm.data, n_particles, n_bins,
                                   hist.as<int>());
    scan_bins<<<1, kScanThreads>>>(hist.as<int>(), offsets.as<int>(),
                                   cursors.as<int>(), run_len.as<int>(), n_bins);
    assign_slots<<<grid_p, kBlock>>>(curr.i_elm.data, n_particles, n_bins,
                                     cursors.as<int>(), slot.as<std::int32_t>());

    /* (3) move the records, the original index and the staged moments */
    jgx_c_scatter(alt_base, curr_base, part_desc, n_particles,
                  slot.as<std::int32_t>());
    scatter_index<<<grid_p, kBlock>>>(orig_alt, orig_curr,
                                      slot.as<std::int32_t>(), n_particles);
    scatter_rows<<<grid_p, kBlock>>>(stg_alt, stg_curr,
                                     slot.as<std::int32_t>(), n_particles, kNVar);
    std::swap(curr_base, alt_base);
    std::swap(orig_curr, orig_alt);
    std::swap(stg_curr, stg_alt);
    curr = part_set::from_soa(curr_base, *part_desc, n_particles);

    /* (4) project, atomic-free */
    proj_accumulate_kernel<<<grid_e, kAccumBlock>>>(
        curr, fields, offsets.as<int>(), run_len.as<int>(), stg_curr,
        rhs, proj);
  }

  /* Undo the sorts: record rec goes back to the slot Fortran gave it. */
  jgx_c_scatter(alt_base, curr_base, part_desc, n_particles, orig_curr);
  std::swap(curr_base, alt_base);

  /* Both the launch error and the completion error, so nothing below reads a
   * buffer a failed kernel never wrote. */
  jgx_c_synchronize();

  /* --- back to the host --------------------------------------------------- */
  /* Both views over the same five extents, differing only in stride order, so
   * the un-permute is the identity in index space. Half a million elements once
   * per group-step, against a kernel that ran nstep times over every particle. */
  double* rhs_dev = new double[rhs_size];
  jgx_c_pull(rhs_dev, rhs_buf.get(), rhs_buf.bytes());
  const jorek::epf_rhs_view_device src(rhs_dev, rhs_ext);
  const jorek::epf_rhs_view        dst(rhs_data, rhs_ext);
  for (std::size_t v = 0; v < rhs_ext[4]; ++v)
    for (std::size_t it = 0; it < rhs_ext[3]; ++it)
      for (std::size_t ie = 0; ie < rhs_ext[2]; ++ie)
        for (std::size_t m = 0; m < rhs_ext[1]; ++m)
          for (std::size_t n = 0; n < rhs_ext[0]; ++n)
            dst(n, m, ie, it, v) += src(n, m, ie, it, v);
  delete[] rhs_dev;

  /* Only the particles were written; the mesh comes home unchanged and is
   * dropped with the packs. The unpack reads whichever block the swaps left the
   * restored copy in, which is not necessarily the pack's own. */
  part_pk.download(part_base, curr_base);

  diagnostics diag;
  jgx_c_pull(&diag, diag_buf.get(), sizeof(diagnostics));
  *not_found  = static_cast<int32_t>(diag.not_found);
  *nf_R       = diag.nf_R;
  *nf_Z       = diag.nf_Z;
  *bad_i_from = static_cast<int32_t>(diag.bad_i_from);
  *bad_i_to   = static_cast<int32_t>(diag.bad_i_to);
}
