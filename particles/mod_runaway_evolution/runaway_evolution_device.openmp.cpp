/* particles/mod_runaway_evolution/runaway_evolution_device.openmp.cpp -- the
 * OpenMP offload launcher of the runaway-electron kernel.
 *
 * A port of runaway_evolution_device.hip.cpp: the same five launches per step,
 * the same decomposition, the same buffers. A HIP block is a team, a HIP thread
 * an iteration of an inner parallel loop, and the end of that loop the barrier.
 * Launches are synchronous.
 */
#include "particles/mod_runaway_evolution/runaway_evolution_device.h"
#include "particles/mod_runaway_evolution/runaway_evolution.h"
#include "particles/mod_fields_linear/fields_linear.h"
#include "particles/particle_types/particle_set.h"

#include "jgx/data/device_pack.h"
#include "jgx/data/device_buffer.h"
#include "jgx/common/device_sort_openmp.h"
#include "jgx/jgx_c_api.h"
#include "jgx/macros.h"
#include "jgx/device.h"

#include <cstddef>
#include <cstdint>
#include <utility>

/* Same switch as the host shim. */
#ifdef DEBUG
static constexpr bool kFindRZNearbyDebug = true;
#else
static constexpr bool kFindRZNearbyDebug = false;
#endif

namespace {

using part_set    = jorek::particle_kin_rel_set_soa;
using fields_set  = jorek::fields_linear_set_soa;
using diagnostics = kinetic_relativistic::push_diagnostics;

constexpr int kNDeg = JGX_N_DEGREES;
constexpr int kNVtx = JGX_N_VERTEX_MAX;
constexpr int kNTor = JGX_N_TOR;
constexpr int kNVar = 3;                     /* P_par, P_perp, j_Phi */

/* One element's share of the feedback array. */
constexpr int kCells = kNDeg * kNVtx * kNTor * kNVar;

/* Particles whose factor vectors are held in team memory at once. */
constexpr int kProjTile = 64;

/* Row widths. The HIP launcher pads them to odd widths for its warp banks,
 * which have no counterpart here. */
constexpr int kShBF = kNDeg * kNVtx;
constexpr int kShHZ = kNTor;
constexpr int kShVW = kNVar;

/* ------------------------------------------------------------------------- */
/*  the three physics kernels                                                 */
/* ------------------------------------------------------------------------- */

/* Projection, phase one: the field evaluation, iteration per particle. */
void proj_stage_kernel(part_set part, fields_set fields, double mass, double time,
                       double* stg_Ppar, double* stg_Pperp, double* stg_jPhi) {
  const std::size_t n = part.n_records;
  #pragma omp target teams distribute parallel for \
      is_device_ptr(stg_Ppar, stg_Pperp, stg_jPhi) map(to: part, fields)
  for (std::size_t ip = 0; ip < n; ++ip) {
    const jorek::re_projection pr = jorek::re_project(part, ip, fields, mass, time);
    stg_Ppar [ip] = pr.m_Ppar  * pr.weight;
    stg_Pperp[ip] = pr.m_Pperp * pr.weight;
    stg_jPhi [ip] = pr.m_jPhi  * pr.weight;
  }
}

/* Projection, phase two: one team per element, one inner iteration per feedback
 * cell. Each tile's factor vectors are worked out once into team memory, and
 * every cell sums the tile into its own Kahan accumulator. */
void proj_accumulate_kernel(part_set part, fields_set fields, std::size_t n_elements,
                            const int* run_start, const int* run_len,
                            const double* stg_Ppar, const double* stg_Pperp,
                            const double* stg_jPhi, jorek::re_rhs_view_device rhs,
                            jorek::re_projection_indices idx) {
  #pragma omp target teams distribute \
      is_device_ptr(run_start, run_len, stg_Ppar, stg_Pperp, stg_jPhi) \
      map(to: part, fields, rhs, idx)
  for (std::size_t ie = 0; ie < n_elements; ++ie) {
    double sh_bf[kProjTile][kShBF];
    double sh_hz[kProjTile][kShHZ];
    double sh_vw[kProjTile][kShVW];
    double sum[kCells];
    double cmp[kCells];

    const int start = run_start[ie];
    const int cnt   = run_len[ie];

    #pragma omp parallel for
    for (int c = 0; c < kCells; ++c) {
      sum[c] = 0.0;
      cmp[c] = 0.0;
    }

    for (int base = 0; base < cnt; base += kProjTile) {
      const int tile = (cnt - base < kProjTile) ? cnt - base : kProjTile;

      /* (a) one iteration per tile particle works out its factor vectors, once.
       * They are evaluated into iteration-local scratch and copied: nvc++ faults
       * when a called routine writes team memory through a pointer. */
      #pragma omp parallel for
      for (int q = 0; q < tile; ++q) {
        const std::size_t p = static_cast<std::size_t>(start + base + q);

        double bf[kShBF];
        double hz[kShHZ];
        const std::size_t bf_ext[2] = { kNDeg, 4 };
        const std::size_t hz_ext[1] = { kNTor };
        const jgx::view<double, 2, jgx::layout_right> HH(bf, bf_ext);
        const jgx::view<double, 1> HZ(hz, hz_ext);

        basisfunctions::basisfunctions_2D_0_T(part.st(p, 0), part.st(p, 1), HH);
        interp::mode_moivre_explicit(part.x(p, 2), HZ, JGX_N_TOR, JGX_N_PERIOD);

        for (int k = 0; k < kShBF; ++k) sh_bf[q][k] = bf[k];
        for (int k = 0; k < kShHZ; ++k) sh_hz[q][k] = hz[k];

        sh_vw[q][0] = stg_Ppar [p];
        sh_vw[q][1] = stg_Pperp[p];
        sh_vw[q][2] = stg_jPhi [p];
      }

      /* (b) each cell sums this tile into its own accumulator. */
      #pragma omp parallel for
      for (int c = 0; c < kCells; ++c) {
        /* c = var + kNVar*(it + kNTor*(m + kNVtx*n)) */
        int d = c;
        const int var = d % kNVar; d /= kNVar;
        const int it  = d % kNTor; d /= kNTor;
        const int m   = d % kNVtx; d /= kNVtx;
        const int nm  = d * kNVtx + m;

        double s = 0.0;
        for (int q = 0; q < tile; ++q)
          s += sh_bf[q][nm] * sh_hz[q][it] * sh_vw[q][var];
        const double y  = s - cmp[c];
        const double tt = sum[c] + y;
        cmp[c] = (tt - sum[c]) - y;
        sum[c] = tt;
      }
    }

    #pragma omp parallel for
    for (int c = 0; c < kCells; ++c) {
      int d = c;
      const int var = d % kNVar; d /= kNVar;
      const int it  = d % kNTor; d /= kNTor;
      const int m   = d % kNVtx; d /= kNVtx;
      const int n   = d;

      const std::size_t pv = (var == 0) ? idx.P_par
                           : (var == 1) ? idx.P_perp
                                        : idx.j_Phi;
      rhs(n, m, ie, it, pv) +=
          sum[c] * fields.element_list.size(ie, static_cast<std::size_t>(m),
                                            static_cast<std::size_t>(n));
    }
  }
}

/* The pusher, iteration per particle, in place. First occurrence of a diagnostic
 * wins: the atomic capture of its claim flag elects the writer. */
void push_kernel(part_set part, fields_set fields, double mass, double time,
                 double timestep, double phi_search, diagnostics* diag, int* claim) {
  const std::size_t n = part.n_records;
  #pragma omp target teams distribute parallel for \
      is_device_ptr(diag, claim) map(to: part, fields)
  for (std::size_t ip = 0; ip < n; ++ip) {
    int ifail = 0;
    diagnostics my_diag;
    kinetic_relativistic::volume_preserving_push_jorek<kFindRZNearbyDebug>(
        part, ip, fields, mass, time, timestep, phi_search, ifail, my_diag);

    if (my_diag.not_found != 0) {
      int taken;
      #pragma omp atomic capture
      { taken = claim[0]; claim[0] = 1; }
      if (taken == 0) {
        diag->not_found = my_diag.not_found;
        diag->nf_R = my_diag.nf_R;
        diag->nf_Z = my_diag.nf_Z;
      }
    }
    if (my_diag.bad_i_to != 0) {
      int taken;
      #pragma omp atomic capture
      { taken = claim[1]; claim[1] = 1; }
      if (taken == 0) {
        diag->bad_i_to = my_diag.bad_i_to;
        diag->bad_i_from = my_diag.bad_i_from;
      }
    }
  }
}

using jgx::device_buffer;

} /* anonymous namespace */


extern "C" void jgx_device_runaway_evolution_evolve_REs(
    void* part_base, const jgx_record_desc* part_desc, size_t n_particles,
    void* el_base,   const jgx_record_desc* el_desc,   size_t n_elements,
    void* nd_base,   const jgx_record_desc* nd_desc,   size_t n_nodes,
    const void* interp_base, const jgx_record_desc* interp_desc,
    double* rhs_data, const size_t* rhs_ext, const size_t* idx,
    double mass, double time, double timestep, int32_t nstep, double phi_search,
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
   * That is also what frees the layout -- see re_rhs_view_device.
   */
  std::size_t rhs_size = 1;
  for (int d = 0; d < 5; ++d) rhs_size *= rhs_ext[d];

  const device_buffer rhs_buf(rhs_size * sizeof(double));
  const jorek::re_rhs_view_device rhs(rhs_buf.as<double>(), rhs_ext);
  const jorek::re_projection_indices proj = { idx[0], idx[1], idx[2] };

  const device_buffer diag_buf(sizeof(diagnostics));
  const device_buffer claim_buf(2 * sizeof(int));   /* not_found, bad_i_to */

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
  /* Three arrays, not seven: (s, t, phi) are read straight from the sorted
   * particle set, which phase two holds anyway. */
  const device_buffer stg_Ppar (n_particles * sizeof(double));
  const device_buffer stg_Pperp(n_particles * sizeof(double));
  const device_buffer stg_jPhi (n_particles * sizeof(double));

  jgx::iota_index(orig_curr, n_particles);

  /* --- the step loop ------------------------------------------------------ */
  for (int k = 0; k < nstep; ++k) {
    part_set curr = part_set::from_soa(curr_base, *part_desc, n_particles);

    /* (1) sort by element */
    jgx::count_bins(curr.i_elm.data, n_particles, n_bins, hist.as<int>());
    jgx::scan_bins(hist.as<int>(), offsets.as<int>(), cursors.as<int>(),
                   run_len.as<int>(), n_bins);
    jgx::assign_slots(curr.i_elm.data, n_particles, n_bins, cursors.as<int>(),
                      slot.as<std::int32_t>());

    /* (2) move the records, and the original index with them */
    jgx_c_scatter(alt_base, curr_base, part_desc, n_particles,
                  slot.as<std::int32_t>());
    jgx::scatter_index(orig_alt, orig_curr, slot.as<std::int32_t>(), n_particles);
    std::swap(curr_base, alt_base);
    std::swap(orig_curr, orig_alt);
    curr = part_set::from_soa(curr_base, *part_desc, n_particles);

    /* (3, 4) project: stage, then accumulate atomic-free */
    proj_stage_kernel(curr, fields, mass, time, stg_Ppar.as<double>(),
                      stg_Pperp.as<double>(), stg_jPhi.as<double>());
    proj_accumulate_kernel(curr, fields, n_elements, offsets.as<int>(),
                           run_len.as<int>(), stg_Ppar.as<double>(),
                           stg_Pperp.as<double>(), stg_jPhi.as<double>(), rhs, proj);

    /* (5) push */
    push_kernel(curr, fields, mass, time, timestep, phi_search,
                static_cast<diagnostics*>(diag_buf.get()), claim_buf.as<int>());
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
  const jorek::re_rhs_view_device src(rhs_dev, rhs_ext);
  const jorek::re_rhs_view        dst(rhs_data, rhs_ext);
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
