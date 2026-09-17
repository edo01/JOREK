/* particles/mod_runaway_evolution/runaway_evolution_device.alpaka.cpp -- the
 * alpaka device launcher of the runaway-electron kernel.
 *
 * Same decomposition as runaway_evolution_device.hip.cpp, launch for launch:
 *
 *   1. count_bins / scan_bins / assign_slots  -- the counting sort by element;
 *   2. jgx_c_scatter                          -- the move itself;
 *   3. proj_stage_kernel                      -- thread per particle: the three
 *      velocity moments, staged;
 *   4. proj_accumulate_kernel                 -- block per element, thread per
 *      feedback cell, tiles in block-shared memory;
 *   5. push_kernel                            -- thread per particle, in place.
 *
 * The particle order is restored before the download.
 */
#include "particles/mod_runaway_evolution/runaway_evolution_device.h"
#include "particles/mod_runaway_evolution/runaway_evolution.h"
#include "particles/mod_fields_linear/fields_linear.h"
#include "particles/particle_types/particle_set.h"

#include "jgx/data/device_pack.h"
#include "jgx/data/device_buffer.h"
#include "jgx/common/device_sort_alpaka.h"
#include "jgx/jgx_c_api.h"
#include "jgx/macros.h"
#include "jgx/device.h"

#include "jgx/cpp/backends/alpaka/alpaka_device.h"

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

using jgx::alpaka_backend::queue;
using jgx::alpaka_backend::thread_index;
using jgx::alpaka_backend::thread_spec;

constexpr int kNDeg = JGX_N_DEGREES;
constexpr int kNVtx = JGX_N_VERTEX_MAX;
constexpr int kNTor = JGX_N_TOR;
constexpr int kNVar = 3;                     /* P_par, P_perp, j_Phi */

/* One element's share of the feedback array. */
constexpr int kCells = kNDeg * kNVtx * kNTor * kNVar;

constexpr unsigned kBlock = 256;             /* the per-particle kernels */

/* One element and one cell per thread, rounded up to a whole warp. */
constexpr unsigned kAccumBlock = ((kCells + 63) / 64) * 64;
static_assert(kAccumBlock >= static_cast<unsigned>(kCells), "one cell per thread");

/* Particles whose factor vectors are held in shared memory at once. */
constexpr int kProjTile = 64;

/* Row widths padded to an odd number of doubles, as in the HIP launcher. */
constexpr int kShBF = (kNDeg * kNVtx) | 1;
constexpr int kShHZ = (kNTor) | 1;
constexpr int kShVW = (kNVar) | 1;

/* ------------------------------------------------------------------------- */
/*  the three physics kernels                                                 */
/* ------------------------------------------------------------------------- */

/* Projection, phase one: the field evaluation, thread per particle. */
struct proj_stage_kernel {
  part_set part;
  fields_set fields;
  double mass;
  double time;
  double* stg_Ppar;
  double* stg_Pperp;
  double* stg_jPhi;

  ALPAKA_FN_ACC void operator()(alpaka::onAcc::concepts::Acc auto const& acc) const {
    const std::size_t ip = thread_index(acc);
    if (ip >= part.n_records) return;

    const jorek::re_projection pr = jorek::re_project(part, ip, fields, mass, time);
    stg_Ppar [ip] = pr.m_Ppar  * pr.weight;
    stg_Pperp[ip] = pr.m_Pperp * pr.weight;
    stg_jPhi [ip] = pr.m_jPhi  * pr.weight;
  }
};

/* Projection, phase two: one block per element, one thread per feedback cell.
 * Each tile's factor vectors are filled cooperatively into block-shared memory,
 * and every cell-thread sums the tile into its own Kahan accumulator. */
struct proj_accumulate_kernel {
  part_set part;
  fields_set fields;
  const int* run_start;
  const int* run_len;
  const double* stg_Ppar;
  const double* stg_Pperp;
  const double* stg_jPhi;
  jorek::re_rhs_view_device rhs;
  jorek::re_projection_indices idx;

  ALPAKA_FN_ACC void operator()(alpaka::onAcc::concepts::Acc auto const& acc) const {
    const std::size_t ie = acc[alpaka::layer::block].idx().x();
    const int n_threads = static_cast<int>(acc[alpaka::layer::thread].count().x());

    auto sh_bf = alpaka::onAcc::declareSharedMdArray<double, alpaka::uniqueId()>(
        acc, alpaka::CVec<std::uint32_t, kProjTile, kShBF>{});
    auto sh_hz = alpaka::onAcc::declareSharedMdArray<double, alpaka::uniqueId()>(
        acc, alpaka::CVec<std::uint32_t, kProjTile, kShHZ>{});
    auto sh_vw = alpaka::onAcc::declareSharedMdArray<double, alpaka::uniqueId()>(
        acc, alpaka::CVec<std::uint32_t, kProjTile, kShVW>{});

    const int start = run_start[ie];
    const int cnt   = run_len[ie];

    const int c = static_cast<int>(acc[alpaka::layer::thread].idx().x());
    const bool mine = (c < kCells);

    /* c = var + kNVar*(it + kNTor*(m + kNVtx*n)) */
    int d = c;
    const int var = d % kNVar; d /= kNVar;
    const int it  = d % kNTor; d /= kNTor;
    const int m   = d % kNVtx; d /= kNVtx;
    const int n   = d;
    const int nm  = n * kNVtx + m;

    double sum = 0.0, cmp = 0.0;

    for (int base = 0; base < cnt; base += kProjTile) {
      const int tile = (cnt - base < kProjTile) ? cnt - base : kProjTile;

      /* (a) one lane per tile particle works out its factor vectors, once. */
      for (int q = c; q < tile; q += n_threads) {
        const std::size_t p = static_cast<std::size_t>(start + base + q);
        const std::uint32_t uq = static_cast<std::uint32_t>(q);

        const std::size_t bf_ext[2] = { kNDeg, 4 };
        const std::size_t hz_ext[1] = { kNTor };
        const jgx::view<double, 2, jgx::layout_right> HH(&sh_bf[alpaka::Vec{uq, 0u}], bf_ext);
        const jgx::view<double, 1> HZ(&sh_hz[alpaka::Vec{uq, 0u}], hz_ext);

        basisfunctions::basisfunctions_2D_0_T(part.st(p, 0), part.st(p, 1), HH);
        interp::mode_moivre_explicit(part.x(p, 2), HZ, JGX_N_TOR, JGX_N_PERIOD);

        sh_vw[alpaka::Vec{uq, 0u}] = stg_Ppar [p];
        sh_vw[alpaka::Vec{uq, 1u}] = stg_Pperp[p];
        sh_vw[alpaka::Vec{uq, 2u}] = stg_jPhi [p];
      }
      alpaka::onAcc::syncBlockThreads(acc);

      /* (b) each cell-thread sums this tile into its own accumulator. */
      if (mine) {
        double s = 0.0;
        for (std::uint32_t q = 0; q < static_cast<std::uint32_t>(tile); ++q)
          s += sh_bf[alpaka::Vec{q, std::uint32_t(nm)}] *
               sh_hz[alpaka::Vec{q, std::uint32_t(it)}] *
               sh_vw[alpaka::Vec{q, std::uint32_t(var)}];
        const double y  = s - cmp;
        const double tt = sum + y;
        cmp = (tt - sum) - y;
        sum = tt;
      }
      alpaka::onAcc::syncBlockThreads(acc);  /* the next tile may overwrite */
    }

    if (!mine) return;

    const std::size_t pv = (var == 0) ? idx.P_par
                         : (var == 1) ? idx.P_perp
                                      : idx.j_Phi;
    rhs(n, m, ie, it, pv) +=
        sum * fields.element_list.size(ie, static_cast<std::size_t>(m),
                                       static_cast<std::size_t>(n));
  }
};

/* The pusher, thread per particle, in place. First occurrence of a diagnostic
 * wins, elected by the compare-and-swap. */
struct push_kernel {
  part_set part;
  fields_set fields;
  double mass;
  double time;
  double timestep;
  double phi_search;
  diagnostics* diag;

  ALPAKA_FN_ACC void operator()(alpaka::onAcc::concepts::Acc auto const& acc) const {
    const std::size_t ip = thread_index(acc);
    if (ip >= part.n_records) return;

    int ifail = 0;
    diagnostics my_diag;
    kinetic_relativistic::volume_preserving_push_jorek<kFindRZNearbyDebug>(
        part, ip, fields, mass, time, timestep, phi_search, ifail, my_diag);

    using flag_type = decltype(diag->not_found);
    if (my_diag.not_found != 0 &&
        alpaka::onAcc::atomicCas(acc, &diag->not_found, flag_type(0),
                                 my_diag.not_found) == 0) {
      diag->nf_R = my_diag.nf_R;
      diag->nf_Z = my_diag.nf_Z;
    }
    using to_type = decltype(diag->bad_i_to);
    if (my_diag.bad_i_to != 0 &&
        alpaka::onAcc::atomicCas(acc, &diag->bad_i_to, to_type(0),
                                 my_diag.bad_i_to) == 0) {
      diag->bad_i_from = my_diag.bad_i_from;
    }
  }
};

using jgx::device_buffer;
using jgx::grid_for;

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

  /* --- the packs ---------------------------------------------------------- */
  const jgx::data::device_pack part_pk(*part_desc, n_particles);
  const jgx::data::device_pack el_pk(*el_desc, n_elements);
  const jgx::data::device_pack nd_pk(*nd_desc, n_nodes);
  part_pk.upload(part_base);
  el_pk.upload(el_base);
  nd_pk.upload(nd_base);

  const device_buffer alt_blk(jgx::data::soa_block_bytes(*part_desc, n_particles));

  void* curr_base = part_pk.base();
  void* alt_base  = alt_blk.get();

  const fields_set fields = jorek::make_fields_set(
      jorek::element_set_soa::from_soa(el_pk.base(), *el_desc, n_elements),
      jorek::node_set_soa::from_soa(nd_pk.base(), *nd_desc, n_nodes),
      jorek::make_fields_interp_linear_set<double>(interp_base, *interp_desc));

  /* --- the feedback array, zeroed; the host adds this call's contribution -- */
  std::size_t rhs_size = 1;
  for (int d = 0; d < 5; ++d) rhs_size *= rhs_ext[d];

  const device_buffer rhs_buf(rhs_size * sizeof(double));
  const jorek::re_rhs_view_device rhs(rhs_buf.as<double>(), rhs_ext);
  const jorek::re_projection_indices proj = { idx[0], idx[1], idx[2] };

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
  const device_buffer stg_Ppar (n_particles * sizeof(double));
  const device_buffer stg_Pperp(n_particles * sizeof(double));
  const device_buffer stg_jPhi (n_particles * sizeof(double));

  const unsigned grid_p = grid_for(n_particles, kBlock);
  const unsigned grid_e = static_cast<unsigned>(n_elements);

  queue().enqueue(thread_spec(grid_p, kBlock), jgx::iota_index{orig_curr, n_particles});

  /* --- the step loop, queued without a synchronisation --------------------- */
  for (int k = 0; k < nstep; ++k) {
    part_set curr = part_set::from_soa(curr_base, *part_desc, n_particles);

    /* (1) sort by element */
    queue().enqueue(thread_spec(grid_p, kBlock),
                    jgx::count_bins{curr.i_elm.data, n_particles, n_bins,
                                    hist.as<int>()});
    queue().enqueue(thread_spec(1, jgx::kScanThreads),
                    jgx::scan_bins{hist.as<int>(), offsets.as<int>(),
                                   cursors.as<int>(), run_len.as<int>(), n_bins});
    queue().enqueue(thread_spec(grid_p, kBlock),
                    jgx::assign_slots{curr.i_elm.data, n_particles, n_bins,
                                      cursors.as<int>(), slot.as<std::int32_t>()});

    /* (2) move the records, and the original index with them */
    jgx_c_scatter(alt_base, curr_base, part_desc, n_particles,
                  slot.as<std::int32_t>());
    queue().enqueue(thread_spec(grid_p, kBlock),
                    jgx::scatter_index{orig_alt, orig_curr,
                                       slot.as<std::int32_t>(), n_particles});
    std::swap(curr_base, alt_base);
    std::swap(orig_curr, orig_alt);
    curr = part_set::from_soa(curr_base, *part_desc, n_particles);

    /* (3, 4) project: stage, then accumulate atomic-free */
    queue().enqueue(thread_spec(grid_p, kBlock),
                    proj_stage_kernel{curr, fields, mass, time,
                                      stg_Ppar.as<double>(),
                                      stg_Pperp.as<double>(),
                                      stg_jPhi.as<double>()});
    queue().enqueue(thread_spec(grid_e, kAccumBlock),
                    proj_accumulate_kernel{curr, fields, offsets.as<int>(),
                                           run_len.as<int>(), stg_Ppar.as<double>(),
                                           stg_Pperp.as<double>(),
                                           stg_jPhi.as<double>(), rhs, proj});

    /* (5) push */
    queue().enqueue(thread_spec(grid_p, kBlock),
                    push_kernel{curr, fields, mass, time, timestep, phi_search,
                                static_cast<diagnostics*>(diag_buf.get())});
  }

  /* Undo the sorts: record rec goes back to the slot Fortran gave it. */
  jgx_c_scatter(alt_base, curr_base, part_desc, n_particles, orig_curr);
  std::swap(curr_base, alt_base);

  jgx_c_synchronize();

  /* --- back to the host --------------------------------------------------- */
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

  part_pk.download(part_base, curr_base);

  diagnostics diag;
  jgx_c_pull(&diag, diag_buf.get(), sizeof(diagnostics));
  *not_found  = static_cast<int32_t>(diag.not_found);
  *nf_R       = diag.nf_R;
  *nf_Z       = diag.nf_Z;
  *bad_i_from = static_cast<int32_t>(diag.bad_i_from);
  *bad_i_to   = static_cast<int32_t>(diag.bad_i_to);
}
