/* particles/mod_runaway_evolution/runaway_evolution_device.kokkos.cpp -- the
 * Kokkos device launcher of the runaway-electron kernel.
 *
 * Same entry point and decomposition as runaway_evolution_device.hip.cpp; the
 * per-particle physics comes from the headers the host arm uses. One step:
 *
 *   1. element_sort::sort  -- bin the particles by element index;
 *   2. jgx_c_scatter       -- move the records into element order;
 *   3. stage               -- per particle: the three weighted velocity moments;
 *   4. accumulate          -- team per element, member per feedback cell: the
 *                             sum over the element's run from tiles in team
 *                             scratch memory, atomic-free;
 *   5. push                -- per particle, in place.
 *
 * The particle order is restored before the download.
 */
#include "particles/mod_runaway_evolution/runaway_evolution_device.h"
#include "particles/mod_runaway_evolution/runaway_evolution.h"
#include "particles/mod_fields_linear/fields_linear.h"
#include "particles/particle_types/particle_set.h"

#include "jgx/data/device_pack.h"
#include "jgx/data/device_buffer.h"
#include "jgx/common/device_sort_kokkos.h"
#include "jgx/jgx_c_api.h"
#include "jgx/macros.h"
#include "jgx/device.h"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>

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

/* One member per feedback cell, rounded up to a whole warp. */
constexpr int kAccumTeam = ((kCells + 63) / 64) * 64;
static_assert(kAccumTeam <= 1024, "one member per cell within the maximum team size");

/* Particles whose factor vectors are held in team scratch memory at once. */
constexpr int kProjTile = 64;

/* Row widths padded to an odd number of doubles, as in the HIP launcher. */
constexpr int kShBF = (kNDeg * kNVtx) | 1;
constexpr int kShHZ = (kNTor) | 1;
constexpr int kShVW = (kNVar) | 1;

using policy_1d   = Kokkos::RangePolicy<Kokkos::IndexType<std::int64_t>>;
using team_policy = Kokkos::TeamPolicy<>;
using team_member = team_policy::member_type;
using team_rows   = Kokkos::View<double**, Kokkos::LayoutRight,
                                 Kokkos::DefaultExecutionSpace::scratch_memory_space,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

/* ------------------------------------------------------------------------- */
/*  the three physics kernels                                                 */
/* ------------------------------------------------------------------------- */

/* Projection, phase one: per particle, the three weighted moments. */
void stage(const part_set& part, const fields_set& fields, double mass,
           double time, double* stg_vw) {
  Kokkos::parallel_for(
      "jorek::re_stage", policy_1d(0, std::int64_t(part.n_records)),
      KOKKOS_LAMBDA(const std::int64_t i) {
        const std::size_t ip = std::size_t(i);
        const jorek::re_projection pr = jorek::re_project(part, ip, fields, mass, time);
        stg_vw[ip*kNVar + 0] = pr.m_Ppar  * pr.weight;
        stg_vw[ip*kNVar + 1] = pr.m_Pperp * pr.weight;
        stg_vw[ip*kNVar + 2] = pr.m_jPhi  * pr.weight;
      });
}

/* Projection, phase two: one team per element, one member per feedback cell.
 *
 * Per tile, the members fill the factor vectors of the tile's particles into
 * team scratch memory, one particle each; after a barrier every member sums its
 * own cell over the tile into a private accumulator, with Kahan compensation.
 * Each cell is written once. */
template <class RunStart, class RunLen>
void accumulate(const part_set& part, const fields_set& fields,
                std::size_t n_elements, const RunStart& run_start,
                const RunLen& run_len, const double* stg_vw,
                const jorek::re_rhs_view_device& rhs,
                const jorek::re_projection_indices& idx) {
  const std::size_t sh_bytes = team_rows::shmem_size(kProjTile, kShBF)
                             + team_rows::shmem_size(kProjTile, kShHZ)
                             + team_rows::shmem_size(kProjTile, kShVW);
  Kokkos::parallel_for(
      "jorek::re_accumulate",
      team_policy(std::int64_t(n_elements), kAccumTeam)
          .set_scratch_size(0, Kokkos::PerTeam(sh_bytes)),
      KOKKOS_LAMBDA(const team_member& team) {
        const std::size_t ie = std::size_t(team.league_rank());
        const team_rows sh_bf(team.team_scratch(0), kProjTile, kShBF);
        const team_rows sh_hz(team.team_scratch(0), kProjTile, kShHZ);
        const team_rows sh_vw(team.team_scratch(0), kProjTile, kShVW);

        const int start = int(run_start(ie));
        const int cnt   = int(run_len(ie));

        const int c = team.team_rank();
        const bool mine = (c < kCells);

        /* c = var + kNVar*(it + kNTor*(m + kNVtx*n)) */
        int d = c;
        const int var = d % kNVar; d /= kNVar;
        const int it  = d % kNTor; d /= kNTor;
        const int m   = d % kNVtx; d /= kNVtx;
        const int n   = d;
        const int nm  = n * kNVtx + m;

        double acc = 0.0, cmp = 0.0;
        for (int base = 0; base < cnt; base += kProjTile) {
          const int tile = (cnt - base < kProjTile) ? cnt - base : kProjTile;

          /* (a) one member per tile particle fills its factor vectors. */
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, tile), [&](const int q) {
                const std::size_t p = std::size_t(start + base + q);
                const std::size_t bf_ext[2] = { kNDeg, kNVtx };
                const std::size_t hz_ext[1] = { kNTor };
                const jgx::view<double, 2, jgx::layout_right> HH(&sh_bf(q, 0), bf_ext);
                const jgx::view<double, 1> HZ(&sh_hz(q, 0), hz_ext);
                basisfunctions::basisfunctions_2D_0_T(part.st(p, 0), part.st(p, 1), HH);
                interp::mode_moivre_explicit(part.x(p, 2), HZ, JGX_N_TOR, JGX_N_PERIOD);
                sh_vw(q, 0) = stg_vw[p*kNVar + 0];
                sh_vw(q, 1) = stg_vw[p*kNVar + 1];
                sh_vw(q, 2) = stg_vw[p*kNVar + 2];
              });
          team.team_barrier();

          /* (b) each member sums its cell over the tile. */
          if (mine) {
            double s = 0.0;
            for (int q = 0; q < tile; ++q)
              s += sh_bf(q, nm) * sh_hz(q, it) * sh_vw(q, var);
            const double y  = s - cmp;
            const double tt = acc + y;
            cmp = (tt - acc) - y;
            acc = tt;
          }
          team.team_barrier();
        }

        if (!mine) return;

        const std::size_t pv = (var == 0) ? idx.P_par
                             : (var == 1) ? idx.P_perp
                                          : idx.j_Phi;
        rhs(n, m, ie, it, pv) +=
            acc * fields.element_list.size(ie, std::size_t(m), std::size_t(n));
      });
}

/* The pusher, per particle, in place. First occurrence of each diagnostic wins,
 * elected by compare-and-swap. */
void push(const part_set& part, const fields_set& fields, double mass,
          double time, double timestep, double phi_search, diagnostics* diag) {
  Kokkos::parallel_for(
      "jorek::re_push", policy_1d(0, std::int64_t(part.n_records)),
      KOKKOS_LAMBDA(const std::int64_t i) {
        int ifail = 0;
        diagnostics my_diag;
        kinetic_relativistic::volume_preserving_push_jorek<kFindRZNearbyDebug>(
            part, std::size_t(i), fields, mass, time, timestep, phi_search,
            ifail, my_diag);

        if (my_diag.not_found != 0 &&
            Kokkos::atomic_compare_exchange(&diag->not_found, 0,
                                            my_diag.not_found) == 0) {
          diag->nf_R = my_diag.nf_R;
          diag->nf_Z = my_diag.nf_Z;
        }
        if (my_diag.bad_i_to != 0 &&
            Kokkos::atomic_compare_exchange(&diag->bad_i_to, 0,
                                            my_diag.bad_i_to) == 0) {
          diag->bad_i_from = my_diag.bad_i_from;
        }
      });
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

  /* --- the packs ------------------------------------------------------------ */
  const jgx::data::device_pack part_pk(*part_desc, n_particles);
  const jgx::data::device_pack el_pk(*el_desc, n_elements);
  const jgx::data::device_pack nd_pk(*nd_desc, n_nodes);
  part_pk.upload(part_base);
  el_pk.upload(el_base);
  nd_pk.upload(nd_base);

  /* The sort's destination block. */
  const device_buffer alt_blk(jgx::data::soa_block_bytes(*part_desc, n_particles));

  void* curr_base = part_pk.base();
  void* alt_base  = alt_blk.get();

  const fields_set fields = jorek::make_fields_set(
      jorek::element_set_soa::from_soa(el_pk.base(), *el_desc, n_elements),
      jorek::node_set_soa::from_soa(nd_pk.base(), *nd_desc, n_nodes),
      jorek::make_fields_interp_linear_set<double>(interp_base, *interp_desc));

  /* --- the feedback array: this call's contribution, added on the host ------- */
  std::size_t rhs_size = 1;
  for (int d = 0; d < 5; ++d) rhs_size *= rhs_ext[d];

  const device_buffer rhs_buf(rhs_size * sizeof(double));
  const jorek::re_rhs_view_device rhs(rhs_buf.as<double>(), rhs_ext);
  const jorek::re_projection_indices proj = { idx[0], idx[1], idx[2] };

  const device_buffer diag_buf(sizeof(diagnostics));

  /* --- the sort's working set ----------------------------------------------- */
  const int n_bins = static_cast<int>(n_elements) + 1;
  jgx::element_sort sorter(n_particles, n_bins);
  const device_buffer slot(n_particles * sizeof(std::int32_t));

  const device_buffer orig_a(n_particles * sizeof(std::int32_t));
  const device_buffer orig_b(n_particles * sizeof(std::int32_t));
  std::int32_t* orig_curr = orig_a.as<std::int32_t>();
  std::int32_t* orig_alt  = orig_b.as<std::int32_t>();

  /* --- the projection staging ----------------------------------------------- */
  const device_buffer stg_vw(n_particles * kNVar * sizeof(double));

  jgx::iota_index(orig_curr, n_particles);

  /* --- the step loop: queued, not synchronised until after the loop ---------- */
  for (int k = 0; k < nstep; ++k) {
    part_set curr = part_set::from_soa(curr_base, *part_desc, n_particles);

    /* (1, 2) sort by element, move the records and the original index */
    sorter.sort(curr.i_elm.data, slot.as<std::int32_t>());
    jgx_c_scatter(alt_base, curr_base, part_desc, n_particles,
                  slot.as<std::int32_t>());
    jgx::scatter_index(orig_alt, orig_curr, slot.as<std::int32_t>(), n_particles);
    std::swap(curr_base, alt_base);
    std::swap(orig_curr, orig_alt);
    curr = part_set::from_soa(curr_base, *part_desc, n_particles);

    /* (3, 4) project */
    stage(curr, fields, mass, time, stg_vw.as<double>());
    accumulate(curr, fields, n_elements, sorter.run_start(), sorter.run_len(),
               stg_vw.as<double>(), rhs, proj);

    /* (5) push */
    push(curr, fields, mass, time, timestep, phi_search,
         static_cast<diagnostics*>(diag_buf.get()));
  }

  /* Undo the sorts. */
  jgx_c_scatter(alt_base, curr_base, part_desc, n_particles, orig_curr);
  std::swap(curr_base, alt_base);

  jgx_c_synchronize();

  /* --- back to the host ----------------------------------------------------- */
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
