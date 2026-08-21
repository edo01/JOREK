/* particles/mod_runaway_evolution/runaway_evolution_device.hip.cpp -- the
 * device launcher of the runaway-electron kernel.
 *
 * evolve_RE itself is untouched and is not duplicated here.
 */
#include "particles/mod_runaway_evolution/runaway_evolution_device.h"
#include "particles/mod_runaway_evolution/runaway_evolution.h"
#include "particles/mod_fields_linear/fields_linear.h"
#include "particles/particle_types/particle_set.h"

#include "jgx/data/device_pack.h"
#include "jgx/jgx_c_api.h"
#include "jgx/macros.h"

#include <cstddef>
#include <cstdint>

/* Same switch as the host shim: find_RZ_nearby takes the DEBUG build as a
 * template parameter rather than reading the preprocessor itself. */
#ifdef DEBUG
static constexpr bool kFindRZNearbyDebug = true;
#else
static constexpr bool kFindRZNearbyDebug = false;
#endif

namespace {

using part_set   = jorek::particle_kin_rel_set_soa;
using fields_set = jorek::fields_linear_set_soa;
using diagnostics = kinetic_relativistic::push_diagnostics;


constexpr unsigned kBlock = 256;

/**
 * One particle per thread, the whole nstep loop inside.
 */
JGX_KERNEL void evolve_REs_kernel(part_set part, fields_set fields,
                                  jorek::re_rhs_view rhs,
                                  jorek::re_projection_indices idx,
                                  double mass, double time, double timestep,
                                  int nstep, double phi_search,
                                  diagnostics* diag) {
  const std::size_t ip = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  if (ip >= part.n_records) return;

  diagnostics my_diag;
  jorek::evolve_RE<kFindRZNearbyDebug>(part, ip, fields, rhs, idx, mass, time,
                                       timestep, nstep, phi_search, my_diag);

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

/* A device buffer's worth of a POD, zeroed, freed with the object. The feedback
 * array and the diagnostics both want exactly this and nothing more. */
class device_buffer {
 public:
  explicit device_buffer(std::size_t n_bytes)
      : n_bytes_(n_bytes), p_(jgx_c_alloc(n_bytes)) {
    jgx_c_memset(p_, 0, n_bytes_);
  }
  ~device_buffer() { jgx_c_free(p_); }
  device_buffer(const device_buffer&) = delete;
  device_buffer& operator=(const device_buffer&) = delete;

  void* get() const noexcept { return p_; }
  std::size_t bytes() const noexcept { return n_bytes_; }

 private:
  std::size_t n_bytes_;
  void* p_;
};

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

  /* --- the packs: one contiguous H2D each, then a transpose on the device --- */
  const jgx::data::device_pack part_pk(*part_desc, n_particles);
  const jgx::data::device_pack el_pk(*el_desc, n_elements);
  const jgx::data::device_pack nd_pk(*nd_desc, n_nodes);
  part_pk.upload(part_base);
  el_pk.upload(el_base);
  nd_pk.upload(nd_base);

  part_set part = jorek::particle_kin_rel_set_soa::from_soa(
      part_pk.base(), *part_desc, n_particles);

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
   * Plain layout_left first, which puts consecutive elements 16 doubles apart --
   * every thread in a warp then atomically updates its own cache line. Making
   * `ie` the fastest axis needs no new machinery, only jgx::layout_perm on the
   * alias in runaway_evolution.h; measure before changing it (section 4.5).
   */
  std::size_t rhs_size = 1;
  for (int d = 0; d < 5; ++d) rhs_size *= rhs_ext[d];

  const device_buffer rhs_buf(rhs_size * sizeof(double));
  const jorek::re_rhs_view rhs(static_cast<double*>(rhs_buf.get()), rhs_ext);

  const device_buffer diag_buf(sizeof(diagnostics));

  /* --- launch ------------------------------------------------------------- */
  const jorek::re_projection_indices proj = { idx[0], idx[1], idx[2] };
  const unsigned grid =
      static_cast<unsigned>((n_particles + kBlock - 1) / kBlock);

  if (grid > 0) {
    evolve_REs_kernel<<<grid, kBlock>>>(
        part, fields, rhs, proj, mass, time, timestep,
        static_cast<int>(nstep), phi_search,
        static_cast<diagnostics*>(diag_buf.get()));
  }

  /* Both the launch error and the completion error, so nothing below reads a
   * buffer a failed kernel never wrote. */
  jgx_c_synchronize();

  /* --- back to the host --------------------------------------------------- */
  double* rhs_dev = new double[rhs_size];
  jgx_c_pull(rhs_dev, rhs_buf.get(), rhs_buf.bytes());
  for (std::size_t i = 0; i < rhs_size; ++i) rhs_data[i] += rhs_dev[i];
  delete[] rhs_dev;

  /* Only the particles were written; the mesh comes home unchanged and is
   * dropped with the packs. */
  part_pk.download(part_base);

  diagnostics diag;
  jgx_c_pull(&diag, diag_buf.get(), sizeof(diagnostics));
  *not_found  = static_cast<int32_t>(diag.not_found);
  *nf_R       = diag.nf_R;
  *nf_Z       = diag.nf_Z;
  *bad_i_from = static_cast<int32_t>(diag.bad_i_from);
  *bad_i_to   = static_cast<int32_t>(diag.bad_i_to);
}
