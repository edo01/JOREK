/* node_variants/fullmhd/node_fullmhd_set.h -- the node container for a
 * full-MHD model.
 */
#ifndef JOREK_NODE_FULLMHD_SET_H
#define JOREK_NODE_FULLMHD_SET_H

#include <cstddef>
#include "datatypes/data_structure/node_set.h"
#include "jgx/jorek/jgx_record_ids.h"
#include "jgx/jgx_record_api.h"
#include "jgx/macros.h"
#include "jgx/field_view.h"
#include "jgx/view.h"

namespace jorek {

/* Ids start at 0: this is a record of its own, not a
 * continuation of node_field. */
enum node_fullmhd_field {
  JGX_NFM_PSI_EQ = 0,
  JGX_NFM_FPROF_EQ,
  JGX_NFM_COUNT
};

template <class L, class Real = double, class Int = int>
struct node_fullmhd_set : node_set<L, Real, Int> {
  view<Real, 2, L> psi_eq;    /* (in, kf) equilibrium flux            */
  view<Real, 2, L> Fprof_eq;  /* (in, kf) equilibrium R*B_phi profile */
};

using node_fullmhd_set_aos = node_fullmhd_set<layout_stride>;
using node_fullmhd_set_soa = node_fullmhd_set<layout_left>;

/* One packed buffer per added field. The base half keeps its own node_soa_ptrs. */
struct node_fullmhd_soa_ptrs {
  void* field[JGX_NFM_COUNT] = {};
};

/* The added half, split out so a further variant could reuse it. */
template <class Real, class Int>
JGX_HD inline void
fill_node_fullmhd_aos(node_fullmhd_set<layout_stride, Real, Int>& s,
                      void* record_base, const jgx_record_desc& r,
                      std::size_t n_nodes) {
  const std::size_t rs = r.record_stride_bytes;

  s.psi_eq   = jgx::aos_field<Real, 2>(record_base, r.field[JGX_NFM_PSI_EQ  ], rs, n_nodes);
  s.Fprof_eq = jgx::aos_field<Real, 2>(record_base, r.field[JGX_NFM_FPROF_EQ], rs, n_nodes);
}

template <class Real, class Int>
JGX_HD inline void
fill_node_fullmhd_soa(node_fullmhd_set<layout_left, Real, Int>& s,
                      void* const* p, const jgx_record_desc& r,
                      std::size_t n_nodes) {
  s.psi_eq   = jgx::soa_field<Real, 2>(p[JGX_NFM_PSI_EQ  ], r.field[JGX_NFM_PSI_EQ  ], n_nodes);
  s.Fprof_eq = jgx::soa_field<Real, 2>(p[JGX_NFM_FPROF_EQ], r.field[JGX_NFM_FPROF_EQ], n_nodes);
}

/* rn describes JGX_REC_NODE, rf describes JGX_REC_NODE_FULLMHD. */
template <class Real = double, class Int = int>
JGX_HD inline node_fullmhd_set<layout_stride, Real, Int>
make_node_fullmhd_set_aos(void* record_base, const jgx_record_desc& rn,
                          const jgx_record_desc& rf, std::size_t n_nodes) {
  node_fullmhd_set<layout_stride, Real, Int> s;
  fill_node_base_aos(s, record_base, rn, n_nodes);
  fill_node_fullmhd_aos(s, record_base, rf, n_nodes);
  return s;
}

template <class Real = double, class Int = int>
JGX_HD inline node_fullmhd_set<layout_left, Real, Int>
make_node_fullmhd_set_soa(const node_soa_ptrs& pn,
                          const node_fullmhd_soa_ptrs& pf,
                          const jgx_record_desc& rn, const jgx_record_desc& rf,
                          std::size_t n_nodes) {
  node_fullmhd_set<layout_left, Real, Int> s;
  fill_node_base_soa(s, pn.field, rn, n_nodes);
  fill_node_fullmhd_soa(s, pf.field, rf, n_nodes);
  return s;
}

/* Address an existing Fortran node array in place, reading both offset tables
 * that mod_jgx_node_record.f90 and mod_jgx_node_variant_record.f90 filled. */
inline node_fullmhd_set_aos node_fullmhd_set_from_registry(void* base,
                                                           std::size_t n_nodes) {
  return make_node_fullmhd_set_aos<>(
      base,
      jgx::registered_record(JGX_REC_NODE,         JGX_NF_COUNT),
      jgx::registered_record(JGX_REC_NODE_FULLMHD, JGX_NFM_COUNT),
      n_nodes);
}

} /* namespace jorek */

#endif /* JOREK_NODE_FULLMHD_SET_H */
