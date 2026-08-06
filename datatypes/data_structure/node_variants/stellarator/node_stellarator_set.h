/* node_variants/stellarator/node_stellarator_set.h -- the node container
 * extension for a stellarator model.
 *
 */
 
 /**
 * @todo The stellarator node has three preprocessor switches 
 * -- JOREK_MODEL 180, USE_DOMM and USE_EXT_FIELD -- and they are replicated here.
 * They shouldn't exist on the fortran side in the first place, 
 * but for now this solution is acceptable since it's constrained to this file
 * only.
 */
#ifndef JOREK_NODE_STELLARATOR_SET_H
#define JOREK_NODE_STELLARATOR_SET_H

#include <cstddef>
#include "datatypes/data_structure/node_set.h"
#include "jgx/jorek/jgx_record_ids.h"
#include "jgx/jgx_record_api.h"
#include "jgx/macros.h"
#include "jgx/field_view.h"
#include "jgx/view.h"

namespace jorek {

/* What the stellarator arm adds, in declaration order. Ids start at 0: this is
 * a record of its own, not a continuation of node_field. */
enum node_stellarator_field {
  JGX_NST_R_TOR_EQ = 0,
#if JOREK_MODEL == 180
  JGX_NST_PRESSURE,
  JGX_NST_J_FIELD,
  JGX_NST_B_FIELD,
#endif
#ifndef USE_DOMM
#ifdef USE_EXT_FIELD
  JGX_NST_B_VAC_FIELD,
#else
  JGX_NST_CHI_CORRECTION,
#endif
#endif
  JGX_NST_J_SOURCE,
  JGX_NST_COUNT
};

template <class L, class Real = double, class Int = int>
struct node_stellarator_set : node_set<L, Real, Int> {
  view<Real, 2, L> r_tor_eq;         /* (in, kf)          sqrt of normalised toroidal flux */
#if JOREK_MODEL == 180
  view<Real, 2, L> pressure;         /* (in, kf)          scalar pressure, from GVEC       */
  view<Real, 4, L> j_field;          /* (in, kc, kf, kd+1) current density, from GVEC      */
  view<Real, 4, L> b_field;          /* (in, kc, kf, kd+1) magnetic field, from GVEC       */
#endif
#ifndef USE_DOMM
#ifdef USE_EXT_FIELD
  view<Real, 4, L> b_vac_field;      /* (in, kc, kf, kd+1) vacuum magnetic field           */
#else
  view<Real, 3, L> chi_correction;   /* (in, kc, kf)      correction to the vacuum field   */
#endif
#endif
  view<Real, 3, L> j_source;         /* (in, kt, kf)      current source                   */
};

using node_stellarator_set_aos = node_stellarator_set<layout_stride>;
using node_stellarator_set_soa = node_stellarator_set<layout_left>;

/* One packed buffer per added field. The base half keeps its own node_soa_ptrs. */
struct node_stellarator_soa_ptrs {
  void* field[JGX_NST_COUNT] = {};
};

template <class Real, class Int>
JGX_HD inline void
fill_node_stellarator_aos(node_stellarator_set<layout_stride, Real, Int>& s,
                          void* record_base, const jgx_record_desc& r,
                          std::size_t n_nodes) {
  const std::size_t rs = r.record_stride_bytes;

  s.r_tor_eq       = jgx::aos_field<Real, 2>(record_base, r.field[JGX_NST_R_TOR_EQ      ], rs, n_nodes);
#if JOREK_MODEL == 180
  s.pressure       = jgx::aos_field<Real, 2>(record_base, r.field[JGX_NST_PRESSURE      ], rs, n_nodes);
  s.j_field        = jgx::aos_field<Real, 4>(record_base, r.field[JGX_NST_J_FIELD       ], rs, n_nodes);
  s.b_field        = jgx::aos_field<Real, 4>(record_base, r.field[JGX_NST_B_FIELD       ], rs, n_nodes);
#endif
#ifndef USE_DOMM
#ifdef USE_EXT_FIELD
  s.b_vac_field    = jgx::aos_field<Real, 4>(record_base, r.field[JGX_NST_B_VAC_FIELD   ], rs, n_nodes);
#else
  s.chi_correction = jgx::aos_field<Real, 3>(record_base, r.field[JGX_NST_CHI_CORRECTION], rs, n_nodes);
#endif
#endif
  s.j_source       = jgx::aos_field<Real, 3>(record_base, r.field[JGX_NST_J_SOURCE      ], rs, n_nodes);
}

template <class Real, class Int>
JGX_HD inline void
fill_node_stellarator_soa(node_stellarator_set<layout_left, Real, Int>& s,
                          void* const* p, const jgx_record_desc& r,
                          std::size_t n_nodes) {
  s.r_tor_eq       = jgx::soa_field<Real, 2>(p[JGX_NST_R_TOR_EQ      ], r.field[JGX_NST_R_TOR_EQ      ], n_nodes);
#if JOREK_MODEL == 180
  s.pressure       = jgx::soa_field<Real, 2>(p[JGX_NST_PRESSURE      ], r.field[JGX_NST_PRESSURE      ], n_nodes);
  s.j_field        = jgx::soa_field<Real, 4>(p[JGX_NST_J_FIELD       ], r.field[JGX_NST_J_FIELD       ], n_nodes);
  s.b_field        = jgx::soa_field<Real, 4>(p[JGX_NST_B_FIELD       ], r.field[JGX_NST_B_FIELD       ], n_nodes);
#endif
#ifndef USE_DOMM
#ifdef USE_EXT_FIELD
  s.b_vac_field    = jgx::soa_field<Real, 4>(p[JGX_NST_B_VAC_FIELD   ], r.field[JGX_NST_B_VAC_FIELD   ], n_nodes);
#else
  s.chi_correction = jgx::soa_field<Real, 3>(p[JGX_NST_CHI_CORRECTION], r.field[JGX_NST_CHI_CORRECTION], n_nodes);
#endif
#endif
  s.j_source       = jgx::soa_field<Real, 3>(p[JGX_NST_J_SOURCE      ], r.field[JGX_NST_J_SOURCE      ], n_nodes);
}

/* rn describes JGX_REC_NODE, rs_ describes JGX_REC_NODE_STELLARATOR. */
template <class Real = double, class Int = int>
JGX_HD inline node_stellarator_set<layout_stride, Real, Int>
make_node_stellarator_set_aos(void* record_base, const jgx_record_desc& rn,
                              const jgx_record_desc& rs_, std::size_t n_nodes) {
  node_stellarator_set<layout_stride, Real, Int> s;
  fill_node_base_aos(s, record_base, rn, n_nodes);
  fill_node_stellarator_aos(s, record_base, rs_, n_nodes);
  return s;
}

template <class Real = double, class Int = int>
JGX_HD inline node_stellarator_set<layout_left, Real, Int>
make_node_stellarator_set_soa(const node_soa_ptrs& pn,
                              const node_stellarator_soa_ptrs& ps,
                              const jgx_record_desc& rn,
                              const jgx_record_desc& rs_,
                              std::size_t n_nodes) {
  node_stellarator_set<layout_left, Real, Int> s;
  fill_node_base_soa(s, pn.field, rn, n_nodes);
  fill_node_stellarator_soa(s, ps.field, rs_, n_nodes);
  return s;
}

/* Address an existing Fortran node array in place, reading both offset tables
 * that mod_jgx_node_record.f90 and mod_jgx_node_variant_record.f90 filled. */
inline node_stellarator_set_aos
node_stellarator_set_from_registry(void* base, std::size_t n_nodes) {
  return make_node_stellarator_set_aos<>(
      base,
      jgx::registered_record(JGX_REC_NODE,             JGX_NF_COUNT),
      jgx::registered_record(JGX_REC_NODE_STELLARATOR, JGX_NST_COUNT),
      n_nodes);
}

} /* namespace jorek */

#endif /* JOREK_NODE_STELLARATOR_SET_H */
