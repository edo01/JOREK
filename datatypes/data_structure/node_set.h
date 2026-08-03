/* datatypes/data_structure/node_set.h -- the node container, in AoS and SoA form.
 *
 * Covers **ONLY the unconditional components** of type_node
 * (datatypes/data_structure/data_structure.f90). The model-guarded ones (psi_eq, Fprof_eq and
 * the STELLARATOR_MODEL block) are out of scope here by design and get their
 * own set when a model needs them. (one offset table cannot describe a field
 * list that moves with a preprocessor switch)
 *
 * axis_node and constrained are also absent: default `logical` has no
 * interoperable kind, so they need logical(c_bool) on the Fortran side first.
 *
 * The field enum and node_set_from_registry() at the bottom are the other half
 * of mod_jgx_node_record.f90; the two field lists are the contract and must
 * stay in the same order. Passing JGX_NF_COUNT to registered_record checks that
 * the two lists are at least the same length; the order is not checked.
 */
#ifndef JOREK_NODE_SET_H
#define JOREK_NODE_SET_H

#include <cstddef>
#include "jgx/jorek/jgx_record_ids.h"
#include "jgx/jgx_record_api.h"
#include "jgx/macros.h"
#include "jgx/field_view.h"
#include "jgx/view.h"

namespace jorek {

using jgx::layout_left;
using jgx::layout_stride;
using jgx::view;

/* type_node, unconditional components only */
enum node_field {
  JGX_NF_X = 0,
  JGX_NF_VALUES,
  JGX_NF_DELTAS,
  JGX_NF_INDEX,
  JGX_NF_PARENTS,
  JGX_NF_BOUNDARY,
  JGX_NF_BOUNDARY_INDEX,
  JGX_NF_AXIS_DOF,
  JGX_NF_PARENT_ELEM,
  JGX_NF_REF_LAMBDA,
  JGX_NF_REF_MU,
  JGX_NF_COUNT
};

template <class L, class Real = double, class Int = int>
struct node_set {
  std::size_t n_nodes = 0;

  view<Real, 4, L> x;              /* (in, kc, kf, kd) */
  view<Real, 4, L> values;         /* (in, kt, kf, kv) */
  view<Real, 4, L> deltas;         /* (in, kt, kf, kv) */
  view<Int , 2, L> index;          /* (in, kf)         */
  view<Int , 2, L> parents;        /* (in, kp)         */
  view<Int , 1, L> boundary;       /* (in)             */
  view<Int , 1, L> boundary_index; /* (in)             */
  view<Int , 1, L> axis_dof;       /* (in)             */
  view<Int , 1, L> parent_elem;    /* (in)             */
  view<Real, 1, L> ref_lambda;     /* (in)             */
  view<Real, 1, L> ref_mu;         /* (in)             */
};

using node_set_aos = node_set<layout_stride>;
using node_set_soa = node_set<layout_left>;

/* Fortran parameters, passed in rather than hardcoded. */
struct node_dims {
  std::size_t n_nodes      = 0;
  std::size_t n_coord_tor  = 0;
  std::size_t n_degrees    = 0;
  std::size_t n_dim        = 0;
  std::size_t n_tor        = 0;
  std::size_t n_values_max = 0;
  std::size_t n_parents    = 2;   /* type_node%parents(2) */
};

/* Byte offsets within one record, measured on the Fortran side. */
struct node_offsets {
  std::size_t record_stride  = 0;
  std::size_t x              = 0;
  std::size_t values         = 0;
  std::size_t deltas         = 0;
  std::size_t index          = 0;
  std::size_t parents        = 0;
  std::size_t boundary       = 0;
  std::size_t boundary_index = 0;
  std::size_t axis_dof       = 0;
  std::size_t parent_elem    = 0;
  std::size_t ref_lambda     = 0;
  std::size_t ref_mu         = 0;
};

/* One packed buffer per field. */
struct node_soa_ptrs {
  void* x              = nullptr;
  void* values         = nullptr;
  void* deltas         = nullptr;
  void* index          = nullptr;
  void* parents        = nullptr;
  void* boundary       = nullptr;
  void* boundary_index = nullptr;
  void* axis_dof       = nullptr;
  void* parent_elem    = nullptr;
  void* ref_lambda     = nullptr;
  void* ref_mu         = nullptr;
};

template <class Real = double, class Int = int>
inline bool node_offsets_are_addressable(const node_offsets& off) {
  const std::size_t rs = off.record_stride;
  return jgx::field_is_addressable<Real>(off.x,              rs)
      && jgx::field_is_addressable<Real>(off.values,         rs)
      && jgx::field_is_addressable<Real>(off.deltas,         rs)
      && jgx::field_is_addressable<Int >(off.index,          rs)
      && jgx::field_is_addressable<Int >(off.parents,        rs)
      && jgx::field_is_addressable<Int >(off.boundary,       rs)
      && jgx::field_is_addressable<Int >(off.boundary_index, rs)
      && jgx::field_is_addressable<Int >(off.axis_dof,       rs)
      && jgx::field_is_addressable<Int >(off.parent_elem,    rs)
      && jgx::field_is_addressable<Real>(off.ref_lambda,     rs)
      && jgx::field_is_addressable<Real>(off.ref_mu,         rs);
}

template <class Real = double, class Int = int>
JGX_HD inline node_set<layout_stride, Real, Int>
make_node_set_aos(void* record_base, const node_offsets& off,
                  const node_dims& d) {
  const std::size_t nn = d.n_nodes;
  const std::size_t e1[1] = { nn };
  const std::size_t ef[2] = { nn, d.n_degrees };
  const std::size_t ep[2] = { nn, d.n_parents };
  const std::size_t ex[4] = { nn, d.n_coord_tor, d.n_degrees, d.n_dim };
  const std::size_t ev[4] = { nn, d.n_tor, d.n_degrees, d.n_values_max };
  const std::size_t rs = off.record_stride;

  node_set<layout_stride, Real, Int> s;
  s.n_nodes        = nn;
  s.x              = jgx::aos_field<Real, 4>(record_base, off.x,              rs, ex);
  s.values         = jgx::aos_field<Real, 4>(record_base, off.values,         rs, ev);
  s.deltas         = jgx::aos_field<Real, 4>(record_base, off.deltas,         rs, ev);
  s.index          = jgx::aos_field<Int , 2>(record_base, off.index,          rs, ef);
  s.parents        = jgx::aos_field<Int , 2>(record_base, off.parents,        rs, ep);
  s.boundary       = jgx::aos_field<Int , 1>(record_base, off.boundary,       rs, e1);
  s.boundary_index = jgx::aos_field<Int , 1>(record_base, off.boundary_index, rs, e1);
  s.axis_dof       = jgx::aos_field<Int , 1>(record_base, off.axis_dof,       rs, e1);
  s.parent_elem    = jgx::aos_field<Int , 1>(record_base, off.parent_elem,    rs, e1);
  s.ref_lambda     = jgx::aos_field<Real, 1>(record_base, off.ref_lambda,     rs, e1);
  s.ref_mu         = jgx::aos_field<Real, 1>(record_base, off.ref_mu,         rs, e1);
  return s;
}

template <class Real = double, class Int = int>
JGX_HD inline node_set<layout_left, Real, Int>
make_node_set_soa(const node_soa_ptrs& p, const node_dims& d) {
  const std::size_t nn = d.n_nodes;
  const std::size_t e1[1] = { nn };
  const std::size_t ef[2] = { nn, d.n_degrees };
  const std::size_t ep[2] = { nn, d.n_parents };
  const std::size_t ex[4] = { nn, d.n_coord_tor, d.n_degrees, d.n_dim };
  const std::size_t ev[4] = { nn, d.n_tor, d.n_degrees, d.n_values_max };

  node_set<layout_left, Real, Int> s;
  s.n_nodes        = nn;
  s.x              = jgx::soa_field<Real, 4>(p.x,              ex);
  s.values         = jgx::soa_field<Real, 4>(p.values,         ev);
  s.deltas         = jgx::soa_field<Real, 4>(p.deltas,         ev);
  s.index          = jgx::soa_field<Int , 2>(p.index,          ef);
  s.parents        = jgx::soa_field<Int , 2>(p.parents,        ep);
  s.boundary       = jgx::soa_field<Int , 1>(p.boundary,       e1);
  s.boundary_index = jgx::soa_field<Int , 1>(p.boundary_index, e1);
  s.axis_dof       = jgx::soa_field<Int , 1>(p.axis_dof,       e1);
  s.parent_elem    = jgx::soa_field<Int , 1>(p.parent_elem,    e1);
  s.ref_lambda     = jgx::soa_field<Real, 1>(p.ref_lambda,     e1);
  s.ref_mu         = jgx::soa_field<Real, 1>(p.ref_mu,         e1);
  return s;
}

/* Address an existing Fortran node array in place. Nothing here assumes an
 * offset or an extent -- both come from the registry, which
 * mod_jgx_node_record.f90 filled with c_loc measurements. */
inline node_set_aos node_set_from_registry(void* base, std::size_t n_nodes) {
  const jgx_record_desc& r = jgx::registered_record(JGX_REC_NODE, JGX_NF_COUNT);

  node_offsets off;
  off.record_stride  = r.record_stride_bytes;
  off.x              = r.field[JGX_NF_X             ].offset_bytes;
  off.values         = r.field[JGX_NF_VALUES        ].offset_bytes;
  off.deltas         = r.field[JGX_NF_DELTAS        ].offset_bytes;
  off.index          = r.field[JGX_NF_INDEX         ].offset_bytes;
  off.parents        = r.field[JGX_NF_PARENTS       ].offset_bytes;
  off.boundary       = r.field[JGX_NF_BOUNDARY      ].offset_bytes;
  off.boundary_index = r.field[JGX_NF_BOUNDARY_INDEX].offset_bytes;
  off.axis_dof       = r.field[JGX_NF_AXIS_DOF      ].offset_bytes;
  off.parent_elem    = r.field[JGX_NF_PARENT_ELEM   ].offset_bytes;
  off.ref_lambda     = r.field[JGX_NF_REF_LAMBDA    ].offset_bytes;
  off.ref_mu         = r.field[JGX_NF_REF_MU        ].offset_bytes;

  node_dims d;
  d.n_nodes      = n_nodes;
  d.n_coord_tor  = r.field[JGX_NF_X      ].intra_extents[0];
  d.n_degrees    = r.field[JGX_NF_X      ].intra_extents[1];
  d.n_dim        = r.field[JGX_NF_X      ].intra_extents[2];
  d.n_tor        = r.field[JGX_NF_VALUES ].intra_extents[0];
  d.n_values_max = r.field[JGX_NF_VALUES ].intra_extents[2];
  d.n_parents    = r.field[JGX_NF_PARENTS].intra_extents[0];

  return make_node_set_aos<>(base, off, d);
}

} /* namespace jorek */

#endif /* JOREK_NODE_SET_H */
