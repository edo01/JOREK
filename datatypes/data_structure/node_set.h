/* datatypes/data_structure/node_set.h -- the node container, in AoS and SoA form.
 *
 * Covers **ONLY the unconditional components** of type_node
 * (datatypes/data_structure/data_structure.f90).
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

/**
 * @todo: The model-guarded ones (psi_eq, Fprof_eq and
 * the STELLARATOR_MODEL block) are out of scope here by design and get their
 * own set when a model needs them. (one offset table cannot describe a field
 * list that moves with a preprocessor switch)
 */

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

/* One packed buffer per field, indexed by node_field. */
struct node_soa_ptrs {
  void* field[JGX_NF_COUNT] = {};
};

template <class Real = double, class Int = int>
JGX_HD inline node_set<layout_stride, Real, Int>
make_node_set_aos(void* record_base, const jgx_record_desc& r,
                  std::size_t n_nodes) {
  const std::size_t rs = r.record_stride_bytes;

  node_set<layout_stride, Real, Int> s;
  s.n_nodes        = n_nodes;
  s.x              = jgx::aos_field<Real, 4>(record_base, r.field[JGX_NF_X             ], rs, n_nodes);
  s.values         = jgx::aos_field<Real, 4>(record_base, r.field[JGX_NF_VALUES        ], rs, n_nodes);
  s.deltas         = jgx::aos_field<Real, 4>(record_base, r.field[JGX_NF_DELTAS        ], rs, n_nodes);
  s.index          = jgx::aos_field<Int , 2>(record_base, r.field[JGX_NF_INDEX         ], rs, n_nodes);
  s.parents        = jgx::aos_field<Int , 2>(record_base, r.field[JGX_NF_PARENTS       ], rs, n_nodes);
  s.boundary       = jgx::aos_field<Int , 1>(record_base, r.field[JGX_NF_BOUNDARY      ], rs, n_nodes);
  s.boundary_index = jgx::aos_field<Int , 1>(record_base, r.field[JGX_NF_BOUNDARY_INDEX], rs, n_nodes);
  s.axis_dof       = jgx::aos_field<Int , 1>(record_base, r.field[JGX_NF_AXIS_DOF      ], rs, n_nodes);
  s.parent_elem    = jgx::aos_field<Int , 1>(record_base, r.field[JGX_NF_PARENT_ELEM   ], rs, n_nodes);
  s.ref_lambda     = jgx::aos_field<Real, 1>(record_base, r.field[JGX_NF_REF_LAMBDA    ], rs, n_nodes);
  s.ref_mu         = jgx::aos_field<Real, 1>(record_base, r.field[JGX_NF_REF_MU        ], rs, n_nodes);
  return s;
}

template <class Real = double, class Int = int>
JGX_HD inline node_set<layout_left, Real, Int>
make_node_set_soa(const node_soa_ptrs& p, const jgx_record_desc& r,
                  std::size_t n_nodes) {
  node_set<layout_left, Real, Int> s;
  s.n_nodes        = n_nodes;
  s.x              = jgx::soa_field<Real, 4>(p.field[JGX_NF_X             ], r.field[JGX_NF_X             ], n_nodes);
  s.values         = jgx::soa_field<Real, 4>(p.field[JGX_NF_VALUES        ], r.field[JGX_NF_VALUES        ], n_nodes);
  s.deltas         = jgx::soa_field<Real, 4>(p.field[JGX_NF_DELTAS        ], r.field[JGX_NF_DELTAS        ], n_nodes);
  s.index          = jgx::soa_field<Int , 2>(p.field[JGX_NF_INDEX         ], r.field[JGX_NF_INDEX         ], n_nodes);
  s.parents        = jgx::soa_field<Int , 2>(p.field[JGX_NF_PARENTS       ], r.field[JGX_NF_PARENTS       ], n_nodes);
  s.boundary       = jgx::soa_field<Int , 1>(p.field[JGX_NF_BOUNDARY      ], r.field[JGX_NF_BOUNDARY      ], n_nodes);
  s.boundary_index = jgx::soa_field<Int , 1>(p.field[JGX_NF_BOUNDARY_INDEX], r.field[JGX_NF_BOUNDARY_INDEX], n_nodes);
  s.axis_dof       = jgx::soa_field<Int , 1>(p.field[JGX_NF_AXIS_DOF      ], r.field[JGX_NF_AXIS_DOF      ], n_nodes);
  s.parent_elem    = jgx::soa_field<Int , 1>(p.field[JGX_NF_PARENT_ELEM   ], r.field[JGX_NF_PARENT_ELEM   ], n_nodes);
  s.ref_lambda     = jgx::soa_field<Real, 1>(p.field[JGX_NF_REF_LAMBDA    ], r.field[JGX_NF_REF_LAMBDA    ], n_nodes);
  s.ref_mu         = jgx::soa_field<Real, 1>(p.field[JGX_NF_REF_MU        ], r.field[JGX_NF_REF_MU        ], n_nodes);
  return s;
}

/* Address an existing Fortran node array in place. Offset or extent come
 * from the registry which mod_jgx_node_record.f90 filled with c_loc measurements. */
inline node_set_aos node_set_from_registry(void* base, std::size_t n_nodes) {
  return make_node_set_aos<>(base, jgx::registered_record(JGX_REC_NODE, JGX_NF_COUNT),
                             n_nodes);
}

} /* namespace jorek */

#endif /* JOREK_NODE_SET_H */
