/* datatypes/data_structure/node_set.h -- the node container, in AoS and SoA form.
 *
 * Covers ONLY the unconditional components of type_node. The field list, the
 * kinds, the ranks and the axis names are in jgx/jorek/records/node_record.def,
 * which mod_jgx_node_record.f90 expands as well -- so the two sides are one
 * list, not two that must be kept in the same order.
 */
#ifndef JOREK_NODE_SET_H
#define JOREK_NODE_SET_H

#include <cstddef>
#include <type_traits>
#include "jgx/jorek/jgx_record_ids.h"
#include "jgx/jgx_record_api.h"
#include "jgx/macros.h"
#include "jgx/field_view.h"
#include "jgx/record_set.h"
#include "jgx/view.h"

namespace jorek {

using jgx::layout_left;
using jgx::layout_stride;
using jgx::view;

enum node_field {
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES) JGX_NF_##TAG,
#include "jgx/jorek/records/node_record.def"
#undef JGX_FIELD
  JGX_NF_COUNT
};

using node_soa_ptrs = jgx::record_ptrs<JGX_NF_COUNT>;

template <class L, class Real = double, class Int = int>
struct node_set : jgx::record_set {
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES) view<T, (RANK) + 1, L> comp;
#include "jgx/jorek/records/node_record.def"
#undef JGX_FIELD

  /* Each maker names the layout it builds rather than the current
   * instantiation, so it returns the right set whichever alias it is reached
   * through. */
  JGX_HD static node_set<layout_stride, Real, Int>
  from_aos(void* record_base, const jgx_record_desc& r, std::size_t n_nodes) {
    node_set<layout_stride, Real, Int> s;
    s.n_records = n_nodes;
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES)                              \
    s.comp = jgx::aos_field<T, (RANK) + 1>(record_base, r.field[JGX_NF_##TAG], \
                                           r.record_stride_bytes, n_nodes);
#include "jgx/jorek/records/node_record.def"
#undef JGX_FIELD
    return s;
  }

  JGX_HD static node_set<layout_left, Real, Int>
  from_soa(const node_soa_ptrs& p, const jgx_record_desc& r,
           std::size_t n_nodes) {
    node_set<layout_left, Real, Int> s;
    s.n_records = n_nodes;
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES)                              \
    s.comp = jgx::soa_field<T, (RANK) + 1>(p.field[JGX_NF_##TAG],              \
                                           r.field[JGX_NF_##TAG], n_nodes);
#include "jgx/jorek/records/node_record.def"
#undef JGX_FIELD
    return s;
  }

  /* The registration, checked field by field against the .def. The one place
   * the descriptor is fetched, so no caller can skip the check. Host-side: the
   * registry lives on the host. */
  static const jgx_record_desc& record() {
    const jgx_record_desc& r =
        jgx::registered_record(JGX_REC_NODE, JGX_NF_COUNT);
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES)                              \
    jgx::check_record_field(r, JGX_NF_##TAG, KIND, RANK, sizeof(T), #comp);
#include "jgx/jorek/records/node_record.def"
#undef JGX_FIELD
    return r;
  }

  /* Address an existing Fortran node array in place. Offsets and extents come
   * from the registry, which mod_jgx_node_record.f90 filled with c_loc
   * measurements. */
  static node_set<layout_stride, Real, Int>
  from_registry(void* record_base, std::size_t n_nodes) {
    return from_aos(record_base, record(), n_nodes);
  }
};

using node_set_aos = node_set<layout_stride>;
using node_set_soa = node_set<layout_left>;

static_assert(std::is_trivially_copyable<node_set_aos>::value
              && std::is_trivially_copyable<node_set_soa>::value,
              "a node_set crosses to a kernel by value");

} /* namespace jorek */

#endif /* JOREK_NODE_SET_H */
