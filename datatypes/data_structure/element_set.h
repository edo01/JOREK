/* datatypes/data_structure/element_set.h -- the element container, in AoS and
 * SoA form.
 *
 * Mirrors the nine components of type_element. The field list, the kinds, the
 * ranks and the axis names are in jgx/jorek/records/element_record.def, which
 * mod_jgx_element_record.f90 expands as well -- so the two sides are one list,
 * not two that must be kept in the same order.
 */
#ifndef JOREK_ELEMENT_SET_H
#define JOREK_ELEMENT_SET_H

#include <cstddef>
#include <type_traits>
#include "jgx/jorek/jgx_record_ids.h"
#include "jgx/data/record_api.h"
#include "jgx/macros.h"
#include "jgx/data/field_view.h"
#include "jgx/data/pack.h"
#include "jgx/data/record_set.h"
#include "jgx/view.h"

namespace jorek {

using jgx::layout_left;
using jgx::layout_stride;
using jgx::view;

enum element_field {
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES) JGX_EF_##TAG,
#include "jgx/jorek/records/element_record.def"
#undef JGX_FIELD
  JGX_EF_COUNT
};

template <class L, class Real = double, class Int = int>
struct element_set : jgx::data::record_set {
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES) view<T, (RANK) + 1, L> comp;
#include "jgx/jorek/records/element_record.def"
#undef JGX_FIELD

  /* Each maker names the layout it builds rather than the current
   * instantiation, so it returns the right set whichever alias it is reached
   * through. */
  JGX_HD static element_set<layout_stride, Real, Int>
  from_aos(void* aos_base, const jgx_record_desc& r,
           std::size_t n_elements) {
    element_set<layout_stride, Real, Int> s;
    s.n_records = n_elements;
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES)                              \
    s.comp = jgx::data::aos_field<T, (RANK) + 1>(aos_base, r.field[JGX_EF_##TAG], \
                                           r.record_stride_bytes, n_elements);
#include "jgx/jorek/records/element_record.def"
#undef JGX_FIELD
    return s;
  }

  JGX_HD static element_set<layout_left, Real, Int>
  from_soa(void* soa_base, const jgx_record_desc& r, std::size_t n_elements) {
    element_set<layout_left, Real, Int> s;
    s.n_records = n_elements;
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES)                              \
    s.comp = jgx::data::soa_field<T, (RANK) + 1>(                                    \
        jgx::data::soa_field_ptr(soa_base, r, JGX_EF_##TAG, n_elements),             \
        r.field[JGX_EF_##TAG], n_elements);
#include "jgx/jorek/records/element_record.def"
#undef JGX_FIELD
    return s;
  }

  /* The registration, checked field by field against the .def. The one place
   * the descriptor is fetched, so no caller can skip the check. Host-side: the
   * registry lives on the host. */
  static const jgx_record_desc& record() {
    const jgx_record_desc& r =
        jgx::data::registered_record(JGX_REC_ELEMENT, JGX_EF_COUNT);
/** @todo: guard this using DEBUG 
 * 
*/
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES)                              \
    jgx::data::check_record_field(r, JGX_EF_##TAG, KIND, RANK, sizeof(T), #comp);
#include "jgx/jorek/records/element_record.def"
#undef JGX_FIELD
    return r;
  }

};

using element_set_aos = element_set<layout_stride>;
using element_set_soa = element_set<layout_left>;

static_assert(std::is_trivially_copyable<element_set_aos>::value
              && std::is_trivially_copyable<element_set_soa>::value,
              "an element_set crosses to a kernel by value");

} /* namespace jorek */

#endif /* JOREK_ELEMENT_SET_H */
