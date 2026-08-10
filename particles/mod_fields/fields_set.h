/* particles/mod_fields/fields_set.h -- the field interpolator.
 *
 * Mirrors fields_base (particles/mod_fields/mod_fields.f90). Each extension is
 * mirrored as C++ inheritance and lives with its own bindings, in the header of
 * the module it comes from.
 *
 * Unlike element_set / node_set / particle_set this is NOT a record: fields_base
 * is a singleton, not an array, so there is no record stride, no field offset
 * table and no AoS/SoA view. It has no record id and no
 * mod_jgx_*_record.f90. What it does have is the two meshes, which it holds *by
 * value* as an element_set and a node_set -- those are views, so a copy is a
 * handful of pointers and extents, cheap enough to pass to a kernel by value and
 * trivially copyable to a device. The layout parameter L is simply the one the
 * two sets were built with; both halves must agree.
 *
 */
#ifndef JOREK_FIELDS_SET_H
#define JOREK_FIELDS_SET_H

#include <cstddef>
#include "datatypes/data_structure/element_set.h"
#include "datatypes/data_structure/node_set.h"
#include "jgx/macros.h"

namespace jorek {

template <class L, class Real = double, class Int = int>
struct fields_base_set {
  element_set<L, Real, Int> element_list;
  node_set<L, Real, Int>    node_list;

  bool is_static        = false;  /* fields_base%static (`static` is a C++ keyword) */
  bool flag_zero_dpsidt = false;
};

using fields_base_set_aos = fields_base_set<layout_stride>;
using fields_base_set_soa = fields_base_set<layout_left>;

/* The inherited half, filled the same way for any interpolator. Layout-free:
 * whichever pair of sets is handed in decides it. */
template <class L, class Real, class Int>
JGX_HD inline void
fill_fields_base(fields_base_set<L, Real, Int>& f,
                 const element_set<L, Real, Int>& el,
                 const node_set<L, Real, Int>& nd,
                 bool is_static, bool flag_zero_dpsidt) {
  f.element_list     = el;
  f.node_list        = nd;
  f.is_static        = is_static;
  f.flag_zero_dpsidt = flag_zero_dpsidt;
}

} /* namespace jorek */

#endif /* JOREK_FIELDS_SET_H */
