/* particles/mod_fields/fields_set.h -- the field interpolator.
 *
 * Mirrors fields_base and its extensions (particles/mod_fields/mod_fields.f90,
 * particles/mod_fields_linear.f90). The extension is mirrored as C++
 * inheritance.
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
#include "node_variant_set.h"
#include "jgx/macros.h"

namespace jorek {

template <class L, class Real = double, class Int = int>
struct fields_base_set {
  using node_type = node_model_set<L, Real, Int>;

  element_set<L, Real, Int> element_list;
  // Node_type is defined at compiled type (reduced, fullmhd, stellarator)
  node_type                 node_list;

  bool is_static        = false;  /* fields_base%static (`static` is a C++ keyword) */
  bool flag_zero_dpsidt = false;
};

template <class L, class Real = double, class Int = int>
struct fields_interp_linear_set : fields_base_set<L, Real, Int> {
  Real time_now  = 0;  /* SI */
  Real time_prev = 0;  /* SI */
};

using fields_base_set_aos = fields_base_set<layout_stride>;
using fields_base_set_soa = fields_base_set<layout_left>;
using fields_interp_linear_set_aos = fields_interp_linear_set<layout_stride>;
using fields_interp_linear_set_soa = fields_interp_linear_set<layout_left>;

/* The inherited half, filled the same way for any interpolator. Layout-free:
 * whichever pair of sets is handed in decides it. */
template <class L, class Real, class Int>
JGX_HD inline void
fill_fields_base(fields_base_set<L, Real, Int>& f,
                 const element_set<L, Real, Int>& el,
                 const node_model_set<L, Real, Int>& nd,
                 bool is_static, bool flag_zero_dpsidt) {
  f.element_list     = el;
  f.node_list        = nd;
  f.is_static        = is_static;
  f.flag_zero_dpsidt = flag_zero_dpsidt;
}

template <class L, class Real = double, class Int = int>
JGX_HD inline fields_interp_linear_set<L, Real, Int>
make_fields_interp_linear_set(const element_set<L, Real, Int>& el,
                              const node_model_set<L, Real, Int>& nd,
                              bool is_static, bool flag_zero_dpsidt,
                              Real time_now, Real time_prev) {
  fields_interp_linear_set<L, Real, Int> f;
  fill_fields_base(f, el, nd, is_static, flag_zero_dpsidt);
  f.time_now  = time_now;
  f.time_prev = time_prev;
  return f;
}

/* Address an existing Fortran jorek_fields_interp_linear in place. The two mesh
 * bases come from inside the caller's select type; their layout comes from the
 * registry, as for any other set. Flags arrive as int32 because Fortran's
 * default logical is not interoperable. */
inline fields_interp_linear_set_aos
fields_interp_linear_set_from_registry(void* el_base, std::size_t n_elements,
                                       void* nd_base, std::size_t n_nodes,
                                       int is_static, int flag_zero_dpsidt,
                                       double time_now, double time_prev) {
  return make_fields_interp_linear_set<layout_stride, double, int>(
      element_set_from_registry(el_base, n_elements),
      node_model_set_from_registry(nd_base, n_nodes),
      is_static != 0, flag_zero_dpsidt != 0, time_now, time_prev);
}

} /* namespace jorek */

#endif /* JOREK_FIELDS_SET_H */
