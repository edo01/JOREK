/* particles/mod_fields/fields_set.h -- the field interpolator.
 *
 * Mirrors type_fields and fields_interpolator_base (particles/mod_fields/mod_fields.f90). 
 * Each extension is mirrored as C++ inheritance and lives with its own bindings, in 
 * the header of the module it comes from.
 *
 * The INTERPOLATOR is a record (JGX_REC_FIELDS_INTERP_LINEAR,
 * mod_fields_linear/mod_jgx_fields_interp_linear_record.f90). It is registered
 * as a record of count 1.
 */
#ifndef JOREK_FIELDS_SET_H
#define JOREK_FIELDS_SET_H

#include <cstddef>
#include <cstdint>
#include "jgx/jgx_record_api.h"
#include "jgx/field_view.h"
#include "datatypes/data_structure/element_set.h"
#include "datatypes/data_structure/node_set.h"
#include "jgx/macros.h"

namespace jorek {

/* fields_interpolator_base, in declaration order. These ids are shared by every
 * interpolator record, so an extension continues the numbering from
 * JGX_FI_BASE_COUNT. Mirrors mod_jgx_fields_interp_linear_record.f90. */
enum fields_interp_field {
  JGX_FI_STATIC = 0,
  JGX_FI_FLAG_ZERO_DPSIDT,
  JGX_FI_BASE_COUNT
};

/* The strategy's inherited half -- what fields_interpolator_base declares.
 *
 * It carries no grid and no layout parameter. The grid belongs to the fields and
 * is handed to interp_PRZ as an argument, exactly as the Fortran deferred
 * interface takes node_list and element_list. */
struct fields_interp_base {
  bool is_static        = false;  /* fields_interpolator_base%static (`static` is a C++ keyword) */
  bool flag_zero_dpsidt = false;
};

/* The inherited half, filled the same way for any interpolator: both flags come
 * from the interpolator record, so every extension reads them identically. */
JGX_HD inline void
fill_fields_interp_base(fields_interp_base& f,
                        const void* interp_base, const jgx_record_desc& r) {
  /* registered as int32; `!= 0` is the contract -- see the header comment. */
  f.is_static        = jgx::record_scalar<std::int32_t>(interp_base, r.field[JGX_FI_STATIC]) != 0;
  f.flag_zero_dpsidt = jgx::record_scalar<std::int32_t>(interp_base, r.field[JGX_FI_FLAG_ZERO_DPSIDT]) != 0;
}

/* type_fields: the grid that defines the fields, and the strategy that
 * interpolates them in time.
 *
 * Composition, mirroring the Fortran -- type_fields *has* an interpolator, it is
 * not one. Interp is a template parameter rather than a base class, so the
 * strategy is resolved statically and the set stays trivially copyable to a
 * device; the Fortran select type at the seam is what picks the instantiation,
 * the same way it picks a concrete particle set. */
template <class L, class Interp, class Real = double, class Int = int>
struct fields_set {
  element_set<L, Real, Int> element_list;
  node_set<L, Real, Int>    node_list;
  Interp                    interp;
};

template <class Interp> using fields_set_aos = fields_set<layout_stride, Interp>;
template <class Interp> using fields_set_soa = fields_set<layout_left, Interp>;

template <class L, class Interp, class Real, class Int>
JGX_HD inline fields_set<L, Interp, Real, Int>
make_fields_set(const element_set<L, Real, Int>& el,
                const node_set<L, Real, Int>& nd,
                const Interp& interp) {
  fields_set<L, Interp, Real, Int> f;
  f.element_list = el;
  f.node_list    = nd;
  f.interp       = interp;
  return f;
}

} /* namespace jorek */

#endif /* JOREK_FIELDS_SET_H */
