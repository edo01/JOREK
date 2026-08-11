/* particles/mod_fields/fields_set.h -- the field interpolator.
 *
 * Mirrors fields_base (particles/mod_fields/mod_fields.f90). Each extension is
 * mirrored as C++ inheritance and lives with its own bindings, in the header of
 * the module it comes from.
 *
 * The fields themselves are NOT a record. type_fields holds two pointers and a
 * polymorphic allocatable, and a record cannot describe either; nor does it need
 * to, because nothing of type_fields is addressed in place. What crosses the
 * seam is three base pointers -- elements, nodes, interpolator -- and the two
 * mesh counts.
 *
 * The INTERPOLATOR is a record (JGX_REC_FIELDS_INTERP_LINEAR,
 * mod_fields_linear/mod_jgx_fields_interp_linear_record.f90). It is registered
 * as a record of count 1: the registry describes the type, so the count is
 * irrelevant to what it recovers -- the compiler-chosen component offsets. That
 * is what lets the strategy be a hierarchy whose scalars are read through one
 * base pointer instead of being unpacked into loose arguments at every call.
 *
 * The two inherited flags are default Fortran logicals registered as JGX_I32 and
 * read back with `!= 0`. That test is the contract, not caution: ifx 2024.1
 * stores .true. as the word -1 on this build, so `== 1` reads every set flag as
 * false. See the registration module for what else this rests on.
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

/* fields_interpolator, in declaration order. These ids are shared by every
 * interpolator record, so an extension continues the numbering from
 * JGX_FI_BASE_COUNT. Mirrors mod_jgx_fields_interp_linear_record.f90. */
enum fields_interp_field {
  JGX_FI_STATIC = 0,
  JGX_FI_FLAG_ZERO_DPSIDT,
  JGX_FI_BASE_COUNT
};

/* The strategy's inherited half -- what fields_interpolator declares.
 *
 * It carries no grid and no layout parameter. The grid belongs to the fields and
 * is handed to interp_PRZ as an argument, exactly as the Fortran deferred
 * interface takes node_list and element_list. A strategy that owns snapshots of
 * its own (Hermite-Birkhoff's ring buffer) holds them here, in its extension. */
struct fields_interp_base_set {
  bool is_static        = false;  /* fields_interpolator%static (`static` is a C++ keyword) */
  bool flag_zero_dpsidt = false;
};

/* The inherited half, filled the same way for any interpolator: both flags come
 * from the interpolator record, so every extension reads them identically. */
JGX_HD inline void
fill_fields_interp_base(fields_interp_base_set& f,
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
