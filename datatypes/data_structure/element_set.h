/* datatypes/data_structure/element_set.h -- the element container, in AoS and SoA form.
 *
 * Mirrors the nine components of type_element (datatypes/data_structure/data_structure.f90).
 * Kernels take element_set by template parameter and name no layout:
 *
 *   template <class ES> JGX_HD void eval(const ES& el, ...)
 *
 * Extents and byte offsets arrive in a jgx_record_desc, measured by Fortran at
 * registration (jgx-design.md section 2). Nothing here assumes a padding layout
 * or a parameter value.
 *
 * The field enum and element_set_from_registry() at the bottom are the other
 * half of mod_jgx_element_record.f90; the two field lists are the contract and
 * must stay in the same order. Passing JGX_EF_COUNT to registered_record checks
 * that the two lists are at least the same length; the order is not checked.
 */
#ifndef JOREK_ELEMENT_SET_H
#define JOREK_ELEMENT_SET_H

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

/* type_element, in declaration order (datatypes/data_structure/data_structure.f90). */
enum element_field {
  JGX_EF_VERTEX = 0,
  JGX_EF_NEIGHBOURS,
  JGX_EF_SIZE,
  JGX_EF_FATHER,
  JGX_EF_N_SONS,
  JGX_EF_N_GEN,
  JGX_EF_SONS,
  JGX_EF_CONTAIN_NODE,
  JGX_EF_NREF,
  JGX_EF_COUNT
};

template <class L, class Real = double, class Int = int>
struct element_set {
  std::size_t n_elements = 0;

  view<Int , 2, L> vertex;        /* (ie, kv)     */
  view<Int , 2, L> neighbours;    /* (ie, kv)     */
  view<Real, 3, L> size;          /* (ie, kv, kf) */
  view<Int , 1, L> father;        /* (ie)         */
  view<Int , 1, L> n_sons;        /* (ie)         */
  view<Int , 1, L> n_gen;         /* (ie)         */
  view<Int , 2, L> sons;          /* (ie, ks)     */
  view<Int , 2, L> contain_node;  /* (ie, kc)     */
  view<Int , 1, L> nref;          /* (ie)         */
};

using element_set_aos = element_set<layout_stride>;
using element_set_soa = element_set<layout_left>;

/* One packed buffer per field, indexed by element_field. */
struct element_soa_ptrs {
  void* field[JGX_EF_COUNT] = {};
};

template <class Real = double, class Int = int>
JGX_HD inline element_set<layout_stride, Real, Int>
make_element_set_aos(void* record_base, const jgx_record_desc& r,
                     std::size_t n_elements) {
  const std::size_t rs = r.record_stride_bytes;

  element_set<layout_stride, Real, Int> s;
  s.n_elements   = n_elements;
  s.vertex       = jgx::aos_field<Int , 2>(record_base, r.field[JGX_EF_VERTEX      ], rs, n_elements);
  s.neighbours   = jgx::aos_field<Int , 2>(record_base, r.field[JGX_EF_NEIGHBOURS  ], rs, n_elements);
  s.size         = jgx::aos_field<Real, 3>(record_base, r.field[JGX_EF_SIZE        ], rs, n_elements);
  s.father       = jgx::aos_field<Int , 1>(record_base, r.field[JGX_EF_FATHER      ], rs, n_elements);
  s.n_sons       = jgx::aos_field<Int , 1>(record_base, r.field[JGX_EF_N_SONS      ], rs, n_elements);
  s.n_gen        = jgx::aos_field<Int , 1>(record_base, r.field[JGX_EF_N_GEN       ], rs, n_elements);
  s.sons         = jgx::aos_field<Int , 2>(record_base, r.field[JGX_EF_SONS        ], rs, n_elements);
  s.contain_node = jgx::aos_field<Int , 2>(record_base, r.field[JGX_EF_CONTAIN_NODE], rs, n_elements);
  s.nref         = jgx::aos_field<Int , 1>(record_base, r.field[JGX_EF_NREF        ], rs, n_elements);
  return s;
}

template <class Real = double, class Int = int>
JGX_HD inline element_set<layout_left, Real, Int>
make_element_set_soa(const element_soa_ptrs& p, const jgx_record_desc& r,
                     std::size_t n_elements) {
  element_set<layout_left, Real, Int> s;
  s.n_elements   = n_elements;
  s.vertex       = jgx::soa_field<Int , 2>(p.field[JGX_EF_VERTEX      ], r.field[JGX_EF_VERTEX      ], n_elements);
  s.neighbours   = jgx::soa_field<Int , 2>(p.field[JGX_EF_NEIGHBOURS  ], r.field[JGX_EF_NEIGHBOURS  ], n_elements);
  s.size         = jgx::soa_field<Real, 3>(p.field[JGX_EF_SIZE        ], r.field[JGX_EF_SIZE        ], n_elements);
  s.father       = jgx::soa_field<Int , 1>(p.field[JGX_EF_FATHER      ], r.field[JGX_EF_FATHER      ], n_elements);
  s.n_sons       = jgx::soa_field<Int , 1>(p.field[JGX_EF_N_SONS      ], r.field[JGX_EF_N_SONS      ], n_elements);
  s.n_gen        = jgx::soa_field<Int , 1>(p.field[JGX_EF_N_GEN       ], r.field[JGX_EF_N_GEN       ], n_elements);
  s.sons         = jgx::soa_field<Int , 2>(p.field[JGX_EF_SONS        ], r.field[JGX_EF_SONS        ], n_elements);
  s.contain_node = jgx::soa_field<Int , 2>(p.field[JGX_EF_CONTAIN_NODE], r.field[JGX_EF_CONTAIN_NODE], n_elements);
  s.nref         = jgx::soa_field<Int , 1>(p.field[JGX_EF_NREF        ], r.field[JGX_EF_NREF        ], n_elements);
  return s;
}

/* Address an existing Fortran element array in place. Offset or extent come
 * from the registry, which mod_jgx_element_record.f90 filled with c_loc measurements. */
inline element_set_aos element_set_from_registry(void* base, std::size_t n_elements) {
  return make_element_set_aos<>(base, jgx::registered_record(JGX_REC_ELEMENT, JGX_EF_COUNT),
                                n_elements);
}

} /* namespace jorek */

#endif /* JOREK_ELEMENT_SET_H */
