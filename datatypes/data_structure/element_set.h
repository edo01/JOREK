/* datatypes/data_structure/element_set.h -- the element container, in AoS and SoA form.
 *
 * Mirrors the nine components of type_element (datatypes/data_structure/data_structure.f90).
 * Kernels take element_set by template parameter and name no layout:
 *
 *   template <class ES> JGX_HD void eval(const ES& el, ...)
 *
 * Extents and byte offsets are supplied by the caller -- they come from Fortran
 * at registration (jgx-design.md section 2). Nothing here assumes a padding
 * layout or a parameter value.
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

/* Fortran parameters, passed in rather than hardcoded. */
struct element_dims {
  std::size_t n_elements     = 0;
  std::size_t n_vertex_max   = 0;
  std::size_t n_degrees      = 0;
  std::size_t n_sons_max     = 4;   /* type_element%sons(4)         */
  std::size_t n_contain_node = 5;   /* type_element%contain_node(5) */
};

/* Byte offsets within one record, measured on the Fortran side. */
struct element_offsets {
  std::size_t record_stride = 0;
  std::size_t vertex        = 0;
  std::size_t neighbours    = 0;
  std::size_t size          = 0;
  std::size_t father        = 0;
  std::size_t n_sons        = 0;
  std::size_t n_gen         = 0;
  std::size_t sons          = 0;
  std::size_t contain_node  = 0;
  std::size_t nref          = 0;
};

/* One packed buffer per field. */
struct element_soa_ptrs {
  void* vertex       = nullptr;
  void* neighbours   = nullptr;
  void* size         = nullptr;
  void* father       = nullptr;
  void* n_sons       = nullptr;
  void* n_gen        = nullptr;
  void* sons         = nullptr;
  void* contain_node = nullptr;
  void* nref         = nullptr;
};

/* True when every component divides evenly into whole-T strides. */
template <class Real = double, class Int = int>
inline bool element_offsets_are_addressable(const element_offsets& off) {
  const std::size_t rs = off.record_stride;
  return jgx::field_is_addressable<Int >(off.vertex,       rs)
      && jgx::field_is_addressable<Int >(off.neighbours,   rs)
      && jgx::field_is_addressable<Real>(off.size,         rs)
      && jgx::field_is_addressable<Int >(off.father,       rs)
      && jgx::field_is_addressable<Int >(off.n_sons,       rs)
      && jgx::field_is_addressable<Int >(off.n_gen,        rs)
      && jgx::field_is_addressable<Int >(off.sons,         rs)
      && jgx::field_is_addressable<Int >(off.contain_node, rs)
      && jgx::field_is_addressable<Int >(off.nref,         rs);
}

template <class Real = double, class Int = int>
JGX_HD inline element_set<layout_stride, Real, Int>
make_element_set_aos(void* record_base, const element_offsets& off,
                     const element_dims& d) {
  const std::size_t ne = d.n_elements;
  const std::size_t e1[1] = { ne };
  const std::size_t ev[2] = { ne, d.n_vertex_max };
  const std::size_t es[3] = { ne, d.n_vertex_max, d.n_degrees };
  const std::size_t eo[2] = { ne, d.n_sons_max };
  const std::size_t ec[2] = { ne, d.n_contain_node };
  const std::size_t rs = off.record_stride;

  element_set<layout_stride, Real, Int> s;
  s.n_elements   = ne;
  s.vertex       = jgx::aos_field<Int , 2>(record_base, off.vertex,       rs, ev);
  s.neighbours   = jgx::aos_field<Int , 2>(record_base, off.neighbours,   rs, ev);
  s.size         = jgx::aos_field<Real, 3>(record_base, off.size,         rs, es);
  s.father       = jgx::aos_field<Int , 1>(record_base, off.father,       rs, e1);
  s.n_sons       = jgx::aos_field<Int , 1>(record_base, off.n_sons,       rs, e1);
  s.n_gen        = jgx::aos_field<Int , 1>(record_base, off.n_gen,        rs, e1);
  s.sons         = jgx::aos_field<Int , 2>(record_base, off.sons,         rs, eo);
  s.contain_node = jgx::aos_field<Int , 2>(record_base, off.contain_node, rs, ec);
  s.nref         = jgx::aos_field<Int , 1>(record_base, off.nref,         rs, e1);
  return s;
}

template <class Real = double, class Int = int>
JGX_HD inline element_set<layout_left, Real, Int>
make_element_set_soa(const element_soa_ptrs& p, const element_dims& d) {
  const std::size_t ne = d.n_elements;
  const std::size_t e1[1] = { ne };
  const std::size_t ev[2] = { ne, d.n_vertex_max };
  const std::size_t es[3] = { ne, d.n_vertex_max, d.n_degrees };
  const std::size_t eo[2] = { ne, d.n_sons_max };
  const std::size_t ec[2] = { ne, d.n_contain_node };

  element_set<layout_left, Real, Int> s;
  s.n_elements   = ne;
  s.vertex       = jgx::soa_field<Int , 2>(p.vertex,       ev);
  s.neighbours   = jgx::soa_field<Int , 2>(p.neighbours,   ev);
  s.size         = jgx::soa_field<Real, 3>(p.size,         es);
  s.father       = jgx::soa_field<Int , 1>(p.father,       e1);
  s.n_sons       = jgx::soa_field<Int , 1>(p.n_sons,       e1);
  s.n_gen        = jgx::soa_field<Int , 1>(p.n_gen,        e1);
  s.sons         = jgx::soa_field<Int , 2>(p.sons,         eo);
  s.contain_node = jgx::soa_field<Int , 2>(p.contain_node, ec);
  s.nref         = jgx::soa_field<Int , 1>(p.nref,         e1);
  return s;
}

/* Address an existing Fortran element array in place. Nothing here assumes an
 * offset or an extent -- both come from the registry, which
 * mod_jgx_element_record.f90 filled with c_loc measurements. */
inline element_set_aos element_set_from_registry(void* base, std::size_t n_elements) {
  const jgx_record_desc& r = jgx::registered_record(JGX_REC_ELEMENT, JGX_EF_COUNT);

  element_offsets off;
  off.record_stride = r.record_stride_bytes;
  off.vertex        = r.field[JGX_EF_VERTEX      ].offset_bytes;
  off.neighbours    = r.field[JGX_EF_NEIGHBOURS  ].offset_bytes;
  off.size          = r.field[JGX_EF_SIZE        ].offset_bytes;
  off.father        = r.field[JGX_EF_FATHER      ].offset_bytes;
  off.n_sons        = r.field[JGX_EF_N_SONS      ].offset_bytes;
  off.n_gen         = r.field[JGX_EF_N_GEN       ].offset_bytes;
  off.sons          = r.field[JGX_EF_SONS        ].offset_bytes;
  off.contain_node  = r.field[JGX_EF_CONTAIN_NODE].offset_bytes;
  off.nref          = r.field[JGX_EF_NREF        ].offset_bytes;

  element_dims d;
  d.n_elements     = n_elements;
  d.n_vertex_max   = r.field[JGX_EF_SIZE        ].intra_extents[0];
  d.n_degrees      = r.field[JGX_EF_SIZE        ].intra_extents[1];
  d.n_sons_max     = r.field[JGX_EF_SONS        ].intra_extents[0];
  d.n_contain_node = r.field[JGX_EF_CONTAIN_NODE].intra_extents[0];

  return make_element_set_aos<>(base, off, d);
}

} /* namespace jorek */

#endif /* JOREK_ELEMENT_SET_H */
