/* particles/particle_types/particle_set.h -- the particle container, in AoS and SoA form.
 *
 * Mirrors particle_kinetic_relativistic and its extensions
 * (particles/particle_types/mod_particle_types.f90).
 * The extension is mirrored here as C++ inheritance, so a kernel names 
 * the components exactly as the Fortran does -- * s.x and s.p, not s.base.x -- 
 * and a routine that only needs the base part takes a particle_base_set
 * and works for every particle type.
 *
 * The base pointer these builders take must come from inside a select type. 
 * For instance:
 *   select type (particles => sim%groups(g)%particles)
 *   type is (particle_kinetic_relativistic)
 * 
 */
#ifndef JOREK_PARTICLE_SET_H
#define JOREK_PARTICLE_SET_H

#include <cstddef>
#include <cstdint>
#include "jgx/jorek/jgx_record_ids.h"
#include "jgx/jgx_record_api.h"
#include "jgx/macros.h"
#include "jgx/field_view.h"
#include "jgx/view.h"

namespace jorek {

using jgx::layout_left;
using jgx::layout_stride;
using jgx::view;

/* particle_base, in declaration order. These ids are shared by every particle
 * record. */
enum particle_base_field {
  JGX_PF_X = 0,
  JGX_PF_ST,
  JGX_PF_WEIGHT,
  JGX_PF_I_ELM,
  JGX_PF_I_LIFE,
  JGX_PF_T_BIRTH,
  JGX_PF_BASE_COUNT
};

enum particle_kin_rel_field {
  JGX_PKR_P = JGX_PF_BASE_COUNT, // to continue the same numbering
  JGX_PKR_Q,
  JGX_PKR_COUNT
};

/* t_birth is real*4 and q is integer*1 in the Fortran types, so neither
 * follows the Real / Int template parameters. */
template <class L, class Real = double, class Int = int>
struct particle_base_set {
  std::size_t n_particles = 0;

  view<Real , 2, L> x;        /* (ip, kc) */
  view<Real , 2, L> st;       /* (ip, kc) */
  view<Real , 1, L> weight;   /* (ip)     */
  view<Int  , 1, L> i_elm;    /* (ip)     */
  view<Int  , 1, L> i_life;   /* (ip)     */
  view<float, 1, L> t_birth;  /* (ip)     */
};

template <class L, class Real = double, class Int = int>
struct particle_kin_rel_set : particle_base_set<L, Real, Int> {
  view<Real        , 2, L> p;  /* (ip, kc) */
  view<std::int8_t , 1, L> q;  /* (ip)     */
};

using particle_kin_rel_set_aos = particle_kin_rel_set<layout_stride>;
using particle_kin_rel_set_soa = particle_kin_rel_set<layout_left>;

/* One packed buffer per field, indexed by the field enums above. */
struct particle_kin_rel_soa_ptrs {
  void* field[JGX_PKR_COUNT] = {};
};

/* The inherited half, filled the same way for any particle record. */
template <class Real, class Int>
JGX_HD inline void
fill_particle_base_aos(particle_base_set<layout_stride, Real, Int>& s,
                       void* record_base, const jgx_record_desc& r,
                       std::size_t n_particles) {
  const std::size_t rs = r.record_stride_bytes;

  s.n_particles = n_particles;
  s.x       = jgx::aos_field<Real , 2>(record_base, r.field[JGX_PF_X      ], rs, n_particles);
  s.st      = jgx::aos_field<Real , 2>(record_base, r.field[JGX_PF_ST     ], rs, n_particles);
  s.weight  = jgx::aos_field<Real , 1>(record_base, r.field[JGX_PF_WEIGHT ], rs, n_particles);
  s.i_elm   = jgx::aos_field<Int  , 1>(record_base, r.field[JGX_PF_I_ELM  ], rs, n_particles);
  s.i_life  = jgx::aos_field<Int  , 1>(record_base, r.field[JGX_PF_I_LIFE ], rs, n_particles);
  s.t_birth = jgx::aos_field<float, 1>(record_base, r.field[JGX_PF_T_BIRTH], rs, n_particles);
}

template <class Real, class Int>
JGX_HD inline void
fill_particle_base_soa(particle_base_set<layout_left, Real, Int>& s,
                       void* const* field, const jgx_record_desc& r,
                       std::size_t n_particles) {
  s.n_particles = n_particles;
  s.x       = jgx::soa_field<Real , 2>(field[JGX_PF_X      ], r.field[JGX_PF_X      ], n_particles);
  s.st      = jgx::soa_field<Real , 2>(field[JGX_PF_ST     ], r.field[JGX_PF_ST     ], n_particles);
  s.weight  = jgx::soa_field<Real , 1>(field[JGX_PF_WEIGHT ], r.field[JGX_PF_WEIGHT ], n_particles);
  s.i_elm   = jgx::soa_field<Int  , 1>(field[JGX_PF_I_ELM  ], r.field[JGX_PF_I_ELM  ], n_particles);
  s.i_life  = jgx::soa_field<Int  , 1>(field[JGX_PF_I_LIFE ], r.field[JGX_PF_I_LIFE ], n_particles);
  s.t_birth = jgx::soa_field<float, 1>(field[JGX_PF_T_BIRTH], r.field[JGX_PF_T_BIRTH], n_particles);
}

template <class Real = double, class Int = int>
JGX_HD inline particle_kin_rel_set<layout_stride, Real, Int>
make_particle_kin_rel_set_aos(void* record_base, const jgx_record_desc& r,
                              std::size_t n_particles) {
  const std::size_t rs = r.record_stride_bytes;

  particle_kin_rel_set<layout_stride, Real, Int> s;
  fill_particle_base_aos(s, record_base, r, n_particles);
  s.p = jgx::aos_field<Real       , 2>(record_base, r.field[JGX_PKR_P], rs, n_particles);
  s.q = jgx::aos_field<std::int8_t, 1>(record_base, r.field[JGX_PKR_Q], rs, n_particles);
  return s;
}

template <class Real = double, class Int = int>
JGX_HD inline particle_kin_rel_set<layout_left, Real, Int>
make_particle_kin_rel_set_soa(const particle_kin_rel_soa_ptrs& p,
                              const jgx_record_desc& r,
                              std::size_t n_particles) {
  particle_kin_rel_set<layout_left, Real, Int> s;
  fill_particle_base_soa(s, p.field, r, n_particles);
  s.p = jgx::soa_field<Real       , 2>(p.field[JGX_PKR_P], r.field[JGX_PKR_P], n_particles);
  s.q = jgx::soa_field<std::int8_t, 1>(p.field[JGX_PKR_Q], r.field[JGX_PKR_Q], n_particles);
  return s;
}

/* Address an existing Fortran particle array in place. Offset or extent come
 * from the registry, which mod_jgx_particle_record.f90 filled with c_loc
 * measurements. */
inline particle_kin_rel_set_aos
particle_kin_rel_set_from_registry(void* base, std::size_t n_particles) {
  return make_particle_kin_rel_set_aos<>(
      base, jgx::registered_record(JGX_REC_PARTICLE_KIN_REL, JGX_PKR_COUNT),
      n_particles);
}

} /* namespace jorek */

#endif /* JOREK_PARTICLE_SET_H */
