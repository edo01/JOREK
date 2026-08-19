/* particles/particle_types/particle_set.h -- the particle container, in AoS and
 * SoA form.
 *
 * Mirrors particle_kinetic_relativistic and its extensions
 * (particles/particle_types/mod_particle_types.f90). The extension is C++
 * inheritance, so a kernel names the components exactly as the Fortran does --
 * s.x and s.p, not s.base.x -- and a routine that only needs the base part
 * takes a particle_base_set and works for every particle type.
 *
 * The field lists are jgx/jorek/records/particle_{base,kin_rel}_record.def,
 * which mod_jgx_particle_record.f90 expands as well. The base list comes first
 * everywhere the whole record is needed, which is what keeps the inherited
 * components at the same ids in every particle record.
 *
 * The base pointer the makers take must come from inside a select type:
 *   select type (particles => sim%groups(g)%particles)
 *   type is (particle_kinetic_relativistic)
 *
 * See jgx/record_set.h for what a record set is and how the makers differ.
 */
#ifndef JOREK_PARTICLE_SET_H
#define JOREK_PARTICLE_SET_H

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "jgx/jorek/jgx_record_ids.h"
#include "jgx/jgx_record_api.h"
#include "jgx/macros.h"
#include "jgx/field_view.h"
#include "jgx/pack.h"
#include "jgx/record_set.h"
#include "jgx/view.h"

namespace jorek {

using jgx::layout_left;
using jgx::layout_stride;
using jgx::view;

/* Shared by every particle record. */
enum particle_base_field {
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES) JGX_PF_##TAG,
#include "jgx/jorek/records/particle_base_record.def"
#undef JGX_FIELD
  JGX_PF_BASE_COUNT
};

/* The whole record: the base list, then what this one adds. Expanding both here
 * is what continues the numbering -- the extension's first id is the base
 * count, with nothing to state by hand. */
enum particle_kin_rel_field {
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES) JGX_PKR_##TAG,
#include "jgx/jorek/records/particle_base_record.def"
#include "jgx/jorek/records/particle_kin_rel_record.def"
#undef JGX_FIELD
  JGX_PKR_COUNT
};

static_assert(JGX_PKR_P == JGX_PF_BASE_COUNT,
              "the extension must continue the base numbering");

template <class L, class Real = double, class Int = int>
struct particle_base_set : jgx::record_set {
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES) view<T, (RANK) + 1, L> comp;
#include "jgx/jorek/records/particle_base_record.def"
#undef JGX_FIELD

  /* The inherited half, filled the same way for any particle record: a concrete
   * record's from_aos / from_soa calls one of these, then adds its own fields.
   * Both take the base the concrete maker was given -- the base fields are
   * ordinary fields of the concrete record, at its own offsets. */
  JGX_HD static void fill_aos(particle_base_set<layout_stride, Real, Int>& s,
                              void* aos_base, const jgx_record_desc& r,
                              std::size_t n_particles) {
    s.n_records = n_particles;
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES)                              \
    s.comp = jgx::aos_field<T, (RANK) + 1>(aos_base, r.field[JGX_PF_##TAG], \
                                           r.record_stride_bytes, n_particles);
#include "jgx/jorek/records/particle_base_record.def"
#undef JGX_FIELD
  }

  JGX_HD static void fill_soa(particle_base_set<layout_left, Real, Int>& s,
                              void* soa_base, const jgx_record_desc& r,
                              std::size_t n_particles) {
    s.n_records = n_particles;
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES)                              \
    s.comp = jgx::soa_field<T, (RANK) + 1>(                                    \
        jgx::soa_field_ptr(soa_base, r, JGX_PF_##TAG, n_particles),            \
        r.field[JGX_PF_##TAG], n_particles);
#include "jgx/jorek/records/particle_base_record.def"
#undef JGX_FIELD
  }
};

template <class L, class Real = double, class Int = int>
struct particle_kin_rel_set : particle_base_set<L, Real, Int> {
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES) view<T, (RANK) + 1, L> comp;
#include "jgx/jorek/records/particle_kin_rel_record.def"
#undef JGX_FIELD

  /* Each maker names the layout it builds rather than the current
   * instantiation, so it returns the right set whichever alias it is reached
   * through. */
  JGX_HD static particle_kin_rel_set<layout_stride, Real, Int>
  from_aos(void* aos_base, const jgx_record_desc& r,
           std::size_t n_particles) {
    particle_kin_rel_set<layout_stride, Real, Int> s;
    particle_base_set<layout_stride, Real, Int>::fill_aos(s, aos_base, r,
                                                          n_particles);
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES)                              \
    s.comp = jgx::aos_field<T, (RANK) + 1>(aos_base, r.field[JGX_PKR_##TAG],\
                                           r.record_stride_bytes, n_particles);
#include "jgx/jorek/records/particle_kin_rel_record.def"
#undef JGX_FIELD
    return s;
  }

  JGX_HD static particle_kin_rel_set<layout_left, Real, Int>
  from_soa(void* soa_base, const jgx_record_desc& r, std::size_t n_particles) {
    particle_kin_rel_set<layout_left, Real, Int> s;
    particle_base_set<layout_left, Real, Int>::fill_soa(s, soa_base, r,
                                                        n_particles);
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES)                              \
    s.comp = jgx::soa_field<T, (RANK) + 1>(                                    \
        jgx::soa_field_ptr(soa_base, r, JGX_PKR_##TAG, n_particles),           \
        r.field[JGX_PKR_##TAG], n_particles);
#include "jgx/jorek/records/particle_kin_rel_record.def"
#undef JGX_FIELD
    return s;
  }

  /* The registration, checked field by field against the .def -- both lists,
   * since both describe this record. The one place the descriptor is fetched,
   * so no caller can skip the check. Host-side: the registry lives on the host. */
  static const jgx_record_desc& record() {
    const jgx_record_desc& r =
        jgx::registered_record(JGX_REC_PARTICLE_KIN_REL, JGX_PKR_COUNT);
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES)                              \
    jgx::check_record_field(r, JGX_PKR_##TAG, KIND, RANK, sizeof(T), #comp);
#include "jgx/jorek/records/particle_base_record.def"
#include "jgx/jorek/records/particle_kin_rel_record.def"
#undef JGX_FIELD
    return r;
  }

};

using particle_kin_rel_set_aos = particle_kin_rel_set<layout_stride>;
using particle_kin_rel_set_soa = particle_kin_rel_set<layout_left>;

static_assert(std::is_trivially_copyable<particle_kin_rel_set_aos>::value
              && std::is_trivially_copyable<particle_kin_rel_set_soa>::value,
              "a particle set crosses to a kernel by value");

} /* namespace jorek */

#endif /* JOREK_PARTICLE_SET_H */
