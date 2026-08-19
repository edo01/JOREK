/* jgx/record_set.h -- what a record set is, and what every one must provide.
 *
 * A record set is the C++ face of a Fortran array of a derived type: one view
 * per registered component, extents and offsets taken from the registration
 * (jgx_record_api.h). The same members exist in two forms -- layout_stride over
 * the Fortran storage(AoS), layout_left(SoA) over a packed block (jgx/pack.h) -- and a
 * kernel takes the set by template parameter, so no assumption on the layout is made
 * by the c++ code thanks to the view mechanism.
 *
 * A record <name> provides an enum <name>_field (the field ids) and one struct
 * deriving jgx::record_set that carries the field views and, as static members,
 * everything that builds it:
 *
 *   <name>_set<L, Real, Int>::from_aos(base, desc, n)  over the Fortran
 *                                                      storage, always AoS
 *                             from_soa(base, desc, n)  over a packed block
 *                             record()                 the checked descriptor
 *
 * They are static rather than free so that the type is the only thing a caller
 * has to find. Each maker names the layout it builds rather than the current
 * instantiation, so <name>_set_soa::from_aos still yields the AoS set -- there
 * is no wrong-layout call to get wrong. record() is host-only, the registry
 * being host storage; the makers are JGX_HD.
 *
 * A set over a Fortran array is from_aos(base, <name>_set::record(), n), spelled
 * out at the call site: naming record() there is what says the descriptor came
 * from the registry, and it costs one argument that the SoA maker needs anyway.
 *
 * All of it is expanded from one <name>_record.def (jgx/jorek/records/), which
 * the Fortran registration includes as well: the field list, its order, the
 * kinds and the ranks are written once and cannot drift.
 *
 * The two makers take the same three arguments, and differ only in what the
 * base points at:
 *
 *   aos  the Fortran array itself. A component sits at its offset inside a
 *        record and repeats at the record stride, so nothing is copied and
 *        nothing is owned -- the set is valid as long as that array is.
 *   soa  one packed block, owned by a jgx::host_pack_buffers or a
 *        jgx::device_pack that must outlive the set. A component sits at
 *        jgx::soa_field_offset() and runs contiguously; the record stride plays
 *        no part here, only the kind and the intra extents do.
 */
#ifndef JGX_RECORD_SET_H
#define JGX_RECORD_SET_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "jgx/jgx_record_api.h"

namespace jgx {

/* The base of every record set: the record count under one name whatever the
 * record is, and the mark that the type follows the contract above. Empty of
 * behaviour on purpose -- a set is data. */
struct record_set {
  std::size_t n_records = 0;
};

/* Check one registered field against what the .def declares. Registration reads
 * the rank and the extents off the storage itself, so this compares what the
 * .def says a component is against what it turned out to be -- a wrong RANK or
 * KIND column, which registered_record's length check cannot see.
 *
 * Host-side and called once, from <name>_set::record() -- the makers stay
 * JGX_HD and free of it. */
inline void check_record_field(const jgx_record_desc& r, int field_id,
                               std::int32_t elem_kind, int intra_rank,
                               std::size_t elem_size, const char* name) {
  const jgx_field_desc& f = r.field[field_id];
  if (f.elem_kind == elem_kind && f.intra_rank == intra_rank
      && jgx_kind_size(elem_kind) == elem_size)
    return;

  std::fprintf(stderr,
      "jgx record_set: field %d (%s) is registered as kind %d (%zu bytes) "
      "rank %d; the .def declares kind %d (%zu bytes) rank %d over a "
      "%zu-byte type\n",
      field_id, name, int(f.elem_kind), jgx_kind_size(f.elem_kind),
      int(f.intra_rank), int(elem_kind), jgx_kind_size(elem_kind),
      intra_rank, elem_size);
  std::abort();
}

} /* namespace jgx */

#endif /* JGX_RECORD_SET_H */
