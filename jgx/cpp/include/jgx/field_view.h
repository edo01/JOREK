/* jgx/field_view.h -- build a view over one component of a record array,
 * either in place (AoS) or over its packed counterpart (SoA).
 *
 * Axis 0 is the record index; the remaining axes are the component's own axes
 * in Fortran order.
 */
#ifndef JGX_FIELD_VIEW_H
#define JGX_FIELD_VIEW_H

#include <cstddef>
#include "jgx/macros.h"
#include "jgx/view.h"

namespace jgx {

/* Component of a Fortran derived-type array, addressed where it lies.
 * ext[0] is the record count, ext[1..] the intra-record extents.
 *
 * offset_bytes and record_stride_bytes are measured on the Fortran side; both
 * must be whole multiples of sizeof(T), which holds for naturally aligned
 * components.
 */
template <class T, int Rank>
JGX_HD inline view<T, Rank, layout_stride>
aos_field(void* record_base, std::size_t offset_bytes,
          std::size_t record_stride_bytes, const std::size_t ext[Rank]) {
  std::size_t str[Rank];
  str[0] = record_stride_bytes / sizeof(T);
  for (int d = 1; d < Rank; ++d) str[d] = (d == 1) ? 1 : str[d - 1] * ext[d - 1];

  T* const p = reinterpret_cast<T*>(
      static_cast<char*>(record_base) + offset_bytes);
  return view<T, Rank, layout_stride>(p, ext, str);
}

/* The same component after packing (on the device), one buffer per field. layout_left over
 * {n_records, intra...} is exactly the pack's j*n_records + rec, so the two
 * cannot drift.
 */
template <class T, int Rank>
JGX_HD inline view<T, Rank, layout_left>
soa_field(void* field_base, const std::size_t ext[Rank]) {
  return view<T, Rank, layout_left>(static_cast<T*>(field_base), ext);
}

/* Whether a component can be addressed by whole-T strides at all. Checked by
 * the caller; the builders above assume it. */
template <class T>
inline bool field_is_addressable(std::size_t offset_bytes,
                                 std::size_t record_stride_bytes) {
  return offset_bytes % sizeof(T) == 0 && record_stride_bytes % sizeof(T) == 0;
}

} /* namespace jgx */

#endif /* JGX_FIELD_VIEW_H */
