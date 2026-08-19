/* jgx/data/field_view.h -- build a view over one component of a record array,
 * either in place (AoS) or over its packed counterpart (SoA).
 *
 * Axis 0 is the record index; the remaining axes are the component's own axes
 * in Fortran order.
 */
#ifndef JGX_DATA_FIELD_VIEW_H
#define JGX_DATA_FIELD_VIEW_H

#include <cstddef>
#include "jgx/macros.h"
#include "jgx/data/record_api.h"
#include "jgx/view.h"

namespace jgx::data {

namespace detail {

/* The extents of a view over one component: the record count, then the
 * component's own extents in Fortran order. A rank-Rank view takes the first
 * Rank-1 of those; a scalar component takes none. */
template <int Rank>
JGX_HD inline void record_extents(const jgx_field_desc& fd, std::size_t n_records,
                                  std::size_t ext[Rank]) {
  static_assert(Rank >= 1 && Rank - 1 <= JGX_MAX_INTRA_RANK,
                "view rank exceeds JGX_MAX_INTRA_RANK + 1");
  ext[0] = n_records;
  for (int d = 1; d < Rank; ++d) ext[d] = fd.intra_extents[d - 1];
}

} /* namespace detail */

/* Component of a Fortran derived-type array, addressed where it lies. Its
 * offset and extents come from the registration, so the caller passes only
 * what varies per array: the base pointer and the record count.
 *
 * The offset and the record stride are whole multiples of sizeof(T);
 * jgx_c_record_add_field refuses a registration where they are not.
 */
template <class T, int Rank>
JGX_HD inline view<T, Rank, layout_stride>
aos_field(void* aos_base, const jgx_field_desc& fd,
          std::size_t record_stride_bytes, std::size_t n_records) {
  std::size_t ext[Rank];
  detail::record_extents<Rank>(fd, n_records, ext);

  std::size_t str[Rank];
  str[0] = record_stride_bytes / sizeof(T);
  for (int d = 1; d < Rank; ++d) str[d] = (d == 1) ? 1 : str[d - 1] * ext[d - 1];

  T* const p = reinterpret_cast<T*>(
      static_cast<char*>(aos_base) + fd.offset_bytes);
  return view<T, Rank, layout_stride>(p, ext, str);
}

/* The same component after packing (on the device), one buffer per field.
 * layout_left over {n_records, intra...} is exactly the pack's
 * j*n_records + rec, so the two cannot drift.
 */
template <class T, int Rank>
JGX_HD inline view<T, Rank, layout_left>
soa_field(void* soa_base, const jgx_field_desc& fd, std::size_t n_records) {
  std::size_t ext[Rank];
  detail::record_extents<Rank>(fd, n_records, ext);
  return view<T, Rank, layout_left>(static_cast<T*>(soa_base), ext);
}

/* A scalar component of a single record, read where it lies.
 *
 * A record whose count is 1 -- an interpolator, say -- has no record axis to
 * make a view over, so its scalar components are read directly through the
 * registered offset. Rank and extents are not consulted: the caller asks for a
 * component it declared scalar at registration.
 */
template <class T>
JGX_HD inline T record_scalar(const void* record_base, const jgx_field_desc& fd) {
  return *reinterpret_cast<const T*>(
      static_cast<const char*>(record_base) + fd.offset_bytes);
}

} /* namespace jgx::data */

#endif /* JGX_DATA_FIELD_VIEW_H */
