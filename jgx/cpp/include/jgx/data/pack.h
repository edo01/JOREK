/* jgx/data/pack.h -- AoS <-> SoA transposition of a record array.
 *
 * A packed block, for a record descriptor r and a record count n, is laid out
 * as follows:
 *
 *   - one contiguous buffer per registered field, in field-id order, each
 *     padded to kSoaFieldAlign;
 *   - within a field's buffer, element j of record rec occupies slot
 *     j*n + rec, the layout_left ordering soa_field() (field_view.h) reads.
 *
 * This header defines those two rules, the corresponding address in the AoS
 * block, and the element copy between them. It defines no traversal: iterating
 * over records belongs to the caller.
 *
 * Every entity is JGX_HD and driven by the descriptor, so host and device share
 * one definition and no per-type code is generated.
 */
#ifndef JGX_DATA_PACK_H
#define JGX_DATA_PACK_H

#include <cstddef>
#include <cstdint>

#include "jgx/data/record_api.h"
#include "jgx/macros.h"

namespace jgx::data {

/* Number of elements one record contributes to field `fd`: the product of the
 * field's intra-record extents, 1 for a scalar. */
JGX_HD inline std::size_t field_elems(const jgx_field_desc& fd) {
  std::size_t n = 1;
  for (int d = 0; d < fd.intra_rank; ++d) n *= fd.intra_extents[d];
  return n;
}

/* Size in bytes of field `fd`'s packed buffer for `n_records` records. */
JGX_HD inline std::size_t field_bytes(const jgx_field_desc& fd,
                                      std::size_t n_records) {
  return n_records * field_elems(fd) * jgx_kind_size(fd.elem_kind);
}

/* Alignment required of every field buffer within a packed block. Each buffer
 * is padded up to it, so a buffer's alignment does not depend on the fields
 * preceding it. */
constexpr std::size_t kSoaFieldAlign = 256;

/* `n` rounded up to a multiple of `a`. */
JGX_HD inline std::size_t align_up(std::size_t n, std::size_t a) {
  return ((n + a - 1) / a) * a;
}

/* Offset in bytes of field `field_id`'s buffer from the base of the packed
 * block: the sum of the padded sizes of the fields before it. Field ids are
 * dense from 0, so the offset is derived rather than stored. */
JGX_HD inline std::size_t soa_field_offset(const jgx_record_desc& r,
                                           int field_id,
                                           std::size_t n_records) {
  std::size_t off = 0;
  for (int f = 0; f < field_id; ++f)
    off += align_up(field_bytes(r.field[f], n_records), kSoaFieldAlign);
  return off;
}

/* Size in bytes of the whole packed block: the offset one past the last
 * field. */
JGX_HD inline std::size_t soa_block_bytes(const jgx_record_desc& r,
                                          std::size_t n_records) {
  return soa_field_offset(r, r.n_fields, n_records);
}

/* Address of field `field_id`'s buffer within the block based at `soa_base`.
 * This is the base soa_field() (field_view.h) takes. */
JGX_HD inline void* soa_field_ptr(void* soa_base, const jgx_record_desc& r,
                                  int field_id, std::size_t n_records) {
  return static_cast<char*>(soa_base)
       + soa_field_offset(r, field_id, n_records);
}

JGX_HD inline const void* soa_field_ptr(const void* soa_base,
                                        const jgx_record_desc& r, int field_id,
                                        std::size_t n_records) {
  return static_cast<const char*>(soa_base)
       + soa_field_offset(r, field_id, n_records);
}

/* Offset in bytes of element j of record `rec` of field `fd` from the base of
 * the AoS block. */
JGX_HD inline std::size_t aos_offset(const jgx_field_desc& fd,
                                     std::size_t record_stride_bytes,
                                     std::size_t rec, std::size_t j) {
  return rec * record_stride_bytes + fd.offset_bytes
       + j * jgx_kind_size(fd.elem_kind);
}

/* Offset in bytes of element j of record `rec` of field `fd` from the base of
 * that field's packed buffer. */
JGX_HD inline std::size_t soa_offset(const jgx_field_desc& fd,
                                     std::size_t n_records,
                                     std::size_t rec, std::size_t j) {
  return (j * n_records + rec) * jgx_kind_size(fd.elem_kind);
}

/* Copies `elem_size` bytes from `src` to `dst`, dispatching on the size rather
 * than on a type: a descriptor carries a kind tag, not a C++ type. Both
 * addresses are required to be elem_size-aligned. */
JGX_HD inline void copy_elem(void* dst, const void* src, std::size_t elem_size) {
  switch (elem_size) {
    case 8: *static_cast<std::uint64_t*>(dst) =
                *static_cast<const std::uint64_t*>(src); break;
    case 4: *static_cast<std::uint32_t*>(dst) =
                *static_cast<const std::uint32_t*>(src); break;
    case 2: *static_cast<std::uint16_t*>(dst) =
                *static_cast<const std::uint16_t*>(src); break;
    case 1: *static_cast<std::uint8_t*>(dst) =
                *static_cast<const std::uint8_t*>(src); break;
    default: {
      char* d = static_cast<char*>(dst);
      const char* s = static_cast<const char*>(src);
      for (std::size_t b = 0; b < elem_size; ++b) d[b] = s[b];
    }
  }
}

/* Copies element j of record `rec` of field `fd` from the AoS block at
 * `aos_base` to that field's packed buffer at `field_base`. */
JGX_HD inline void pack_elem(void* field_base, const void* aos_base,
                             const jgx_field_desc& fd,
                             std::size_t record_stride_bytes,
                             std::size_t n_records,
                             std::size_t rec, std::size_t j) {
  copy_elem(static_cast<char*>(field_base) + soa_offset(fd, n_records, rec, j),
            static_cast<const char*>(aos_base)
                + aos_offset(fd, record_stride_bytes, rec, j),
            jgx_kind_size(fd.elem_kind));
}

/* Copies element j of record `rec` of field `fd` from that field's packed
 * buffer at `field_base` to the AoS block at `aos_base`. */
JGX_HD inline void unpack_elem(void* aos_base, const void* field_base,
                               const jgx_field_desc& fd,
                               std::size_t record_stride_bytes,
                               std::size_t n_records,
                               std::size_t rec, std::size_t j) {
  copy_elem(static_cast<char*>(aos_base)
                + aos_offset(fd, record_stride_bytes, rec, j),
            static_cast<const char*>(field_base)
                + soa_offset(fd, n_records, rec, j),
            jgx_kind_size(fd.elem_kind));
}

} /* namespace jgx::data */

#endif /* JGX_DATA_PACK_H */
