/* jgx/cpp/backends/kokkos/jgx_pack_kokkos.kokkos.cpp -- the Kokkos kernels for
 * the AoS<->SoA transposition and the permutation of a packed block.
 */
#include "jgx/jgx_c_api.h"
#include "jgx/data/record_api.h"
#include "jgx/data/pack.h"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>

namespace {

/* The per-field quantities a kernel needs, computed once on the host. */
struct record_spans {
  std::size_t off[JGX_MAX_RECORD_FIELDS];        /* offset within a packed block */
  std::size_t n_elems[JGX_MAX_RECORD_FIELDS];    /* n_records * elements per record */
  std::size_t elem_size[JGX_MAX_RECORD_FIELDS];
  std::size_t widest;
};

record_spans spans_of(const jgx_record_desc& r, std::size_t n_records) {
  record_spans s{};
  for (int f = 0; f < r.n_fields; ++f) {
    s.off[f]       = jgx::data::soa_field_offset(r, f, n_records);
    s.n_elems[f]   = n_records * jgx::data::field_elems(r.field[f]);
    s.elem_size[f] = jgx_kind_size(r.field[f].elem_kind);
    if (s.n_elems[f] > s.widest) s.widest = s.n_elems[f];
  }
  return s;
}

/* One iterate per (field, component element), the second index covering the
 * widest field. */
using record_policy =
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<std::int64_t>>;

record_policy record_range(const jgx_record_desc& r, const record_spans& s) {
  return record_policy({0, 0}, {std::int64_t(r.n_fields), std::int64_t(s.widest)});
}

} /* anonymous namespace */

extern "C" {

void jgx_c_pack(void* soa_base, const jgx_record_desc* r,
                const void* aos_base, size_t n_records) {
  if (n_records == 0) return;
  const jgx_record_desc rd = *r;
  const record_spans s = spans_of(rd, n_records);
  const char* aos = static_cast<const char*>(aos_base);
  char* soa = static_cast<char*>(soa_base);
  Kokkos::parallel_for(
      "jgx_c_pack", record_range(rd, s),
      KOKKOS_LAMBDA(const std::int64_t f, const std::int64_t i) {
        if (std::size_t(i) >= s.n_elems[f]) return;
        const std::size_t rec = std::size_t(i) % n_records, j = std::size_t(i) / n_records;
        jgx::data::copy_elem(
            soa + s.off[f] + jgx::data::soa_offset(rd.field[f], n_records, rec, j),
            aos + jgx::data::aos_offset(rd.field[f], rd.record_stride_bytes, rec, j),
            s.elem_size[f]);
      });
}

void jgx_c_unpack(void* aos_base, const void* soa_base,
                  const jgx_record_desc* r, size_t n_records) {
  if (n_records == 0) return;
  const jgx_record_desc rd = *r;
  const record_spans s = spans_of(rd, n_records);
  char* aos = static_cast<char*>(aos_base);
  const char* soa = static_cast<const char*>(soa_base);
  Kokkos::parallel_for(
      "jgx_c_unpack", record_range(rd, s),
      KOKKOS_LAMBDA(const std::int64_t f, const std::int64_t i) {
        if (std::size_t(i) >= s.n_elems[f]) return;
        const std::size_t rec = std::size_t(i) % n_records, j = std::size_t(i) / n_records;
        jgx::data::copy_elem(
            aos + jgx::data::aos_offset(rd.field[f], rd.record_stride_bytes, rec, j),
            soa + s.off[f] + jgx::data::soa_offset(rd.field[f], n_records, rec, j),
            s.elem_size[f]);
      });
}

/* Element j of record `rec` goes to element j of record dst_slot[rec]. */
void jgx_c_scatter(void* dst_soa, const void* src_soa, const jgx_record_desc* r,
                   size_t n_records, const int32_t* dst_slot) {
  if (n_records == 0) return;
  const jgx_record_desc rd = *r;
  const record_spans s = spans_of(rd, n_records);
  char* dst = static_cast<char*>(dst_soa);
  const char* src = static_cast<const char*>(src_soa);
  Kokkos::parallel_for(
      "jgx_c_scatter", record_range(rd, s),
      KOKKOS_LAMBDA(const std::int64_t f, const std::int64_t i) {
        if (std::size_t(i) >= s.n_elems[f]) return;
        const std::size_t rec = std::size_t(i) % n_records, j = std::size_t(i) / n_records;
        jgx::data::copy_elem(
            dst + s.off[f] + jgx::data::soa_offset(rd.field[f], n_records,
                                                   std::size_t(dst_slot[rec]), j),
            src + s.off[f] + jgx::data::soa_offset(rd.field[f], n_records, rec, j),
            s.elem_size[f]);
      });
}

} /* extern "C" */
