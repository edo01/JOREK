/* jgx/cpp/backends/openmp/jgx_pack_openmp.openmp.cpp -- the OpenMP offload
 * kernels for the AoS<->SoA transposition and the permutation of a packed block.
 */
#include "jgx/jgx_c_api.h"
#include "jgx/data/record_api.h"
#include "jgx/data/pack.h"
#include "jgx/device.h"

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

} /* anonymous namespace */

extern "C" {

/* One iteration per (field, record component element); the element range
 * covers the widest field, so a narrower field leaves some with nothing to do. */
void jgx_c_pack(void* soa_base, const jgx_record_desc* r,
                const void* aos_base, size_t n_records) {
  if (n_records == 0) return;
  const jgx_record_desc rd = *r;
  const record_spans s = spans_of(rd, n_records);
  const int n_fields = rd.n_fields;
  const std::size_t widest = s.widest;
  char* const soa = static_cast<char*>(soa_base);
  const char* const aos = static_cast<const char*>(aos_base);

  #pragma omp target teams distribute parallel for collapse(2) \
      is_device_ptr(soa, aos) map(to: rd, s)
  for (int f = 0; f < n_fields; ++f)
    for (std::size_t i = 0; i < widest; ++i) {
      if (i >= s.n_elems[f]) continue;
      const std::size_t rec = i % n_records, j = i / n_records;
      jgx::data::copy_elem(
          soa + s.off[f] + jgx::data::soa_offset(rd.field[f], n_records, rec, j),
          aos + jgx::data::aos_offset(rd.field[f], rd.record_stride_bytes, rec, j),
          s.elem_size[f]);
    }
}

void jgx_c_unpack(void* aos_base, const void* soa_base,
                  const jgx_record_desc* r, size_t n_records) {
  if (n_records == 0) return;
  const jgx_record_desc rd = *r;
  const record_spans s = spans_of(rd, n_records);
  const int n_fields = rd.n_fields;
  const std::size_t widest = s.widest;
  char* const aos = static_cast<char*>(aos_base);
  const char* const soa = static_cast<const char*>(soa_base);

  #pragma omp target teams distribute parallel for collapse(2) \
      is_device_ptr(soa, aos) map(to: rd, s)
  for (int f = 0; f < n_fields; ++f)
    for (std::size_t i = 0; i < widest; ++i) {
      if (i >= s.n_elems[f]) continue;
      const std::size_t rec = i % n_records, j = i / n_records;
      jgx::data::copy_elem(
          aos + jgx::data::aos_offset(rd.field[f], rd.record_stride_bytes, rec, j),
          soa + s.off[f] + jgx::data::soa_offset(rd.field[f], n_records, rec, j),
          s.elem_size[f]);
    }
}

/* Element j of record `rec` goes to element j of record dst_slot[rec]. */
void jgx_c_scatter(void* dst_soa, const void* src_soa, const jgx_record_desc* r,
                   size_t n_records, const int32_t* dst_slot) {
  if (n_records == 0) return;
  const jgx_record_desc rd = *r;
  const record_spans s = spans_of(rd, n_records);
  const int n_fields = rd.n_fields;
  const std::size_t widest = s.widest;
  char* const dst = static_cast<char*>(dst_soa);
  const char* const src = static_cast<const char*>(src_soa);

  #pragma omp target teams distribute parallel for collapse(2) \
      is_device_ptr(dst, src, dst_slot) map(to: rd, s)
  for (int f = 0; f < n_fields; ++f)
    for (std::size_t i = 0; i < widest; ++i) {
      if (i >= s.n_elems[f]) continue;
      const std::size_t rec = i % n_records, j = i / n_records;
      jgx::data::copy_elem(
          dst + s.off[f] + jgx::data::soa_offset(rd.field[f], n_records,
                                                 std::size_t(dst_slot[rec]), j),
          src + s.off[f] + jgx::data::soa_offset(rd.field[f], n_records, rec, j),
          s.elem_size[f]);
    }
}

} /* extern "C" */
