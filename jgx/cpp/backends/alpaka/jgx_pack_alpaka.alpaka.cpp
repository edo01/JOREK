/* jgx/cpp/backends/alpaka/jgx_pack_alpaka.alpaka.cpp -- the alpaka kernels for
 * the AoS<->SoA transposition and the permutation of a packed block.
 */
#include "jgx/jgx_c_api.h"
#include "jgx/data/record_api.h"
#include "jgx/data/pack.h"

#include "jgx/cpp/backends/alpaka/alpaka_device.h"

#include <cstddef>
#include <cstdint>

namespace {

constexpr unsigned kPackBlock = 256;

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

/* One thread per (record, component element) of ONE field, the field being the
 * block's y index; x covers the widest field, so a narrower field leaves some of
 * its blocks with nothing to do. */
auto record_spec(const jgx_record_desc& r, const record_spans& s) {
  const unsigned grid_x = static_cast<unsigned>((s.widest + kPackBlock - 1) / kPackBlock);
  return alpaka::onHost::ThreadSpec{
      alpaka::Vec{static_cast<unsigned>(r.n_fields), grid_x},
      alpaka::Vec{1u, kPackBlock}, jgx::alpaka_backend::kExecutor};
}

/* The field and the element index of a thread. */
struct field_elem {
  int f;
  std::size_t i;
};

ALPAKA_FN_ACC inline field_elem field_elem_of(
    alpaka::onAcc::concepts::Acc auto const& acc) {
  const auto b = acc[alpaka::layer::block].idx();
  const auto t = acc[alpaka::layer::thread].idx();
  const auto d = acc[alpaka::layer::thread].count();
  return {static_cast<int>(b.y()),
          std::size_t(b.x()) * std::size_t(d.x()) + std::size_t(t.x())};
}

struct pack_kernel {
  char* soa;
  const char* aos;
  jgx_record_desc rd;
  record_spans s;
  std::size_t n_records;

  ALPAKA_FN_ACC void operator()(alpaka::onAcc::concepts::Acc auto const& acc) const {
    const field_elem fe = field_elem_of(acc);
    if (fe.i >= s.n_elems[fe.f]) return;
    const std::size_t rec = fe.i % n_records, j = fe.i / n_records;
    jgx::data::copy_elem(
        soa + s.off[fe.f] + jgx::data::soa_offset(rd.field[fe.f], n_records, rec, j),
        aos + jgx::data::aos_offset(rd.field[fe.f], rd.record_stride_bytes, rec, j),
        s.elem_size[fe.f]);
  }
};

struct unpack_kernel {
  char* aos;
  const char* soa;
  jgx_record_desc rd;
  record_spans s;
  std::size_t n_records;

  ALPAKA_FN_ACC void operator()(alpaka::onAcc::concepts::Acc auto const& acc) const {
    const field_elem fe = field_elem_of(acc);
    if (fe.i >= s.n_elems[fe.f]) return;
    const std::size_t rec = fe.i % n_records, j = fe.i / n_records;
    jgx::data::copy_elem(
        aos + jgx::data::aos_offset(rd.field[fe.f], rd.record_stride_bytes, rec, j),
        soa + s.off[fe.f] + jgx::data::soa_offset(rd.field[fe.f], n_records, rec, j),
        s.elem_size[fe.f]);
  }
};

/* Element j of record `rec` goes to element j of record dst_slot[rec]. */
struct scatter_kernel {
  char* dst;
  const char* src;
  jgx_record_desc rd;
  record_spans s;
  std::size_t n_records;
  const std::int32_t* dst_slot;

  ALPAKA_FN_ACC void operator()(alpaka::onAcc::concepts::Acc auto const& acc) const {
    const field_elem fe = field_elem_of(acc);
    if (fe.i >= s.n_elems[fe.f]) return;
    const std::size_t rec = fe.i % n_records, j = fe.i / n_records;
    jgx::data::copy_elem(
        dst + s.off[fe.f] + jgx::data::soa_offset(rd.field[fe.f], n_records,
                                                  std::size_t(dst_slot[rec]), j),
        src + s.off[fe.f] + jgx::data::soa_offset(rd.field[fe.f], n_records, rec, j),
        s.elem_size[fe.f]);
  }
};

} /* anonymous namespace */

extern "C" {

/* Asynchronous, as the HIP launches. */
void jgx_c_pack(void* soa_base, const jgx_record_desc* r,
                const void* aos_base, size_t n_records) {
  if (n_records == 0) return;
  const record_spans s = spans_of(*r, n_records);
  jgx::alpaka_backend::queue().enqueue(
      record_spec(*r, s),
      pack_kernel{static_cast<char*>(soa_base), static_cast<const char*>(aos_base),
                  *r, s, n_records});
}

void jgx_c_unpack(void* aos_base, const void* soa_base,
                  const jgx_record_desc* r, size_t n_records) {
  if (n_records == 0) return;
  const record_spans s = spans_of(*r, n_records);
  jgx::alpaka_backend::queue().enqueue(
      record_spec(*r, s),
      unpack_kernel{static_cast<char*>(aos_base), static_cast<const char*>(soa_base),
                    *r, s, n_records});
}

void jgx_c_scatter(void* dst_soa, const void* src_soa, const jgx_record_desc* r,
                   size_t n_records, const int32_t* dst_slot) {
  if (n_records == 0) return;
  const record_spans s = spans_of(*r, n_records);
  jgx::alpaka_backend::queue().enqueue(
      record_spec(*r, s),
      scatter_kernel{static_cast<char*>(dst_soa), static_cast<const char*>(src_soa),
                     *r, s, n_records, dst_slot});
}

} /* extern "C" */
