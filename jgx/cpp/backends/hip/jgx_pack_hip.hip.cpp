/* jgx/cpp/backends/hip/jgx_pack_hip.hip.cpp -- the HIP kernels for the the AoS<->SoA 
 * transposition and the permutation of a packed block.
 */
#include "jgx/jgx_c_api.h"
#include "jgx/data/record_api.h"
#include "jgx/data/pack.h"

#include "jgx/cpp/backends/hip/hip_check.h"

// Kernel code for the AoS<->SoA transposition
namespace {

constexpr unsigned kPackBlock = 256;

/* One thread per (record, component element) of ONE field, the field being
 * blockIdx.y -- so a thread never searches for which field it is in, and the
 * field's offset within the packed block is worked out once per thread instead
 * of once per element copied.
 *
 * Fields have different element counts, so the grid is sized for the largest
 * and each field grid-strides over its own.
 */
struct field_span {
  const jgx_field_desc* fd;
  std::size_t off;        /* the field's offset within a packed block */
  std::size_t n_elems;    /* n_records * elements per record */
  std::size_t elem_size;
};

__device__ inline field_span span_of(const jgx_record_desc& r, int f,
                                     std::size_t n_records) {
  field_span s;
  s.fd        = &r.field[f];
  s.off       = jgx::data::soa_field_offset(r, f, n_records);
  s.n_elems   = n_records * jgx::data::field_elems(r.field[f]);
  s.elem_size = jgx_kind_size(r.field[f].elem_kind);
  return s;
}

__global__ void pack_record(char* soa_base, const char* aos_base,
                            const jgx_record_desc r, std::size_t n_records) {
  const field_span s = span_of(r, blockIdx.y, n_records);
  char* field_base = soa_base + s.off;
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       i < s.n_elems; i += std::size_t(gridDim.x) * blockDim.x) {
    const std::size_t rec = i % n_records, j = i / n_records;
    jgx::data::copy_elem(
        field_base + jgx::data::soa_offset(*s.fd, n_records, rec, j),
        aos_base + jgx::data::aos_offset(*s.fd, r.record_stride_bytes, rec, j),
        s.elem_size);
  }
}

__global__ void unpack_record(char* aos_base, const char* soa_base,
                              const jgx_record_desc r, std::size_t n_records) {
  const field_span s = span_of(r, blockIdx.y, n_records);
  const char* field_base = soa_base + s.off;
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       i < s.n_elems; i += std::size_t(gridDim.x) * blockDim.x) {
    const std::size_t rec = i % n_records, j = i / n_records;
    jgx::data::copy_elem(
        aos_base + jgx::data::aos_offset(*s.fd, r.record_stride_bytes, rec, j),
        field_base + jgx::data::soa_offset(*s.fd, n_records, rec, j),
        s.elem_size);
  }
}

/* Both sides are packed blocks, so the move stays inside one field's buffer:
 * element j of record `rec` goes to element j of record dst_slot[rec]. Reads are
 * contiguous across a warp; the writes are not, which is the cost of reordering
 * and the reason it is done here rather than by gathering at every later read. */
__global__ void scatter_record(char* dst_base, const char* src_base,
                               const jgx_record_desc r, std::size_t n_records,
                               const int32_t* dst_slot) {
  const field_span s = span_of(r, blockIdx.y, n_records);
  const char* src = src_base + s.off;
  char*       dst = dst_base + s.off;
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       i < s.n_elems; i += std::size_t(gridDim.x) * blockDim.x) {
    const std::size_t rec = i % n_records, j = i / n_records;
    jgx::data::copy_elem(
        dst + jgx::data::soa_offset(*s.fd, n_records,
                                    static_cast<std::size_t>(dst_slot[rec]), j),
        src + jgx::data::soa_offset(*s.fd, n_records, rec, j),
        s.elem_size);
  }
}

/* x covers the largest field, y is the field. A field with fewer elements
 * simply leaves some of its x blocks with nothing to do. */
dim3 record_grid(const jgx_record_desc& r, std::size_t n_records) {
  std::size_t widest = 0;
  for (int f = 0; f < r.n_fields; ++f)
    widest = max(widest, n_records * jgx::data::field_elems(r.field[f]));
  return dim3(static_cast<unsigned>((widest + kPackBlock - 1) / kPackBlock),
              static_cast<unsigned>(r.n_fields));
}

} /* anonymous namespace */

extern "C" {

/*
 * Call the *asynchronous* kernel that perform the transposition
 * on the device.
 */
void jgx_c_pack(void* soa_base, const jgx_record_desc* r,
                const void* aos_base, size_t n_records) {
  if (n_records == 0) return;
  pack_record<<<record_grid(*r, n_records), kPackBlock>>>(
      static_cast<char*>(soa_base), static_cast<const char*>(aos_base),
      *r, n_records);
  JGX_HIP_CHECK(hipGetLastError());
}

/*
 * Call the *asynchronous* kernel that perform the transposition
 * on the device.
 */
void jgx_c_unpack(void* aos_base, const void* soa_base,
                  const jgx_record_desc* r, size_t n_records) {
  if (n_records == 0) return;
  unpack_record<<<record_grid(*r, n_records), kPackBlock>>>(
      static_cast<char*>(aos_base), static_cast<const char*>(soa_base),
      *r, n_records);
  JGX_HIP_CHECK(hipGetLastError());
}

/*
 * Call the *asynchronous* kernel that permutes a packed block.
 */
void jgx_c_scatter(void* dst_soa, const void* src_soa, const jgx_record_desc* r,
                   size_t n_records, const int32_t* dst_slot) {
  if (n_records == 0) return;
  scatter_record<<<record_grid(*r, n_records), kPackBlock>>>(
      static_cast<char*>(dst_soa), static_cast<const char*>(src_soa),
      *r, n_records, dst_slot);
  JGX_HIP_CHECK(hipGetLastError());
}

} /* extern "C" */
