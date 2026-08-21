/* jgx/cpp/backends/hip/jgx_backend_hip.hip.cpp -- the HIP implementation
 * of jgx_c_api.h.
 */
#include "jgx/jgx_c_api.h"
#include "jgx/data/record_api.h"
#include "jgx/data/pack.h"

#include "jgx/cpp/backends/hip/hip_check.h"

// Kernel code for the AoS<->SoA transposition
namespace {

constexpr unsigned kPackBlock = 256;

/* One thread per (field, record, component element), one launch for the whole
 * record.
 */
struct slot_id {
  int         field;
  std::size_t slot;
};

__device__ inline slot_id locate(const jgx_record_desc& r,
                                 std::size_t n_records, std::size_t k) {
  slot_id s{0, k};
  for (; s.field < r.n_fields; ++s.field) {
    const std::size_t n = n_records * jgx::data::field_elems(r.field[s.field]);
    if (s.slot < n) break;
    s.slot -= n;
  }
  return s;
}

__global__ void pack_record(char* soa_base, const char* aos_base,
                            const jgx_record_desc r, std::size_t n_records,
                            std::size_t n_slots) {
  const std::size_t k = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  if (k >= n_slots) return;
  const slot_id s = locate(r, n_records, k);
  jgx::data::pack_elem(jgx::data::soa_field_ptr(soa_base, r, s.field, n_records),
                 aos_base, r.field[s.field], r.record_stride_bytes,
                 n_records, s.slot % n_records, s.slot / n_records);
}

__global__ void unpack_record(char* aos_base, const char* soa_base,
                              const jgx_record_desc r, std::size_t n_records,
                              std::size_t n_slots) {
  const std::size_t k = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  if (k >= n_slots) return;
  const slot_id s = locate(r, n_records, k);
  jgx::data::unpack_elem(aos_base,
                   jgx::data::soa_field_ptr(soa_base, r, s.field, n_records),
                   r.field[s.field], r.record_stride_bytes, n_records,
                   s.slot % n_records, s.slot / n_records);
}

std::size_t record_slots(const jgx_record_desc& r, std::size_t n_records) {
  std::size_t n = 0;
  for (int f = 0; f < r.n_fields; ++f) n += n_records * jgx::data::field_elems(r.field[f]);
  return n;
}

unsigned n_blocks(std::size_t n_slots) {
  return static_cast<unsigned>((n_slots + kPackBlock - 1) / kPackBlock);
}

} /* anonymous namespace */

extern "C" {

const char* jgx_c_backend_name(void) { return "hip"; }

/* Which layout the buffers this backend hands out are in. The pack writes
 * slot = j*n_records + rec, i.e. layout_left over {n_records, intra...}
 * (jgx/field_view.h), so the answer is fixed at compile time for the backend as
 * a whole and not per buffer -- which is why it is a string and not carried
 * alongside each pointer. */
const char* jgx_c_layout_tag(void) { return "colmajor"; }

/* Rank->device binding happens above this call (jgx-gpu-backend.md, section
 * 4.6: kinetic_main, from the node-local rank) -- this just pins the process
 * to the device it was told to use. */
void jgx_c_init_device(int device_id) { JGX_HIP_CHECK(hipSetDevice(device_id)); }

/* Symmetric with jgx_c_init_device: releases everything the runtime holds for this
 * device on this process, rather than leaving it to process exit. */
void jgx_c_finalize_device(void) { JGX_HIP_CHECK(hipDeviceReset()); }

/* hipMalloc(&p, 0) is not guaranteed a null p across HIP versions, and every
 * caller already routes n_bytes==0 through here (jgx_backend_push/pull guard
 * it, jgx_backend_alloc does not) -- so it is handled once, on the spot,
 * instead of trusting every caller to skip a zero-sized buffer. */
void* jgx_c_alloc(size_t n_bytes) {
  if (n_bytes == 0) return nullptr;
  void* p = nullptr;
  JGX_HIP_CHECK(hipMalloc(&p, n_bytes));
  return p;
}

void jgx_c_free(void* device_ptr) { JGX_HIP_CHECK(hipFree(device_ptr)); }

void jgx_c_push(void* device_ptr, const void* host_ptr, size_t n_bytes) {
  JGX_HIP_CHECK(hipMemcpy(device_ptr, host_ptr, n_bytes, hipMemcpyHostToDevice));
}

void jgx_c_pull(void* host_ptr, const void* device_ptr, size_t n_bytes) {
  JGX_HIP_CHECK(hipMemcpy(host_ptr, device_ptr, n_bytes, hipMemcpyDeviceToHost));
}

/* Synchronous here, like the two copies above: it precedes a launch that must
 * see the fill, and the null stream would order them anyway. */
void jgx_c_memset(void* device_ptr, int value, size_t n_bytes) {
  JGX_HIP_CHECK(hipMemset(device_ptr, value, n_bytes));
}

/* Both the errors a launch can produce: the one it reported on the spot, and the
 * one it only reports on completion. Checking them here is what lets a caller
 * launch a kernel and then call nothing but jgx_c_*. */
void jgx_c_synchronize(void) {
  JGX_HIP_CHECK(hipGetLastError());
  JGX_HIP_CHECK(hipDeviceSynchronize());
}

/*
 * Call the *asynchronous* kernel that perform the transposition
 * on the device.
 */
void jgx_c_pack(void* soa_base, const jgx_record_desc* r,
                const void* aos_base, size_t n_records) {
  if (n_records == 0) return;
  const std::size_t n_slots = record_slots(*r, n_records);
  pack_record<<<n_blocks(n_slots), kPackBlock>>>(
      static_cast<char*>(soa_base), static_cast<const char*>(aos_base),
      *r, n_records, n_slots);
  JGX_HIP_CHECK(hipGetLastError());
}

/*
 * Call the *asynchronous* kernel that perform the transposition
 * on the device.
 */
void jgx_c_unpack(void* aos_base, const void* soa_base,
                  const jgx_record_desc* r, size_t n_records) {
  if (n_records == 0) return;
  const std::size_t n_slots = record_slots(*r, n_records);
  unpack_record<<<n_blocks(n_slots), kPackBlock>>>(
      static_cast<char*>(aos_base), static_cast<const char*>(soa_base),
      *r, n_records, n_slots);
  JGX_HIP_CHECK(hipGetLastError());
}

void jgx_c_push_symbol(const void* symbol, const void* host_ptr, size_t n_bytes) {
  JGX_HIP_CHECK(hipMemcpyToSymbol(symbol, host_ptr, n_bytes));
}

void jgx_c_pull_symbol(void* host_ptr, const void* symbol, size_t n_bytes) {
  JGX_HIP_CHECK(hipMemcpyFromSymbol(host_ptr, symbol, n_bytes));
}

} /* extern "C" */
