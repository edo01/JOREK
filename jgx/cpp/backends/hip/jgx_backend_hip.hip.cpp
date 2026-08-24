/* jgx/cpp/backends/hip/jgx_backend_hip.hip.cpp -- the HIP implementation
 * of jgx_c_api.h: device binding, allocation, and the host<->device copies.
 */
#include "jgx/jgx_c_api.h"

#include "jgx/cpp/backends/hip/hip_check.h"

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

void jgx_c_push_symbol(const void* symbol, const void* host_ptr, size_t n_bytes) {
  JGX_HIP_CHECK(hipMemcpyToSymbol(symbol, host_ptr, n_bytes));
}

void jgx_c_pull_symbol(void* host_ptr, const void* symbol, size_t n_bytes) {
  JGX_HIP_CHECK(hipMemcpyFromSymbol(host_ptr, symbol, n_bytes));
}

} /* extern "C" */
