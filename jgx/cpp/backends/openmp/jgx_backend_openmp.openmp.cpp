/* jgx/cpp/backends/openmp/jgx_backend_openmp.openmp.cpp -- the OpenMP target
 * offload implementation of jgx_c_api.h: device binding, allocation, and the
 * host<->device copies.
 */
#include "jgx/jgx_c_api.h"
#include "jgx/device.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

void check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "jgx openmp: %s failed\n", what);
    std::abort();
  }
}

} /* anonymous namespace */

extern "C" {

const char* jgx_c_backend_name(void) { return "openmp"; }

/* Same packed layout as every backend: slot = j*n_records + rec. */
const char* jgx_c_layout_tag(void) { return "colmajor"; }

void jgx_c_init_device(int device_id) {
  const int n_devices = omp_get_num_devices();
  if (n_devices < 1) {
    std::fprintf(stderr, "%s:%d: no OpenMP offload device visible to this rank\n",
                 __FILE__, __LINE__);
    std::abort();
  }
  omp_set_default_device(device_id % n_devices);
}

/* The runtime releases the device at process exit. */
void jgx_c_finalize_device(void) {}

void* jgx_c_alloc(size_t n_bytes) {
  if (n_bytes == 0) return nullptr;
  void* p = omp_target_alloc(n_bytes, omp_get_default_device());
  check(p != nullptr, "omp_target_alloc");
  return p;
}

void jgx_c_free(void* device_ptr) {
  if (device_ptr != nullptr) omp_target_free(device_ptr, omp_get_default_device());
}

void jgx_c_push(void* device_ptr, const void* host_ptr, size_t n_bytes) {
  check(omp_target_memcpy(device_ptr, const_cast<void*>(host_ptr), n_bytes, 0, 0,
                          omp_get_default_device(), omp_get_initial_device()) == 0,
        "omp_target_memcpy (push)");
}

void jgx_c_pull(void* host_ptr, const void* device_ptr, size_t n_bytes) {
  check(omp_target_memcpy(host_ptr, const_cast<void*>(device_ptr), n_bytes, 0, 0,
                          omp_get_initial_device(), omp_get_default_device()) == 0,
        "omp_target_memcpy (pull)");
}

void jgx_c_memset(void* device_ptr, int value, size_t n_bytes) {
  if (n_bytes == 0) return;
  unsigned char* const p = static_cast<unsigned char*>(device_ptr);
  const unsigned char v = static_cast<unsigned char>(value);
  #pragma omp target teams distribute parallel for is_device_ptr(p)
  for (size_t i = 0; i < n_bytes; ++i) p[i] = v;
}

/* Target regions are synchronous here: nothing is left queued. */
void jgx_c_synchronize(void) {}

/* A declare-target variable has a host copy and a device copy; both are written,
 * so that host code in a device translation unit reads the pushed value too. */
void jgx_c_push_symbol(const void* symbol, const void* host_ptr, size_t n_bytes) {
  char* const s = const_cast<char*>(static_cast<const char*>(symbol));
  std::memcpy(s, host_ptr, n_bytes);
  #pragma omp target update to(s[0:n_bytes])
}

void jgx_c_pull_symbol(void* host_ptr, const void* symbol, size_t n_bytes) {
  char* const s = const_cast<char*>(static_cast<const char*>(symbol));
  #pragma omp target update from(s[0:n_bytes])
  std::memcpy(host_ptr, s, n_bytes);
}

} /* extern "C" */
