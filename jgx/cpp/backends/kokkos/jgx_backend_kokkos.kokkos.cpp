/* jgx/cpp/backends/kokkos/jgx_backend_kokkos.kokkos.cpp -- the Kokkos
 * implementation of jgx_c_api.h: runtime lifetime, allocation, and the
 * host<->device copies.
 */
#include "jgx/jgx_c_api.h"

#include <Kokkos_Core.hpp>

#include <cstdio>
#include <cstdlib>

namespace {

using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

using host_bytes   = Kokkos::View<const char*, Kokkos::HostSpace,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using host_bytes_w = Kokkos::View<char*, Kokkos::HostSpace,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using dev_bytes    = Kokkos::View<const char*, memory_space,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using dev_bytes_w  = Kokkos::View<char*, memory_space,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

} /* anonymous namespace */

extern "C" {

const char* jgx_c_backend_name(void) { return "kokkos"; }

/* Same packed layout as every backend: slot = j*n_records + rec. */
const char* jgx_c_layout_tag(void) { return "colmajor"; }

void jgx_c_init_device(int device_id) {
  if (Kokkos::is_initialized()) return;
  const int n_devices = Kokkos::num_devices();
  if (n_devices < 1) {
    std::fprintf(stderr, "%s:%d: no Kokkos device visible to this rank\n",
                 __FILE__, __LINE__);
    std::abort();
  }
  Kokkos::initialize(
      Kokkos::InitializationSettings().set_device_id(device_id % n_devices));
}

void jgx_c_finalize_device(void) {
  if (Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
}

void* jgx_c_alloc(size_t n_bytes) {
  if (n_bytes == 0) return nullptr;
  return Kokkos::kokkos_malloc<memory_space>("jgx_c_alloc", n_bytes);
}

void jgx_c_free(void* device_ptr) {
  if (device_ptr != nullptr) Kokkos::kokkos_free<memory_space>(device_ptr);
}

/* deep_copy without an execution space is synchronous, as the HIP copies are. */
void jgx_c_push(void* device_ptr, const void* host_ptr, size_t n_bytes) {
  if (n_bytes == 0) return;
  Kokkos::deep_copy(dev_bytes_w(static_cast<char*>(device_ptr), n_bytes),
                    host_bytes(static_cast<const char*>(host_ptr), n_bytes));
}

void jgx_c_pull(void* host_ptr, const void* device_ptr, size_t n_bytes) {
  if (n_bytes == 0) return;
  Kokkos::deep_copy(host_bytes_w(static_cast<char*>(host_ptr), n_bytes),
                    dev_bytes(static_cast<const char*>(device_ptr), n_bytes));
}

void jgx_c_memset(void* device_ptr, int value, size_t n_bytes) {
  if (n_bytes == 0) return;
  Kokkos::deep_copy(dev_bytes_w(static_cast<char*>(device_ptr), n_bytes),
                    static_cast<char>(value));
}

void jgx_c_synchronize(void) { Kokkos::fence("jgx_c_synchronize"); }

/* TODO: Kokkos has no device global, so the symbol copies call the runtime of
 * the backend Kokkos is built on. */
void jgx_c_push_symbol(const void* symbol, const void* host_ptr, size_t n_bytes) {
  if (Kokkos::is_initialized()) Kokkos::fence("jgx_c_push_symbol");
#if defined(KOKKOS_ENABLE_CUDA)
  const cudaError_t err = cudaMemcpyToSymbol(symbol, host_ptr, n_bytes);
  const bool ok = (err == cudaSuccess);
#elif defined(KOKKOS_ENABLE_HIP)
  const hipError_t err = hipMemcpyToSymbol(symbol, host_ptr, n_bytes);
  const bool ok = (err == hipSuccess);
#else
#  error "jgx_c_push_symbol: no device global for this Kokkos backend"
#endif
  if (!ok) {
    std::fprintf(stderr, "%s:%d: symbol copy to device failed\n", __FILE__,
                 __LINE__);
    std::abort();
  }
}

void jgx_c_pull_symbol(void* host_ptr, const void* symbol, size_t n_bytes) {
  if (Kokkos::is_initialized()) Kokkos::fence("jgx_c_pull_symbol");
#if defined(KOKKOS_ENABLE_CUDA)
  const cudaError_t err = cudaMemcpyFromSymbol(host_ptr, symbol, n_bytes);
  const bool ok = (err == cudaSuccess);
#elif defined(KOKKOS_ENABLE_HIP)
  const hipError_t err = hipMemcpyFromSymbol(host_ptr, symbol, n_bytes);
  const bool ok = (err == hipSuccess);
#else
#  error "jgx_c_pull_symbol: no device global for this Kokkos backend"
#endif
  if (!ok) {
    std::fprintf(stderr, "%s:%d: symbol copy from device failed\n", __FILE__,
                 __LINE__);
    std::abort();
  }
}

} /* extern "C" */
