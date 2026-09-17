/* jgx/cpp/backends/alpaka/jgx_backend_alpaka.alpaka.cpp -- the alpaka
 * implementation of jgx_c_api.h: device binding, allocation, and the
 * host<->device copies.
 */
#include "jgx/jgx_c_api.h"

#include "jgx/cpp/backends/alpaka/alpaka_device.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace {

using jgx::alpaka_backend::device_type;
using jgx::alpaka_backend::kApi;
using jgx::alpaka_backend::kDeviceKind;
using jgx::alpaka_backend::queue_type;

using byte_buffer = decltype(alpaka::onHost::alloc<char>(
    std::declval<device_type&>(), alpaka::Vec{std::size_t{}}));

std::optional<device_type> g_device;
std::optional<queue_type>  g_queue;

/* alpaka buffers own their memory; the C ABI hands out the raw pointer, so the
 * buffer is kept here until jgx_c_free. */
std::mutex g_buffers_mutex;
std::unordered_map<void*, byte_buffer> g_buffers;

auto device_bytes(void* p, std::size_t n) {
  return alpaka::makeView(kApi, static_cast<char*>(p), alpaka::Vec{n});
}

auto host_bytes(void* p, std::size_t n) {
  return alpaka::makeView(alpaka::api::host, static_cast<char*>(p), alpaka::Vec{n});
}

} /* anonymous namespace */

namespace jgx {
namespace alpaka_backend {

device_type& device() { return *g_device; }
queue_type& queue() { return *g_queue; }

} /* namespace alpaka_backend */
} /* namespace jgx */

extern "C" {

const char* jgx_c_backend_name(void) { return "alpaka"; }

/* Same packed layout as every backend: slot = j*n_records + rec. */
const char* jgx_c_layout_tag(void) { return "colmajor"; }

void jgx_c_init_device(int device_id) {
  if (g_device) return;
  auto selector =
      alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{kApi, kDeviceKind});
  const int n_devices = static_cast<int>(selector.getDeviceCount());
  if (n_devices < 1) {
    std::fprintf(stderr, "%s:%d: no alpaka device visible to this rank\n",
                 __FILE__, __LINE__);
    std::abort();
  }
  g_device.emplace(selector.makeDevice(device_id % n_devices));
  g_queue.emplace(g_device->makeQueue());
}

void jgx_c_finalize_device(void) {
  if (!g_device) return;
  alpaka::onHost::wait(*g_queue);
  {
    const std::lock_guard<std::mutex> lock(g_buffers_mutex);
    g_buffers.clear();
  }
  g_queue.reset();
  g_device.reset();
}

void* jgx_c_alloc(size_t n_bytes) {
  if (n_bytes == 0) return nullptr;
  byte_buffer buf = alpaka::onHost::alloc<char>(*g_device, alpaka::Vec{n_bytes});
  void* p = alpaka::onHost::data(buf);
  const std::lock_guard<std::mutex> lock(g_buffers_mutex);
  g_buffers.emplace(p, std::move(buf));
  return p;
}

/* Freed only once the queue has drained, as hipFree waits for the device. */
void jgx_c_free(void* device_ptr) {
  if (device_ptr == nullptr) return;
  alpaka::onHost::wait(*g_queue);
  const std::lock_guard<std::mutex> lock(g_buffers_mutex);
  g_buffers.erase(device_ptr);
}

/* The copies and the fill are synchronous, as the HIP ones are. */
void jgx_c_push(void* device_ptr, const void* host_ptr, size_t n_bytes) {
  if (n_bytes == 0) return;
  alpaka::onHost::memcpy(*g_queue, device_bytes(device_ptr, n_bytes),
                         host_bytes(const_cast<void*>(host_ptr), n_bytes));
  alpaka::onHost::wait(*g_queue);
}

void jgx_c_pull(void* host_ptr, const void* device_ptr, size_t n_bytes) {
  if (n_bytes == 0) return;
  alpaka::onHost::memcpy(*g_queue, host_bytes(host_ptr, n_bytes),
                         device_bytes(const_cast<void*>(device_ptr), n_bytes));
  alpaka::onHost::wait(*g_queue);
}

void jgx_c_memset(void* device_ptr, int value, size_t n_bytes) {
  if (n_bytes == 0) return;
  alpaka::onHost::memset(*g_queue, device_bytes(device_ptr, n_bytes),
                         static_cast<std::uint8_t>(value));
  alpaka::onHost::wait(*g_queue);
}

void jgx_c_synchronize(void) { alpaka::onHost::wait(*g_queue); }

/* TODO: device globals through ALPAKA_DEVICE_GLOBAL; meanwhile the symbol copies
 * call the runtime of the vendor API, on the device the queue is bound to. */
void jgx_c_push_symbol(const void* symbol, const void* host_ptr, size_t n_bytes) {
  if (g_queue) alpaka::onHost::wait(*g_queue);
#if ALPAKA_LANG_CUDA
  if (g_device) cudaSetDevice(g_device->getNativeHandle());
  const bool ok = (cudaMemcpyToSymbol(symbol, host_ptr, n_bytes) == cudaSuccess);
#elif ALPAKA_LANG_HIP
  if (g_device) hipSetDevice(g_device->getNativeHandle());
  const bool ok = (hipMemcpyToSymbol(symbol, host_ptr, n_bytes) == hipSuccess);
#else
#  error "jgx_c_push_symbol: no device global for this alpaka API"
#endif
  if (!ok) {
    std::fprintf(stderr, "%s:%d: symbol copy to device failed\n", __FILE__,
                 __LINE__);
    std::abort();
  }
}

void jgx_c_pull_symbol(void* host_ptr, const void* symbol, size_t n_bytes) {
  if (g_queue) alpaka::onHost::wait(*g_queue);
#if ALPAKA_LANG_CUDA
  if (g_device) cudaSetDevice(g_device->getNativeHandle());
  const bool ok = (cudaMemcpyFromSymbol(host_ptr, symbol, n_bytes) == cudaSuccess);
#elif ALPAKA_LANG_HIP
  if (g_device) hipSetDevice(g_device->getNativeHandle());
  const bool ok = (hipMemcpyFromSymbol(host_ptr, symbol, n_bytes) == hipSuccess);
#else
#  error "jgx_c_pull_symbol: no device global for this alpaka API"
#endif
  if (!ok) {
    std::fprintf(stderr, "%s:%d: symbol copy from device failed\n", __FILE__,
                 __LINE__);
    std::abort();
  }
}

} /* extern "C" */
