/* jgx/cpp/backends/alpaka/alpaka_device.h -- the device, queue and launch shape
 * of the alpaka backend, shared by its translation units.
 */
#ifndef JGX_BACKENDS_ALPAKA_DEVICE_H
#define JGX_BACKENDS_ALPAKA_DEVICE_H

#include "jgx/device.h"

#include <cstddef>
#include <utility>

namespace jgx {
namespace alpaka_backend {

/* One vendor API per build, selected by the compiler that builds the unit. */
#if ALPAKA_LANG_HIP
inline constexpr auto kApi        = alpaka::api::hip;
inline constexpr auto kDeviceKind = alpaka::deviceKind::amdGpu;
inline constexpr auto kExecutor   = alpaka::exec::gpuHip;
#else
inline constexpr auto kApi        = alpaka::api::cuda;
inline constexpr auto kDeviceKind = alpaka::deviceKind::nvidiaGpu;
inline constexpr auto kExecutor   = alpaka::exec::gpuCuda;
#endif

using device_type = decltype(alpaka::onHost::makeDeviceSelector(
                                 alpaka::onHost::DeviceSpec{kApi, kDeviceKind})
                                 .makeDevice(0));
using queue_type = decltype(std::declval<device_type&>().makeQueue());

/* Valid between jgx_c_init_device and jgx_c_finalize_device. Launches on the
 * queue are asynchronous, as HIP launches on the null stream are. */
device_type& device();
queue_type& queue();

/* Exactly n_blocks x n_threads, as a HIP launch; a frame specification would let
 * alpaka reshape it, which the block-shared memory of a kernel does not allow. */
inline auto thread_spec(unsigned n_blocks, unsigned n_threads) {
  return alpaka::onHost::ThreadSpec{alpaka::Vec{n_blocks}, alpaka::Vec{n_threads},
                                    kExecutor};
}

/* The global index of a thread in a one-dimensional launch. */
ALPAKA_FN_ACC inline std::size_t thread_index(
    alpaka::onAcc::concepts::Acc auto const& acc) {
  return std::size_t(acc[alpaka::layer::block].idx().x()) *
             std::size_t(acc[alpaka::layer::thread].count().x()) +
         std::size_t(acc[alpaka::layer::thread].idx().x());
}

} /* namespace alpaka_backend */
} /* namespace jgx */

#endif /* JGX_BACKENDS_ALPAKA_DEVICE_H */
