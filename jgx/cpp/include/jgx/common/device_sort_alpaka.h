/* jgx/common/device_sort_alpaka.h -- the counting sort that puts a particle
 * array in element order on the device. alpaka only.
 *
 * TODO: factor with jgx/common/device_sort.h.
 * TODO: aggregate the atomics per warp, as jgx/common/device_sort.h does; here
 * every particle takes its own atomic.
 *
 * A caller runs, per step:
 *
 *   count_bins / scan_bins / assign_slots -- the run start and length of every
 *      element, and the destination slot of every particle;
 *   jgx_c_scatter                         -- the move itself;
 *   scatter_index                         -- the original index, travelling with
 *      the record.
 */
#ifndef JGX_COMMON_DEVICE_SORT_ALPAKA_H
#define JGX_COMMON_DEVICE_SORT_ALPAKA_H

#include "jgx/device.h"
#include "jgx/macros.h"

#if defined(JGX_DEVICE_ALPAKA)

#include "jgx/cpp/backends/alpaka/alpaka_device.h"

#include <cstddef>
#include <cstdint>

namespace jgx {

/* Bins are the elements plus one for "lost": a valid 1-based i_elm goes to
 * i_elm-1, anything else to the last bin. */
ALPAKA_FN_ACC inline int bin_of(int i_elm, int n_bins) {
  const int n_elements = n_bins - 1;
  return (i_elm >= 1 && i_elm <= n_elements) ? i_elm - 1 : n_elements;
}

struct count_bins {
  const int* i_elm;
  std::size_t n;
  int n_bins;
  int* hist;

  ALPAKA_FN_ACC void operator()(alpaka::onAcc::concepts::Acc auto const& acc) const {
    const std::size_t j = alpaka_backend::thread_index(acc);
    if (j >= n) return;
    alpaka::onAcc::atomicAdd(acc, &hist[bin_of(i_elm[j], n_bins)], 1);
  }
};

/* The one-block scan over the bins. */
constexpr unsigned kScanThreads = 1024;

/* Exclusive scan of the histogram, one block, chunk by chunk. It emits the run
 * starts twice (offsets for the accumulate, cursors for assign_slots), the
 * counts, and zeroes the histogram behind the read. */
struct scan_bins {
  int* hist;
  int* offsets;
  int* cursors;
  int* run_len;
  int n_bins;

  ALPAKA_FN_ACC void operator()(alpaka::onAcc::concepts::Acc auto const& acc) const {
    auto buf = alpaka::onAcc::declareSharedMdArray<int, alpaka::uniqueId()>(
        acc, alpaka::CVec<std::uint32_t, kScanThreads>{});
    auto& carry = alpaka::onAcc::declareSharedVar<int, alpaka::uniqueId()>(acc);
    const std::uint32_t t = acc[alpaka::layer::thread].idx().x();
    if (t == 0) carry = 0;
    alpaka::onAcc::syncBlockThreads(acc);

    for (int base = 0; base < n_bins; base += kScanThreads) {
      const int i = base + static_cast<int>(t);
      const int v = (i < n_bins) ? hist[i] : 0;
      buf[alpaka::Vec{t}] = v;
      alpaka::onAcc::syncBlockThreads(acc);

      for (std::uint32_t off = 1; off < kScanThreads; off <<= 1) {
        const int prev = (t >= off) ? buf[alpaka::Vec{t - off}] : 0;
        alpaka::onAcc::syncBlockThreads(acc);
        buf[alpaka::Vec{t}] += prev;
        alpaka::onAcc::syncBlockThreads(acc);
      }

      if (i < n_bins) {
        const int excl = carry + buf[alpaka::Vec{t}] - v;  /* inclusive -> exclusive */
        offsets[i] = excl;
        cursors[i] = excl;
        run_len[i] = v;
        hist[i]    = 0;
      }
      alpaka::onAcc::syncBlockThreads(acc);
      if (t == kScanThreads - 1) carry += buf[alpaka::Vec{kScanThreads - 1}];
      alpaka::onAcc::syncBlockThreads(acc);
    }
  }
};

/* Where each particle goes: the atomic returns the cursor before the increment,
 * so an element's run is packed densely in an unspecified order. */
struct assign_slots {
  const int* i_elm;
  std::size_t n;
  int n_bins;
  int* cursors;
  std::int32_t* dst_slot;

  ALPAKA_FN_ACC void operator()(alpaka::onAcc::concepts::Acc auto const& acc) const {
    const std::size_t j = alpaka_backend::thread_index(acc);
    if (j >= n) return;
    dst_slot[j] = static_cast<std::int32_t>(
        alpaka::onAcc::atomicAdd(acc, &cursors[bin_of(i_elm[j], n_bins)], 1));
  }
};

/* An index rides the same permutation as the record it belongs to. */
struct scatter_index {
  std::int32_t* dst;
  const std::int32_t* src;
  const std::int32_t* dst_slot;
  std::size_t n;

  ALPAKA_FN_ACC void operator()(alpaka::onAcc::concepts::Acc auto const& acc) const {
    const std::size_t j = alpaka_backend::thread_index(acc);
    if (j < n) dst[dst_slot[j]] = src[j];
  }
};

struct iota_index {
  std::int32_t* v;
  std::size_t n;

  ALPAKA_FN_ACC void operator()(alpaka::onAcc::concepts::Acc auto const& acc) const {
    const std::size_t j = alpaka_backend::thread_index(acc);
    if (j < n) v[j] = static_cast<std::int32_t>(j);
  }
};

} /* namespace jgx */

#endif /* JGX_DEVICE_ALPAKA */

#endif /* JGX_COMMON_DEVICE_SORT_ALPAKA_H */
