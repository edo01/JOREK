/* jgx/common/device_sort_openmp.h -- the counting sort that puts a particle
 * array in element order on the device. OpenMP offload only.
 *
 * TODO: factor with jgx/common/device_sort.h.
 * TODO: every particle takes its own atomic; jgx/common/device_sort.h aggregates
 * them per warp.
 *
 * A caller runs, per step:
 *
 *   count_bins / scan_bins / assign_slots -- the run start and length of every
 *      element, and the destination slot of every particle;
 *   jgx_c_scatter                         -- the move itself;
 *   scatter_index                         -- the original index, travelling with
 *      the record.
 *
 * Each function below is one launch.
 */
#ifndef JGX_COMMON_DEVICE_SORT_OPENMP_H
#define JGX_COMMON_DEVICE_SORT_OPENMP_H

#include "jgx/device.h"
#include "jgx/macros.h"

#if defined(JGX_DEVICE_OPENMP)

#include <cstddef>
#include <cstdint>

namespace jgx {

/* Bins are the elements plus one for "lost": a valid 1-based i_elm goes to
 * i_elm-1, anything else to the last bin. */
inline int bin_of(int i_elm, int n_bins) {
  const int n_elements = n_bins - 1;
  return (i_elm >= 1 && i_elm <= n_elements) ? i_elm - 1 : n_elements;
}

inline void count_bins(const int* i_elm, std::size_t n, int n_bins, int* hist) {
  #pragma omp target teams distribute parallel for is_device_ptr(i_elm, hist)
  for (std::size_t j = 0; j < n; ++j) {
    const int b = bin_of(i_elm[j], n_bins);
    #pragma omp atomic update
    hist[b] += 1;
  }
}

/* The threads of the one-team scan over the bins. */
constexpr int kScanThreads = 1024;

/* Exclusive scan of the histogram, one team, chunk by chunk. It emits the run
 * starts twice (offsets for the accumulate, cursors for assign_slots), the
 * counts, and zeroes the histogram behind the read.
 *
 * Each inner loop ends at the team's barrier; the reads of a doubling pass are
 * staged in `prev` so that its writes do not race them. */
inline void scan_bins(int* hist, int* offsets, int* cursors, int* run_len,
                      int n_bins) {
  #pragma omp target teams num_teams(1) is_device_ptr(hist, offsets, cursors, run_len)
  {
    int buf[kScanThreads];
    int prev[kScanThreads];
    int carry = 0;

    for (int base = 0; base < n_bins; base += kScanThreads) {
      #pragma omp parallel for
      for (int t = 0; t < kScanThreads; ++t)
        buf[t] = (base + t < n_bins) ? hist[base + t] : 0;

      for (int off = 1; off < kScanThreads; off <<= 1) {
        #pragma omp parallel for
        for (int t = 0; t < kScanThreads; ++t)
          prev[t] = (t >= off) ? buf[t - off] : 0;
        #pragma omp parallel for
        for (int t = 0; t < kScanThreads; ++t) buf[t] += prev[t];
      }

      #pragma omp parallel for
      for (int t = 0; t < kScanThreads; ++t) {
        const int i = base + t;
        if (i >= n_bins) continue;
        const int v = hist[i];
        const int excl = carry + buf[t] - v;  /* inclusive -> exclusive */
        offsets[i] = excl;
        cursors[i] = excl;
        run_len[i] = v;
        hist[i]    = 0;
      }
      carry += buf[kScanThreads - 1];
    }
  }
}

/* Where each particle goes: the atomic captures the cursor before the increment,
 * so an element's run is packed densely in an unspecified order. */
inline void assign_slots(const int* i_elm, std::size_t n, int n_bins, int* cursors,
                         std::int32_t* dst_slot) {
  #pragma omp target teams distribute parallel for is_device_ptr(i_elm, cursors, dst_slot)
  for (std::size_t j = 0; j < n; ++j) {
    const int b = bin_of(i_elm[j], n_bins);
    int pos;
    #pragma omp atomic capture
    { pos = cursors[b]; cursors[b] += 1; }
    dst_slot[j] = static_cast<std::int32_t>(pos);
  }
}

/* An index rides the same permutation as the record it belongs to. */
inline void scatter_index(std::int32_t* dst, const std::int32_t* src,
                          const std::int32_t* dst_slot, std::size_t n) {
  #pragma omp target teams distribute parallel for is_device_ptr(dst, src, dst_slot)
  for (std::size_t j = 0; j < n; ++j) dst[dst_slot[j]] = src[j];
}

inline void iota_index(std::int32_t* v, std::size_t n) {
  #pragma omp target teams distribute parallel for is_device_ptr(v)
  for (std::size_t j = 0; j < n; ++j) v[j] = static_cast<std::int32_t>(j);
}

} /* namespace jgx */

#endif /* JGX_DEVICE_OPENMP */

#endif /* JGX_COMMON_DEVICE_SORT_OPENMP_H */
