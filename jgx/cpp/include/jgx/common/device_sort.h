/* jgx/common/device_sort.h -- the counting sort that puts a particle array in
 * element order on the device. HIP and CUDA only.
 *
 *
 * A caller runs, per step:
 *
 *   count_bins  / scan_bins / assign_slots  -- the run start and length of every
 *      element, and the destination slot of every particle;
 *   jgx_c_scatter                           -- the move itself, every field of
 *      the record, driven by the descriptor;
 *   scatter_index / scatter_rows            -- anything travelling alongside the
 *      record: the original index, staged per-particle values.
 *
 */
#ifndef JGX_COMMON_DEVICE_SORT_H
#define JGX_COMMON_DEVICE_SORT_H

#include "jgx/device.h"
#include "jgx/macros.h"

#if defined(JGX_DEVICE_CUDA) || defined(JGX_DEVICE_HIP)

#include <cstddef>
#include <cstdint>

namespace jgx {
namespace {

/* Bins are the elements plus one for "lost": a valid 1-based i_elm goes to
 * i_elm-1, anything else to the last bin. An accumulate walks the element bins
 * only, so the lost run is simply never read. */
JGX_HD inline int bin_of(int i_elm, int n_bins) {
  const int n_elements = n_bins - 1;
  return (i_elm >= 1 && i_elm <= n_elements) ? i_elm - 1 : n_elements;
}

/* The three warp-collective primitives the aggregation needs, spelled per
 * platform. CUDA dropped the maskless __shfl at sm_70 and HIP-on-NVIDIA does not
 * supply the AMD spellings, so neither side's names work for both; the warp is
 * also 32 lanes wide on one and 64 on the other. Confined to these two wrappers.
 */
#if defined(__HIP_PLATFORM_AMD__)
constexpr int kWarp = 64;
__device__ inline unsigned long long warp_ballot(bool p) { return __ballot(p); }
__device__ inline int warp_bcast(int v, int src) { return __shfl(v, src, kWarp); }
#else
constexpr int kWarp = 32;
__device__ inline unsigned long long warp_ballot(bool p) {
  return static_cast<unsigned long long>(__ballot_sync(0xffffffffu, p));
}
__device__ inline int warp_bcast(int v, int src) {
  return __shfl_sync(0xffffffffu, v, src);
}
#endif

/* Aggregate the lanes of a warp that share a bin into one global atomic.
 *
 * Worth the trouble precisely because the input is nearly sorted already -- the
 * particles were sorted last step and few crossed an element -- so a warp's 32
 * lanes usually want the *same* counter, which is the case a plain per-lane
 * atomicAdd serialises 32 ways.
 *
 * Every thread of the warp runs every iteration, including the ones with no
 * particle (have = false): the ballots are warp-wide and must not be reached
 * under divergence. `pos_out`, when given, receives the lane's own slot within
 * the run its group reserved.
 */
__device__ inline void warp_bin_add(int* counter, bool have, int key,
                                    int* pos_out) {
  const int lane = static_cast<int>(threadIdx.x) % kWarp;
  const unsigned long long lt = (1ull << lane) - 1ull;

  unsigned long long active = warp_ballot(have);
  while (active) {
    const int leader = __ffsll(active) - 1;
    const int lkey   = warp_bcast(key, leader);
    const unsigned long long peers = warp_ballot(have && key == lkey) & active;

    int base = 0;
    if (lane == leader) base = atomicAdd(&counter[lkey], __popcll(peers));
    base = warp_bcast(base, leader);

    if (pos_out != nullptr && ((peers >> lane) & 1ull))
      *pos_out = base + __popcll(peers & lt);

    active &= ~peers;
  }
}

__global__ void count_bins(const int* __restrict__ i_elm, std::size_t n,
                           int n_bins, int* __restrict__ hist) {
  const std::size_t j = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  const bool have = (j < n);
  const int key = have ? bin_of(i_elm[j], n_bins) : 0;
  warp_bin_add(hist, have, key, nullptr);
}

/* The one-block scan over the bins. */
constexpr unsigned kScanThreads = 1024;

/* Exclusive scan of the histogram, one block, chunk by chunk.
 *
 * A single block is enough: the bins are the elements, tens of thousands at
 * most, and this runs once per pusher step against a push that reads the mesh
 * for every particle. A multi-block scan would add two more launches to a loop
 * that is launch-bound already.
 *
 * It emits four things from the one read of the histogram, so that the step loop
 * needs no host-side memset or device-to-device copy -- both of those are
 * synchronous in the jgx C ABI, and a synchronous call per pusher step would
 * drain the queue tens of thousands of times per fluid step:
 *   offsets  the run start, what assign_slots reserves from and the accumulate
 *            reads;
 *   cursors  the same values, to be consumed by assign_slots;
 *   run_len  the counts, preserved for the accumulate;
 *   hist     zeroed behind the read, ready for the next step's count_bins.
 */
__global__ void scan_bins(int* __restrict__ hist, int* __restrict__ offsets,
                          int* __restrict__ cursors, int* __restrict__ run_len,
                          int n_bins) {
  __shared__ int buf[kScanThreads];
  __shared__ int carry;
  if (threadIdx.x == 0) carry = 0;
  __syncthreads();

  for (int base = 0; base < n_bins; base += kScanThreads) {
    const int i = base + static_cast<int>(threadIdx.x);
    const int v = (i < n_bins) ? hist[i] : 0;
    buf[threadIdx.x] = v;
    __syncthreads();

    for (unsigned off = 1; off < kScanThreads; off <<= 1) {
      const int t = (threadIdx.x >= off) ? buf[threadIdx.x - off] : 0;
      __syncthreads();
      buf[threadIdx.x] += t;
      __syncthreads();
    }

    if (i < n_bins) {
      const int excl = carry + buf[threadIdx.x] - v;  /* inclusive -> exclusive */
      offsets[i] = excl;
      cursors[i] = excl;
      run_len[i] = v;
      hist[i]    = 0;
    }
    __syncthreads();
    if (threadIdx.x == kScanThreads - 1) carry += buf[kScanThreads - 1];
    __syncthreads();
  }
}

/* Where each particle goes. `cursors` starts as a copy of the run starts, so a
 * group's reservation packs its element's run densely; the order within a run is
 * unspecified, which the accumulate does not care about. */
__global__ void assign_slots(const int* __restrict__ i_elm, std::size_t n,
                             int n_bins, int* __restrict__ cursors,
                             std::int32_t* __restrict__ dst_slot) {
  const std::size_t j = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  const bool have = (j < n);
  const int key = have ? bin_of(i_elm[j], n_bins) : 0;
  int pos = 0;
  warp_bin_add(cursors, have, key, &pos);
  if (have) dst_slot[j] = static_cast<std::int32_t>(pos);
}

/* An index rides the same permutation as the record it belongs to. */
__global__ void scatter_index(std::int32_t* __restrict__ dst,
                              const std::int32_t* __restrict__ src,
                              const std::int32_t* __restrict__ dst_slot,
                              std::size_t n) {
  const std::size_t j = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  if (j < n) dst[dst_slot[j]] = src[j];
}

/* So do per-particle values staged outside the record: n_rows arrays of n
 * doubles, laid out row-major so that a row is contiguous over the particles.
 * One launch for all of them -- the step loop is launch-bound, and a kernel per
 * row would add one per staged quantity per step. */
__global__ void scatter_rows(double* __restrict__ dst,
                             const double* __restrict__ src,
                             const std::int32_t* __restrict__ dst_slot,
                             std::size_t n, int n_rows) {
  const std::size_t j = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  if (j >= n) return;
  const std::size_t to = static_cast<std::size_t>(dst_slot[j]);
  for (int r = 0; r < n_rows; ++r) dst[r*n + to] = src[r*n + j];
}

__global__ void iota_index(std::int32_t* __restrict__ v, std::size_t n) {
  const std::size_t j = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  if (j < n) v[j] = static_cast<std::int32_t>(j);
}

} /* anonymous namespace */
} /* namespace jgx */

#endif /* device build */

#endif /* JGX_COMMON_DEVICE_SORT_H */
