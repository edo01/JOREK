/* jgx/common/device_sort_kokkos.h -- the sort that puts a particle array in
 * element order on the device. Kokkos only.
 *
 * TODO: factor with jgx/common/device_sort.h.
 *
 * A caller runs, per step:
 *
 *   element_sort::sort   -- the run start and length of every element, and the
 *      destination slot of every particle;
 *   jgx_c_scatter        -- the move itself, every field of the record;
 *   scatter_index        -- the original index, travelling with the record.
 */
#ifndef JGX_COMMON_DEVICE_SORT_KOKKOS_H
#define JGX_COMMON_DEVICE_SORT_KOKKOS_H

#include "jgx/device.h"
#include "jgx/macros.h"

#if defined(JGX_DEVICE_KOKKOS)

#include <Kokkos_Sort.hpp>

#include <cstddef>
#include <cstdint>

namespace jgx {

using device_ints = Kokkos::View<int*, Kokkos::DefaultExecutionSpace>;

/* Bins are the elements plus one for "lost": a valid 1-based i_elm goes to
 * i_elm-1, anything else to the last bin. */
KOKKOS_INLINE_FUNCTION int bin_of(int i_elm, int n_bins) {
  const int n_elements = n_bins - 1;
  return (i_elm >= 1 && i_elm <= n_elements) ? i_elm - 1 : n_elements;
}

/* The keys already hold the bin. */
struct element_bin_op {
  int n_bins = 0;

  int max_bins() const { return n_bins; }

  template <class ViewType>
  KOKKOS_INLINE_FUNCTION int bin(ViewType& keys, const int& i) const {
    return keys(i);
  }

  template <class ViewType, typename iType1, typename iType2>
  KOKKOS_INLINE_FUNCTION bool operator()(ViewType& keys, iType1& i1,
                                         iType2& i2) const {
    return keys(i1) < keys(i2);
  }
};

/* Allocated once per call of the launcher, reused by every step. */
class element_sort {
 public:
  using bin_sort = Kokkos::BinSort<device_ints, element_bin_op,
                                   Kokkos::DefaultExecutionSpace, int>;

  element_sort(std::size_t n, int n_bins)
      : n_(n), n_bins_(n_bins),
        keys_("jgx::element_sort::keys", n),
        bs_(keys_, element_bin_op{n_bins}) {}

  /* Bins by `i_elm` and writes dst_slot[rec], the position of record rec in
   * element order. Asynchronous. */
  void sort(const int* i_elm, std::int32_t* dst_slot) {
    const device_ints keys = keys_;
    const int n_bins = n_bins_;
    Kokkos::parallel_for(
        "jgx::element_sort::keys", n_,
        KOKKOS_LAMBDA(const std::size_t j) { keys(j) = bin_of(i_elm[j], n_bins); });

    /* create_permute_vector accumulates into the counts of the previous call. */
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{}, bs_.bin_count_atomic, 0);
    bs_.create_permute_vector(Kokkos::DefaultExecutionSpace{});

    const auto order = bs_.get_permute_vector();
    Kokkos::parallel_for(
        "jgx::element_sort::slots", n_,
        KOKKOS_LAMBDA(const std::size_t p) { dst_slot[order(p)] = std::int32_t(p); });
  }

  /* Run start and length of every bin, valid once sort() has run. */
  typename bin_sort::offset_type run_start() const { return bs_.get_bin_offsets(); }
  typename bin_sort::bin_count_type run_len() const { return bs_.get_bin_count(); }

 private:
  std::size_t n_;
  int n_bins_;
  device_ints keys_;
  bin_sort bs_;
};

/* An index rides the same permutation as the record it belongs to. */
inline void scatter_index(std::int32_t* dst, const std::int32_t* src,
                          const std::int32_t* dst_slot, std::size_t n) {
  Kokkos::parallel_for(
      "jgx::scatter_index", n,
      KOKKOS_LAMBDA(const std::size_t j) { dst[dst_slot[j]] = src[j]; });
}

inline void iota_index(std::int32_t* v, std::size_t n) {
  Kokkos::parallel_for(
      "jgx::iota_index", n,
      KOKKOS_LAMBDA(const std::size_t j) { v[j] = std::int32_t(j); });
}

} /* namespace jgx */

#endif /* JGX_DEVICE_KOKKOS */

#endif /* JGX_COMMON_DEVICE_SORT_KOKKOS_H */
