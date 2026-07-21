/* jgx/view.h -- A lightweight, non-owning, mdspan-like accessor over raw
 * device/host memory. Header-only C++17, no dependency on any
 * backend. JGX owns the memory; views only index it.
 *
 * Canonical index order is Fortran order: a Fortran array a(n1,n2,n3) maps to
 * view<T,3,layout_left> with extent{n1,n2,n3} and a(i,j,k) [1-based] ==
 * v(i-1,j-1,k-1). This should resolve Fortran<->C indexing bug.
 */
#ifndef JGX_VIEW_H
#define JGX_VIEW_H

#include <cstddef>
#include <array>
#include "jgx/macros.h"

namespace jgx {

struct layout_left  {};   /* column-major: first index fastest (Fortran)  */
struct layout_right {};   /* row-major:    last index fastest  (C)         */

/* Arbitrary axis order : the axes listed
 * FASTEST-FIRST. layout_perm<0,1,2> == layout_left, layout_perm<2,1,0> ==
 * layout_right, and every ordering in between is expressible
 */
template <int... P> struct layout_perm {};

/*
 *  One util function for each layout_left, layout_right and layout_perm
 *  to fill the data following the right axis order. 
 */
namespace detail {
  template <int Rank>
  JGX_HD inline void fill_strides(const std::size_t ext[Rank],
                                  std::size_t str[Rank], layout_left) {
    std::size_t acc = 1;
    for (int d = 0; d < Rank; ++d) { str[d] = acc; acc *= ext[d]; }
  }
  template <int Rank>
  JGX_HD inline void fill_strides(const std::size_t ext[Rank],
                                  std::size_t str[Rank], layout_right) {
    std::size_t acc = 1;
    for (int d = Rank - 1; d >= 0; --d) { str[d] = acc; acc *= ext[d]; }
  }
  template <int Rank, int... P>
  JGX_HD inline void fill_strides(const std::size_t ext[Rank],
                                  std::size_t str[Rank], layout_perm<P...>) {
    static_assert(sizeof...(P) == Rank,
                  "layout_perm arity must equal the view Rank");
    const int perm[Rank] = { P... };
    std::size_t acc = 1;
    for (int d = 0; d < Rank; ++d) { str[perm[d]] = acc; acc *= ext[perm[d]]; }
  }
}

template <class T, int Rank, class Layout = layout_left>
struct view {
  T*          data = nullptr;
  std::size_t extent[Rank] = {};
  std::size_t stride[Rank] = {};

  view() = default;

  JGX_HD view(T* p, const std::size_t ext[Rank]) : data(p) {
    for (int d = 0; d < Rank; ++d) extent[d] = ext[d];
    detail::fill_strides<Rank>(extent, stride, Layout{});
  }

  JGX_HD T& operator()(std::size_t i0) const {
    static_assert(Rank == 1, "jgx::view: index count must equal Rank");
    return data[i0*stride[0]];
  }
  JGX_HD T& operator()(std::size_t i0, std::size_t i1) const {
    static_assert(Rank == 2, "jgx::view: index count must equal Rank");
    return data[i0*stride[0] + i1*stride[1]];
  }
  JGX_HD T& operator()(std::size_t i0, std::size_t i1, std::size_t i2) const {
    static_assert(Rank == 3, "jgx::view: index count must equal Rank");
    return data[i0*stride[0] + i1*stride[1] + i2*stride[2]];
  }
  JGX_HD T& operator()(std::size_t i0, std::size_t i1,
                       std::size_t i2, std::size_t i3) const {
    static_assert(Rank == 4, "jgx::view: index count must equal Rank");
    return data[i0*stride[0] + i1*stride[1] + i2*stride[2] + i3*stride[3]];
  }
  JGX_HD T& operator()(std::size_t i0, std::size_t i1, std::size_t i2,
                       std::size_t i3, std::size_t i4) const {
    static_assert(Rank == 5, "jgx::view: index count must equal Rank");
    return data[i0*stride[0] + i1*stride[1] + i2*stride[2]
              + i3*stride[3] + i4*stride[4]];
  }
};

template <class T, int Rank, class Layout = layout_left>
JGX_HD inline view<T, Rank, Layout> make_view(void* p,
                                              std::array<std::size_t, Rank> ext) {
  return view<T, Rank, Layout>(static_cast<T*>(p), ext.data());
}

/* Build a view straight from a jgx_buf_desc's device pointer + extents.
 * extents are Fortran order, so layout_left is the natural choice but it
 * can be changed by either using layout_right or layout_permutation.
 */
template <class T, int Rank, class Layout = layout_left, class Desc>
JGX_HD inline view<T, Rank, Layout> view_from_desc(const Desc& d) {
  std::size_t ext[Rank];
  for (int i = 0; i < Rank; ++i) ext[i] = d.extents[i];
  return view<T, Rank, Layout>(static_cast<T*>(d.device_ptr), ext);
}

} /* namespace jgx */

#endif /* JGX_VIEW_H */
