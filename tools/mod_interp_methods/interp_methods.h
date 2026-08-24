/* tools/mod_interp_methods/interp_methods.h -- the interpolation methods of
 * mod_interp_methods.f90, next door.
 *
 * DUPLICATED FROM mod_interp_methods.f90 -- keep the two in step.  The Fortran
 * body stays: it is the general path for callers outside the ported kernels,
 * and test_mod_ccoll_relativistic_jorek compares this against it.
 */
#ifndef JOREK_INTERP_METHODS_H
#define JOREK_INTERP_METHODS_H

#include <cmath>
#include <cstddef>

#include "jgx/macros.h"

namespace interp_methods {

/**
 * An abscissa read through log10.
 *
 * A caller whose grid is equidistant in log10 -- the collision table's u and
 * theta are -- would otherwise have to carry a second, logged copy of both
 * abscissae, which on a device means two more buffers to allocate, fill and
 * upload.  interp_bilinear reads only four abscissa entries per call, so taking
 * their log10 here costs four evaluations and gives bit-identical doubles to
 * the whole-array temporary the Fortran builds at every call site.
 */
template<class AView>
struct log10_abscissa {
  AView a;
  JGX_HD double operator()(const std::size_t i) const { return log10(a(i)); }
};

template<class AView>
JGX_HD inline log10_abscissa<AView> as_log10(const AView& a) { return { a }; }

/**
 * mod_interp_methods::interp_bilinear -- bilinear interpolation on an
 * equidistant mesh.
 *
 * The Fortran takes the extents from the assumed-shape arguments; here they are
 * passed, as everywhere across the seam.
 *
 * One deliberate difference: the cell indices are clamped to the last cell.
 * The Fortran reads f(ix+1, ...) with ix = nx when xq lands exactly on the
 * upper edge of the mesh, which is out of bounds -- reachable from
 * interp_L0L1, whose guard is `u <= u(nu)`.  Clamping returns the edge value
 * instead and changes nothing anywhere inside the mesh.
 *
 * @param x, nx   first abscissa and its length; equidistant, ascending
 * @param y, ny   second abscissa and its length
 * @param f       function values, (nx, ny), first index fastest
 * @param xq, yq  the queried point
 */
template<class XView, class YView, class FView>
JGX_HD inline double interp_bilinear(const XView& x, const int nx,
                                     const YView& y, const int ny,
                                     const FView& f,
                                     const double xq, const double yq) {
  const double x0 = x(0), y0 = y(0);
  const double dx = x(1) - x0;
  const double dy = y(1) - y0;

  int ix = static_cast<int>(floor((xq - x0)/dx));
  int iy = static_cast<int>(floor((yq - y0)/dy));
  if (ix < 0)      ix = 0;
  if (ix > nx - 2) ix = nx - 2;
  if (iy < 0)      iy = 0;
  if (iy > ny - 2) iy = ny - 2;

  const std::size_t i = static_cast<std::size_t>(ix);
  const std::size_t j = static_cast<std::size_t>(iy);
  const double xl = x(i), xr = x(i + 1);
  const double yl = y(j), yr = y(j + 1);

  return ( f(i,     j    )*(xr - xq)*(yr - yq)
         + f(i + 1, j    )*(xq - xl)*(yr - yq)
         + f(i,     j + 1)*(xr - xq)*(yq - yl)
         + f(i + 1, j + 1)*(xq - xl)*(yq - yl) ) / (dx*dy);
} // interp_bilinear

} /* namespace interp_methods */

#endif /* JOREK_INTERP_METHODS_H */
