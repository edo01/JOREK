/* The C entry point of interp_methods.h.  The arrays cross as bare pointers
 * with their extents and the view is built here, as everywhere the data is not
 * a registered record. */
#include "tools/mod_interp_methods/interp_methods.h"

#include "jgx/view.h"

#include <cstddef>
#include <cstdint>

extern "C" {

/* mod_interp_methods::interp_bilinear */
double jgx_host_interp_bilinear(const std::int32_t nx, const std::int32_t ny,
                                const double* x, const double* y, const double* f,
                                const double xq, const double yq) {
  const std::size_t xe[1] = { static_cast<std::size_t>(nx) };
  const std::size_t ye[1] = { static_cast<std::size_t>(ny) };
  const std::size_t fe[2] = { static_cast<std::size_t>(nx),
                              static_cast<std::size_t>(ny) };
  const jgx::view<const double, 1> xv(x, xe), yv(y, ye);
  const jgx::view<const double, 2> fv(f, fe);
  return interp_methods::interp_bilinear(xv, nx, yv, ny, fv, xq, yq);
}

} /* extern "C" */
