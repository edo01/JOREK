#include "elements/mod_basisfunctions/basisfunctions.h"
#include "models/mod_settings/mod_settings.h"

/* The bodies index (degree, vertex). The transposed Fortran entries hand them a
 * layout_left view over H(n_degrees,4); the non-transposed ones a layout_right
 * view over H(4,n_degrees), whose strides {4,1} are that same buffer read
 * transposed. One body, both orders, no duplicated arithmetic.
 */
namespace {
    constexpr std::size_t ext[2] = {JGX_N_DEGREES, 4};

    using view_T = jgx::view<double, 2>;                     // H(n_degrees,4)
    using view_N = jgx::view<double, 2, jgx::layout_right>;   // H(4,n_degrees)
}

extern "C" {
    void jgx_host_basisfunctions_2D_0(const double s, const double t, double* H) {
        basisfunctions::basisfunctions_2D_0_T(s, t, view_N(H, ext));
    }

    void jgx_host_basisfunctions_2D_1(const double s, const double t,
                                      double* H, double* H_s, double* H_t) {
        basisfunctions::basisfunctions_2D_1_T(s, t, view_N(H, ext), view_N(H_s, ext),
                                              view_N(H_t, ext));
    }

    void jgx_host_basisfunctions_2D_1_T(const double s, const double t,
                                        double* H, double* H_s, double* H_t) {
        basisfunctions::basisfunctions_2D_1_T(s, t, view_T(H, ext), view_T(H_s, ext),
                                              view_T(H_t, ext));
    }
}
