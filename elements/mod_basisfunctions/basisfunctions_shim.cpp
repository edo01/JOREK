#include "elements/mod_basisfunctions/basisfunctions.h"
#include "jgx/jorek/jorek_settings.h"

extern "C" {
    void jgx_host_basisfunctions_2D_1_T(const double s, const double t,
                                        double* H, double* H_s, double* H_t) {
        // Preparation of the views
        const std::size_t ext[2] = {JGX_N_DEGREES,4};
        const jgx::view<double, 2> H_view(H, ext);
        const jgx::view<double, 2> H_s_view(H_s, ext);
        const jgx::view<double, 2> H_t_view(H_t, ext);

        basisfunctions::basisfunctions_2D_1_T(s, t, H_view, H_s_view, H_t_view);
    }
}