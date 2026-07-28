#include "elements/mod_interp/interp.h"

#include <cstddef>


extern "C" {
    void jgx_host_sincosperiod_moivre_explicit(const double phi,
                                               double* HZ, double* dHZ,
                                               const int n_tor_in, const int n_period_in) {

        // Preparation of the views: HZ/dHZ are rank-1, of extent n_tor_in
        const std::size_t ext[1] = { static_cast<std::size_t>(n_tor_in) };
        const jgx::view<double, 1> HZ_view (HZ,  ext);
        const jgx::view<double, 1> dHZ_view(dHZ, ext);

        interp::sincosperiod_moivre_explicit(phi, HZ_view, dHZ_view, n_tor_in, n_period_in);
    }
}
