#ifndef INTERP_H
#define INTERP_H

#include <cmath>

#include "jgx/view.h"
#include "jgx/macros.h"

namespace interp
{
    template<class BasisFunctionsView>
    JGX_HD inline void sincosperiod_moivre_explicit(const double phi, 
                                                    BasisFunctionsView HZ_view, BasisFunctionsView dHZ_view, 
                                                    const int n_tor_in, const int n_period_in) {
        const int n_mode = (n_tor_in - 1)/2; // number of modes excluding 0

        HZ_view (0) = 1.0;
        dHZ_view(0) = 0.0;

        for (int i = 1; i <= n_mode; ++i) {
            const double n     = static_cast<double>(n_period_in*i);
            const double phase = n*phi;
            // exp(i*phase) = cos(phase) + i*sin(phase)
            const double re    = cos(phase);
            const double im    = sin(phase);
            HZ_view (2*i - 1) =    re;
            HZ_view (2*i    ) =    im;
            dHZ_view(2*i - 1) = -n*im;
            dHZ_view(2*i    ) =  n*re;
        }
    } // sincosperiod_moivre_explicit
} // namespace interp


#endif //INTERP_H