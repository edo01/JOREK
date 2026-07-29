#ifndef BASISFUNCTIONS_H
#define BASISFUNCTIONS_H

#include "jgx/view.h"
#include "jgx/macros.h"

/**
 * @todo: Fix parameters definitions
 */


namespace basisfunctions{

    template<class BasisFunctionsZView>
    JGX_HD inline void basisfunctions_2D_1_T(const double s, const double t, 
                                            BasisFunctionsZView H_view, BasisFunctionsZView H_s_view, 
                                            BasisFunctionsZView H_t_view) {
        // --- CUBIC
        //if (n_order .eq. 3) then
        const double sm  = s - 1.0;
        const double tm  = t - 1.0;
        const double sm2 = sm*sm;
        const double tm2 = tm*tm;
        const double s2  = s*s;
        const double t2  = t*t;

        //---------------------------------------------------------- vertex (1)
        H_view  (0,0) =       sm2*(1.0 + 2.0*s)*tm2*(1.0 + 2.0*t);
        H_s_view(0,0) =  6.0*sm*s            *tm2*(1.0 + 2.0*t);
        H_t_view(0,0) =  6.0*sm2*(1.0 + 2.0*s)*tm*t;
        H_view  (1,0) =  3.0*sm2*s            *tm2*(1.0 + 2.0*t);
        H_s_view(1,0) =  3.0*sm*(3.0*s - 1.0) *tm2*(1.0 + 2.0*t);
        H_t_view(1,0) = 18.0*sm2*s            *tm*t;
        H_view  (2,0) =  3.0*sm2*(1.0 + 2.0*s)*tm2*t;
        H_s_view(2,0) = 18.0*sm*s            *tm2*t;
        H_t_view(2,0) =  3.0*sm2*(1.0 + 2.0*s)*tm*(3.0*t - 1.0);
        H_view  (3,0) =  9.0*sm2*s            *tm2*t;
        H_s_view(3,0) =  9.0*sm*(3.0*s - 1.0) *tm2*t;
        H_t_view(3,0) =  9.0*sm2*s            *tm*(3.0*t - 1.0);
        //---------------------------------------------------------- vertex (2)
        H_view  (0,1) = -       s2*(2.0*s - 3.0)*tm2*(1.0 + 2.0*t);
        H_s_view(0,1) = -  6.0*sm*s             *tm2*(1.0 + 2.0*t);
        H_t_view(0,1) = -  6.0*s2*(2.0*s - 3.0)*tm*t;
        H_view  (1,1) = -  3.0*sm*s2            *tm2*(1.0 + 2.0*t);
        H_s_view(1,1) = -  3.0*s*(3.0*s - 2.0)  *tm2*(1.0 + 2.0*t);
        H_t_view(1,1) = - 18.0*sm*s2            *tm*t;
        H_view  (2,1) = -  3.0*s2*(2.0*s - 3.0)*tm2*t;
        H_s_view(2,1) = - 18.0*sm*s             *tm2*t;
        H_t_view(2,1) =    3.0*s2*(2.0*s - 3.0)*(1.0 - 3.0*t)*tm;
        H_view  (3,1) = -  9.0*sm*s2            *tm2*t;
        H_s_view(3,1) = -  9.0*s*(3.0*s - 2.0)  *tm2*t;
        H_t_view(3,1) =    9.0*sm*s2            *(1.0 - 3.0*t)*tm;
        //---------------------------------------------------------- vertex (3)
        H_view  (0,2) =       s2*(2.0*s - 3.0)*t2*(2.0*t - 3.0);
        H_s_view(0,2) =  6.0*sm*s             *t2*(2.0*t - 3.0);
        H_t_view(0,2) =  6.0*s2*(2.0*s - 3.0)*tm*t;
        H_view  (1,2) =  3.0*sm*s2            *t2*(2.0*t - 3.0);
        H_s_view(1,2) =  3.0*s*(3.0*s - 2.0)  *t2*(2.0*t - 3.0);
        H_t_view(1,2) = 18.0*sm*s2            *tm*t;
        H_view  (2,2) =  3.0*s2*(2.0*s - 3.0)*tm*t2;
        H_s_view(2,2) = 18.0*sm*s             *tm*t2;
        H_t_view(2,2) =  3.0*s2*(2.0*s - 3.0)*t*(3.0*t - 2.0);
        H_view  (3,2) =  9.0*sm*s2            *tm*t2;
        H_s_view(3,2) =  9.0*s*(3.0*s - 2.0)  *tm*t2;
        H_t_view(3,2) =  9.0*sm*s2            *t*(3.0*t - 2.0);
        //---------------------------------------------------------- vertex (4)
        H_view  (0,3) = -       sm2*(1.0 + 2.0*s)*t2*(2.0*t - 3.0);
        H_s_view(0,3) = -  6.0*sm*s             *t2*(2.0*t - 3.0);
        H_t_view(0,3) = -  6.0*sm2*(1.0 + 2.0*s)*tm*t;
        H_view  (1,3) = -  3.0*sm2*s             *t2*(2.0*t - 3.0);
        H_s_view(1,3) =    3.0*(1.0 - 3.0*s)*sm  *t2*(2.0*t - 3.0);
        H_t_view(1,3) = - 18.0*sm2*s             *tm*t;
        H_view  (2,3) = -  3.0*sm2*(1.0 + 2.0*s)*tm*t2;
        H_s_view(2,3) = - 18.0*sm*s             *tm*t2;
        H_t_view(2,3) = -  3.0*sm2*(1.0 + 2.0*s)*t*(3.0*t - 2.0);
        H_view  (3,3) = -  9.0*sm2*s             *tm*t2;
        H_s_view(3,3) =    9.0*(1.0 - 3.0*s)*sm  *tm*t2;
        H_t_view(3,3) = -  9.0*sm2*s             *t*(3.0*t - 2.0);
        /*}else{
            //  QUINTIC MISSING
        }*/

    } // basisfunctions_2D_1_T
}

#endif /*BASISFUNCTIONS_H*/
