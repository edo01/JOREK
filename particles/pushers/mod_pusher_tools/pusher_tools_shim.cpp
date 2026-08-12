/* particles/pushers/mod_pusher_tools/pusher_tools_shim.cpp -- C entry points
 * onto pusher_tools.h, for the Fortran interfaces in mod_pusher_tools.f90.
 */
#include "particles/pushers/mod_pusher_tools/pusher_tools.h"

extern "C" {
    /* mod_pusher_tools::cayley_transform.
     *
     * M is the Fortran M(3,3): column-major, so M(i+1,j+1) sits at M[i + 3*j],
     * the transpose of the kernel's row-major M[i][j]. */
    void jgx_host_pusher_tools_cayley_transform(const double alpha,
                                                const double* vec, double* M) {
        double m[3][3];
        pusher_tools::cayley_transform(alpha, vec, m);

        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                M[i + 3*j] = m[i][j];
    }
}
