/* The C entry points of bessel.h.  Scalars cross by value; the Fortran bodies
 * these mirror are still live in mod_bessel.f90, and
 * test_mod_ccoll_relativistic_jorek compares the two. */
#include "tools/mod_bessel/bessel.h"

extern "C" {

double jgx_host_bessel_k0exp(const double x) { return bessel::bessel_k0exp(x); }
double jgx_host_bessel_k1exp(const double x) { return bessel::bessel_k1exp(x); }
double jgx_host_bessel_k2exp(const double x) { return bessel::bessel_k2exp(x); }

} /* extern "C" */
