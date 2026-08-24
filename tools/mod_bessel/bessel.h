/* tools/mod_bessel/bessel.h -- the scaled modified Bessel functions of
 * mod_bessel.f90, next door.
 *
 * Rational minimax approximations by Russon and Blair, as implemented in
 * SPECFUN (Cody, Stoltz) and transcribed to Fortran90 by Burkardt.
 *
 * DUPLICATED FROM mod_bessel.f90 -- keep the two in step.  Only the scaled
 * branch (jint = 2, f_scaled = exp(x)*f_exact) is here, because that is the
 * only one the module's three public functions ask for; the unscaled branch and
 * its overflow guard stay in the Fortran, which is also the reference
 * test_mod_ccoll_relativistic_jorek compares this against.
 */
#ifndef JOREK_BESSEL_H
#define JOREK_BESSEL_H

#include <cmath>

#include "jgx/macros.h"

namespace bessel {

/* Machine-dependent constants of calck0 / calck1.  xmax guards the unscaled
 * branch only, so it does not appear here. */
constexpr double kXSmall = 1.11e-16;
constexpr double kXLeast = 2.23e-308;
constexpr double kXInf   = 1.79e+308;

/* mod_bessel::bessel_k0exp -- exp(x)*K0(x). */
JGX_HD inline double bessel_k0exp(const double x) {
  constexpr double p[6] = { 5.8599221412826100000e-04, 1.3166052564989571850e-01,
                            1.1999463724910714109e+01, 4.6850901201934832188e+02,
                            5.9169059852270512312e+03, 2.4708152720399552679e+03 };
  constexpr double q[2] = {-2.4994418972832303646e+02, 2.1312714303849120380e+04 };
  constexpr double f[4] = {-1.6414452837299064100e+00,-2.9601657892958843866e+02,
                           -1.7733784684952985886e+04,-4.0320340761145482298e+05 };
  constexpr double g[3] = {-2.5064972445877992730e+02, 2.9865713163054025489e+04,
                           -1.6128136304458193998e+06 };
  constexpr double pp[10] = { 1.1394980557384778174e+02, 3.6832589957340267940e+03,
                              3.1075408980684392399e+04, 1.0577068948034021957e+05,
                              1.7398867902565686251e+05, 1.5097646353289914539e+05,
                              7.1557062783764037541e+04, 1.8321525870183537725e+04,
                              2.3444738764199315021e+03, 1.1600249425076035558e+02 };
  constexpr double qq[10] = { 2.0013443064949242491e+02, 4.4329628889746408858e+03,
                              3.1474655750295278825e+04, 9.7418829762268075784e+04,
                              1.5144644673520157801e+05, 1.2689839587977598727e+05,
                              5.8824616785857027752e+04, 1.4847228371802360957e+04,
                              1.8821890840982713696e+03, 9.2556599177304839811e+01 };

  if (x <= 0.0) return kXInf;

  if (x <= 1.0) {
    const double temp = log(x);
    if (x < kXSmall) return p[5]/q[1] - temp;

    const double xx = x*x;
    const double sump = ((((p[0]*xx + p[1])*xx + p[2])*xx + p[3])*xx + p[4])*xx + p[5];
    const double sumq = (xx + q[0])*xx + q[1];
    const double sumf = ((f[0]*xx + f[1])*xx + f[2])*xx + f[3];
    const double sumg = ((xx + g[0])*xx + g[1])*xx + g[2];
    return (sump/sumq - xx*sumf*temp/sumg - temp) * exp(x);
  }

  const double xx = 1.0/x;
  double sump = pp[0];
  for (int i = 1; i < 10; ++i) sump = sump*xx + pp[i];
  double sumq = xx;
  for (int i = 0; i < 9; ++i) sumq = (sumq + qq[i])*xx;
  sumq += qq[9];
  return sump/sumq/sqrt(x);
} // bessel_k0exp

/* mod_bessel::bessel_k1exp -- exp(x)*K1(x). */
JGX_HD inline double bessel_k1exp(const double x) {
  constexpr double p[5] = { 4.8127070456878442310e-1, 9.9991373567429309922e+1,
                            7.1885382604084798576e+3, 1.7733324035147015630e+5,
                            7.1938920065420586101e+5 };
  constexpr double q[3] = {-2.8143915754538725829e+2, 3.7264298672067697862e+4,
                           -2.2149374878243304548e+6 };
  constexpr double f[5] = {-2.2795590826955002390e-1,-5.3103913335180275253e+1,
                           -4.5051623763436087023e+3,-1.4758069205414222471e+5,
                           -1.3531161492785421328e+6 };
  constexpr double g[3] = {-3.0507151578787595807e+2, 4.3117653211351080007e+4,
                           -2.7062322985570842656e+6 };
  constexpr double pp[11] = { 6.4257745859173138767e-2, 7.5584584631176030810e+0,
                              1.3182609918569941308e+2, 8.1094256146537402173e+2,
                              2.3123742209168871550e+3, 3.4540675585544584407e+3,
                              2.8590657697910288226e+3, 1.3319486433183221990e+3,
                              3.4122953486801312910e+2, 4.4137176114230414036e+1,
                              2.2196792496874548962e+0 };
  constexpr double qq[9] = { 3.6001069306861518855e+1, 3.3031020088765390854e+2,
                             1.2082692316002348638e+3, 2.1181000487171943810e+3,
                             1.9448440788918006154e+3, 9.6929165726802648634e+2,
                             2.5951223655579051357e+2, 3.4552228452758912848e+1,
                             1.7710478032601086579e+0 };

  if (x < kXLeast) return kXInf;

  if (x <= 1.0) {
    if (x < kXSmall) return 1.0/x;

    const double xx = x*x;
    /* q[2], not a p coefficient: the constant term of sump is shared with sumq
     * in the Fortran table, and the transcription keeps that. */
    const double sump = ((((p[0]*xx + p[1])*xx + p[2])*xx + p[3])*xx + p[4])*xx + q[2];
    const double sumq = ((xx + q[0])*xx + q[1])*xx + q[2];
    const double sumf = (((f[0]*xx + f[1])*xx + f[2])*xx + f[3])*xx + f[4];
    const double sumg = ((xx + g[0])*xx + g[1])*xx + g[2];
    return ((xx*log(x)*sumf/sumg + sump/sumq)/x) * exp(x);
  }

  const double xx = 1.0/x;
  double sump = pp[0];
  for (int i = 1; i < 11; ++i) sump = sump*xx + pp[i];
  double sumq = xx;
  for (int i = 0; i < 8; ++i) sumq = (sumq + qq[i])*xx;
  sumq += qq[8];
  return sump/sumq/sqrt(x);
} // bessel_k1exp

/* mod_bessel::bessel_k2exp -- exp(x)*K2(x), from the recurrence.  At x = 0 the
 * recurrence divides by zero, so the Fortran returns K0's overflow value there
 * and this does the same. */
JGX_HD inline double bessel_k2exp(const double x) {
  if (x == 0.0) return bessel_k0exp(x);
  return bessel_k0exp(x) + 2.0*bessel_k1exp(x)/x;
} // bessel_k2exp

} /* namespace bessel */

#endif /* JOREK_BESSEL_H */
