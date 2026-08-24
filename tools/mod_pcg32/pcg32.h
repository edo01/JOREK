/* tools/mod_pcg32/pcg32.h -- the pcg32 generator behind mod_pcg32.f90, next
 * door.
 *
 * PCG Random Number Generation for C, Copyright 2014 Melissa O'Neill
 * <oneill@pcg-random.org>, Apache License 2.0.  See http://www.pcg-random.org.
 * Transcribed from the reference C implementation (formerly tools/pcg_basic.c,
 * which this header replaced) so that one body serves the Fortran, the host
 * kernels and the device: the C entry points mod_pcg32.f90 binds to are one
 * file out, in pcg32_shim.cpp.
 *
 * The state is 16 bytes and trivially copyable, so a kernel can hold one per
 * particle in a device buffer and permute it with the particle it belongs to.
 */
#ifndef JOREK_PCG32_H
#define JOREK_PCG32_H

#include <cmath>
#include <cstdint>

#include "jgx/macros.h"

namespace pcg32 {

/* pcg_state_setseq_64 -- mirrors the bind(C) type of mod_pcg32.f90, which is
 * what lets a Fortran-held state and a kernel-held one be the same 16 bytes. */
struct state {
  std::uint64_t s   = 0;  /*< the LCG state; all values are legal */
  std::uint64_t inc = 0;  /*< the stream selector; must always be odd */
};

constexpr std::uint64_t kMultiplier = 6364136223846793005ULL;

/* pcg32_random_r -- one uniformly distributed 32-bit draw. */
JGX_HD inline std::uint32_t next_u32(state& rng) {
  const std::uint64_t old = rng.s;
  rng.s = old * kMultiplier + rng.inc;
  const std::uint32_t xorshifted =
      static_cast<std::uint32_t>(((old >> 18u) ^ old) >> 27u);
  const std::uint32_t rot = static_cast<std::uint32_t>(old >> 59u);
  /* The reference writes the rotate as (xorshifted << ((-rot) & 31)); the form
   * below is the same for every rot in [0,31] and does not negate an unsigned. */
  return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

/* pcg32_srandom_r -- seed a stream from a state initialiser and a stream id. */
JGX_HD inline void srandom(state& rng, const std::uint64_t initstate,
                           const std::uint64_t initseq) {
  rng.s   = 0U;
  rng.inc = (initseq << 1u) | 1u;
  next_u32(rng);
  rng.s += initstate;
  next_u32(rng);
}

/* pcg32_random_double_r -- a uniform double in [0,1), 2^-32 precision. */
JGX_HD inline double next_double(state& rng) {
  return ldexp(static_cast<double>(next_u32(rng)), -32);
}

/* pcg_advance_lcg_64 -- jump the LCG delta steps in O(log delta); Brown,
 * "Random Number Generation with Arbitrary Stride", Trans. Am. Nucl. Soc.
 * (1994).  delta is unsigned, so a negative jump is spelled as the long way
 * round, which is what mod_pcg32's pcg32_jumpahead does. */
JGX_HD inline std::uint64_t advance_lcg_64(std::uint64_t s, std::uint64_t delta,
                                           std::uint64_t cur_mult,
                                           std::uint64_t cur_plus) {
  std::uint64_t acc_mult = 1u;
  std::uint64_t acc_plus = 0u;
  while (delta > 0) {
    if (delta & 1) {
      acc_mult *= cur_mult;
      acc_plus = acc_plus * cur_mult + cur_plus;
    }
    cur_plus = (cur_mult + 1) * cur_plus;
    cur_mult *= cur_mult;
    delta /= 2;
  }
  return acc_mult * s + acc_plus;
}

/* pcg32_advance_r */
JGX_HD inline void advance(state& rng, const std::uint64_t delta) {
  rng.s = advance_lcg_64(rng.s, delta, kMultiplier, rng.inc);
}

} /* namespace pcg32 */

#endif /* JOREK_PCG32_H */
