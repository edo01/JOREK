/* The C entry points of pcg32.h.
 *
 * These keep the names of the reference implementation, not the jgx_host_
 * prefix: mod_pcg32.f90 has bound to them since before the port, and the
 * interface must not change with the body behind it.  jgx_host_pcg32_next_double
 * is the exception -- it exists only so a test can reach the generator the way
 * a kernel does, through the state rather than through the Fortran type.
 */
#include "tools/mod_pcg32/pcg32.h"

#include <cstdint>

extern "C" {

void pcg32_srandom_r(pcg32::state* rng, std::uint64_t initstate,
                     std::uint64_t initseq) {
  pcg32::srandom(*rng, initstate, initseq);
}

std::uint32_t pcg32_random_r(pcg32::state* rng) { return pcg32::next_u32(*rng); }

double pcg32_random_double_r(pcg32::state* rng) { return pcg32::next_double(*rng); }

std::uint64_t pcg_advance_lcg_64(std::uint64_t state, std::uint64_t delta,
                                 std::uint64_t cur_mult, std::uint64_t cur_plus) {
  return pcg32::advance_lcg_64(state, delta, cur_mult, cur_plus);
}

void pcg32_advance_r(pcg32::state* rng, std::uint64_t delta) {
  pcg32::advance(*rng, delta);
}

/* One draw from a stream seeded here, so a test can check the kernel-side
 * generator against the one the Fortran gets. */
double jgx_host_pcg32_next_double(std::uint64_t initstate, std::uint64_t initseq,
                                  std::int32_t n_skip) {
  pcg32::state rng;
  pcg32::srandom(rng, initstate, initseq);
  for (std::int32_t i = 0; i < n_skip; ++i) pcg32::next_u32(rng);
  return pcg32::next_double(rng);
}

} /* extern "C" */
