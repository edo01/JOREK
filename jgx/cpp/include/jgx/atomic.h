/* jgx/atomic.h -- atomic definitions for host and device compilation.
 */
#ifndef JGX_ATOMIC_H
#define JGX_ATOMIC_H

#include "jgx/device.h"
#include "jgx/macros.h"

namespace jgx {

JGX_HD inline void atomic_add(double* dst, double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  /* Native double atomicAdd since sm_60 and on every GFX9 and later, so no
   * compare-and-swap fallback is compiled here. */
  atomicAdd(dst, value);
#else
  *dst += value;
#endif
}

} /* namespace jgx */

#endif /* JGX_ATOMIC_H */
