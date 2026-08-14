/* jgx/device.h -- the device language header for whichever backend is compiled.
 *
 * Empty outside a device build, because JGX_DEVICE_* is declared PRIVATE in cmake
 * so a host TU may include it unconditionally.
 */
#ifndef JGX_DEVICE_H
#define JGX_DEVICE_H

#include "jgx/macros.h"

#if defined(JGX_DEVICE_HIP)
#  include <hip/hip_runtime.h>
#elif defined(JGX_DEVICE_CUDA)
#  include <cuda_runtime.h>
#endif
// you can add other libraries here

#endif /* JGX_DEVICE_H */
