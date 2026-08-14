/* jgx/cpp/backends/hip/hip_check.h -- HIP backend's specific macros.
 */
#ifndef JGX_BACKENDS_HIP_CHECK_H
#define JGX_BACKENDS_HIP_CHECK_H

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#define JGX_HIP_CHECK(call)                                                    \
  do {                                                                         \
    const hipError_t jgx_err_ = (call);                                        \
    if (jgx_err_ != hipSuccess) {                                              \
      std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #call,     \
                   hipGetErrorString(jgx_err_));                               \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

#endif /* JGX_BACKENDS_HIP_CHECK_H */
