/* 
 * jgx/macros.h -- JGX_HD annotation
 */
#ifndef JGX_MACROS_H
#define JGX_MACROS_H

/* JGX_HD carries ONLY the host/device annotation, never `inline` -- the code
 * writes `JGX_HD inline` explicitly (needed for ODR on the serial/CUDA/HIP
 * header-only functions). So the Kokkos form is KOKKOS_FUNCTION (annotation
 * only), not KOKKOS_INLINE_FUNCTION (which would double the inline). */
#if defined(JGX_DEVICE_CUDA) || defined(JGX_DEVICE_HIP)
#  define JGX_HD __host__ __device__
#elif defined(JGX_DEVICE_KOKKOS)
#  include <Kokkos_Macros.hpp>
#  define JGX_HD KOKKOS_FUNCTION
#else
#  define JGX_HD
#endif

#endif /* JGX_MACROS_H */
