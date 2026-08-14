/*
 * jgx/macros.h -- device annotations.
 */
#ifndef JGX_MACROS_H
#define JGX_MACROS_H

/* JGX_HD carries ONLY the host/device annotation, never `inline` -- the code
 * writes `JGX_HD inline` explicitly. */
#if defined(JGX_DEVICE_CUDA) || defined(JGX_DEVICE_HIP)
#  define JGX_HD __host__ __device__
// to add kokkos just:
/*#elif defined(JGX_DEVICE_KOKKOS)
 *#  include <Kokkos_Macros.hpp>
 *#  define JGX_HD KOKKOS_FUNCTION*/
#else
#  define JGX_HD
#endif

/* A variable living in device global memory. Declared in the header the readers
 * include, defined in one device translation unit. Written through jgx_c_push_symbol.
 *
 * Both expand to nothing outside a device build, where such a variable has no
 * definition -- the declarations are guarded by the backend switch at the point
 * of use. */
#if defined(JGX_DEVICE_CUDA) || defined(JGX_DEVICE_HIP)
#  define JGX_DEVICE_VAR __device__
#  define JGX_SYMBOL(x)  (&(x))
#  define JGX_KERNEL     __global__
#endif
// ad here other backends symbols

#endif /* JGX_MACROS_H */
