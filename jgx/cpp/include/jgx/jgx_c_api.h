/* jgx_c_api.h -- the C interface to Fortran.
 *
 * Everything above this line line with any Fortran
 * compiler; everything below with any C++17 compiler or a backend compiler.
 * 
 * 
 */
#ifndef JGX_C_API_H
#define JGX_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define JGX_ABI_VERSION 1
#define JGX_MAX_RANK    5

/* elem_kind tags (mirror jgx_kinds.f90). */
enum {
  JGX_F64 = 0,
  JGX_F32 = 1,
  JGX_I32 = 2,
  JGX_I64 = 3
};

/* layout_tag values (mirror jgx_kinds.f90). */
enum {
  JGX_LAYOUT_COLMAJOR = 0,   /* Fortran / layout_left */
  JGX_LAYOUT_ROWMAJOR = 1    /* C / layout_right      */
};

/* flags bits (mirror jgx_kinds.f90). */
enum {
  JGX_FLAG_DIRTY_HOST   = 1,
  JGX_FLAG_DIRTY_DEVICE = 2,
  JGX_FLAG_ON_DEVICE    = 4
};

/* This POD layout must match the
 * bind(C) type jgx_buf_desc in jgx_types.f90 (this is checked by the ABI test). */
typedef struct {
  void*    device_ptr;
  void*    host_ptr;
  size_t   n_bytes;
  int32_t  elem_kind;
  int32_t  rank;
  size_t   extents[JGX_MAX_RANK];   /* Fortran (column-major) order */
  int32_t  layout_tag;
  int32_t  flags;
} jgx_buf_desc;

/* ---- runtime / memory management (raw pointers + sizes) ---------------- */
int         jgx_c_abi_version(void);
const char* jgx_c_backend_name(void);
const char* jgx_c_layout_tag(void);   /* compile-time layout policy string */
void        jgx_c_init(int device_id);
void        jgx_c_finalize(void);

void*       jgx_c_alloc(size_t n_bytes);
void        jgx_c_free(void* device_ptr);
void        jgx_c_push(void* device_ptr, const void* host_ptr, size_t n_bytes);  /* H2D */
void        jgx_c_pull(void* host_ptr, const void* device_ptr, size_t n_bytes);  /* D2H */
void        jgx_c_synchronize(void);

/* Introspection for the ABI struct-size cross-check. */
size_t      jgx_c_sizeof_buf_desc(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* JGX_C_API_H */
