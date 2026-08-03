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

#include "jgx/jgx_abi.def"   /* JGX_MAX_RANK, the record limits */

/* The tag values are not restated here -- the .def files are the single source
 * the Fortran side reads too, so C and Fortran cannot drift apart. */

enum {   /* elem_kind tags */
#define JGX_ELEM_KIND(name, tag, bytes) name = tag,
#include "jgx/enums/elem_kind.def"
#undef JGX_ELEM_KIND
  JGX_ELEM_KIND_COUNT
};

/* It returns the size of a jgx kind*/
static inline size_t jgx_kind_size(int32_t elem_kind) {
  switch (elem_kind) {
#define JGX_ELEM_KIND(name, tag, bytes) case name: return bytes;
#include "jgx/enums/elem_kind.def"
#undef JGX_ELEM_KIND
    default: return 0;
  }
}

enum {   /* layout_tag values */
#define JGX_ENUM_ENTRY(name, value) name = value,
#include "jgx/enums/layout_tag.def"
#undef JGX_ENUM_ENTRY
  JGX_LAYOUT_COUNT
};

enum {   /* flags bits -- disjoint, so no COUNT */
#define JGX_ENUM_ENTRY(name, value) name = value,
#include "jgx/enums/flags.def"
#undef JGX_ENUM_ENTRY
  JGX_FLAG_NONE = 0
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
const char* jgx_c_backend_name(void);
const char* jgx_c_layout_tag(void);   /* compile-time layout policy string */
void        jgx_c_init(int device_id);
void        jgx_c_finalize(void);

void*       jgx_c_alloc(size_t n_bytes);
void        jgx_c_free(void* device_ptr);
void        jgx_c_push(void* device_ptr, const void* host_ptr, size_t n_bytes);  /* H2D */
void        jgx_c_pull(void* host_ptr, const void* device_ptr, size_t n_bytes);  /* D2H */
void        jgx_c_synchronize(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* JGX_C_API_H */
