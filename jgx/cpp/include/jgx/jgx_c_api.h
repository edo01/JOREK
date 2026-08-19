/* jgx_c_api.h -- device jgx api for device, memory and runtime management.
 *
 * Declarations only. The definitions live in exactly one backend directory
 * (jgx/cpp/backends/<backend>/), which is the only place a vendor runtime is
 * named.
 */
#ifndef JGX_C_API_H
#define JGX_C_API_H

#include <stddef.h>
#include <stdint.h>

#include "jgx/macros.h"    

#ifdef __cplusplus
extern "C" {
#endif

enum {   /* elem_kind tags */
#define JGX_ELEM_KIND(name, tag, bytes) name = tag,
#include "jgx/enums/elem_kind.def"
#undef JGX_ELEM_KIND
  JGX_ELEM_KIND_COUNT
};

/* It returns the size of a jgx kind */
static JGX_HD inline size_t jgx_kind_size(int32_t elem_kind) {
  switch (elem_kind) {
#define JGX_ELEM_KIND(name, tag, bytes) case name: return bytes;
#include "jgx/enums/elem_kind.def"
#undef JGX_ELEM_KIND
    default: return 0;
  }
}

/* ---- runtime / memory management (raw pointers + sizes) ---------------- */
const char* jgx_c_backend_name(void);
const char* jgx_c_layout_tag(void);   /* compile-time layout policy string */
void        jgx_c_init_device(int device_id);
void        jgx_c_finalize_device(void);

void*       jgx_c_alloc(size_t n_bytes);
void        jgx_c_free(void* device_ptr);
void        jgx_c_push(void* device_ptr, const void* host_ptr, size_t n_bytes);  /* H2D */
void        jgx_c_pull(void* host_ptr, const void* device_ptr, size_t n_bytes);  /* D2H */
void        jgx_c_memset(void* device_ptr, int value, size_t n_bytes);
void        jgx_c_synchronize(void);

/* ---- device globals ---------------------------------------------------- */
/*
 * The push and pull above moves data the kernel is *handed*: a buffer whose device
 * pointer travels in a jgx_buf_desc and arrives as an argument. Ambient state
 * cannot work that way and must be handled differently.
 *
 * So it lives in a JGX_DEVICE_VAR (jgx/macros.h) instead, and these two move
 * bytes in and out of one. `symbol` is JGX_SYMBOL(the variable), taken in the
 * translation unit that defines it.
 */
void        jgx_c_push_symbol(const void* symbol, const void* host_ptr, size_t n_bytes);
void        jgx_c_pull_symbol(void* host_ptr, const void* symbol, size_t n_bytes);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* JGX_C_API_H */
