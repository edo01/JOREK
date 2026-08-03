/* jgx_record_api.h -- registration of a Fortran derived type's memory layout.
 *                     (C/C++ side of jgx_record.f90)
 *
 * A Fortran array of a derived type is a strided of components of
 * different kinds at compiler-chosen offsets. Fortran measures them once
 * with c_loc and hands them over to C/C++.
 * 
 * NOTE: Fortran data structures are always assumed AoS and all the fields
 * must be known at compile time (no allocatables).
 *
 * What is registered is the TYPE, not an instance: offsets, the record stride
 * and the intra-record extents are the same for every array of that type. Only
 * the base pointer and the record count vary per call, and those travel as
 * ordinary arguments. That is why registration happens once at startup and no
 * kernel needs a handle.
 * 
 * The coherence check between the Fortran and the C/C++ data structures is
 * handled as follows:
 * - the Fortran derived type is registered specifying n_fields, the number of
 *   fields it declares, and
 * - "*_set_from_registry", which extracts an instance of a registered type,
 *   passes the length of its own field enum to registered_record, which aborts
 *   if the two differ.
 * Only the counts are compared; keeping the two field lists in the same order
 * is up to the caller.
 *
 * FOR JOREK:
 * Record and field ids are plain integers owned by the caller; this header
 * knows nothing about JOREK's types. Ids are handed out in
 * jgx/jorek/jgx_record_ids.def -- jgx/jorek is the application's side of the
 * seam and is the only part of jgx/ allowed to name a JOREK type.
 */
#ifndef JGX_RECORD_API_H
#define JGX_RECORD_API_H

#include <stddef.h>
#include <stdint.h>

#include "jgx/jgx_c_api.h"   /* JGX_F64 ... elem_kind tags */

/* JGX_MAX_INTRA_RANK, JGX_MAX_RECORD_FIELDS, JGX_MAX_RECORD_TYPES arrive with
 * jgx_c_api.h, which includes jgx_abi.def. */

#ifdef __cplusplus
extern "C" {
#endif

/* One component of the recorded derived type. 
 * Please note that at the moment no AoSoA can be recorded. */
typedef struct {
  size_t  offset_bytes; // offset of the field in the derived type
  size_t  intra_extents[JGX_MAX_INTRA_RANK]; // dimensions
  int32_t elem_kind; // kind of the field
  int32_t intra_rank; // rank of the field
  int32_t reserved[2]; // extra flags
} jgx_field_desc;

/* The recorded derived type as a whole. */
typedef struct {
  size_t         record_stride_bytes; // distance between consecutive records
  int32_t        n_fields; // number of fields in the record
  int32_t        complete; // whether the registration is finished
  jgx_field_desc field[JGX_MAX_RECORD_FIELDS]; /* the fields descriptor, 
                                                  indexed by field id */
} jgx_record_desc;

/* Fortran calls these once per record type, in field-id order. */
void    jgx_c_record_begin(int32_t record_id, size_t record_stride_bytes,
                           int32_t n_fields);
void    jgx_c_record_add_field(int32_t record_id, int32_t field_id,
                               size_t offset_bytes, int32_t elem_kind,
                               int32_t intra_rank, const size_t* intra_extents);
void    jgx_c_record_end(int32_t record_id);
int32_t jgx_c_record_is_registered(int32_t record_id);

#ifdef __cplusplus
}  /* extern "C" */

namespace jgx {
/* Read a completed registration. Aborts if record_id was never registered, or
 * if expect_n_fields -- the length of the caller's own field list -- differs
 * from the count given at begin(). */
const jgx_record_desc& registered_record(int record_id, int expect_n_fields);
}  /* namespace jgx */

#endif /* __cplusplus */

#endif /* JGX_RECORD_API_H */
