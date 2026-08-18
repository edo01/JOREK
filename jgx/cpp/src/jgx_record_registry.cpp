/* jgx_record_registry.cpp
 *
 * One table per record type, filled once from Fortran at startup and read-only
 * afterwards. The number of record types is bounded by JGX_MAX_RECORD_TYPES in 
 * jgx_record_api.h. */
#include "jgx/jgx_record_api.h"

#include <cstdio>
#include <cstdlib>

namespace { // functions for internal usage

jgx_record_desc g_record[JGX_MAX_RECORD_TYPES] = {};

void fail(const char* what, int32_t record_id) {
  std::fprintf(stderr, "jgx_record: %s (record_id=%d)\n", what, int(record_id));
  std::abort();
}

bool valid_id(int32_t record_id) {
  return record_id >= 0 && record_id < JGX_MAX_RECORD_TYPES;
}

}

extern "C" {

void jgx_c_record_begin(int32_t record_id, size_t record_stride_bytes,
                        int32_t n_fields) {
  if (!valid_id(record_id))                    fail("record_id out of range", record_id);
  if (n_fields < 0 || n_fields > JGX_MAX_RECORD_FIELDS)
                                               fail("too many fields", record_id);
  if (record_stride_bytes == 0)                fail("zero record stride", record_id);

  /* Ids are handed out by hand in jgx_record_ids.h */
  if (g_record[record_id].complete)            fail("record_id already registered", record_id);

  jgx_record_desc& r = g_record[record_id];
  r = jgx_record_desc{};
  r.record_stride_bytes = record_stride_bytes;
  r.n_fields = n_fields;
  r.complete = 0;
}

void jgx_c_record_add_field(int32_t record_id, int32_t field_id,
                            size_t offset_bytes, int32_t elem_kind,
                            int32_t intra_rank, const size_t* intra_extents) {
  if (!valid_id(record_id)) fail("record_id out of range", record_id);
  jgx_record_desc& r = g_record[record_id];

  if (r.record_stride_bytes == 0)          fail("add_field before begin", record_id);
  if (field_id < 0 || field_id >= r.n_fields) fail("field_id out of range", record_id);
  if (intra_rank < 0 || intra_rank > JGX_MAX_INTRA_RANK)
                                           fail("intra_rank out of range", record_id);
  if (offset_bytes >= r.record_stride_bytes) fail("field offset outside record", record_id);

  /* Views address in units of T, so both the offset and the record stride must
   * divide evenly. This holds for a naturally aligned component; when it does
   * not, the stride would be truncated and read plausible garbage. */
  const size_t es = jgx_kind_size(elem_kind);
  if (es == 0)                             fail("unknown elem_kind", record_id);
  if (offset_bytes % es != 0)              fail("field offset is not a whole number of elements", record_id);
  if (r.record_stride_bytes % es != 0)     fail("record stride is not a whole number of elements", record_id);

  jgx_field_desc& f = r.field[field_id];
  f.offset_bytes = offset_bytes;
  f.elem_kind    = elem_kind;
  f.intra_rank   = intra_rank;
  for (int d = 0; d < JGX_MAX_INTRA_RANK; ++d)
    f.intra_extents[d] = (d < intra_rank) ? intra_extents[d] : 1;
}

void jgx_c_record_end(int32_t record_id) {
  if (!valid_id(record_id)) fail("record_id out of range", record_id);
  jgx_record_desc& r = g_record[record_id];
  if (r.record_stride_bytes == 0) fail("end before begin", record_id);

  /* Every declared field must have been filled. */
  for (int i = 0; i < r.n_fields; ++i)
    if (r.field[i].intra_extents[0] == 0) fail("field left unregistered", record_id);

  r.complete = 1;
}

} /* extern "C" */

namespace jgx {

const jgx_record_desc& registered_record(int record_id, int expect_n_fields) {
  if (!valid_id(int32_t(record_id)) || !g_record[record_id].complete)
    fail("record type used before registration", int32_t(record_id));

  const jgx_record_desc& r = g_record[record_id];
  if (r.n_fields != expect_n_fields)
    fail("field count differs from the caller's field enum", int32_t(record_id));
  return r;
}

} /* namespace jgx */
