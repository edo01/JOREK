/* jgx/jorek/jgx_record_ids.h -- the C++ view of the record ids taken from jgx_record_ids.def */
#ifndef JOREK_JGX_RECORD_IDS_H
#define JOREK_JGX_RECORD_IDS_H

#include "jgx/jgx_record_api.h"   /* JGX_MAX_RECORD_TYPES */

namespace jorek {

#define JGX_RECORD_ID(name, value) name = value,
enum record_id {
#include "jgx/jorek/jgx_record_ids.def"
  JGX_REC_COUNT
};
#undef JGX_RECORD_ID

/* Static check that the number of registered types doesn't exceed the size 
   of the registry table*/
static_assert(JGX_REC_COUNT <= JGX_MAX_RECORD_TYPES,
              "more record ids than JGX_MAX_RECORD_TYPES");

} /* namespace jorek */

#endif /* JOREK_JGX_RECORD_IDS_H */
