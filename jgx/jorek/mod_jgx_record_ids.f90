!> The Fortran view of the record ids taken from jgx_record_ids.def
module mod_jgx_record_ids
  use, intrinsic :: iso_c_binding, only: c_int32_t
  implicit none
  public

#define JGX_RECORD_ID(name, value) integer(c_int32_t), parameter :: name = value
#include "jgx/jorek/jgx_record_ids.def"
#undef JGX_RECORD_ID

end module mod_jgx_record_ids
