!> Registers type_node's memory layout with jgx.
!>
!> The field list is jgx/jorek/records/node_record.def, expanded here and by
!> node_set.h -- one list, so nothing has to be kept in step by hand. This
!> module contributes the offsets, which only Fortran can measure.
!>
!> Offsets are taken from a local array of the record type, not from the real
!> mesh, because they are properties of the type.
module mod_jgx_node_record
  use, intrinsic :: iso_c_binding
  use data_structure,     only: type_node
  use mod_jgx_record_ids, only: JGX_REC_NODE
  use mod_jgx_record
  implicit none
  private
  public :: jgx_register_node_record

contains

  subroutine jgx_register_node_record()
    type(type_node), target :: n(2)
    integer(c_int32_t) :: f
    integer(c_size_t)  :: stride

    !> Counted from the same list that registers below, so begin() cannot
    !> disagree with what follows.
    f = 0
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES) f = f + 1
#include "jgx/jorek/records/node_record.def"
#undef JGX_FIELD

    stride = transfer(c_loc(n(2)), 1_c_size_t) - transfer(c_loc(n(1)), 1_c_size_t)
    call jgx_c_record_begin(JGX_REC_NODE, stride, f)

    f = 0
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES) \
    call jgx_add_field(JGX_REC_NODE, f, c_loc(n(1)%comp), c_loc(n(1)), KIND, shape(n(1)%comp)); f = f + 1
#include "jgx/jorek/records/node_record.def"
#undef JGX_FIELD

    call jgx_c_record_end(JGX_REC_NODE)
  end subroutine jgx_register_node_record

end module mod_jgx_node_record
