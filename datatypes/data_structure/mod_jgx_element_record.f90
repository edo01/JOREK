!> Registers type_element's memory layout with jgx.
!>
!> The field list is jgx/jorek/records/element_record.def, expanded here and by
!> element_set.h -- one list, so nothing has to be kept in step by hand. This
!> module contributes the offsets, which only Fortran can measure.
module mod_jgx_element_record
  use, intrinsic :: iso_c_binding
  use data_structure,     only: type_element
  use mod_jgx_record_ids, only: JGX_REC_ELEMENT
  use mod_jgx_record
  implicit none
  private
  public :: jgx_register_element_record

contains

  subroutine jgx_register_element_record()
    type(type_element), target :: e(2)
    integer(c_int32_t) :: f
    integer(c_size_t)  :: stride

    !> Counted from the same list that registers below, so begin() cannot
    !> disagree with what follows.
    f = 0
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES) f = f + 1
#include "jgx/jorek/records/element_record.def"
#undef JGX_FIELD

    stride = transfer(c_loc(e(2)), 1_c_size_t) - transfer(c_loc(e(1)), 1_c_size_t)
    call jgx_c_record_begin(JGX_REC_ELEMENT, stride, f)

    f = 0
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES) \
    call jgx_add_field(JGX_REC_ELEMENT, f, c_loc(e(1)%comp), c_loc(e(1)), KIND, shape(e(1)%comp)); f = f + 1
#include "jgx/jorek/records/element_record.def"
#undef JGX_FIELD

    call jgx_c_record_end(JGX_REC_ELEMENT)
  end subroutine jgx_register_element_record

end module mod_jgx_element_record
