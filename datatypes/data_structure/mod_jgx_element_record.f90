!> Registers type_element's memory layout with jgx.
!>
!> The field ids below are the contract with element_set.h: the two lists must
!> stay in the same order.
!>
!> Offsets are measured from a local array of the record type, not from the real
!> mesh, because they are properties of the type.
module mod_jgx_element_record
  use, intrinsic :: iso_c_binding
  use data_structure,     only: type_element
  use mod_jgx_record_ids, only: JGX_REC_ELEMENT
  use mod_jgx_record
  implicit none
  private
  public :: jgx_register_element_record

  !> type_element fields -- must match jorek::element_field
  integer(c_int32_t), parameter :: JGX_EF_VERTEX       = 0
  integer(c_int32_t), parameter :: JGX_EF_NEIGHBOURS   = 1
  integer(c_int32_t), parameter :: JGX_EF_SIZE         = 2
  integer(c_int32_t), parameter :: JGX_EF_FATHER       = 3
  integer(c_int32_t), parameter :: JGX_EF_N_SONS       = 4
  integer(c_int32_t), parameter :: JGX_EF_N_GEN        = 5
  integer(c_int32_t), parameter :: JGX_EF_SONS         = 6
  integer(c_int32_t), parameter :: JGX_EF_CONTAIN_NODE = 7
  integer(c_int32_t), parameter :: JGX_EF_NREF         = 8
  integer(c_int32_t), parameter :: JGX_EF_COUNT        = 9

contains

  subroutine jgx_register_element_record()
    type(type_element), target :: e(2)
    integer(c_size_t) :: base, stride, ext(3)

    base   = transfer(c_loc(e(1)), 1_c_size_t)
    stride = transfer(c_loc(e(2)), 1_c_size_t) - base
    call jgx_c_record_begin(JGX_REC_ELEMENT, stride, JGX_EF_COUNT)

    ext(1) = size(e(1)%vertex, kind=c_size_t)
    call jgx_c_record_add_field(JGX_REC_ELEMENT, JGX_EF_VERTEX, &
         jgx_offset_of(c_loc(e(1)%vertex(1)), c_loc(e(1))), JGX_I32, 1, ext)

    ext(1) = size(e(1)%neighbours, kind=c_size_t)
    call jgx_c_record_add_field(JGX_REC_ELEMENT, JGX_EF_NEIGHBOURS, &
         jgx_offset_of(c_loc(e(1)%neighbours(1)), c_loc(e(1))), JGX_I32, 1, ext)

    ext(1) = size(e(1)%size, 1, kind=c_size_t)
    ext(2) = size(e(1)%size, 2, kind=c_size_t)
    call jgx_c_record_add_field(JGX_REC_ELEMENT, JGX_EF_SIZE, &
         jgx_offset_of(c_loc(e(1)%size(1,1)), c_loc(e(1))), JGX_F64, 2, ext)

    ext(1) = 1
    call jgx_c_record_add_field(JGX_REC_ELEMENT, JGX_EF_FATHER, &
         jgx_offset_of(c_loc(e(1)%father), c_loc(e(1))), JGX_I32, 1, ext)
    call jgx_c_record_add_field(JGX_REC_ELEMENT, JGX_EF_N_SONS, &
         jgx_offset_of(c_loc(e(1)%n_sons), c_loc(e(1))), JGX_I32, 1, ext)
    call jgx_c_record_add_field(JGX_REC_ELEMENT, JGX_EF_N_GEN, &
         jgx_offset_of(c_loc(e(1)%n_gen), c_loc(e(1))), JGX_I32, 1, ext)
    call jgx_c_record_add_field(JGX_REC_ELEMENT, JGX_EF_NREF, &
         jgx_offset_of(c_loc(e(1)%nref), c_loc(e(1))), JGX_I32, 1, ext)

    ext(1) = size(e(1)%sons, kind=c_size_t)
    call jgx_c_record_add_field(JGX_REC_ELEMENT, JGX_EF_SONS, &
         jgx_offset_of(c_loc(e(1)%sons(1)), c_loc(e(1))), JGX_I32, 1, ext)

    ext(1) = size(e(1)%contain_node, kind=c_size_t)
    call jgx_c_record_add_field(JGX_REC_ELEMENT, JGX_EF_CONTAIN_NODE, &
         jgx_offset_of(c_loc(e(1)%contain_node(1)), c_loc(e(1))), JGX_I32, 1, ext)

    call jgx_c_record_end(JGX_REC_ELEMENT)
  end subroutine jgx_register_element_record

end module mod_jgx_element_record
