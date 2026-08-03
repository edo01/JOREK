!> Registers type_node's memory layout with jgx.
!>
!> The field ids below are the contract with node_set.h: the two lists must stay
!> in the same order. Only the unconditional components are covered -- the
!> model-guarded ones get their own record type, since one offset table cannot
!> describe a field list that moves with a preprocessor switch.
!>
!> Offsets are measured from a local array of the record type, not from the real
!> mesh, because they are properties of the type. That makes this callable at
!> any point in startup, with no ordering constraint against grid construction.
module mod_jgx_node_record
  use, intrinsic :: iso_c_binding
  use data_structure,     only: type_node
  use mod_jgx_record_ids, only: JGX_REC_NODE
  use jgx_record
  implicit none
  private
  public :: jgx_register_node_record

  !> type_node fields -- must match jorek::node_field
  integer(c_int32_t), parameter :: JGX_NF_X              = 0
  integer(c_int32_t), parameter :: JGX_NF_VALUES         = 1
  integer(c_int32_t), parameter :: JGX_NF_DELTAS         = 2
  integer(c_int32_t), parameter :: JGX_NF_INDEX          = 3
  integer(c_int32_t), parameter :: JGX_NF_PARENTS        = 4
  integer(c_int32_t), parameter :: JGX_NF_BOUNDARY       = 5
  integer(c_int32_t), parameter :: JGX_NF_BOUNDARY_INDEX = 6
  integer(c_int32_t), parameter :: JGX_NF_AXIS_DOF       = 7
  integer(c_int32_t), parameter :: JGX_NF_PARENT_ELEM    = 8
  integer(c_int32_t), parameter :: JGX_NF_REF_LAMBDA     = 9
  integer(c_int32_t), parameter :: JGX_NF_REF_MU         = 10
  integer(c_int32_t), parameter :: JGX_NF_COUNT          = 11

contains

  subroutine jgx_register_node_record()
    type(type_node), target :: n(2)
    integer(c_size_t) :: base, stride, ext(3)

    base   = transfer(c_loc(n(1)), 1_c_size_t)
    stride = transfer(c_loc(n(2)), 1_c_size_t) - base
    call jgx_c_record_begin(JGX_REC_NODE, stride, JGX_NF_COUNT)

    ext(1) = size(n(1)%x, 1, kind=c_size_t)
    ext(2) = size(n(1)%x, 2, kind=c_size_t)
    ext(3) = size(n(1)%x, 3, kind=c_size_t)
    call jgx_c_record_add_field(JGX_REC_NODE, JGX_NF_X, &
         jgx_offset_of(c_loc(n(1)%x(1,1,1)), c_loc(n(1))), JGX_F64, 3, ext)

    ext(1) = size(n(1)%values, 1, kind=c_size_t)
    ext(2) = size(n(1)%values, 2, kind=c_size_t)
    ext(3) = size(n(1)%values, 3, kind=c_size_t)
    call jgx_c_record_add_field(JGX_REC_NODE, JGX_NF_VALUES, &
         jgx_offset_of(c_loc(n(1)%values(1,1,1)), c_loc(n(1))), JGX_F64, 3, ext)
    call jgx_c_record_add_field(JGX_REC_NODE, JGX_NF_DELTAS, &
         jgx_offset_of(c_loc(n(1)%deltas(1,1,1)), c_loc(n(1))), JGX_F64, 3, ext)

    ext(1) = size(n(1)%index, kind=c_size_t)
    ext(2) = 1; ext(3) = 1
    call jgx_c_record_add_field(JGX_REC_NODE, JGX_NF_INDEX, &
         jgx_offset_of(c_loc(n(1)%index(1)), c_loc(n(1))), JGX_I32, 1, ext)

    ext(1) = size(n(1)%parents, kind=c_size_t)
    call jgx_c_record_add_field(JGX_REC_NODE, JGX_NF_PARENTS, &
         jgx_offset_of(c_loc(n(1)%parents(1)), c_loc(n(1))), JGX_I32, 1, ext)

    ext(1) = 1
    call jgx_c_record_add_field(JGX_REC_NODE, JGX_NF_BOUNDARY, &
         jgx_offset_of(c_loc(n(1)%boundary), c_loc(n(1))), JGX_I32, 1, ext)
    call jgx_c_record_add_field(JGX_REC_NODE, JGX_NF_BOUNDARY_INDEX, &
         jgx_offset_of(c_loc(n(1)%boundary_index), c_loc(n(1))), JGX_I32, 1, ext)
    call jgx_c_record_add_field(JGX_REC_NODE, JGX_NF_AXIS_DOF, &
         jgx_offset_of(c_loc(n(1)%axis_dof), c_loc(n(1))), JGX_I32, 1, ext)
    call jgx_c_record_add_field(JGX_REC_NODE, JGX_NF_PARENT_ELEM, &
         jgx_offset_of(c_loc(n(1)%parent_elem), c_loc(n(1))), JGX_I32, 1, ext)
    call jgx_c_record_add_field(JGX_REC_NODE, JGX_NF_REF_LAMBDA, &
         jgx_offset_of(c_loc(n(1)%ref_lambda), c_loc(n(1))), JGX_F64, 1, ext)
    call jgx_c_record_add_field(JGX_REC_NODE, JGX_NF_REF_MU, &
         jgx_offset_of(c_loc(n(1)%ref_mu), c_loc(n(1))), JGX_F64, 1, ext)

    call jgx_c_record_end(JGX_REC_NODE)
  end subroutine jgx_register_node_record

end module mod_jgx_node_record
