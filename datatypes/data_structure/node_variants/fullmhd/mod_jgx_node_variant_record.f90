!> Registers what a full-MHD model adds to type_node.
!>
!> One of the interchangeable implementations of mod_jgx_node_variant_record;
!> the model picks the directory (cmake/JorekModelConfig.cmake,
!> JOREK_NODE_VARIANT), so this file is compiled only where psi_eq and Fprof_eq
!> exist and needs no #ifdef of its own.
!>
!> The field ids below are the contract with node_fullmhd_set.h: the two lists
!> must stay in the same order. They start at 0 because this is a record of its
!> own over the node array -- mod_jgx_node_record registers the unconditional
!> components, unchanged, under JGX_REC_NODE.
!>
!> Offsets are measured from a local array of the record type, not from the real
!> mesh, because they are properties of the type.
module mod_jgx_node_variant_record
  use, intrinsic :: iso_c_binding
  use data_structure,     only: type_node
  use mod_jgx_record_ids, only: JGX_REC_NODE_FULLMHD
  use jgx_record
  implicit none
  private
  public :: jgx_register_node_variant_record

  !> must match jorek::node_fullmhd_field
  integer(c_int32_t), parameter :: JGX_NFM_PSI_EQ   = 0
  integer(c_int32_t), parameter :: JGX_NFM_FPROF_EQ = 1
  integer(c_int32_t), parameter :: JGX_NFM_COUNT    = 2

contains

  subroutine jgx_register_node_variant_record()
    type(type_node), target :: n(2)
    integer(c_size_t) :: base, stride, ext(3)

    base   = transfer(c_loc(n(1)), 1_c_size_t)
    stride = transfer(c_loc(n(2)), 1_c_size_t) - base
    call jgx_c_record_begin(JGX_REC_NODE_FULLMHD, stride, JGX_NFM_COUNT)

    ext(1) = size(n(1)%psi_eq, kind=c_size_t)
    ext(2) = 1; ext(3) = 1
    call jgx_c_record_add_field(JGX_REC_NODE_FULLMHD, JGX_NFM_PSI_EQ, &
         jgx_offset_of(c_loc(n(1)%psi_eq(1)), c_loc(n(1))), JGX_F64, 1, ext)

    ext(1) = size(n(1)%Fprof_eq, kind=c_size_t)
    call jgx_c_record_add_field(JGX_REC_NODE_FULLMHD, JGX_NFM_FPROF_EQ, &
         jgx_offset_of(c_loc(n(1)%Fprof_eq(1)), c_loc(n(1))), JGX_F64, 1, ext)

    call jgx_c_record_end(JGX_REC_NODE_FULLMHD)
  end subroutine jgx_register_node_variant_record

end module mod_jgx_node_variant_record
