!> Registers what a stellarator model adds to type_node.
!>
!> One of the interchangeable implementations of mod_jgx_node_variant_record;
!> the model picks the directory (cmake/JorekModelConfig.cmake,
!> JOREK_NODE_VARIANT), so this file is compiled only where r_tor_eq and
!> j_source exist.
!>
!> The field ids below are the contract with node_stellarator_set.h: the two
!> lists must stay in the same order. They start at 0 because this is a record
!> of its own over the node array -- mod_jgx_node_record registers the
!> unconditional components, unchanged, under JGX_REC_NODE.
!>
!> Offsets are measured from a local array of the record type, not from the real
!> mesh, because they are properties of the type.
module mod_jgx_node_variant_record
  use, intrinsic :: iso_c_binding
  use data_structure,     only: type_node
  use mod_jgx_record_ids, only: JGX_REC_NODE_STELLARATOR
  use jgx_record
  implicit none
  private
  public :: jgx_register_node_variant_record

  !> must match jorek::node_stellarator_field
  integer(c_int32_t), parameter :: JGX_NST_R_TOR_EQ        = 0
#if JOREK_MODEL == 180
  integer(c_int32_t), parameter :: JGX_NST_PRESSURE        = JGX_NST_R_TOR_EQ + 1
  integer(c_int32_t), parameter :: JGX_NST_J_FIELD         = JGX_NST_PRESSURE + 1
  integer(c_int32_t), parameter :: JGX_NST_B_FIELD         = JGX_NST_J_FIELD + 1
  integer(c_int32_t), parameter :: JGX_NST_AFTER_GVEC      = JGX_NST_B_FIELD + 1
#else
  integer(c_int32_t), parameter :: JGX_NST_AFTER_GVEC      = JGX_NST_R_TOR_EQ + 1
#endif
#ifndef USE_DOMM
#ifdef USE_EXT_FIELD
  integer(c_int32_t), parameter :: JGX_NST_B_VAC_FIELD     = JGX_NST_AFTER_GVEC
#else
  integer(c_int32_t), parameter :: JGX_NST_CHI_CORRECTION  = JGX_NST_AFTER_GVEC
#endif
  integer(c_int32_t), parameter :: JGX_NST_AFTER_VAC       = JGX_NST_AFTER_GVEC + 1
#else
  integer(c_int32_t), parameter :: JGX_NST_AFTER_VAC       = JGX_NST_AFTER_GVEC
#endif
  integer(c_int32_t), parameter :: JGX_NST_J_SOURCE        = JGX_NST_AFTER_VAC
  integer(c_int32_t), parameter :: JGX_NST_COUNT           = JGX_NST_J_SOURCE + 1

contains

  subroutine jgx_register_node_variant_record()
    type(type_node), target :: n(2)
    integer(c_size_t) :: base, stride, ext(3)

    base   = transfer(c_loc(n(1)), 1_c_size_t)
    stride = transfer(c_loc(n(2)), 1_c_size_t) - base
    call jgx_c_record_begin(JGX_REC_NODE_STELLARATOR, stride, JGX_NST_COUNT)

    ext(1) = size(n(1)%r_tor_eq, kind=c_size_t)
    ext(2) = 1; ext(3) = 1
    call jgx_c_record_add_field(JGX_REC_NODE_STELLARATOR, JGX_NST_R_TOR_EQ, &
         jgx_offset_of(c_loc(n(1)%r_tor_eq(1)), c_loc(n(1))), JGX_F64, 1, ext)

#if JOREK_MODEL == 180
    ext(1) = size(n(1)%pressure, kind=c_size_t)
    call jgx_c_record_add_field(JGX_REC_NODE_STELLARATOR, JGX_NST_PRESSURE, &
         jgx_offset_of(c_loc(n(1)%pressure(1)), c_loc(n(1))), JGX_F64, 1, ext)

    ext(1) = size(n(1)%j_field, 1, kind=c_size_t)
    ext(2) = size(n(1)%j_field, 2, kind=c_size_t)
    ext(3) = size(n(1)%j_field, 3, kind=c_size_t)
    call jgx_c_record_add_field(JGX_REC_NODE_STELLARATOR, JGX_NST_J_FIELD, &
         jgx_offset_of(c_loc(n(1)%j_field(1,1,1)), c_loc(n(1))), JGX_F64, 3, ext)
    call jgx_c_record_add_field(JGX_REC_NODE_STELLARATOR, JGX_NST_B_FIELD, &
         jgx_offset_of(c_loc(n(1)%b_field(1,1,1)), c_loc(n(1))), JGX_F64, 3, ext)
#endif

#ifndef USE_DOMM
#ifdef USE_EXT_FIELD
    ext(1) = size(n(1)%b_vac_field, 1, kind=c_size_t)
    ext(2) = size(n(1)%b_vac_field, 2, kind=c_size_t)
    ext(3) = size(n(1)%b_vac_field, 3, kind=c_size_t)
    call jgx_c_record_add_field(JGX_REC_NODE_STELLARATOR, JGX_NST_B_VAC_FIELD, &
         jgx_offset_of(c_loc(n(1)%b_vac_field(1,1,1)), c_loc(n(1))), JGX_F64, 3, ext)
#else
    ext(1) = size(n(1)%chi_correction, 1, kind=c_size_t)
    ext(2) = size(n(1)%chi_correction, 2, kind=c_size_t)
    ext(3) = 1
    call jgx_c_record_add_field(JGX_REC_NODE_STELLARATOR, JGX_NST_CHI_CORRECTION, &
         jgx_offset_of(c_loc(n(1)%chi_correction(1,1)), c_loc(n(1))), JGX_F64, 2, ext)
#endif
#endif

    ext(1) = size(n(1)%j_source, 1, kind=c_size_t)
    ext(2) = size(n(1)%j_source, 2, kind=c_size_t)
    call jgx_c_record_add_field(JGX_REC_NODE_STELLARATOR, JGX_NST_J_SOURCE, &
         jgx_offset_of(c_loc(n(1)%j_source(1,1)), c_loc(n(1))), JGX_F64, 2, ext)

    call jgx_c_record_end(JGX_REC_NODE_STELLARATOR)
  end subroutine jgx_register_node_variant_record

end module mod_jgx_node_variant_record
