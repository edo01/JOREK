!> Registers particle_kinetic_relativistic's memory layout with jgx.
!>
!> The field ids below are the contract with the C++ field enum: the two lists
!> must stay in the same order. Ids 0..5 are the components inherited from
!> particle_base and keep the same numbering in every particle record, so C++
!> code that only needs the base part is written once.
!>
!> Inheritance is not visible to the registry: the parent's components are
!> ordinary fields of the concrete type, at offsets the compiler chose. They are
!> measured like all the others -- sizeof(particle_base) is not a valid stand-in
!> (ifx leaves 4 bytes of padding between t_birth and p).
!>
!> Offsets are measured from a local array of the record type, not from a
!> particle group, because they are properties of the type. The group is
!> polymorphic and only yields a concrete base pointer inside a select type;
!> that is a call-site concern, not a registration one.
module mod_jgx_particle_record
  use, intrinsic :: iso_c_binding
  use mod_particle_types, only: particle_kinetic_relativistic
  use mod_jgx_record_ids, only: JGX_REC_PARTICLE_KIN_REL
  use mod_jgx_record
  implicit none
  private
  public :: jgx_register_particle_kin_rel_record

  !> particle_base fields -- the same ids in every particle record
  integer(c_int32_t), parameter :: JGX_PF_X          = 0
  integer(c_int32_t), parameter :: JGX_PF_ST         = 1
  integer(c_int32_t), parameter :: JGX_PF_WEIGHT     = 2
  integer(c_int32_t), parameter :: JGX_PF_I_ELM      = 3
  integer(c_int32_t), parameter :: JGX_PF_I_LIFE     = 4
  integer(c_int32_t), parameter :: JGX_PF_T_BIRTH    = 5
  integer(c_int32_t), parameter :: JGX_PF_BASE_COUNT = 6

  !> what particle_kinetic_relativistic adds
  integer(c_int32_t), parameter :: JGX_PKR_P     = JGX_PF_BASE_COUNT
  integer(c_int32_t), parameter :: JGX_PKR_Q     = JGX_PF_BASE_COUNT + 1
  integer(c_int32_t), parameter :: JGX_PKR_COUNT = JGX_PF_BASE_COUNT + 2

contains

  subroutine jgx_register_particle_kin_rel_record()
    type(particle_kinetic_relativistic), target :: pk(2)
    integer(c_size_t) :: base, stride, ext(1)

    base   = transfer(c_loc(pk(1)), 1_c_size_t)
    stride = transfer(c_loc(pk(2)), 1_c_size_t) - base
    call jgx_c_record_begin(JGX_REC_PARTICLE_KIN_REL, stride, JGX_PKR_COUNT)

    !> inherited from particle_base
    ext(1) = size(pk(1)%x, kind=c_size_t)
    call jgx_c_record_add_field(JGX_REC_PARTICLE_KIN_REL, JGX_PF_X, &
         jgx_offset_of(c_loc(pk(1)%x(1)), c_loc(pk(1))), JGX_F64, 1, ext)

    ext(1) = size(pk(1)%st, kind=c_size_t)
    call jgx_c_record_add_field(JGX_REC_PARTICLE_KIN_REL, JGX_PF_ST, &
         jgx_offset_of(c_loc(pk(1)%st(1)), c_loc(pk(1))), JGX_F64, 1, ext)

    ext(1) = 1
    call jgx_c_record_add_field(JGX_REC_PARTICLE_KIN_REL, JGX_PF_WEIGHT, &
         jgx_offset_of(c_loc(pk(1)%weight), c_loc(pk(1))), JGX_F64, 1, ext)
    call jgx_c_record_add_field(JGX_REC_PARTICLE_KIN_REL, JGX_PF_I_ELM, &
         jgx_offset_of(c_loc(pk(1)%i_elm), c_loc(pk(1))), JGX_I32, 1, ext)
    call jgx_c_record_add_field(JGX_REC_PARTICLE_KIN_REL, JGX_PF_I_LIFE, &
         jgx_offset_of(c_loc(pk(1)%i_life), c_loc(pk(1))), JGX_I32, 1, ext)
    call jgx_c_record_add_field(JGX_REC_PARTICLE_KIN_REL, JGX_PF_T_BIRTH, &
         jgx_offset_of(c_loc(pk(1)%t_birth), c_loc(pk(1))), JGX_F32, 1, ext)

    !> own components
    ext(1) = size(pk(1)%p, kind=c_size_t)
    call jgx_c_record_add_field(JGX_REC_PARTICLE_KIN_REL, JGX_PKR_P, &
         jgx_offset_of(c_loc(pk(1)%p(1)), c_loc(pk(1))), JGX_F64, 1, ext)

    ext(1) = 1
    call jgx_c_record_add_field(JGX_REC_PARTICLE_KIN_REL, JGX_PKR_Q, &
         jgx_offset_of(c_loc(pk(1)%q), c_loc(pk(1))), JGX_I8, 1, ext)

    call jgx_c_record_end(JGX_REC_PARTICLE_KIN_REL)
  end subroutine jgx_register_particle_kin_rel_record

end module mod_jgx_particle_record
