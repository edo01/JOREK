!> Registers jorek_fields_interp_linear's memory layout with jgx.
!>
!> The field ids below are the contract with the C++ field enum: the two lists
!> must stay in the same order. Ids 0..1 are the components inherited from
!> fields_interpolator_base and keep the same numbering in every interpolator record,
!> so C++ code that only needs the base part is written once -- the same
!> arrangement particle_base has inside every particle record.
!>
!> An interpolator is registered as a record of count 1. The registry describes
!> the TYPE, so a count of one is no special case: what it recovers is the set of
!> compiler-chosen component offsets, and those do not depend on how many
!> instances exist.
!>
!> static and flag_zero_dpsidt are default logicals registered as JGX_I32. That
!> is deliberate and rests on two assumptions this build satisfies: a default
!> logical occupies 4 bytes, and .true. is never stored as a zero word. C++ reads
!> them back with `!= 0`.
!>
!> Measured with ifx 2024.1 on this build: .true. is stored as the word -1, not
!> 1. So `!= 0` is not defensive phrasing, it is the only correct test --
!> narrowing it to `== 1` reads every .true. flag as false.
!>
!> Offsets are measured from a local array of the record type, not from a live
!> fields object, because they are properties of the type. sim%fields%interp is
!> polymorphic and only yields a concrete base pointer inside a select type; that
!> is a call-site concern, not a registration one.
module mod_jgx_fields_interp_linear_record
  use, intrinsic :: iso_c_binding
  use mod_fields_linear, only: jorek_fields_interp_linear
  use mod_jgx_record_ids, only: JGX_REC_FIELDS_INTERP_LINEAR
  use mod_jgx_record
  implicit none
  private
  public :: jgx_register_fields_interp_linear_record

  !> fields_interpolator_base fields -- the same ids in every interpolator record
  integer(c_int32_t), parameter :: JGX_FI_STATIC           = 0
  integer(c_int32_t), parameter :: JGX_FI_FLAG_ZERO_DPSIDT = 1
  integer(c_int32_t), parameter :: JGX_FI_BASE_COUNT       = 2

  !> what jorek_fields_interp_linear adds
  integer(c_int32_t), parameter :: JGX_FIL_TIME_NOW  = JGX_FI_BASE_COUNT
  integer(c_int32_t), parameter :: JGX_FIL_TIME_PREV = JGX_FI_BASE_COUNT + 1
  integer(c_int32_t), parameter :: JGX_FIL_COUNT     = JGX_FI_BASE_COUNT + 2

contains

  subroutine jgx_register_fields_interp_linear_record()
    type(jorek_fields_interp_linear), target :: f(2)
    integer(c_size_t) :: base, stride, ext(1)

    base   = transfer(c_loc(f(1)), 1_c_size_t)
    stride = transfer(c_loc(f(2)), 1_c_size_t) - base
    call jgx_c_record_begin(JGX_REC_FIELDS_INTERP_LINEAR, stride, JGX_FIL_COUNT)

    ext(1) = 1

    !> inherited from fields_interpolator_base
    call jgx_c_record_add_field(JGX_REC_FIELDS_INTERP_LINEAR, JGX_FI_STATIC, &
         jgx_offset_of(c_loc(f(1)%static), c_loc(f(1))), JGX_I32, 1, ext)
    call jgx_c_record_add_field(JGX_REC_FIELDS_INTERP_LINEAR, JGX_FI_FLAG_ZERO_DPSIDT, &
         jgx_offset_of(c_loc(f(1)%flag_zero_dpsidt), c_loc(f(1))), JGX_I32, 1, ext)

    !> own components
    call jgx_c_record_add_field(JGX_REC_FIELDS_INTERP_LINEAR, JGX_FIL_TIME_NOW, &
         jgx_offset_of(c_loc(f(1)%time_now), c_loc(f(1))), JGX_F64, 1, ext)
    call jgx_c_record_add_field(JGX_REC_FIELDS_INTERP_LINEAR, JGX_FIL_TIME_PREV, &
         jgx_offset_of(c_loc(f(1)%time_prev), c_loc(f(1))), JGX_F64, 1, ext)

    call jgx_c_record_end(JGX_REC_FIELDS_INTERP_LINEAR)
  end subroutine jgx_register_fields_interp_linear_record

end module mod_jgx_fields_interp_linear_record
