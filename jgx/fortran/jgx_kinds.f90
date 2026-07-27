module jgx_kinds
  use, intrinsic :: iso_c_binding, only: c_int, c_int32_t, c_int64_t, &
                                         c_float, c_double, c_size_t
  implicit none
  public

  !> Real / integer kinds
  integer, parameter :: r64 = c_double   !< real(8) <-> C double
  integer, parameter :: r32 = c_float    !< real(4) <-> C float
  integer, parameter :: i32 = c_int32_t  !< integer(4) <-> C int32_t
  integer, parameter :: i64 = c_int64_t  !< integer(8) <-> C int64_t

  !> Element-kind tags for jgx_buf%elem_kind (mirror JGX_* in jgx_c_api.h).
  integer(c_int), parameter :: JGX_F64 = 0
  integer(c_int), parameter :: JGX_F32 = 1
  integer(c_int), parameter :: JGX_I32 = 2
  integer(c_int), parameter :: JGX_I64 = 3

  !> Layout tags for jgx_buf%layout_tag (see design §4).
  integer(c_int), parameter :: JGX_LAYOUT_COLMAJOR = 0  !< Fortran / layout_left
  integer(c_int), parameter :: JGX_LAYOUT_ROWMAJOR = 1  !< C / layout_right

  !> Maximum logical rank a jgx_buf can describe.
  integer, parameter :: JGX_MAX_RANK = 5

  !> dirty-bit flags packed into jgx_buf_desc%flags across the C ABI.
  integer(c_int), parameter :: JGX_FLAG_DIRTY_HOST   = 1
  integer(c_int), parameter :: JGX_FLAG_DIRTY_DEVICE = 2
  integer(c_int), parameter :: JGX_FLAG_ON_DEVICE    = 4

contains

  !> Bytes occupied by one element of the given kind.
  pure function jgx_kind_size(kind_tag) result(nb)
    integer(c_int), intent(in) :: kind_tag
    integer(c_size_t)          :: nb
    select case (kind_tag)
    case (JGX_F64); nb = 8_c_size_t
    case (JGX_F32); nb = 4_c_size_t
    case (JGX_I32); nb = 4_c_size_t
    case (JGX_I64); nb = 8_c_size_t
    case default;   nb = 0_c_size_t
    end select
  end function jgx_kind_size

end module jgx_kinds
