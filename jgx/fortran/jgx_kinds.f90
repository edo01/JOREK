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

  !> Everything below comes from the .def files the C headers expand.
  !> Only the Fortran kind is chosen here.
#include "jgx/jgx_abi.def"

  !> Element-kind tags for jgx_buf%elem_kind.
#define JGX_ELEM_KIND(name, tag, bytes) integer(c_int), parameter :: name = tag
#include "jgx/enums/elem_kind.def"
#undef JGX_ELEM_KIND

  !> Layout tags for jgx_buf%layout_tag (see design §4).
  !> dirty-bit flags packed into jgx_buf_desc%flags across the C ABI.
#define JGX_ENUM_ENTRY(name, value) integer(c_int), parameter :: name = value
#include "jgx/enums/layout_tag.def"
#include "jgx/enums/flags.def"
#undef JGX_ENUM_ENTRY

  !> JGX_MAX_RANK is deliberately left as the macro from jgx_abi.def rather than
  !> re-declared as a parameter here: cpp is case-sensitive and Fortran is not,
  !> so a parameter of that name would be shadowed by the macro at every
  !> uppercase use site.

contains

  !> Bytes occupied by one element of the given kind. Generated from the same
  !> list as the tags, so a new kind cannot arrive without its size.
  pure function jgx_kind_size(kind_tag) result(nb)
    integer(c_int), intent(in) :: kind_tag
    integer(c_size_t)          :: nb
    select case (kind_tag)
#define JGX_ELEM_KIND(name, tag, bytes) case (name); nb = int(bytes, c_size_t)
#include "jgx/enums/elem_kind.def"
#undef JGX_ELEM_KIND
    case default; nb = 0_c_size_t
    end select
  end function jgx_kind_size

end module jgx_kinds
