!> Fortran side of jgx_record_api.h -- registering a derived type's layout.
!>
!> Fortran is the only side that can measure a component offset inside a
!> derived type, so it does that once with c_loc and passes the offsets over
!> to the C type.
!>
!> A type is registered once at the beginning of the program and it's done once
!> for all the instances of that type.
module jgx_record
  use, intrinsic :: iso_c_binding
  implicit none
  public

  !> elem_kind tags. Not mirrored by hand: the values come from the same .def
  !> file the C header expands, so the two sides cannot drift. Only the Fortran
  !> kind is chosen here.
#define JGX_ELEM_KIND(name, tag, bytes) integer(c_int32_t), parameter :: name = tag
#include "jgx/enums/elem_kind.def"
#undef JGX_ELEM_KIND

  interface
    subroutine jgx_c_record_begin(record_id, record_stride_bytes, n_fields) &
        bind(C, name="jgx_c_record_begin")
      import :: c_int32_t, c_size_t
      integer(c_int32_t), value :: record_id, n_fields
      integer(c_size_t),  value :: record_stride_bytes
    end subroutine

    subroutine jgx_c_record_add_field(record_id, field_id, offset_bytes, &
                                      elem_kind, intra_rank, intra_extents) &
        bind(C, name="jgx_c_record_add_field")
      import :: c_int32_t, c_size_t
      integer(c_int32_t), value      :: record_id, field_id, elem_kind
      integer(c_int32_t), value      :: intra_rank
      integer(c_size_t),  value      :: offset_bytes
      integer(c_size_t),  intent(in) :: intra_extents(*)
    end subroutine

    subroutine jgx_c_record_end(record_id) bind(C, name="jgx_c_record_end")
      import :: c_int32_t
      integer(c_int32_t), value :: record_id
    end subroutine
  end interface

contains

  !> Byte offset of a component relative to the record base.
  pure function jgx_offset_of(component_ptr, base_ptr) result(off)
    type(c_ptr), intent(in) :: component_ptr, base_ptr
    integer(c_size_t) :: off
    off = transfer(component_ptr, 1_c_size_t) - transfer(base_ptr, 1_c_size_t)
  end function jgx_offset_of

end module jgx_record
