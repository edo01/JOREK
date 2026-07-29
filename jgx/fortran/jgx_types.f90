module jgx_types

  use, intrinsic :: iso_c_binding
  use jgx_kinds
  implicit none
  public

  !> JGX_MAX_RANK is a macro, not a parameter -- see the note in jgx_kinds.f90.
#include "jgx/jgx_abi.def"

  !> A wrapper to the raw memory and its description.
  !> For example an integer(8) a(100,3) becomes:
  !> rank = 2, extents = [100,3,0,0,0] 
  !> elem_kind = JGX_I64, layout_tag = JGX_LAYOUT_COLMAJOR
  !> The filling logic can be found in fill_buf in jgx_api.f90
  type :: jgx_buf
    type(c_ptr)       :: device_ptr = c_null_ptr  !< device memory (aliases host on cpu backend)
    type(c_ptr)       :: host_ptr   = c_null_ptr  !< associated host array (may be null)
    integer(c_size_t) :: n_bytes    = 0
    integer(c_int)    :: elem_kind  = JGX_F64     !< buffer type (see jgx_kinds.f90)
    integer(c_int)    :: rank       = 0           
    integer(c_size_t) :: extents(JGX_MAX_RANK) = 0   !< array specifing the dimensions of the buffer
    integer(c_int)    :: layout_tag = JGX_LAYOUT_COLMAJOR !< which data layout is the buffer using
    logical           :: is_on_device = .false.   !< the buf is present on the device
    logical           :: dirty_host   = .false.   !< host modified since last push
    logical           :: dirty_device = .false.   !< device modified since last pull
  end type jgx_buf


  !> POD descriptor that crosses the C ABI (mirror of the struct
  !! jgx_buf_desc in jgx_c_api.h). Built from a jgx_buf by jgx_buf_to_desc.
  !! The duplication is necessary because the jgx_buf cannot be
  !! made a bind(C) without losing important fortran features such 
  !! as logicals, defaults, type bound procedures etc...
  type, bind(C) :: jgx_buf_desc
    type(c_ptr)       :: device_ptr
    type(c_ptr)       :: host_ptr
    integer(c_size_t) :: n_bytes
    integer(c_int)    :: elem_kind
    integer(c_int)    :: rank
    integer(c_size_t) :: extents(JGX_MAX_RANK)
    integer(c_int)    :: layout_tag
    integer(c_int)    :: flags
  end type jgx_buf_desc

contains

  !> Pack a jgx_buf handle into its POD C descriptor jgx_buf_desc.
  pure function jgx_buf_to_desc(buf) result(d)
    type(jgx_buf), intent(in) :: buf
    type(jgx_buf_desc)        :: d
    d%device_ptr = buf%device_ptr
    d%host_ptr   = buf%host_ptr
    d%n_bytes    = buf%n_bytes
    d%elem_kind  = buf%elem_kind
    d%rank       = buf%rank
    d%extents    = buf%extents
    d%layout_tag = buf%layout_tag
    d%flags      = 0
    if (buf%dirty_host)   d%flags = ior(d%flags, JGX_FLAG_DIRTY_HOST)
    if (buf%dirty_device) d%flags = ior(d%flags, JGX_FLAG_DIRTY_DEVICE)
    if (buf%is_on_device) d%flags = ior(d%flags, JGX_FLAG_ON_DEVICE)
  end function jgx_buf_to_desc

end module jgx_types