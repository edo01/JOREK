!> CPU backend: pointer aliasing and no overhead.
!!
!! Every operation is a pointer assignment or a return. No C++ is compiled and
!! no iso_c_binding interface to any external symbol is referenced.
module jgx_backend
  use, intrinsic :: iso_c_binding
  use jgx_kinds
  use jgx_types
  implicit none
  private
  public :: jgx_backend_name, jgx_backend_layout_tag
  public :: jgx_backend_alloc, jgx_backend_free
  public :: jgx_backend_push,  jgx_backend_pull
  public :: jgx_backend_synchronize
  public :: jgx_backend_init,   jgx_backend_finalize

contains

  pure function jgx_backend_name() result(name)
    character(len=:), allocatable :: name
    name = "cpu"
  end function jgx_backend_name

  pure function jgx_backend_layout_tag() result(tag)
    character(len=:), allocatable :: tag
    tag = "host-aliased (no device layout)"
  end function jgx_backend_layout_tag

  subroutine jgx_backend_init(device_id)
    integer, intent(in) :: device_id
    ! No device to select on the CPU backend.
    if (device_id < 0) continue
  end subroutine jgx_backend_init

  subroutine jgx_backend_finalize()
  end subroutine jgx_backend_finalize

  !> "Allocate" the device mirror. On CPU the device pointer simply aliases
  !! the already-bound host pointer — no memory is allocated.
  subroutine jgx_backend_alloc(buf)
    type(jgx_buf), intent(inout) :: buf
    buf%device_ptr   = buf%host_ptr
    buf%is_on_device = .true.
  end subroutine jgx_backend_alloc

  subroutine jgx_backend_free(buf)
    type(jgx_buf), intent(inout) :: buf
    buf%device_ptr   = c_null_ptr
    buf%is_on_device = .false.
  end subroutine jgx_backend_free

  subroutine jgx_backend_push(buf)   ! host -> device
    type(jgx_buf), intent(inout) :: buf
    if (buf%n_bytes == 0) continue   ! no-op: device aliases host
  end subroutine jgx_backend_push

  subroutine jgx_backend_pull(buf)   ! device -> host
    type(jgx_buf), intent(inout) :: buf
    if (buf%n_bytes == 0) continue   ! no-op: device aliases host
  end subroutine jgx_backend_pull

  subroutine jgx_backend_synchronize()
    ! No device: nothing to wait for.
  end subroutine jgx_backend_synchronize

end module jgx_backend
