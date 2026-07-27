!> interface to every C++ backend (serial-cpp / cuda / hip / kokkos):
!! it only knows jgx_c_api.h, never which C++ technology is linked below it.
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

  interface
    function c_alloc(n) bind(C, name="jgx_c_alloc") result(p)
      import :: c_ptr, c_size_t
      integer(c_size_t), value :: n
      type(c_ptr)              :: p
    end function c_alloc
    subroutine c_free(p) bind(C, name="jgx_c_free")
      import :: c_ptr
      type(c_ptr), value :: p
    end subroutine c_free
    subroutine c_push(d, h, n) bind(C, name="jgx_c_push")
      import :: c_ptr, c_size_t
      type(c_ptr), value       :: d, h
      integer(c_size_t), value :: n
    end subroutine c_push
    subroutine c_pull(h, d, n) bind(C, name="jgx_c_pull")
      import :: c_ptr, c_size_t
      type(c_ptr), value       :: h, d
      integer(c_size_t), value :: n
    end subroutine c_pull
    subroutine c_sync() bind(C, name="jgx_c_synchronize")
    end subroutine c_sync
    subroutine c_init(id) bind(C, name="jgx_c_init")
      import :: c_int
      integer(c_int), value :: id
    end subroutine c_init
    subroutine c_finalize() bind(C, name="jgx_c_finalize")
    end subroutine c_finalize
    function c_name() bind(C, name="jgx_c_backend_name") result(s)
      import :: c_ptr
      type(c_ptr) :: s
    end function c_name
    function c_layout() bind(C, name="jgx_c_layout_tag") result(s)
      import :: c_ptr
      type(c_ptr) :: s
    end function c_layout
  end interface

contains

  function jgx_backend_name() result(name)
    character(len=:), allocatable :: name
    name = c_string(c_name())
  end function jgx_backend_name

  function jgx_backend_layout_tag() result(tag)
    character(len=:), allocatable :: tag
    tag = c_string(c_layout())
  end function jgx_backend_layout_tag

  subroutine jgx_backend_init(device_id)
    integer, intent(in) :: device_id
    call c_init(int(device_id, c_int))
  end subroutine jgx_backend_init

  subroutine jgx_backend_finalize()
    call c_finalize()
  end subroutine jgx_backend_finalize

  subroutine jgx_backend_alloc(buf)
    type(jgx_buf), intent(inout) :: buf
    buf%device_ptr   = c_alloc(buf%n_bytes)
    buf%is_on_device = .true.
  end subroutine jgx_backend_alloc

  subroutine jgx_backend_free(buf)
    type(jgx_buf), intent(inout) :: buf
    if (c_associated(buf%device_ptr)) call c_free(buf%device_ptr)
    buf%device_ptr   = c_null_ptr
    buf%is_on_device = .false.
  end subroutine jgx_backend_free

  subroutine jgx_backend_push(buf)   ! host -> device
    type(jgx_buf), intent(inout) :: buf
    if (buf%n_bytes > 0) call c_push(buf%device_ptr, buf%host_ptr, buf%n_bytes)
  end subroutine jgx_backend_push

  subroutine jgx_backend_pull(buf)   ! device -> host
    type(jgx_buf), intent(inout) :: buf
    if (buf%n_bytes > 0) call c_pull(buf%host_ptr, buf%device_ptr, buf%n_bytes)
  end subroutine jgx_backend_pull

  subroutine jgx_backend_synchronize()
    call c_sync()
  end subroutine jgx_backend_synchronize

  !> Convert a null-terminated C string to a Fortran allocatable string.
  function c_string(cptr) result(f)
    type(c_ptr), intent(in)        :: cptr
    character(len=:), allocatable  :: f
    character(kind=c_char), pointer :: arr(:)
    integer :: n, i
    if (.not. c_associated(cptr)) then
      f = ""
      return
    end if
    call c_f_pointer(cptr, arr, [256])
    n = 0
    do while (n < 256)
      if (arr(n+1) == c_null_char) exit
      n = n + 1
    end do
    allocate(character(len=n) :: f)
    do i = 1, n
      f(i:i) = arr(i)
    end do
  end function c_string

end module jgx_backend
