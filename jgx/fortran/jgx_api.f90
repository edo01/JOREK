!! Overloaded jgx_alloc binds a contiguous host array to a jgx_buf 
!! and asks the linked backend to establish the device mirror. 
!! push/pull/mirror drive the dirty-bit transfer discipline. 
!! The backend (`jgx_backend`) at compile time by CMake. This 
!! This module acts only as the interface to the backend.
!!
!! Arrays passed here must be `contiguous` and `target`.
module jgx_api
  use, intrinsic :: iso_c_binding
  use jgx_kinds
  use jgx_types
  use jgx_backend
  implicit none
  private

  public :: jgx_alloc, jgx_free
  public :: jgx_push, jgx_pull, jgx_mirror
  public :: jgx_mark_host_dirty, jgx_mark_device_dirty
  public :: jgx_synchronize, jgx_init, jgx_finalize
  public :: jgx_backend_name, jgx_backend_layout_tag

  !> Bind a host array + allocate its device mirror.
  interface jgx_alloc
    module procedure alloc_r64_1, alloc_r64_2, alloc_r64_3, alloc_r64_4, &
                     alloc_r64_5
    module procedure alloc_i32_1, alloc_i32_2, alloc_i32_3, alloc_i32_4, &
                     alloc_i32_5
  end interface jgx_alloc

contains

  subroutine jgx_init(device_id)
    integer, intent(in), optional :: device_id
    integer :: dev
    dev = 0
    if (present(device_id)) dev = device_id
    call jgx_backend_init(dev)
  end subroutine jgx_init

  subroutine jgx_finalize()
    call jgx_backend_finalize()
  end subroutine jgx_finalize

  ! ---- transfers / sync -------------------------------------------------

  !> host -> device (no-op on cpu). Clears the host-dirty flag.
  subroutine jgx_push(buf)
    type(jgx_buf), intent(inout) :: buf
    call jgx_backend_push(buf)
    buf%dirty_host = .false.
  end subroutine jgx_push

  !> device -> host (no-op on cpu). Clears the device-dirty flag.
  subroutine jgx_pull(buf)
    type(jgx_buf), intent(inout) :: buf
    call jgx_backend_pull(buf)
    buf%dirty_device = .false.
  end subroutine jgx_pull

  !> Lazy sync: push if host-dirty, pull if device-dirty, else nothing (§3.2).
  subroutine jgx_mirror(buf)
    type(jgx_buf), intent(inout) :: buf
    if (buf%dirty_host)   call jgx_push(buf)
    if (buf%dirty_device) call jgx_pull(buf)
  end subroutine jgx_mirror

  subroutine jgx_mark_host_dirty(buf)
    type(jgx_buf), intent(inout) :: buf
    buf%dirty_host = .true.
  end subroutine jgx_mark_host_dirty

  subroutine jgx_mark_device_dirty(buf)
    type(jgx_buf), intent(inout) :: buf
    buf%dirty_device = .true.
  end subroutine jgx_mark_device_dirty

  subroutine jgx_synchronize()
    call jgx_backend_synchronize()
  end subroutine jgx_synchronize

  subroutine jgx_free(buf)
    type(jgx_buf), intent(inout) :: buf
    call jgx_backend_free(buf)
    buf%host_ptr = c_null_ptr
    buf%n_bytes  = 0
    buf%rank     = 0
    buf%extents  = 0
  end subroutine jgx_free

  ! ---- private fill descriptor fields ----------------------------------

  subroutine fill_buf(buf, cptr, ext, nrank, kind_tag)
    type(jgx_buf),     intent(inout) :: buf
    type(c_ptr),       intent(in)    :: cptr
    integer,           intent(in)    :: ext(:)
    integer,           intent(in)    :: nrank
    integer(c_int),    intent(in)    :: kind_tag
    integer :: k
    buf%host_ptr   = cptr
    buf%elem_kind  = kind_tag
    buf%rank       = nrank
    buf%layout_tag = JGX_LAYOUT_COLMAJOR
    buf%extents    = 0
    buf%n_bytes    = jgx_kind_size(kind_tag)
    do k = 1, nrank
      buf%extents(k) = int(ext(k), c_size_t)
      buf%n_bytes    = buf%n_bytes * int(ext(k), c_size_t)
    end do
    call jgx_backend_alloc(buf)
  end subroutine fill_buf

  ! ---- jgx_alloc overloads ---------------------------------------------

  subroutine alloc_r64_1(buf, a)
    type(jgx_buf),                   intent(inout) :: buf
    real(r64), contiguous, target,   intent(in)    :: a(:)
    call fill_buf(buf, c_loc(a), shape(a), 1, JGX_F64)
  end subroutine alloc_r64_1

  subroutine alloc_r64_2(buf, a)
    type(jgx_buf),                   intent(inout) :: buf
    real(r64), contiguous, target,   intent(in)    :: a(:,:)
    call fill_buf(buf, c_loc(a), shape(a), 2, JGX_F64)
  end subroutine alloc_r64_2

  subroutine alloc_r64_3(buf, a)
    type(jgx_buf),                   intent(inout) :: buf
    real(r64), contiguous, target,   intent(in)    :: a(:,:,:)
    call fill_buf(buf, c_loc(a), shape(a), 3, JGX_F64)
  end subroutine alloc_r64_3

  subroutine alloc_r64_4(buf, a)
    type(jgx_buf),                   intent(inout) :: buf
    real(r64), contiguous, target,   intent(in)    :: a(:,:,:,:)
    call fill_buf(buf, c_loc(a), shape(a), 4, JGX_F64)
  end subroutine alloc_r64_4

  subroutine alloc_i32_1(buf, a)
    type(jgx_buf),                   intent(inout) :: buf
    integer(i32), contiguous, target, intent(in)   :: a(:)
    call fill_buf(buf, c_loc(a), shape(a), 1, JGX_I32)
  end subroutine alloc_i32_1

  subroutine alloc_i32_2(buf, a)
    type(jgx_buf),                   intent(inout) :: buf
    integer(i32), contiguous, target, intent(in)   :: a(:,:)
    call fill_buf(buf, c_loc(a), shape(a), 2, JGX_I32)
  end subroutine alloc_i32_2

  subroutine alloc_i32_3(buf, a)
    type(jgx_buf),                   intent(inout) :: buf
    integer(i32), contiguous, target, intent(in)   :: a(:,:,:)
    call fill_buf(buf, c_loc(a), shape(a), 3, JGX_I32)
  end subroutine alloc_i32_3

  subroutine alloc_i32_4(buf, a)
    type(jgx_buf),                   intent(inout) :: buf
    integer(i32), contiguous, target, intent(in)   :: a(:,:,:,:)
    call fill_buf(buf, c_loc(a), shape(a), 4, JGX_I32)
  end subroutine alloc_i32_4

  subroutine alloc_i32_5(buf, a)
    type(jgx_buf),                   intent(inout) :: buf
    integer(i32), contiguous, target, intent(in)   :: a(:,:,:,:,:)
    call fill_buf(buf, c_loc(a), shape(a), 5, JGX_I32)
  end subroutine alloc_i32_5


end module jgx_api
