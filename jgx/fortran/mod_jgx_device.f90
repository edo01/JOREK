!> Fortran side of the device backend's lifetime: binding this process to one
!> device, and letting the runtime go of it again.
!>
!> Application-agnostic, like the rest of jgx/ outside jgx/jorek: it does not
!> choose the device, the caller does. Which index is the right one is site
!> policy (on JOREK it is the node-local rank), and this module has no business
!> knowing it.
!>
!> Callers use jgx_device_init / jgx_device_finalize, which are no-ops in a host
!> build. The `#ifdef JGX_HAS_DEVICE` lives here and nowhere else, so no caller
!> has to know whether a backend was configured in.
module mod_jgx_device
  implicit none
  private
  public :: jgx_device_init, jgx_device_finalize

#ifdef JGX_HAS_DEVICE
  interface
    !> Pin this process to a device, by index within its node.
    subroutine jgx_c_init_device(device_id) bind(C, name="jgx_c_init_device")
      use, intrinsic :: iso_c_binding, only: c_int
      implicit none
      integer(c_int), value, intent(in) :: device_id
    end subroutine

    !> Release everything the runtime holds for this device on this process,
    !> rather than leaving it to process exit.
    subroutine jgx_c_finalize_device() bind(C, name="jgx_c_finalize_device")
      implicit none
    end subroutine
  end interface
#endif

contains

  !> Bind this process to one device. `device_id` is an index into the devices
  !> visible to this process, so on a multi-rank node it must be the caller's
  !> **node-local** rank -- device ids are numbered per node, and the
  !> MPI_COMM_WORLD rank is the wrong number as soon as there is a second node.
  subroutine jgx_device_init(device_id)
#ifdef JGX_HAS_DEVICE
    use, intrinsic :: iso_c_binding, only: c_int
#endif
    integer, intent(in) :: device_id
#ifdef JGX_HAS_DEVICE
    call jgx_c_init_device(int(device_id, c_int))
#endif
  end subroutine jgx_device_init

  !> Symmetric with jgx_device_init.
  subroutine jgx_device_finalize()
#ifdef JGX_HAS_DEVICE
    call jgx_c_finalize_device()
#endif
  end subroutine jgx_device_finalize

end module mod_jgx_device
