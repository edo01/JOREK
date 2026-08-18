!> JOREK's side of the device backend's lifetime: which GPU this rank uses, and
!> when the runtime lets go of it.
module mod_jgx_device
  use, intrinsic :: iso_c_binding, only: c_int
  implicit none
  private
  public :: jgx_c_init, jgx_c_finalize

  interface
    !> Pin this process to a device. Takes the node-local index, not the rank.
    subroutine jgx_c_init(device_id) bind(C, name="jgx_c_init")
      import :: c_int
      implicit none
      integer(c_int), value, intent(in) :: device_id
    end subroutine

    !> Release everything the runtime holds for this device on this process,
    !> rather than leaving it to process exit.
    subroutine jgx_c_finalize() bind(C, name="jgx_c_finalize")
      implicit none
    end subroutine
  end interface

end module mod_jgx_device
