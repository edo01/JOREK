module mod_find_rz_nearby
  implicit none
  private
  public :: find_rz_nearby, find_RZ_nearby_gpu, find_RZ_nearby_gpu_inlined
contains
!> Optimized subroutine to find st coordinates corresponding to x_new=[R_new, Z_new] using
!! The previous values x_old=[R_old, Z_old], st_old(2), i_elm_old and checking adjacent elements first
!! It first checks the current element, using newton iteration to find st_try corresponding to x_new
!! Once this is found it calls check_element_boundary to see if we have crossed an element boundary
!! If this is the case, the search is restarted within the new element.
!! This procedure continues until the required tolerance is reached, or element_try_max is reached.
!!
!! Example:
!! (o is the starting point, T is the target, x is the guess)
!! |^^^^^^|^^^^^^|  |^^^^^^|^^^^^^|  |^^^^^^|^^^^^^|
!! |1     |2     |  |1     |2     |  |1     |2     |
!! |      |  T   |  |      |  T   |  |      | xT   |
!! |  o   |      |  |  o   |x     |  |  o   |      |
!! |______|______|  |______|______|  |______|______|
!! We start out knowing the old position (R,Z) x_old, the new position (R,Z) x_new,
!! the old position and element number st_old, i_elm_old.
!!
!! We are now in element 1, and element_try_index=1. Using 2D newton iteration
!! with backtracking we find the (s,t) position st_try corresponding to x_new, starting
!! from x_old, in the basis functions of the current element. (Which are not
!! guaranteed to be nice outside of the element)
!! 
!! When we have found the position st_try, we check whether it is outside of the
!! current element. (middle figure) If so, we switch to this element (element 2), perform a linear transform
!! of the coordinates and start the process again.
!! If it is inside of the element and we have found a good st_try the routine is done.
!! 
!! There are some cases where this method does not work, notably near element
!! boundaries and the magnetic axis. Here, overshoot of the newton's method near
!! the element boundary causes errors. If too many elements are crossed, i.e.
!! element_try_max is exceeded, or if too many iterations are needed, i.e.
!! newton_iter_max is exceeded, we stop the routine and call find_RZ, which is
!! extremely slow but works for all positions in the domain.
!! In that case, ifail=2:5 is returned.
!! If ifail=-1 the particle is lost
subroutine find_RZ_nearby(node_list, element_list, R_old, Z_old, s_old, t_old, i_elm_old, &
        R_new, Z_new, s_new, t_new, i_elm_new, ifail, phi)
use data_structure
use mod_neighbours
use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
implicit none
!> Input parameters
type (type_node_list),    intent(in)    :: node_list
type (type_element_list), intent(in)    :: element_list
real*8,                   intent(in)    :: R_old, Z_old !< The old R,Z location
real*8,                   intent(in)    :: R_new, Z_new !< The new R,Z location
real*8,                   intent(in)    :: s_old, t_old !< The old st location (used to compute a guess)
integer,                  intent(in)    :: i_elm_old
real*8,                   intent(out)   :: s_new, t_new !< The found new coordinates
integer,                  intent(out)   :: i_elm_new
integer,                  intent(out)   :: ifail !< if ifail = -1 the position could not be found in the grid.
!< ifail > 0 indicates various other cases
real*8, optional,         intent(in)    :: phi

!> Accuracy defaults (tolerances are squared!, units of element size)
real*8,  parameter :: element_tolerance   = 1.d-24 !< Tolerance for finding a position inside an element
integer, parameter :: newton_iter_max     = 8 !< Number of iterations to try

!> Internal variables
real*8  :: p              
integer :: newton_iter_number, i_elm_tmp
real*8  :: inv_st_jac_det, R_s, R_t, Z_s, Z_t
real*8  :: st_step(2), x_step(2), x_tmp(2), st_new(2), x_new(2) ! x_step = (R,Z) of trial position
real*8  :: err2, err2_old, dist(2), fact

! Check if element is valid
if (i_elm_old .lt. 1 .or. i_elm_old .gt. element_list%n_elements) then
#if STELLARATOR_MODEL
  write(*,*) "ERROR: find_RZ is not implemented for stellarators"
  stop
#else
  call find_RZ(node_list,element_list,R_new,Z_new,x_step(1),x_step(2),i_elm_new,s_new,t_new,ifail)
  return
#endif
end if

if (present(phi)) then
  p = phi
else
#if STELLARATOR_MODEL
  write(*,*) "Toroidal angle phi must be defined for stellarator models"
  stop
#endif
  p = 0.0
endif

! Setup initial values
x_step = [R_old,Z_old] ! start at the current position
i_elm_new = i_elm_old ! start in the current element
st_new = [s_old,t_old] ! start at the old position
x_new = [R_new,Z_new]
! Find the jacobian at the current s and t position
call try_interp(node_list,element_list,i_elm_new,st_new,p,x_step,R_s,R_t,Z_s,Z_t,inv_st_jac_det)
err2 = dot_product(x_step-x_new,x_step-x_new)
ifail=0


! Newton iteration to find s and t in or out of this element
do newton_iter_number = 1, newton_iter_max
  ! Perform newton iteration by calculating the inverse of the jacobian matrix explicitly
  err2_old = err2

  ! Calculate the trial newton step
  st_step(1) = ( Z_t * (x_new(1)-x_step(1)) - R_t * (x_new(2)-x_step(2))) * inv_st_jac_det
  st_step(2) = (-Z_s * (x_new(1)-x_step(1)) + R_s * (x_new(2)-x_step(2))) * inv_st_jac_det

  ! Limit this step if it goes outside of the element
  dist = merge(1-st_new,st_new,st_step .gt. 0) ! dist = 1-s if step > 0, s if step < 0 (distance to 0 or 1)
  fact = maxval(abs(st_step)/max(dist,1d-30)) ! if fact>=1 we are on the boundary
  ! (it is the overshoot: i.e. how many times we overshoot the boundary with one st_step)

  if (fact .ge. 1.d0-1d-12) then
    st_new = st_new + st_step/fact
#ifdef DEBUG
    call try_interp(node_list,element_list,i_elm_new,st_new,p,x_tmp,R_s,R_t,Z_s,Z_t,inv_st_jac_det)
#endif
    i_elm_tmp = i_elm_new
    call coord_in_neighbour(node_list,element_list,i_elm_tmp,i_elm_new,st_new)
    if (i_elm_new .lt. 0) then
#if STELLARATOR_MODEL
      write(*,*) "ERROR: find_RZ is not implemented for stellarators"
      stop
#else
      call find_RZ(node_list,element_list,x_new(1),x_new(2),x_step(1),x_step(2),i_elm_new,s_new,t_new,ifail)
      if (ifail .ne. 0) i_elm_new = 0
#endif
    end if
    if (i_elm_new .eq. 0) then ! No element on that side, particle is lost
      i_elm_new = - i_elm_tmp ! Save position of particle
      ! Calculate new R and Z in x_new
      call try_interp(node_list,element_list,i_elm_tmp,st_new,p,x_new,R_s,R_t,Z_s,Z_t,inv_st_jac_det)
      ! Set new element-local coordinates for the point on the axis
      s_new = st_new(1)
      t_new = st_new(2)
      ifail = -1
      return
    end if

    call try_interp(node_list,element_list,i_elm_new,st_new,p,x_step,R_s,R_t,Z_s,Z_t,inv_st_jac_det)
#ifdef DEBUG 
    if (norm2(x_step-x_tmp) .gt. 1d-8) then
      !write(*,*) "ERROR on element edge crossing", x_step, x_tmp, norm2(x_step-x_tmp), &
      !i_elm_new, i_elm_old, i_elm_tmp, "deleting particle"
      !call exit(1)
      i_elm_new = 0
      return
    end if
#endif
  else
    st_new = st_new + st_step
    call try_interp(node_list,element_list,i_elm_new,st_new,p,x_step,R_s,R_t,Z_s,Z_t,inv_st_jac_det)
  end if
  err2 = dot_product(x_step-x_new,x_step-x_new)
  s_new = st_new(1)
  t_new = st_new(2)

  if (err2 < element_tolerance) exit
enddo


if (ieee_is_nan(err2)) then
#if STELLARATOR_MODEL
  write(*,*) "ERROR: find_RZ is not implemented for stellarators"
  stop
#else
  !write(*,*) "WARNING: NaN encountered after newton iteration, using find_RZ"
  call find_RZ(node_list,element_list,x_new(1),x_new(2),x_step(1),x_step(2),i_elm_new,s_new,t_new,ifail)
  if (ifail .eq. 0) ifail=2
  return
#endif
endif
if (newton_iter_number .gt. newton_iter_max) then
#if STELLARATOR_MODEL
  write(*,*) "ERROR: find_RZ is not implemented for stellarators"
  stop
#else
  !write(*,"(A,i4,A,i5,A,2g14.6,A,3g14.6)") "WARNING: iteration for st did not converge after", newton_iter_max, " tries in element ", i_elm_new, &
  !" using find_RZ", x_new, "err2(old)/convergence: ", err2, err2_old, err2_old/err2
    !write(*,"(A,2g16.8)") "Find_RZ at ", x_new
  call find_RZ(node_list,element_list,x_new(1),x_new(2),x_step(1),x_step(2),i_elm_new,s_new,t_new,ifail)
  if (ifail .eq. 0) ifail=3
  return
#endif
endif
end subroutine find_RZ_nearby


!> Auxiliary subroutine for find_RZ_nearby
pure subroutine try_interp(node_list,element_list,i_elm,st,p,x,R_s,R_t,Z_s,Z_t,inv_st_jac_det)
use data_structure
use mod_interp
implicit none
!> Input parameters
type (type_node_list),    intent(in)    :: node_list
type (type_element_list), intent(in)    :: element_list
real*8,                   intent(in)    :: st(2), p
integer,                  intent(in)    :: i_elm
real*8,                   intent(out)   :: x(2), R_s, R_t, Z_s, Z_t, inv_st_jac_det

real*8 :: R_p, Z_p
real*8 :: jac

call interp_RZP(node_list,element_list,i_elm,st(1),st(2),p,x(1),R_s,R_t,R_p,x(2),Z_s,Z_t,Z_p)
! Guard against the determinant being close to zero
jac = R_s * Z_t - R_t * Z_s
if (abs(jac) .lt. 1d-8) then
  inv_st_jac_det = sign(1d8, jac) 
else
  inv_st_jac_det = 1.d0/(jac)
end if
end subroutine try_interp

subroutine find_RZ_nearby_gpu(node_list, element_list, R_old, Z_old, s_old, t_old, i_elm_old, &
        R_new, Z_new, s_new, t_new, i_elm_new, ifail, phi)
use data_structure
use mod_neighbours
use data_structure, only: type_node_list_gpu
use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
implicit none
!> Input parameters
type (type_node_list_gpu),intent(in)    :: node_list
type (type_element_list), intent(in)    :: element_list
real*8,                   intent(in)    :: R_old, Z_old !< The old R,Z location
real*8,                   intent(in)    :: R_new, Z_new !< The new R,Z location
real*8,                   intent(in)    :: s_old, t_old !< The old st location (used to compute a guess)
integer,                  intent(in)    :: i_elm_old
real*8,                   intent(out)   :: s_new, t_new !< The found new coordinates
integer,                  intent(out)   :: i_elm_new
integer,                  intent(out)   :: ifail !< if ifail = -1 the position could not be found in the grid.
!< ifail > 0 indicates various other cases
real*8, optional,         intent(in)    :: phi

!> Accuracy defaults (tolerances are squared!, units of element size)
real*8,  parameter :: element_tolerance   = 1.d-24 !< Tolerance for finding a position inside an element
integer, parameter :: newton_iter_max     = 8 !< Number of iterations to try

!> Internal variables
real*8  :: p              
integer :: newton_iter_number, i_elm_tmp
real*8  :: inv_st_jac_det, R_s, R_t, Z_s, Z_t
real*8  :: st_step(2), x_step(2), x_tmp(2), st_new(2), x_new(2) ! x_step = (R,Z) of trial position
real*8  :: err2, err2_old, dist(2), fact

#ifdef USE_GPU
!$omp declare target
#endif

! Check if element is valid
if (i_elm_old .lt. 1 .or. i_elm_old .gt. element_list%n_elements) then
#if STELLARATOR_MODEL
  write(*,*) "ERROR: find_RZ is not implemented for stellarators"
  stop
#else

#ifdef USE_GPU_TEST
  write(*,*) "WARNING: element not valid, but find_RZ is not implemented for GPU"
#endif 
  i_elm_new = 0
  return
#endif
end if

if (present(phi)) then
  p = phi
else
#if STELLARATOR_MODEL
  write(*,*) "Toroidal angle phi must be defined for stellarator models"
  stop
#endif
  p = 0.0
endif

! Setup initial values
x_step = [R_old,Z_old] ! start at the current position
i_elm_new = i_elm_old ! start in the current element
st_new = [s_old,t_old] ! start at the old position
x_new = [R_new,Z_new]
! Find the jacobian at the current s and t position
call try_interp_gpu(node_list,element_list,i_elm_new,st_new,p,x_step,R_s,R_t,Z_s,Z_t,inv_st_jac_det)
err2 = dot_product(x_step-x_new,x_step-x_new)
ifail=0


! Newton iteration to find s and t in or out of this element
do newton_iter_number = 1, newton_iter_max
  ! Perform newton iteration by calculating the inverse of the jacobian matrix explicitly
  err2_old = err2

  ! Calculate the trial newton step
  st_step(1) = ( Z_t * (x_new(1)-x_step(1)) - R_t * (x_new(2)-x_step(2))) * inv_st_jac_det
  st_step(2) = (-Z_s * (x_new(1)-x_step(1)) + R_s * (x_new(2)-x_step(2))) * inv_st_jac_det

  ! Limit this step if it goes outside of the element
  dist = merge(1-st_new,st_new,st_step .gt. 0) ! dist = 1-s if step > 0, s if step < 0 (distance to 0 or 1)
  fact = maxval(abs(st_step)/max(dist,1d-30)) ! if fact>=1 we are on the boundary
  ! (it is the overshoot: i.e. how many times we overshoot the boundary with one st_step)

  if (fact .ge. 1.d0-1d-12) then
    st_new = st_new + st_step/fact
    i_elm_tmp = i_elm_new
    call coord_in_neighbour_gpu(node_list,element_list,i_elm_tmp,i_elm_new,st_new)
    if (i_elm_new .lt. 0) then
#if STELLARATOR_MODEL
      write(*,*) "ERROR: find_RZ is not implemented for stellarators"
      stop
#else

#ifdef USE_GPU_TEST
    write(*,*) "WARNING: impossible finding a neighbour, but find_RZ is not implemented for GPU"
#endif
    i_elm_new = 0

    if (ifail .ne. 0) i_elm_new = 0
#endif
    end if
    if (i_elm_new .eq. 0) then ! No element on that side, particle is lost
      i_elm_new = - i_elm_tmp ! Save position of particle
      ! Calculate new R and Z in x_new
      call try_interp_gpu(node_list,element_list,i_elm_tmp,st_new,p,x_new,R_s,R_t,Z_s,Z_t,inv_st_jac_det)
      ! Set new element-local coordinates for the point on the axis
      s_new = st_new(1)
      t_new = st_new(2)
      ifail = -1
      return
    end if

    call try_interp_gpu(node_list,element_list,i_elm_new,st_new,p,x_step,R_s,R_t,Z_s,Z_t,inv_st_jac_det)

  else ! if not on the boundary
    st_new = st_new + st_step
    call try_interp_gpu(node_list,element_list,i_elm_new,st_new,p,x_step,R_s,R_t,Z_s,Z_t,inv_st_jac_det)

  end if
  err2 = dot_product(x_step-x_new,x_step-x_new)
  s_new = st_new(1)
  t_new = st_new(2)

  if (err2 < element_tolerance) exit
enddo

if (ieee_is_nan(err2)) then
#if STELLARATOR_MODEL
  write(*,*) "ERROR: find_RZ is not implemented for stellarators"
  stop
#else

#ifdef USE_GPU_TEST
  write(*,*) "WARNING: NaN encountered after newton iteration ", newton_iter_number, " and element ", i_elm_new
#endif
  i_elm_new = 0

  return
#endif
endif
if (newton_iter_number .gt. newton_iter_max) then
#if STELLARATOR_MODEL
  write(*,*) "ERROR: find_RZ is not implemented for stellarators"
  stop
#else

#ifdef USE_GPU_TEST
  write(*,*) "WARNING: iteration for st did not converge after", newton_iter_max, " tries in element ", i_elm_new, &
              " using find_RZ, but find_RZ is not implemented for GPU"
#endif
  i_elm_new = 0

  return
#endif
endif
end subroutine find_RZ_nearby_gpu


subroutine find_RZ_nearby_gpu_inlined(node_list, element_list, R_old, Z_old, s_old, t_old, i_elm_old, &
        R_new, Z_new, s_new, t_new, i_elm_new, ifail, phi)
use data_structure
use mod_neighbours
use data_structure, only: type_node_list_gpu
use phys_module, only: mode, mode_coord
use mod_basisfunctions, only: basisfunctions
use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
implicit none
!> Input parameters
type (type_node_list_gpu),intent(in)    :: node_list
type (type_element_list), intent(in)    :: element_list
real*8,                   intent(in)    :: R_old, Z_old !< The old R,Z location
real*8,                   intent(in)    :: R_new, Z_new !< The new R,Z location
real*8,                   intent(in)    :: s_old, t_old !< The old st location (used to compute a guess)
integer,                  intent(in)    :: i_elm_old
real*8,                   intent(out)   :: s_new, t_new !< The found new coordinates
integer,                  intent(out)   :: i_elm_new
integer,                  intent(out)   :: ifail !< if ifail = -1 the position could not be found in the grid.
!< ifail > 0 indicates various other cases
real*8, optional,         intent(in)    :: phi

!> Accuracy defaults (tolerances are squared!, units of element size)
real*8,  parameter :: element_tolerance   = 1.d-24 !< Tolerance for finding a position inside an element
integer, parameter :: newton_iter_max     = 8 !< Number of iterations to try

!> Internal variables
real*8  :: p              
integer :: newton_iter_number, i_elm_tmp
real*8  :: inv_st_jac_det, R_s, R_t, Z_s, Z_t
real*8  :: st_step(2), x_step(2), x_tmp(2), st_new(2), x_new(2) ! x_step = (R,Z) of trial position
real*8  :: err2, err2_old, dist(2), fact

!> Inline subroutine variables
! try_interp_gpu
real*8 :: R_p, Z_p
real*8 :: jac
! intep_RZP_1_gpu
real*8  :: G(4,n_degrees), G_s(4,n_degrees), G_t(4,n_degrees)
real*8  :: HZ_coord(n_coord_tor), HZ_coord_p(n_coord_tor), HZ_coord_pp(n_coord_tor)
real*8  :: xx1, xx2, ss
real*8  :: R, Z
integer :: kv, iv, kf, i_tor
! coord_in_neighbour_gpu
integer :: q_from, q_to !< Quadrants
logical :: nb, co !< .true. if they have the same orientation (co or counter clockwise)
real*8  :: x
! neighbours_side_co_counter
integer :: i, n1(2), n2(2)

#ifdef USE_GPU
!$omp declare target
#endif

! Check if element is valid
if (i_elm_old .lt. 1 .or. i_elm_old .gt. element_list%n_elements) then
#if STELLARATOR_MODEL
  write(*,*) "ERROR: find_RZ is not implemented for stellarators"
  stop
#else

#ifdef USE_GPU_TEST
  write(*,*) "WARNING: element not valid, but find_RZ is not implemented for GPU"
#endif 
  i_elm_new = 0
  return
#endif
end if

if (present(phi)) then
  p = phi
else
#if STELLARATOR_MODEL
  write(*,*) "Toroidal angle phi must be defined for stellarator models"
  stop
#endif
  p = 0.0
endif

! Setup initial values
x_step = [R_old,Z_old] ! start at the current position
i_elm_new = i_elm_old ! start in the current element
st_new = [s_old,t_old] ! start at the old position
x_new = [R_new,Z_new]
! Find the jacobian at the current s and t position
call try_interp_gpu(node_list,element_list,i_elm_new,st_new,p,x_step,R_s,R_t,Z_s,Z_t,inv_st_jac_det)
err2 = dot_product(x_step-x_new,x_step-x_new)
ifail=0


! Newton iteration to find s and t in or out of this element
do newton_iter_number = 1, newton_iter_max
  ! Perform newton iteration by calculating the inverse of the jacobian matrix explicitly
  err2_old = err2

  ! Calculate the trial newton step
  st_step(1) = ( Z_t * (x_new(1)-x_step(1)) - R_t * (x_new(2)-x_step(2))) * inv_st_jac_det
  st_step(2) = (-Z_s * (x_new(1)-x_step(1)) + R_s * (x_new(2)-x_step(2))) * inv_st_jac_det

  ! Limit this step if it goes outside of the element
  dist = merge(1-st_new,st_new,st_step .gt. 0) ! dist = 1-s if step > 0, s if step < 0 (distance to 0 or 1)
  fact = maxval(abs(st_step)/max(dist,1d-30)) ! if fact>=1 we are on the boundary
  ! (it is the overshoot: i.e. how many times we overshoot the boundary with one st_step)

  if (fact .ge. 1.d0-1d-12) then
    st_new = st_new + st_step/fact
    i_elm_tmp = i_elm_new
    !call coord_in_neighbour_gpu(node_list,element_list,%VAL(i_elm_tmp),i_elm_new,st_new)
    !> coord_in_neighbour_gpu {
    if (st_new(1) .gt. st_new(2)) then ! 1 or 2
      if (1.d0 - st_new(1) .gt. st_new(2)) then
        q_from = 1
      else
        q_from = 2
      end if
    else
      if (1.d0 - st_new(1) .le. st_new(2)) then
        q_from = 3
      else
        q_from = 4
      end if
    end if

    i_elm_new = element_list%element(i_elm_tmp)%neighbours(q_from)
    if (i_elm_new .le. 0) return

    ! Check once more that they are neighbours and determine the orientation
    !call neighbours_side_co_counter(element_list,i_elm_tmp,i_elm_new,q_from,q_to,nb,co)
    !> neighbours_side_co_counter {
    co = .false. ! does not mean anything if not neighbours
    nb = .false.

    ! Find the side in i_elm_new pointing to i_elm_tmp
    ! If i_elm_new has no neighbour == i_elm_tmp use the last one that is 0 (i.e. the one on
    ! the axis itself.)
    q_to = 0
    do i=1,n_vertex_max
      if (element_list%element(i_elm_new)%neighbours(i) .eq. i_elm_tmp) then
        q_to = i
        exit ! loop
      end if
    end do
    if (q_to .gt. 0) then
      nb = .true.

      ! Determine node numbers of the sides
      ! node numbers are related to sides as node1=mod(side-1,4)+1, node2=mod(side,4)+1
      n1 = element_list%element(i_elm_tmp)%vertex(mod([q_from-1,q_from],4)+1)
      n2 = element_list%element(i_elm_new)%vertex(mod([q_to-1,q_to],4)+1)
      ! Find if match cross or straight (i.e. 1->2/2->1 or 1->1/2->2)
      ! We do not need to check the node position since there is only one option
      ! and one of the node numbers must match.
      if      (n1(1) .eq. n2(2)) then
        co = .true.
      else if (n1(2) .eq. n2(1)) then
        co = .true.
      else if (n1(1) .eq. n2(1)) then
        co = .false.
      else if (n1(2) .eq. n2(2)) then
        co = .false.
      end if
    end if
    !> } neighbours_side_co_counter

    if (.not. nb .or. q_to .eq. 0) then
      !write(*,"(A,i5,A,i5)") "ERROR IN element_list%element(", i_elm_tmp, ")%neighbours to ", i_elm_new
      i_elm_new = 0
      return
    end if

    ! x is the coordinate along the boundary from vertex i_side to i_side+1
    select case (q_from)
    case (1)
      x = st_new(1)
    case (2)
      x = st_new(2)
    case (3)
      x = 1.d0 - st_new(1)
    case (4)
      x = 1.d0 - st_new(2)
    end select

    if (co) x = 1.d0-x ! if the vectors along the boundary are antiparallel

    select case (q_to)
    case (1)
      st_new = [x,0.d0]
    case (2)
      st_new = [1.d0,x]
    case (3)
      st_new = [1.d0-x,1.d0]
    case (4)
      st_new = [0.d0,1.d0-x]
    end select
    !> } coord_in_neighbour_gpu

    if (i_elm_new .lt. 0) then
#if STELLARATOR_MODEL
      write(*,*) "ERROR: find_RZ is not implemented for stellarators"
      stop
#else

#ifdef USE_GPU_TEST
    write(*,*) "WARNING: impossible finding a neighbour, but find_RZ is not implemented for GPU"
#endif
    i_elm_new = 0

    if (ifail .ne. 0) i_elm_new = 0
#endif
    end if
    if (i_elm_new .eq. 0) then ! No element on that side, particle is lost
      i_elm_new = - i_elm_tmp ! Save position of particle
      ! Calculate new R and Z in x_new
      call try_interp_gpu(node_list,element_list,i_elm_tmp,st_new,p,x_new,R_s,R_t,Z_s,Z_t,inv_st_jac_det)
      ! Set new element-local coordinates for the point on the axis
      s_new = st_new(1)
      t_new = st_new(2)
      ifail = -1
      return
    end if

    !call try_interp_gpu(node_list,element_list,i_elm_new,st_new,p,x_step,R_s,R_t,Z_s,Z_t,inv_st_jac_det)
    !> try_interp_gpu {
    !call interp_RZP_1_gpu(node_list,element_list,i_elm_new,st_new(1),st_new(2),p,x_step(1),R_s,R_t,R_p,x_step(2),Z_s,Z_t,Z_p)
    !> interp_RZP_1_gpu {
    call basisfunctions(st_new(1),st_new(2),G,G_s,G_t)
    HZ_coord(1)      = 1.d0
    HZ_coord_p(1)    = 0.d0
    do i_tor=1,(n_coord_tor-1)/2
      HZ_coord(2*i_tor)        = + cos(mode_coord(2*i_tor)  *p)
      HZ_coord_p(2*i_tor)      = - float(mode_coord(2*i_tor))      * sin(mode_coord(2*i_tor)  *p)
      HZ_coord(2*i_tor+1)      = - sin(mode_coord(2*i_tor+1)*p)
      HZ_coord_p(2*i_tor+1)    = - float(mode_coord(2*i_tor+1))    * cos(mode_coord(2*i_tor+1)*p)
    enddo

    R = 0.d0; R_s = 0.d0; R_t = 0.d0; R_p = 0.d0
    Z = 0.d0; Z_s = 0.d0; Z_t = 0.d0; Z_p = 0.d0

    do kv = 1,n_vertex_max  ! 4 vertices
      iv = element_list%element(i_elm_new)%vertex(kv)  ! the node number
      do kf = 1, n_degrees       ! 4 basis functions
        do i_tor=1, n_coord_tor
          xx1 = node_list%node(iv)%x(i_tor,kf,1)
          xx2 = node_list%node(iv)%x(i_tor,kf,2)
          ss  = element_list%element(i_elm_new)%size(kv,kf)
          
          R    = R    + xx1 * ss * G(kv,kf)    * HZ_coord(i_tor)
          R_s  = R_s  + xx1 * ss * G_s(kv,kf)  * HZ_coord(i_tor)
          R_t  = R_t  + xx1 * ss * G_t(kv,kf)  * HZ_coord(i_tor)
          R_p  = R_p  + xx1 * ss * G(kv,kf)    * HZ_coord_p(i_tor)

          Z    = Z    + xx2 * ss * G(kv,kf)    * HZ_coord(i_tor)
          Z_s  = Z_s  + xx2 * ss * G_s(kv,kf)  * HZ_coord(i_tor)
          Z_t  = Z_t  + xx2 * ss * G_t(kv,kf)  * HZ_coord(i_tor)
          Z_p  = Z_p  + xx2 * ss * G(kv,kf)    * HZ_coord_p(i_tor)
        end do
      end do
    end do
    x_step(1) = R
    x_step(2) = Z
    !> } interp_RZP_1_gpu 
    ! Guard against the determinant being close to zero
    jac = R_s * Z_t - R_t * Z_s
    if (abs(jac) .lt. 1d-8) then
      inv_st_jac_det = sign(1d8, jac) 
    else
      inv_st_jac_det = 1.d0/(jac)
    end if
    !> } try_interp_gpu

  else ! if not on the boundary
    st_new = st_new + st_step
    !call try_interp_gpu(node_list,element_list,i_elm_new,st_new,p,x_step,R_s,R_t,Z_s,Z_t,inv_st_jac_det)
    !> try_interp_gpu {
    !call interp_RZP_1_gpu(node_list,element_list,i_elm_new,st_new(1),st_new(2),p,x_step(1),R_s,R_t,R_p,x_step(2),Z_s,Z_t,Z_p)
    !> interp_RZP_1_gpu {
    call basisfunctions(st_new(1),st_new(2),G,G_s,G_t)
    HZ_coord(1)      = 1.d0
    HZ_coord_p(1)    = 0.d0
    do i_tor=1,(n_coord_tor-1)/2
      HZ_coord(2*i_tor)        = + cos(mode_coord(2*i_tor)  *p)
      HZ_coord_p(2*i_tor)      = - float(mode_coord(2*i_tor))      * sin(mode_coord(2*i_tor)  *p)
      HZ_coord(2*i_tor+1)      = - sin(mode_coord(2*i_tor+1)*p)
      HZ_coord_p(2*i_tor+1)    = - float(mode_coord(2*i_tor+1))    * cos(mode_coord(2*i_tor+1)*p)
    enddo

    R = 0.d0; R_s = 0.d0; R_t = 0.d0; R_p = 0.d0
    Z = 0.d0; Z_s = 0.d0; Z_t = 0.d0; Z_p = 0.d0

    do kv = 1,n_vertex_max  ! 4 vertices
      iv = element_list%element(i_elm_new)%vertex(kv)  ! the node number
      do kf = 1, n_degrees       ! 4 basis functions
        do i_tor=1, n_coord_tor
          xx1 = node_list%node(iv)%x(i_tor,kf,1)
          xx2 = node_list%node(iv)%x(i_tor,kf,2)
          ss  = element_list%element(i_elm_new)%size(kv,kf)
          
          R    = R    + xx1 * ss * G(kv,kf)    * HZ_coord(i_tor)
          R_s  = R_s  + xx1 * ss * G_s(kv,kf)  * HZ_coord(i_tor)
          R_t  = R_t  + xx1 * ss * G_t(kv,kf)  * HZ_coord(i_tor)
          R_p  = R_p  + xx1 * ss * G(kv,kf)    * HZ_coord_p(i_tor)

          Z    = Z    + xx2 * ss * G(kv,kf)    * HZ_coord(i_tor)
          Z_s  = Z_s  + xx2 * ss * G_s(kv,kf)  * HZ_coord(i_tor)
          Z_t  = Z_t  + xx2 * ss * G_t(kv,kf)  * HZ_coord(i_tor)
          Z_p  = Z_p  + xx2 * ss * G(kv,kf)    * HZ_coord_p(i_tor)
        end do
      end do
    end do
    x_step(1) = R
    x_step(2) = Z
    !> } interp_RZP_1_gpu 
    ! Guard against the determinant being close to zero
    jac = R_s * Z_t - R_t * Z_s
    if (abs(jac) .lt. 1d-8) then
      inv_st_jac_det = sign(1d8, jac) 
    else
      inv_st_jac_det = 1.d0/(jac)
    end if
    !> } try_interp_gpu
  end if
  err2 = dot_product(x_step-x_new,x_step-x_new)
  s_new = st_new(1)
  t_new = st_new(2)

  if (err2 < element_tolerance) exit
enddo


if (ieee_is_nan(err2)) then
#if STELLARATOR_MODEL
  write(*,*) "ERROR: find_RZ is not implemented for stellarators"
  stop
#else

#ifdef USE_GPU_TEST
  write(*,*) "WARNING: NaN encountered after newton iteration ", newton_iter_number, " and element ", i_elm_new
#endif
  i_elm_new = 0

  return
#endif
endif
if (newton_iter_number .gt. newton_iter_max) then
#if STELLARATOR_MODEL
  write(*,*) "ERROR: find_RZ is not implemented for stellarators"
  stop
#else

#ifdef USE_GPU_TEST
  write(*,*) "WARNING: iteration for st did not converge after", newton_iter_max, " tries in element ", i_elm_new, &
              " using find_RZ, but find_RZ is not implemented for GPU"
#endif
  i_elm_new = 0

  return
#endif
endif
end subroutine find_RZ_nearby_gpu_inlined


!> Auxiliary subroutine for find_RZ_nearby_gpu
pure subroutine try_interp_gpu(node_list,element_list,i_elm,st,p,x,R_s,R_t,Z_s,Z_t,inv_st_jac_det)
use data_structure
use mod_interp
use data_structure, only: type_node_list_gpu
implicit none
!> Input parameters
type (type_node_list_gpu),intent(in) :: node_list
type (type_element_list), intent(in)    :: element_list
real*8,                   intent(in)    :: st(2), p
integer,                  intent(in)    :: i_elm
real*8,                   intent(out)   :: x(2), R_s, R_t, Z_s, Z_t, inv_st_jac_det

real*8 :: R_p, Z_p
real*8 :: jac
#ifdef USE_GPU
!$omp declare target
#endif

call interp_RZP_1_gpu(node_list,element_list,i_elm,st(1),st(2),p,x(1),R_s,R_t,R_p,x(2),Z_s,Z_t,Z_p)
! Guard against the determinant being close to zero
jac = R_s * Z_t - R_t * Z_s
if (abs(jac) .lt. 1d-8) then
  inv_st_jac_det = sign(1d8, jac) 
else
  inv_st_jac_det = 1.d0/(jac)
end if
end subroutine try_interp_gpu
end module mod_find_rz_nearby