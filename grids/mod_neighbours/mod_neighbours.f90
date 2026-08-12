module mod_neighbours
  use data_structure
  use mod_element_rtree, only: get_vertex_pos_in_rtree_plane
  implicit none
  private

  !> Interface only -- the body is in C++ (grids/mod_neighbours/neighbours_shim.cpp).
  interface
    subroutine jgx_host_coord_in_neighbour(el_base, n_elements, i_from0, i_to, st, bad_i_to) &
        bind(C, name="jgx_host_coord_in_neighbour")
      use, intrinsic :: iso_c_binding, only: c_double, c_int32_t, c_ptr
      implicit none
      type(c_ptr),        value, intent(in)  :: el_base
      integer(c_int32_t), value, intent(in)  :: n_elements, i_from0
      integer(c_int32_t),        intent(out) :: i_to, bad_i_to
      real(c_double),          intent(inout) :: st(2)
    end subroutine
  end interface

  public :: neighbours, update_neighbours
  public :: coord_in_neighbour
contains
logical function neighbours(node_list,elm1,elm2,inb1,inb2)
!-------------------------------------------------------
! function to check if the two elements elm1 and elm2
! are neighbours. Two elements are neighbours if they
! share a side, i.e. if two nodes are the same.
! inb1 -> the index of the shared neighbour of elm1
! inb2 -> the index of the shared neighbour of elm2
!   note : does not work for unequal sized neighbours
!-------------------------------------------------------
implicit none

type (type_node_list), intent(in) :: node_list
type (type_element), intent(in)   :: elm1, elm2
integer, intent(out)              :: inb1, inb2

integer :: n1(2), n2(2) ! nodes of side i of elm1 or of side j of elm2
integer :: i, j
neighbours = .false.

! First test by node number
do i=1,4 ! Loop over sides of element 1
  n1 = elm1%vertex(mod([i-1,i],4)+1)
  do j=1,4 ! Loop over sides of element 2
    ! node numbers are related to sides as node1=mod(side-1,4)+1, node2=mod(side,4)+1
    n2 = elm2%vertex(mod([j-1,j],4)+1)
    ! If match cross or straight (i.e. 1->2/2->1 or 1->1/2->2)
    ! Four different cases
    if            (node_same_pos(node_list, n1(1), n2(2))) then
      neighbours = node_same_pos(node_list, n1(2), n2(1))
    else if       (node_same_pos(node_list, n1(2), n2(1))) then
      neighbours = node_same_pos(node_list, n1(1), n2(2))
    else if       (node_same_pos(node_list, n1(1), n2(1))) then
      neighbours = node_same_pos(node_list, n1(2), n2(2))
    else if       (node_same_pos(node_list, n1(2), n2(2))) then
      neighbours = node_same_pos(node_list, n1(1), n2(1))
    end if
    if (neighbours) then
      inb1 = i
      inb2 = j
      return
    end if
  enddo
enddo
end function neighbours

pure function node_same_pos(node_list, i1, i2) result(same)
  type (type_node_list), intent(in) :: node_list
  integer, intent(in) :: i1, i2
  logical :: same
  real*8, parameter :: tol=1.d-8
  real*8 :: pos1(2), pos2(2)
  if (i1 .eq. i2) then
    same = .true.
  else
    pos1 = get_vertex_pos_in_rtree_plane(node_list%node(i1)%x(1:n_coord_tor,1,1:2))
    pos2 = get_vertex_pos_in_rtree_plane(node_list%node(i2)%x(1:n_coord_tor,1,1:2))
    if (norm2(pos1-pos2) .lt. tol) then
      same = .true.
    else
      same = .false.
    end if
  end if
end function node_same_pos




!> Convert s,t coordinates on the boundary of element i_from into s,t coordinates
!> on the boundary of i_to.
!> Element side and node numbering:
!> 4______3 1
!> |\  3 /|
!> | \  / |
!> |  \/  | t
!> |4 /\ 2|
!> | /  \ |
!> |/  1 \| 0
!> 1^^^^^^2 
!>
!> 0   s  1
!>
!> On the boundary between elements the following is guaranteed:
!> * One of the local coordinates (s,t) is either 0 or 1 (1: t=0, 2: s=1, 3: t=1, 4: s=0)
!> * The other coordinates x_i, x_j (elements i and j) are related: |dx_i/dx_j| = 1
!> Facade only -- the body is in C++ (grids/mod_neighbours/neighbours_shim.cpp), together with
!> the private neighbours_side_co_counter it used.
subroutine coord_in_neighbour(node_list,element_list,i_from,i_to,st)
  use, intrinsic :: iso_c_binding, only: c_int32_t, c_ptr, c_loc
  type (type_node_list), intent(in)            :: node_list !< unused, kept for the callers
  type (type_element_list), target, intent(in) :: element_list
  integer, intent(in)                  :: i_from
  integer, intent(out)                 :: i_to !<  >0 if a neighbour exists on that side, -1 if search
  real*8, intent(inout)                :: st(2)

  integer(c_int32_t) :: c_i_to, bad_i_to

  call jgx_host_coord_in_neighbour(c_loc(element_list%element(1)),          &
                                   int(element_list%n_elements, c_int32_t), &
                                   int(i_from - 1, c_int32_t), c_i_to, st, bad_i_to)

  i_to = int(c_i_to)

  ! The kernel cannot print, so it hands the offending element back instead.
  if (bad_i_to /= 0) write(*,"(A,i5,A,i5)") &
    "ERROR IN element_list%element(", i_from, ")%neighbours to ", bad_i_to
end subroutine coord_in_neighbour



subroutine update_neighbours(node_list, element_list, force_rtree_initialize)
#ifdef USE_NO_TREE  
use mod_no_tree
#elif USE_QUADTREE
use mod_quadtree
#else
use mod_element_rtree, only: rtree_initialized, populate_element_rtree, nearby_elements  
#endif
implicit none

type (type_node_list), intent(in)       :: node_list
type (type_element_list), intent(inout) :: element_list
logical, intent(in), optional           :: force_rtree_initialize !< default false

type (type_element)      :: elm_i, elm_j
integer                  :: inb_i, inb_j, i, j, k, iv1, iv2
integer                  :: i_elm, j_elm, i_node1, i_node2
real*8                   :: s_i, t_i, R_i, Rs_i, Rt_i, Rst_i, Rss_i, Rtt_i, Z_i, Zs_i, Zt_i, Zst_i,Zss_i,Ztt_i
real*8                   :: s_j, t_j, R_j, Rs_j, Rt_j, Rst_j, Rss_j, Rtt_j, Z_j, Zs_j, Zt_j, Zst_j,Zss_j,Ztt_j
real*8                   :: pos1(2), pos2(2)
integer, dimension(:), allocatable :: i_nearby
real*8 :: t0, t1, t2

call cpu_time(t0)

#ifdef USE_NO_TREE
call no_tree_init(node_list, element_list)
#elif USE_QUADTREE
if (present(force_rtree_initialize)) then
  if (force_rtree_initialize) then
    call quadtree_free(quadtree)
    call quadtree_init(node_list, element_list)
  endif
else if (quadtree%depth .eq. 0) then
call quadtree_init(node_list, element_list)
end if
#else 
! Be careful here. If the grid changes the information will be incorrect and you
! need to manually call populate_element_rtree
if (present(force_rtree_initialize)) then
  if (force_rtree_initialize) call populate_element_rtree(node_list, element_list)
else if (.not. rtree_initialized) then
  call populate_element_rtree(node_list, element_list)
end if
#endif

call cpu_time(t1)

!$omp parallel default(none) private(i,k,j,i_node1,i_node2,inb_i,inb_j,i_nearby, iv1, iv2, pos1, pos2) shared(element_list,node_list)
!$omp do
do i=1, element_list%n_elements
  element_list%element(i)%neighbours(:) = 0
#ifdef USE_NO_TREE
  call nearby_elements_no_tree(node_list, element_list, i, i_nearby)
#elif USE_QUADTREE
  call nearby_elements_quadtree(node_list, element_list, i, i_nearby)
#else
  call nearby_elements(node_list, element_list, i, i_nearby)
#endif
  do k=1,size(i_nearby,1)
    j = i_nearby(k)
    if (i .eq. j) cycle

    if  (neighbours(node_list, element_list%element(i), element_list%element(j),inb_i,inb_j)) then
      element_list%element(i)%neighbours(inb_i) = j
    endif
  enddo

! check if all neighbours have been found
  do j=1,4
    if (element_list%element(i)%neighbours(j) .le. 0) then
      iv1 = element_list%element(i)%vertex(j)
      iv2 = element_list%element(i)%vertex(mod(j,4)+1)
      pos1 = get_vertex_pos_in_rtree_plane(node_list%node(iv1)%x(1:n_coord_tor,1,1:2))
      pos2 = get_vertex_pos_in_rtree_plane(node_list%node(iv2)%x(1:n_coord_tor,1,1:2))
      if ((node_list%node(iv1)%boundary .ne. 0) .and. (node_list%node(iv2)%boundary .ne.0) ) then
        ;     ! this is a boundary, should not have a neighbour
      elseif ((sqrt( (pos1(1) - pos2(1))**2 &
                    +(pos1(2) - pos2(2))**2) .lt. 1d-8) ) then
        ;     ! these two corners of the element are coinciding, thus no neighbour
      else
        write(*,*) 'ERROR neighbours : ',i,j,element_list%element(i)%neighbours(j)
        write(*,*) 'boundary check ',iv1,iv2,node_list%node(iv1)%boundary,node_list%node(iv2)%boundary
      endif
    endif
  enddo
enddo
!$omp end do

! Force search for element crossings over the axis
!$omp do
do i=1, element_list%n_elements
  do j=1, n_vertex_max
    i_node1 = element_list%element(i)%vertex(j)
    i_node2 = element_list%element(i)%vertex(mod(j,4)+1)

    ! If the two nodes for this side are in the same position (degenerate element)
    if (norm2(node_list%node(i_node1)%x(1,1,1:2) - node_list%node(i_node2)%x(1,1,1:2)) .lt. 1d-8) then
      ! Force search for the opposite side
      element_list%element(i)%neighbours(j) = -1
      do k=1,n_vertex_max
        ! If any of the other sides was not found force search there as well
        if(element_list%element(i)%neighbours(k).eq.0) then
           element_list%element(i)%neighbours(k) = -1
        endif
      enddo
    endif
  enddo
enddo
!$omp end do
!$omp end parallel
call cpu_time(t2)

write(*,'(A,e12.4)') 'neighbours cpu_time, init tree      : ',t1-t0 
write(*,'(A,e12.4)') 'neighbours cpu_time, init neighbours: ',t2-t1 
end subroutine update_neighbours
end module mod_neighbours
