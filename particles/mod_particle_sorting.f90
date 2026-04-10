!> Module for sorting particles based on their location
!> Currently only contains an implementation for sorting particles into their elements
module mod_particle_sorting
  use mod_particle_types
  use phys_module, only: use_manual_random_seed
  use mpi
  
  !$ use omp_lib

  implicit none
   
  private
  public :: indices_in_elm, sort_particles_in_elm
  public sort_particles, particle_global_sort, particle_simd_lane_sort

  !> enumerator of the particle type 
  enum, bind(C)
  enumerator :: particle_global_sort = 1, particle_simd_lane_sort = 2
  endenum

  !> an object containing the array of particle indices of particles in this element 
  !> s.t. an array of these objects can store all particle indices as array per element
  type :: indices_in_elm
    integer, dimension(:), allocatable :: pa_ind
  end type

contains

!> Deterministically sort the particle indices (of the particle array pa given as input)
!> into arrays by their element number
!> Also return the number of particles in each element (as that is a useful intermediate result)
subroutine sort_particles_in_elm(pa, n_elm, sorted_ind_arr, pa_in_elm_arr)
  implicit none

  class(particle_base),  dimension(:), allocatable, intent(in)  :: pa             !< particle array to be sorted (n_pa)
  integer,                                          intent(in)  :: n_elm          !< number of elements in the grid
  type(indices_in_elm),  dimension(:), allocatable, intent(out) :: sorted_ind_arr !< object containing all particle indices as arrays per element number (n_elm)
  integer,               dimension(:), allocatable, intent(out) :: pa_in_elm_arr  !< number of particles in each element (n_elm)
  
  ! arrays used for determining sorted_ind_arr
  integer, dimension(:),       allocatable :: pa_elm_arr        !< precalculated pa(:)%i_elm for faster masking over i_elm  (n_pa)
  integer, dimension(:,:),     allocatable :: pa_in_thread_arr  !< number of particles in the thread for this element (n_elm, n_thread)
  integer, dimension(:,:),     allocatable :: offset_thread_arr !< offset of this thread's contribution to sorted_ind_arr for this element (n_elm, n_thread)
  integer, dimension(:,:),     allocatable :: i_loc_thread_arr  !< ith particle index insertion into sorted_ind_arr for this element by this thread (n_elm, n_thread)

  integer :: i, i_elm, i_loc
  integer :: i_thread, n_thread

  ! --- start of code
  
  !initialisation
  i_thread = 1 !default if not using OMP
  n_thread = 1
  !$ n_thread = omp_get_max_threads()
  
  allocate(pa_elm_arr(size(pa)),source=0)
  allocate(pa_in_elm_arr(n_elm),source=0)
  allocate(pa_in_thread_arr(n_elm,n_thread),source=0)
  
  !find out how many particles per element and thread
  !$omp parallel do default(none)  &
  !$omp shared(pa,pa_elm_arr,pa_in_thread_arr)      &
  !$omp private(i_elm, i_thread)   &
  !$omp schedule(static,100)
  do i=1,size(pa)
    !$ i_thread = omp_get_thread_num()+1
    i_elm = pa(i)%i_elm
    if(i_elm < 1) cycle
    pa_elm_arr(i) = i_elm
    pa_in_thread_arr(i_elm,i_thread) = pa_in_thread_arr(i_elm,i_thread) + 1
  end do
  !$omp end parallel do

  !find out how many particles per element total, and getting the offset of each thread 
  !so that each thread writes to its own part of the sorted_ind_arr(i_elm)%pa_ind(:) array later on
  if(allocated(offset_thread_arr)) deallocate(offset_thread_arr)
  allocate(offset_thread_arr(n_elm,n_thread))
  !$omp parallel do default(none)  &
  !$omp shared(pa_in_elm_arr,pa_in_thread_arr,offset_thread_arr,n_thread,n_elm) &
  !$omp private(i_thread)
  do i_elm=1,n_elm
    pa_in_elm_arr(i_elm) = sum(pa_in_thread_arr(i_elm,:))
    
    !the offset must be the sum over the particles in that element of lower thread numbers, we can determine this iteratively from the previous offset
    offset_thread_arr(i_elm,1) = 0
    do i_thread=2,n_thread
      offset_thread_arr(i_elm,i_thread) = offset_thread_arr(i_elm,i_thread - 1) + pa_in_thread_arr(i_elm,i_thread - 1)
    end do
  end do
  !$omp end parallel do

  !allocating sorted_ind_arr object
  if(.not. allocated(sorted_ind_arr)) allocate(sorted_ind_arr(n_elm))
  !$omp parallel do default(none) &
  !$omp shared(pa_in_elm_arr, sorted_ind_arr, n_elm)
  do i_elm=1,n_elm
    if(allocated(sorted_ind_arr(i_elm)%pa_ind)) deallocate(sorted_ind_arr(i_elm)%pa_ind) ! this can be done here as we know size(sorted_ind_arr)=n_elm is fixed
    allocate(sorted_ind_arr(i_elm)%pa_ind(pa_in_elm_arr(i_elm)))
  end do
  !$omp end parallel do

  !filling sorted_ind_arr object
  if(allocated(i_loc_thread_arr)) deallocate(i_loc_thread_arr)
  allocate(i_loc_thread_arr(n_elm,n_thread),source=1) !index 1
  !$omp parallel do default(none)                     &
  !$omp shared(pa_elm_arr, offset_thread_arr, i_loc_thread_arr, sorted_ind_arr) &
  !$omp private(i_elm, i_loc, i_thread) &
  !$omp schedule(static,100)
  do i=1,size(pa_elm_arr)
    !$ i_thread = omp_get_thread_num()+1
    i_elm = pa_elm_arr(i)
    if(i_elm < 1) cycle
    !find out where to insert this particles' index in the element array (i_loc) from the thread offset + the number already filled in by this thread
    i_loc = offset_thread_arr(i_elm,i_thread) + i_loc_thread_arr(i_elm,i_thread) 
    !update the number of indices filled in by this thread for this element
    i_loc_thread_arr(i_elm,i_thread) = i_loc_thread_arr(i_elm,i_thread) + 1
    !insert the particle index in the element array (at the right spot i_loc)
    sorted_ind_arr(i_elm)%pa_ind(i_loc) = i
  end do
  !$omp end parallel do

end subroutine sort_particles_in_elm

!> Sort particles using the given method. In addition to that,
!> it returns the number of alive particles (i_elm>0)
subroutine sort_particles(sim, group_num, method, alive_particle_count)
  use mod_event, only: mpi_minmeanmax
  use mpi
  use mod_particle_sim
  implicit none

  ! Arguments
  class(particle_sim), target, intent(inout)                :: sim
  integer, intent(in) :: group_num
  integer, intent(in) :: method
  integer, optional, intent(out) :: alive_particle_count

  ! Local variables
  integer, allocatable :: sorted_indices(:)
  integer :: local_alive_count
  double precision :: start_time, end_time, tot_time(3)

  start_time = MPI_WTIME()

  select case (method)
  case (particle_simd_lane_sort)
    if (sim%my_id .eq. 0) then
      write(*,*) "INFO: Using SIMD-lane based particle sorting."
    end if
    call i_elm_sort_simd_lane(sorted_indices, sim%groups(group_num)%particles, local_alive_count)
  case (particle_global_sort)
    if (sim%my_id .eq. 0) then
      write(*,*) "INFO: Using global particle sorting."
    end if
    call i_elm_sort_global(sorted_indices, sim%groups(group_num)%particles, local_alive_count)
  case default
    if (sim%my_id .eq. 0) then
      write(*,*) "ERROR: Unsupported particle sorting method: ", method
    end if
    stop
  end select

  if (present(alive_particle_count)) alive_particle_count = local_alive_count

  ! Reorder particles based on sorted indices
  call reorder_particles(sim%groups(group_num)%particles, sorted_indices)

  end_time = MPI_WTIME()
  tot_time = mpi_minmeanmax(end_time-start_time)
  if (sim%my_id .eq. 0) then
      write(*,"(A,3f10.3,A)") , "Time taken for ordering (min/mean/max): ", tot_time, " seconds"
  endif
end subroutine sort_particles

!> Sort particles by their i_elm field by SIMD-lane grouping the positive i_elm. In this way,
!> particles in a SIMD lane will have strictly increasing i_elm values. This strategy might 
!> improve memory access patterns on GPUs.
subroutine i_elm_sort_simd_lane(sorted_indices, particles, last_positive_idx)
  implicit none

  ! Arguments
  integer, allocatable, intent(out) :: sorted_indices(:)
  class(particle_base), intent(in) :: particles(:)
  integer, intent(out) :: last_positive_idx

  integer :: SIMD_LANE_SIZE
  ! Use a sentinel value to sort particles with negative i_elm to the end
  integer, parameter :: NEGATIVE_I_ELM_SENTINEL = HUGE(0)

  ! Local variables
  integer :: n_particles, i, n_pos
  integer, allocatable :: temp_indices(:)
  integer, allocatable :: sort_keys(:)

  integer :: num_placed, output_idx
  integer :: group_member_count
  integer :: last_val_in_group
  integer, allocatable :: next_available(:), prev_available(:)
  integer :: head_of_list, current_node, node_to_remove

  n_particles = size(particles, 1)
  SIMD_LANE_SIZE = 32
  if (.not. allocated(sorted_indices)) allocate(sorted_indices(n_particles))

  if (n_particles == 0) then
    return
  end if

  ! --- STEP 1: UNIFIED GLOBAL SORT ---
  ! Create sort keys for all particles and sort them in one pass. This avoids
  ! manual partitioning and multiple allocations.
  allocate(temp_indices(n_particles), sort_keys(n_particles))
  n_pos = 0
  do i = 1, n_particles
    temp_indices(i) = i
    if (particles(i)%i_elm > 0) then
      sort_keys(i) = particles(i)%i_elm
      n_pos = n_pos + 1
    else
      sort_keys(i) = NEGATIVE_I_ELM_SENTINEL
    end if
  end do

  ! This external call sorts `temp_indices` based on `sort_keys`.
  ! The first `n_pos` indices will be the sorted positive particles.
  call sort_indices(sort_keys, temp_indices)
  deallocate(sort_keys)


  ! --- STEP 2: GROUP POSITIVE PARTICLES ---
  ! A linked list of available indices is used to avoid the slow O(N^2) scan.
  if (n_pos > 0) then
    ! Create a doubly linked list of available indices (from 1 to n_pos)
    allocate(next_available(n_pos), prev_available(n_pos))
    do i = 1, n_pos
      next_available(i) = i + 1
      prev_available(i) = i - 1
    end do
    next_available(n_pos) = 0 ! End of list marker
    prev_available(1) = 0   ! Start of list marker
    head_of_list = 1

    num_placed = 0
    output_idx = 1

    do while (num_placed < n_pos)
      ! Start a new group with the first available particle in the list
      current_node = head_of_list
      sorted_indices(output_idx) = temp_indices(current_node)
      last_val_in_group = particles(sorted_indices(output_idx))%i_elm
      group_member_count = 1

      ! Remove this node from the linked list
      head_of_list = next_available(current_node)
      if (head_of_list /= 0) then
        prev_available(head_of_list) = 0
      end if

      ! Scan the REST of the available particles for this group
      do while (group_member_count < SIMD_LANE_SIZE .and. current_node /= 0)
        current_node = next_available(current_node)
        if (current_node == 0) exit ! Reached end of the list

        if (particles(temp_indices(current_node))%i_elm > last_val_in_group) then
          ! Add this particle to the current group
          group_member_count = group_member_count + 1
          sorted_indices(output_idx + group_member_count - 1) = temp_indices(current_node)
          last_val_in_group = particles(temp_indices(current_node))%i_elm

          ! This particle is now used, so remove it from the list
          node_to_remove = current_node
          if (next_available(node_to_remove) /= 0) then
            prev_available(next_available(node_to_remove)) = prev_available(node_to_remove)
          end if
          if (prev_available(node_to_remove) /= 0) then
            next_available(prev_available(node_to_remove)) = next_available(node_to_remove)
          else
            ! This case should not be hit as we already advanced head_of_list
            head_of_list = next_available(node_to_remove)
          end if
        end if
      end do

      num_placed = num_placed + group_member_count
      output_idx = output_idx + group_member_count
    end do
    deallocate(next_available, prev_available)
  end if ! (end if n_pos > 0)


  ! --- STEP 3: APPEND NEGATIVE PARTICLES ---
  ! The sort automatically placed the negative-i_elm particles at the end of temp_indices.
  last_positive_idx = n_particles
  if (n_pos < n_particles) then
    sorted_indices(n_pos + 1:n_particles) = temp_indices(n_pos + 1:n_particles)
    last_positive_idx = n_pos
  end if

  ! --- CLEANUP ---
  deallocate(temp_indices)
end subroutine i_elm_sort_simd_lane

!> Sort particles globally by their i_elm field, appending negative i_elm particles at the end.
subroutine i_elm_sort_global(sorted_indices, particles, last_positive_idx)
  implicit none
  integer, allocatable, intent(out) :: sorted_indices(:)
  class(particle_base), intent(in) :: particles(:)
  integer, intent(out) :: last_positive_idx

  integer :: i, j, n_particles
  integer :: positive_count, negative_count
  integer, allocatable :: field_values(:)
  integer, allocatable :: positive_indices(:), negative_indices(:)
  n_particles = size(particles, 1)

  ! For efficiency, copy the field values into a simple integer array
  allocate(field_values(n_particles))
  do i = 1, n_particles
    field_values(i) = particles(i)%i_elm
  end do
  
  positive_count = count(field_values > 0)
  last_positive_idx = positive_count
  negative_count = n_particles - positive_count

  if (.not. allocated(sorted_indices)) allocate(sorted_indices(n_particles))

  ! Allocate temporary arrays for the two groups
  allocate(positive_indices(positive_count))
  allocate(negative_indices(negative_count))

  ! Partition indices 
  positive_count = 0 ! Reuse as an index counter
  negative_count = 0 ! Reuse as an index counter
  do i = 1, n_particles
    if (field_values(i) > 0) then
      positive_count = positive_count + 1
      positive_indices(positive_count) = i
    else if (field_values(i) <= 0) then
      negative_count = negative_count + 1
      negative_indices(negative_count) = i
    end if
  end do

  call sort_indices(field_values, positive_indices)

  ! Combine sorted positives first, then unsorted negatives
  j = 0
  ! Append sorted positive indices
  do i = 1, size(positive_indices)
    j = j + 1
    sorted_indices(j) = positive_indices(i)
  end do
  ! Append unsorted negative indices
  do i = 1, size(negative_indices)
    j = j + 1
    sorted_indices(j) = negative_indices(i)
  end do

  ! Clean up memory
  deallocate(field_values)
  deallocate(positive_indices)
  deallocate(negative_indices)
end subroutine i_elm_sort_global

!> Counting sort: stably sort `indices` by `field_values(indices(i))`.
!> Keys must be positive integers or HUGE(0) (sentinel for dead particles).
!> Sentinel-keyed entries are placed at the end; all others in ascending order.
!>
!> Replaces the former recursive quicksort which caused a stack overflow on
!> large particle counts: i_elm keys are bounded integers (1..n_elements) with
!> ~n_particles/n_elements duplicates per key, leading to O(n_elements)
!> recursion depth and O(n_elements * frame_size) stack usage.
subroutine sort_indices(field_values, indices)
  implicit none
  integer, intent(in)    :: field_values(:)
  integer, intent(inout) :: indices(:)

  integer :: i, key, n, key_max, pos
  integer, allocatable :: starts(:), indices_copy(:)

  n = size(indices)
  if (n <= 1) return

  ! --- Pass 1: find maximum non-sentinel key ---
  key_max = 0
  do i = 1, n
    key = field_values(indices(i))
    if (key /= HUGE(0) .and. key > key_max) key_max = key
  end do

  ! starts(k) will hold the next write position for key k.
  ! Bucket 0 is reserved for HUGE(0) sentinel (dead particles), placed last.
  ! Buckets 1..key_max are for live particles sorted by i_elm.
  allocate(starts(0:key_max), source=0)

  ! --- Pass 2: count occurrences per key ---
  do i = 1, n
    key = field_values(indices(i))
    if (key == HUGE(0)) then
      starts(0) = starts(0) + 1
    else
      starts(key) = starts(key) + 1
    end if
  end do

  ! --- Convert counts to exclusive start positions (1-based output array) ---
  ! Live keys occupy positions 1 .. sum(starts(1:key_max))
  ! Sentinel bucket follows immediately after.
  pos = 1
  do i = 1, key_max
    key = starts(i)        ! save count
    starts(i) = pos        ! overwrite with start position
    pos = pos + key
  end do
  starts(0) = pos          ! sentinel bucket starts after all live entries

  ! --- Pass 3: stable scatter ---
  allocate(indices_copy(n))
  do i = 1, n
    key = field_values(indices(i))
    if (key == HUGE(0)) then
      indices_copy(starts(0)) = indices(i)
      starts(0) = starts(0) + 1
    else
      indices_copy(starts(key)) = indices(i)
      starts(key) = starts(key) + 1
    end if
  end do

  indices = indices_copy

  deallocate(starts, indices_copy)
end subroutine sort_indices


end module mod_particle_sorting
