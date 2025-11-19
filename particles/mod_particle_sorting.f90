module mod_particle_sorting
use mod_particle_types

implicit none
public sort_particles

!> enumerator of the particle type 
enum, bind(C)
enumerator :: particle_global_sort = 1, particle_simd_lane_sort = 2
endenum

contains

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
  double precision :: start_time, end_time, tot_time(3)
  
  start_time = MPI_WTIME()

  select case (method)
  case (particle_simd_lane_sort)
    write(*,*) "INFO: Using SIMD-lane based particle sorting."
    call i_elm_sort_simd_lane(sorted_indices, sim%groups(group_num)%particles, alive_particle_count)
  case (particle_global_sort)
    write(*,*) "INFO: Using global particle sorting."
    call i_elm_sort_global(sorted_indices, sim%groups(group_num)%particles, alive_particle_count)
  case default
    write(*,*) "ERROR: Unsupported particle sorting method: ", method
    stop
  end select

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
    if (particles(i)%i_elm >= 0) then
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

!> A simple quicksort implementation to sort indices based on field values
recursive subroutine sort_indices(field_values, indices)
  implicit none
  integer, intent(in) :: field_values(:)
  integer, intent(inout) :: indices(:)

  integer :: pivot_index, pivot_value
  integer :: i, j
  integer, allocatable :: low_indices(:), equal_indices(:), high_indices(:)
  integer :: low_count, equal_count, high_count

  if (size(indices) > 1) then
    ! Choose pivot (simple choice: middle element)
    pivot_index = indices(size(indices) / 2)
    pivot_value = field_values(pivot_index)

    ! Partitioning
    low_count = count(field_values(indices) < pivot_value)
    equal_count = count(field_values(indices) == pivot_value)
    high_count = count(field_values(indices) > pivot_value)

    allocate(low_indices(low_count))
    allocate(equal_indices(equal_count))
    allocate(high_indices(high_count))

    low_count = 0
    equal_count = 0
    high_count = 0
    do i = 1, size(indices)
      if (field_values(indices(i)) < pivot_value) then
        low_count = low_count + 1
        low_indices(low_count) = indices(i)
      else if (field_values(indices(i)) == pivot_value) then
        equal_count = equal_count + 1
        equal_indices(equal_count) = indices(i)
      else if (field_values(indices(i)) > pivot_value) then
        high_count = high_count + 1
        high_indices(high_count) = indices(i)
      end if
    end do

    ! Recursively sort the partitions
    call sort_indices(field_values, low_indices)
    call sort_indices(field_values, high_indices)

    ! Combine the results
    j = 0
    do i = 1, low_count
      j = j + 1
      indices(j) = low_indices(i)
    end do
    do i = 1, equal_count
      j = j + 1
      indices(j) = equal_indices(i)
    end do
    do i = 1, high_count
      j = j + 1
      indices(j) = high_indices(i)
    end do

    deallocate(low_indices)
    deallocate(equal_indices)
    deallocate(high_indices)
  end if
end subroutine sort_indices


end module mod_particle_sorting