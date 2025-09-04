module mod_particle_sorting
use mod_particle_types

implicit none
private
public sort_indices_by_i_elm_warped, sort_indices_by_i_elm, sort_indices_by_i_elm_batched, sort_indices_by_i_elm_batched_warped
public sort_indices_by_i_elm_bucket

contains
! This subroutine reorders particle indices into groups, in a highly optimized manner.
! The process is:
! 1. Perform a single global sort of all particle indices. A large sentinel value
!    is used as the sort key for particles with negative 'i_elm' to move them to the end.
! 2. Build groups from the sorted positive particles. This is done using a linked
!    list of available indices to avoid a slow O(N^2) search, reducing the
!    grouping complexity to O(N).
! 3. The final 'sorted_indices' array contains these groups concatenated, followed by
!    the indices of particles with negative 'i_elm'.
subroutine sort_indices_by_i_elm_warped(sorted_indices, particles)
  implicit none

  ! Arguments
  integer, allocatable, intent(out) :: sorted_indices(:)
  type(particle_kinetic_relativistic), intent(in) :: particles(:)
  ! Local constants
  integer :: GROUP_SIZE
  ! Use a sentinel value to sort particles with negative i_elm to the end
  integer, parameter :: NEGATIVE_I_ELM_SENTINEL = HUGE(0)

  ! Local variables
  integer :: n_particles, i, n_pos
  integer, allocatable :: temp_indices(:)
  integer, allocatable :: sort_keys(:)

  ! For the efficient O(N) grouping algorithm
  integer :: num_placed, output_idx
  integer :: group_member_count
  integer :: last_val_in_group
  integer, allocatable :: next_available(:), prev_available(:)
  integer :: head_of_list, current_node, node_to_remove

  n_particles = size(particles, 1)
  GROUP_SIZE  = 32
  allocate(sorted_indices(n_particles))

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


  ! --- STEP 2: EFFICIENTLY GROUP POSITIVE PARTICLES ---
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
      do while (group_member_count < GROUP_SIZE .and. current_node /= 0)
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
  if (n_pos < n_particles) then
    sorted_indices(n_pos + 1:n_particles) = temp_indices(n_pos + 1:n_particles)
  end if

  ! --- CLEANUP ---
  deallocate(temp_indices)

end subroutine sort_indices_by_i_elm_warped

subroutine sort_indices_by_i_elm_bucket(sorted_indices, particles)
  implicit none
  integer, allocatable, intent(out) :: sorted_indices(:)
  type(particle_kinetic_relativistic), intent(in) :: particles(:)

  integer :: i, j, n_particles, n_pos, bucket, group_size
  integer, allocatable :: field_values(:)
  integer, allocatable :: pos_indices(:), neg_indices(:)
  integer, allocatable :: bucket_ids(:), bucket_counts(:), bucket_offsets(:)
  integer :: n_buckets, max_bucket
  integer :: idx

  integer :: lo, hi, size_bucket, start, finish

  n_particles = size(particles, 1)
  allocate(sorted_indices(n_particles))
  allocate(field_values(n_particles))

  ! Warp size
  group_size = 32

  ! Gather i_elm values
  do i = 1, n_particles
    field_values(i) = particles(i)%i_elm
  end do

  ! Count positives vs negatives
  n_pos = count(field_values >= 0)

  allocate(pos_indices(n_pos))
  allocate(neg_indices(n_particles - n_pos))
  allocate(bucket_ids(n_pos))

  j = 0
  idx = 0
  do i = 1, n_particles
    if (field_values(i) >= 0) then
      j = j + 1
      pos_indices(j) = i
      bucket_ids(j) = field_values(i) / group_size
    else
      idx = idx + 1
      neg_indices(idx) = i
    end if
  end do

  ! Determine max bucket ID
  if (n_pos > 0) then
    max_bucket = maxval(bucket_ids)
  else
    max_bucket = -1
  end if
  n_buckets = max_bucket + 1

  ! Count number of entries per bucket
  allocate(bucket_counts(n_buckets))
  bucket_counts = 0
  do i = 1, n_pos
    bucket_counts(bucket_ids(i)+1) = bucket_counts(bucket_ids(i)+1) + 1
  end do

  ! Compute bucket offsets for stable placement
  allocate(bucket_offsets(n_buckets))
  bucket_offsets(1) = 1
  do i = 2, n_buckets
    bucket_offsets(i) = bucket_offsets(i-1) + bucket_counts(i-1)
  end do

  ! Place particles into buckets (stable ordering)
  do i = 1, n_pos
    bucket = bucket_ids(i)
    idx = bucket_offsets(bucket+1)
    sorted_indices(idx) = pos_indices(i)
    field_values(idx) = particles(pos_indices(i))%i_elm
    bucket_offsets(bucket+1) = bucket_offsets(bucket+1) + 1
  end do

 ! Within each bucket, sort by i_elm ascending and split into warps
  do i = 0, n_buckets-1
    if (bucket_counts(i+1) > 0) then
      ! Compute range of this bucket
      lo = sum(bucket_counts(1:i)) + 1
      hi = sum(bucket_counts(1:i+1))
      size_bucket = hi - lo + 1

      ! Sort bucket by i_elm
      if (size_bucket > 1) then
        call sort_bucket(sorted_indices, field_values, lo, hi)
      end if

      ! (Optional) enforce warp chunking
      ! Already sorted, just keep warps contiguous:
      ! Fortran arrays are already ordered, so unlike Python
      ! no explicit chunking loop is strictly necessary here.
    end if
  end do


  ! Append negatives at the end
  do i = 1, size(neg_indices)
    sorted_indices(n_pos + i) = neg_indices(i)
  end do

  deallocate(pos_indices, neg_indices, bucket_ids, bucket_counts, bucket_offsets, field_values)
contains
  subroutine sort_bucket(indices, values, lo, hi)
    integer, intent(inout) :: indices(:), values(:)
    integer, intent(in) :: lo, hi
    integer :: i, j, key_idx, key_val

    ! Simple insertion sort for each bucket
    do i = lo+1, hi
      key_idx = indices(i)
      key_val = values(i)
      j = i - 1
      do while (j >= lo .and. values(j) > key_val)
        indices(j+1) = indices(j)
        values(j+1) = values(j)
        j = j - 1
      end do
      indices(j+1) = key_idx
      values(j+1) = key_val
    end do
  end subroutine sort_bucket
end subroutine sort_indices_by_i_elm_bucket

subroutine sort_indices_by_i_elm(sorted_indices, particles)
  implicit none
  integer, allocatable, intent(out) :: sorted_indices(:)
  type(particle_kinetic_relativistic), intent(in) :: particles(:)

  integer :: i, n_particles
  integer, allocatable :: field_values(:)

  n_particles = size(particles, 1)
  allocate(sorted_indices(n_particles))
  allocate(field_values(n_particles))

  ! Initialize the indices and field_values
  do i = 1, n_particles
    sorted_indices(i) = i
    field_values(i) = particles(i)%i_elm
  end do

  ! Sort the indices based on the i_elm field values
  call sort_indices(field_values, sorted_indices)

  deallocate(field_values)
end subroutine sort_indices_by_i_elm

subroutine sort_indices_by_i_elm_batched(sorted_indices, particles, n_batches)
  implicit none
  integer, allocatable, intent(out) :: sorted_indices(:)
  type(particle_kinetic_relativistic), intent(in) :: particles(:)
  integer, intent(in) :: n_batches

  integer :: i, j, n_particles, n_batches, n_particles_per_batch
  integer, allocatable :: field_values(:), batch_indices(:)

  n_particles = size(particles, 1)
  n_particles_per_batch = n_particles / n_batches

  allocate(sorted_indices(n_particles))
  allocate(field_values(n_particles))

  ! Initialize global indices and field_values
  do i = 1, n_particles
    sorted_indices(i) = i
    field_values(i) = particles(i)%i_elm
  end do

  ! Sort each batch independently
  do i = 0, n_batches - 1
    ! Work on one batch
    allocate(batch_indices(n_particles_per_batch))

    do j = 1, n_particles_per_batch
      batch_indices(j) = sorted_indices(i * n_particles_per_batch + j)
    end do

    ! Sort indices of this batch based on field_values
    call sort_indices(field_values, batch_indices)

    ! Put back into sorted_indices
    do j = 1, n_particles_per_batch
      sorted_indices(i * n_particles_per_batch + j) = batch_indices(j)
    end do

    deallocate(batch_indices)
  end do

  deallocate(field_values)
end subroutine sort_indices_by_i_elm_batched

subroutine sort_indices_by_i_elm_batched_warped(sorted_indices, particles, n_batches)
  implicit none
  integer, allocatable, intent(out) :: sorted_indices(:)
  type(particle_kinetic_relativistic), intent(in) :: particles(:)
  integer, intent(in) :: n_batches

  integer :: i, j, n_particles, n_batches, n_particles_per_batch
  integer, allocatable :: field_values(:), batch_indices(:)

  n_particles = size(particles, 1)
  n_particles_per_batch = n_particles / n_batches

  allocate(sorted_indices(n_particles))

  ! Initialize global indices and field_values
  do i = 1, n_particles
    sorted_indices(i) = i
  end do

  ! Sort each batch independently
  do i = 0, n_batches - 1
    ! Work on one batch
    allocate(batch_indices(n_particles_per_batch))

    do j = 1, n_particles_per_batch
      batch_indices(j) = sorted_indices(i * n_particles_per_batch + j)
    end do

    ! Sort indices of this batch based on field_values
    call sort_indices_by_i_elm_warped(batch_indices, particles(i * n_particles_per_batch:n_particles_per_batch))

    ! Put back into sorted_indices
    do j = 1, n_particles_per_batch
      sorted_indices(i * n_particles_per_batch + j) = batch_indices(j)
    end do

    deallocate(batch_indices)
  end do

end subroutine sort_indices_by_i_elm_batched_warped

! This recursive sorting subroutine is used to perform the initial global sort.
! It remains unchanged from the original provided code.
recursive subroutine sort_indices(field_values, indices)
  implicit none
  integer, intent(in) :: field_values(:)
  integer, intent(inout) :: indices(:)

  integer :: pivot_index, pivot_value
  integer :: i, j
  integer, allocatable :: low_indices(:), high_indices(:)
  integer :: low_count, high_count

  if (size(indices) > 1) then
    ! Choose pivot (simple choice: middle element)
    pivot_index = indices(size(indices) / 2)
    pivot_value = field_values(pivot_index)

    ! Partitioning
    low_count = count(field_values(indices) < pivot_value)
    high_count = count(field_values(indices) > pivot_value)

    allocate(low_indices(low_count))
    allocate(high_indices(high_count))

    low_count = 0
    high_count = 0
    do i = 1, size(indices)
      if (field_values(indices(i)) < pivot_value) then
        low_count = low_count + 1
        low_indices(low_count) = indices(i)
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
    do i = 1, size(indices)
        if (field_values(indices(i)) == pivot_value) then
          j = j + 1
          indices(j) = indices(i)
        end if
    end do
    do i = 1, high_count
      j = j + 1
      indices(j) = high_indices(i)
    end do

    deallocate(low_indices)
    deallocate(high_indices)
  end if
end subroutine sort_indices



end module mod_particle_sorting