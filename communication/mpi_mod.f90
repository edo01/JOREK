module mpi_mod

! backwards compatibility
#ifdef LAHEY
#define INCLUDE_MPIFH 1
#endif

#ifndef INCLUDE_MPIFH
#ifdef MPI_F08
  use mpi_f08
#else
  use mpi
#endif
#endif

  implicit none

  ! --- If 
#ifdef INCLUDE_MPIFH
  include 'mpif.h'
#endif
  !> Number of mpi tasks per node. 
  !> Will be determined by calling get_tasks_per_node.
  integer, private :: mpi_tasks_per_node = -1 
  !> Rank of this task within its node.
  !> Will be determined by calling get_node_local_rank.
  integer, private :: mpi_node_local_rank = -1
  
  contains
  
  !> Returns the number of MPI tasks per node.
  !! The number of tasks is determined at the first call to this function,
  !! and cached into the mpi_tasks_per_node module variable.
  !! The first call has a collective communication, therefore must be called by every task.
  integer function get_tasks_per_node()
    integer :: comm, ierr, myrank
    if (mpi_tasks_per_node < 0) then
      ! First call.
#if MPI_VERSION >= 3
      ! We create a shared memory communicator, and count the number of tasks in it.
      call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, comm, ierr)
      call MPI_Comm_size(comm, mpi_tasks_per_node, ierr)
      call MPI_Comm_free(comm, ierr)
#else
      ! MPI_COMM_TYPE_SHARED is only available since MPI 3.0
      ierr = 1
#endif
      if (ierr.ne.0 .or. mpi_tasks_per_node < 1) then
        ! We set a safe default in case there was an error
        mpi_tasks_per_node = 1
        call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
        if (myrank == 0) then
          write(*,*) 'Could not determine the number of MPI tasks per node.'
          write(*,*) 'This information would be used to limit the number of PaStiX threads per node.'
          write(*,*) 'In case you are using pastix_maxthrd, then you should compile with MPI 3,'
          write(*,*) 'otherwise you can ignore this message.'
        endif
      endif
    endif
    get_tasks_per_node = mpi_tasks_per_node
  end function get_tasks_per_node

  !> Returns the rank of this task within its own node, in [0, tasks_per_node).
  !! Determined at the first call and cached, like get_tasks_per_node above --
  !! and like it, the first call is collective and must be made by every task.
  !!
  !! This is the index an accelerator runtime wants: device ids are numbered per
  !! node, so binding from the MPI_COMM_WORLD rank would send every task on the
  !! second node past the end of its device list.
  integer function get_node_local_rank()
    integer :: comm, ierr
    if (mpi_node_local_rank < 0) then
#if MPI_VERSION >= 3
      call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, comm, ierr)
      call MPI_Comm_rank(comm, mpi_node_local_rank, ierr)
      call MPI_Comm_free(comm, ierr)
#else
      ierr = 1
#endif
      if (ierr .ne. 0 .or. mpi_node_local_rank < 0) then
        ! Same fallback as get_tasks_per_node: assume one task per node. Every
        ! task then takes device 0, which is what happened before this existed.
        mpi_node_local_rank = 0
      endif
    endif
    get_node_local_rank = mpi_node_local_rank
  end function get_node_local_rank

end module mpi_mod
