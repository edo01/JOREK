!> Mod_particle_type contains the default particle type, type_particle
!> If you wish to add a new particle type, create one here, inheriting from
!> [[particle_base]], and update [[mod_particle_io]] (search for `particle_kinetic`
!> and add your particle at each spot)
module mod_particle_types
  use, intrinsic :: iso_c_binding
  implicit none
  private
  public :: particle_base_id,particle_fieldline_id,particle_gc_id
  public :: particle_gc_vpar_id,particle_gc_Qin_id,particle_kinetic_id
  public :: particle_kinetic_leapfrog_id,particle_kinetic_relativistic_id
  public :: particle_gc_relativistic_id
  public :: particle_base_size,particle_fieldline_size,particle_gc_size
  public :: particle_gc_vpar_size,particle_gc_Qin_size,particle_kinetic_size
  public :: particle_kinetic_leapfrog_size,particle_kinetic_relativistic_size
  public :: particle_gc_relativistic_size
  public :: particle_base, particle_kinetic, particle_kinetic_leapfrog
  public :: particle_gc, particle_fieldline
  public :: particle_kinetic_relativistic, particle_gc_relativistic
  public :: particle_gc_vpar, particle_gc_Qin
  public :: particle_get_q
  public :: copy_particle
  public :: copy_particle_base
  public :: copy_particle_kinetic_leapfrog
  public :: reorder_particles
  public :: codify_particle_type
  public :: find_active_particle_id
  public :: particle_arrays_from_list,particle_list_from_arrays
  public :: initialize_particle_list_to_zero,initialize_particle_to_zero
  public :: deallocate_particle_arrays

#include "optimization_defines.h"
#if USE_GPU
  public :: particle_SoA_kinetic_relativistic_c
  public :: particle_group_c
  public :: node_list_SoA_c
  public :: element_list_SoA_c
  public :: jorek_fields_interp_linear_c
  public :: particle_sim_c
  public :: particles_AoS_to_SoA
  public :: particles_SoA_to_AoS
  public :: node_list_to_SoA
  public :: element_list_to_SoA
  public :: dealloc_particle_SoA
  public :: dealloc_node_list_SoA
  public :: dealloc_element_list_SoA
#endif

  !> publicity only for unit testing
#ifdef UNIT_TESTS
  public :: find_active_particle_id_seq
  public :: find_active_particle_id_openmp
#endif

  !> buffer size in bits for each particle type
  integer,parameter :: particle_base_size                 = 480
  integer,parameter :: particle_fieldline_size            = 736
  integer,parameter :: particle_gc_size                   = 640
  integer,parameter :: particle_gc_vpar_size              = 640
  integer,parameter :: particle_gc_Qin_size               = 2496
  integer,parameter :: particle_kinetic_size              = 704
  integer,parameter :: particle_kinetic_leapfrog_size     = 704
  integer,parameter :: particle_kinetic_relativistic_size = 704
  integer,parameter :: particle_gc_relativistic_size      = 640
  !> enumerator of the particle type 
  enum, bind(C)
  enumerator :: particle_base_id=0,particle_fieldline_id,particle_gc_id,&
                particle_gc_vpar_id,particle_gc_Qin_id,particle_kinetic_id,&
                particle_kinetic_leapfrog_id,particle_kinetic_relativistic_id,&
                particle_gc_relativistic_id
  endenum

  !> The base type for all other particles. Includes only the position and weight elements
  !> Integration in a 2D finite element method is included in the form of 2 coordinates
  !> and an element index.
  type, abstract :: particle_base
    real*8    :: x(3)    = 0.d0   !< particle position in real space
    real*8    :: st(2)   = 0.d0   !< particle position in the element
    real*8    :: weight = 1.0     !< weight (i.e. number of particles)
    integer*4 :: i_elm = 0        !< index in element_list. Negative indices indicate lost particles on the edge of - that element.
    integer*4 :: i_life = 0       !< particle lifetime index (i.e. is this still the same particle?)
    real*4    :: t_birth = 0.0    !< birth time of this particle
    !< zero means lost without location specification.
  contains
    procedure :: copy => copy_particle
    generic :: assignment(=) => copy
  end type particle_base

  !> A simple type just for fieldline tracing in two-step methods (Adams Bashforth) or for forward euler
  type, extends(particle_base) :: particle_fieldline
    real*8    :: B_hat_prev(3) = 0.d0 !< Field direction at previous timestep
    real*8    :: v = 0.d0 !< Parallel velocity along the fieldline
  end type particle_fieldline

  !> A simple guiding-center particle type.
  type, extends(particle_base) :: particle_gc
    real*8    :: E  = 0.d0 !< The particle energy [eV]
    real*8    :: mu = 0.d0 !< The magnetic moment [eV/T]. Sign determines sign of v_par
    integer*1 :: q  = 0_1  !< Charge [e]
  end type particle_gc

  !> A simple guiding-center particle type.
  type, extends(particle_base) :: particle_gc_vpar
    real*8    :: vpar   = 0.d0 !< Guiding centre parallel velocity [m/s]
    real*8    :: mu     = 0.d0 !< The magnetic moment [eV/T] 
    real*8    :: B_norm = 0.d0 !< norm of total magnetic field [T]
    integer*1 :: q      = 0_1  !< Charge [e]
  end type particle_gc_vpar

  !> A simple guiding-center particle type.
  type, extends(particle_gc_vpar) :: particle_gc_Qin
    real*8    :: x_m(3)        !< position (previous step)
    real*8    :: vpar_m        !< parallel velocity (previous step)
    real*8    :: Astar_m(3)    !< A* (previous step)
    real*8    :: Astar_k(3)    !< A*  (current step)
    real*8    :: dAstar_k(3,3) !< dA* (current step)
    real*8    :: Bn_k          !< B   (amplitude, current step)
    real*8    :: dBn_k(3)      !< dB  (derivatives of Bn, current step)
    real*8    :: Bnorm_k(3)    !< normalised B (current step)
    real*8    :: E_k(3)        !< electric field (current step)
  end type particle_gc_Qin

  !> For most kinetic methods the velocity is required at time \(t\)
  type, extends(particle_base) :: particle_kinetic
    real*8, dimension(3) :: v = 0.d0 !< Velocity [m/s]
    integer*1            :: q = 0_1 !< charge [e]
  end type particle_kinetic

  !> Leapfrog methods define the particle velocity at time \(t^{n-1/2}\)
  !> and are therefore incompatible with normal kinetic methods (but a conversion
  !> function should not be too difficult)
  type, extends(particle_base) :: particle_kinetic_leapfrog
    real*8, dimension(3) :: v = 0.d0 !< Velocity [m/s] at t=t^(n-1/2) (where the position is known at t^n)
    integer*1            :: q = 0_1 !< charge [e]
  end type particle_kinetic_leapfrog

  !> This particle type is used for computing the full orbit trajectory
  !> of a relativistic particle. Particle position and momentum are given at time \(t\)
 type, extends(particle_base) :: particle_kinetic_relativistic
    real(kind=8),dimension(3) :: p !< Momentum in Cartesian coordinates (p_x,p_y,p_z) in [AMU*m/s]
    integer(kind=1)           :: q !< charge [e]
 end type particle_kinetic_relativistic

 !> This particle type is used for computing the guiding center trajectory of a
 !> relativistic particle. GC position and momentum are given at the time (\t\)
 type, extends(particle_base) :: particle_gc_relativistic
    real(kind=8), dimension(2) :: p  !< 1: parallel momentum [AMU m/s], 2: magnetic moment [(AMU*m**2)/(T*s**2)]
    integer(kind=1) :: q !< charge [e]
 end type particle_gc_relativistic

#if USE_GPU
! =====================================================================================
!  bind(C) Structure-of-Arrays types for GPU interoperability
!  These match the C structs in kernel_re_evolution.hip.cpp exactly.
! =====================================================================================

  !> SoA particle data for relativistic kinetic particles (bind(C))
  type, bind(C) :: particle_SoA_kinetic_relativistic_c
    type(c_ptr) :: x       = c_null_ptr !< (num_particles, 3) position (R, Z, phi)
    type(c_ptr) :: p       = c_null_ptr !< (num_particles, 3) momentum (Cartesian)
    type(c_ptr) :: st      = c_null_ptr !< (num_particles, 2) element-local coords
    type(c_ptr) :: weight  = c_null_ptr !< (num_particles)    macro-particle weight
    type(c_ptr) :: i_elm   = c_null_ptr !< (num_particles)    element index (C_INT)
  end type particle_SoA_kinetic_relativistic_c

  !> Particle group for GPU (bind(C))
  type, bind(C) :: particle_group_c
    real(c_double) :: mass                  !< species mass in AMU
    real(c_double) :: charge                !< charge number (e.g. -1.0 for electrons)
    integer(c_int) :: num_particles         !< total number of particles allocated
    type(particle_SoA_kinetic_relativistic_c) :: particles !< SoA particle data
  end type particle_group_c

  !> Node list in SoA layout for GPU (bind(C))
  type, bind(C) :: node_list_SoA_c
    integer(c_int) :: n_nodes  !< total number of nodes
    type(c_ptr)    :: x        = c_null_ptr !< (n_dim, n_degrees, n_coord_tor, n_nodes) — n_nodes slowest for stride-1 it-loop
    type(c_ptr)    :: values   = c_null_ptr !< (n_var, n_degrees, n_tor, n_nodes)       — n_nodes slowest for stride-1 it-loop
    type(c_ptr)    :: deltas   = c_null_ptr !< (n_var, n_degrees, n_tor, n_nodes)       — n_nodes slowest for stride-1 it-loop
  end type node_list_SoA_c

  !> Element list in SoA layout for GPU (bind(C))
  type, bind(C) :: element_list_SoA_c
    integer(c_int) :: n_elements !< total number of elements
    type(c_ptr)    :: vertex     = c_null_ptr !< (n_vertex_max, n_elements) — n_elements fastest for coalesced warp access
    type(c_ptr)    :: neighbours = c_null_ptr !< (n_vertex_max, n_elements) — n_elements fastest for coalesced warp access
    type(c_ptr)    :: size       = c_null_ptr !< (n_degrees, n_vertex_max, n_elements) — n_elements fastest
  end type element_list_SoA_c

  !> Fields accessor for GPU (bind(C))
  type, bind(C) :: jorek_fields_interp_linear_c
    type(node_list_SoA_c)    :: node_list
    type(element_list_SoA_c) :: element_list
    real(c_double) :: time_now
    real(c_double) :: time_prev
    integer(c_int) :: flag_static
    integer(c_int) :: flag_zero_dpsidt
    real(c_double) :: F0
    real(c_double) :: t_norm
    type(c_ptr)    :: mode_coord = c_null_ptr !< (n_coord_tor)
  end type jorek_fields_interp_linear_c

  !> Top-level simulation context for GPU (bind(C))
  type, bind(C) :: particle_sim_c
    type(jorek_fields_interp_linear_c) :: fields
    type(particle_group_c)             :: group
    real(c_double) :: sim_time
    integer(c_int) :: my_id
    integer(c_int) :: n_mpi
  end type particle_sim_c

#endif /* USE_GPU */

!> interfaces ------------------------------------------------------------------------------
interface codify_particle_type
  module procedure codify_single_particle_type
  module procedure codify_particle_list_alloc_type
end interface codify_particle_type

interface find_active_particle_id
  module procedure find_active_particle_id_base
  module procedure find_active_particle_id_type
end interface find_active_particle_id

contains
  !> Convenience function to obtain q if it exists, or 0 otherwise
  !> Here also because of https://gcc.gnu.org/bugzilla/show_bug.cgi?id=82064
  !> which means that we cannot use the same derived type in too many modules which will be
  !> imported in the main program (roughly)
  pure function particle_get_q(in) result(q)
    class(particle_base), intent(in) :: in
    integer*1 :: q
    select type (p => in)
    type is (particle_kinetic)
      q = p%q
    type is (particle_kinetic_leapfrog)
      q = p%q
    type is (particle_gc)
      q = p%q
    type is (particle_gc_vpar)
      q = p%q
    type is (particle_gc_Qin)
      q = p%q
    type is (particle_kinetic_relativistic)
      q = p%q
    type is (particle_gc_relativistic)
      q=p%q
    class default
      q = 0
    end select
  end function particle_get_q
  !> Copy the base variables from one particle of class(particle_base) to another
  
  pure subroutine copy_particle_base(in, out)
    class(particle_base), intent(in)    :: in
    class(particle_base), intent(inout) :: out
    out%x      = in%x
    out%st     = in%st
    out%weight = in%weight
    out%i_elm  = in%i_elm
    out%i_life = in%i_life
    out%t_birth= in%t_birth
  end subroutine copy_particle_base

  !> Copy one particle of a type kinetic_leapfrog to another
  pure subroutine copy_particle_kinetic_leapfrog(in, out)
    type(particle_kinetic_leapfrog), intent(in)    :: in
    type(particle_kinetic_leapfrog), intent(inout) :: out
    out%x       = in%x
    out%st      = in%st
    out%weight  = in%weight
    out%i_elm   = in%i_elm
    out%i_life  = in%i_life
    out%t_birth = in%t_birth
    out%v       = in%v
    out%q       = in%q
  end subroutine copy_particle_kinetic_leapfrog

  !> Copy a descendant of particle_base into another descendant of particle_base
  !> as requested by the types of the input and output parameters.
  !> We do not transform any of the quantities between eachother, just copy them
  !> if the particles are of the same type.
  !> To do a proper transform we typically need additional parameters, like the
  !> mass or magnetic field. See [[mod_boris.f90]] for some examples.
  !> See https://stackoverflow.com/a/19082934
  subroutine copy_particle(particle_out, particle_in)
    class(particle_base), intent(out) :: particle_out !< Particle to copy attributes into
    class(particle_base), intent(in)  :: particle_in  !< Particle to copy attributes from

    particle_out%x        = particle_in%x
    particle_out%st       = particle_in%st
    particle_out%weight   = particle_in%weight
    particle_out%i_elm    = particle_in%i_elm
    particle_out%i_life   = particle_in%i_life
    particle_out%t_birth  = particle_in%t_birth

    select type (p_out => particle_out)
    type is (particle_fieldline)
      select type (p_in => particle_in)
      type is (particle_fieldline) ! this is a straight copy, simple
        p_out%B_hat_prev = p_in%B_hat_prev
        p_out%v = p_in%v
      class default
        ! Maybe we should warn here instead of just putting zeros,
        ! or put nothing so the compiler can catch the uninitialized value
        p_out%B_hat_prev = [0.d0, 0.d0, 0.d0]
        p_out%v = 0.d0
      end select
    type is (particle_gc)
      select type (p_in => particle_in)
      type is (particle_gc)
        p_out%E  = p_in%E
        p_out%mu = p_in%mu
        p_out%q  = p_in%q
      class default
        p_out%E  = 0.d0
        p_out%mu = 0.d0
        p_out%q  = 0
      end select
    type is (particle_gc_vpar)
      select type (p_in => particle_in)
      type is (particle_gc_vpar)
        p_out%vpar    = p_in%vpar
        p_out%mu      = p_in%mu
        p_out%B_norm  = p_in%B_norm
        p_out%q       = p_in%q
      class default
        p_out%vpar    = 0.d0
        p_out%mu      = 0.d0
        p_out%B_norm  = 0.d0
        p_out%q       = 0
      end select
    type is (particle_gc_Qin)
    select type (p_in => particle_in)
    type is (particle_gc_Qin)
      p_out%vpar     = p_in%vpar
      p_out%mu       = p_in%mu
      p_out%q        = p_in%q
      p_out%x_m      = p_in%x_m
      p_out%vpar_m   = p_in%vpar_m
      p_out%Astar_m  = p_in%Astar_m
      p_out%Astar_k  = p_in%Astar_k
      p_out%dAstar_k = p_in%dAstar_k
      p_out%Bn_k     = p_in%Bn_k
      p_out%dBn_k    = p_in%dBn_k
      p_out%Bnorm_k  = p_in%Bnorm_k
      p_out%E_k      = p_in%E_k
    class default
      p_out%vpar     = 0.d0
      p_out%mu       = 0.d0
      p_out%q        = 0
      p_out%x_m      = 0.d0
      p_out%vpar_m   = 0.d0
      p_out%Astar_m  = 0.d0
      p_out%Astar_k  = 0.d0
      p_out%dAstar_k = 0.d0
      p_out%Bn_k     = 0.d0
      p_out%dBn_k    = 0.d0
      p_out%Bnorm_k  = 0.d0
      p_out%E_k      = 0.d0
    end select
    type is (particle_kinetic)
      select type (p_in => particle_in)
      type is (particle_kinetic)
        p_out%v  = p_in%v
        p_out%q  = p_in%q
      class default
        p_out%v  = [0.d0, 0.d0, 0.d0]
        p_out%q  = 0
      end select
    type is (particle_kinetic_leapfrog)
      select type (p_in => particle_in)
      type is (particle_kinetic_leapfrog)
        p_out%v  = p_in%v !copy_particle(particle_out, particle_in)
        p_out%q  = p_in%q
      class default
        ! the transformation from kinetic to kinetic_leapfrog could be done with a small error here
        p_out%v  = [0.d0, 0.d0, 0.d0]
        p_out%q  = 0
      end select
    type is (particle_kinetic_relativistic)
      select type (p_in => particle_in)
      type is (particle_kinetic_relativistic)
        p_out%p  = p_in%p
        p_out%q  = p_in%q
      class default
        p_out%p = [0.d0,0.d0,0.d0]
        p_out%q = 0
      end select     
     type is (particle_gc_relativistic)
       select type (p_in => particle_in)
       type is (particle_gc_relativistic)
         p_out%p = p_in%p
         p_out%q = p_in%q
     class default
        p_out%p  = [0.d0, 0.d0]
        p_out%q  = 0
      end select
    end select
  end subroutine copy_particle

  !> Reorder the particles according to a new array of indices
  subroutine reorder_particles(particle_list, new_order)
    class(particle_base), allocatable, dimension(:), intent(inout) :: particle_list
    integer, dimension(:), intent(in) :: new_order
    integer :: i
    class(particle_base), allocatable, dimension(:) :: particle_list_copy
    
    ! Create a copy of the original particle list
    call copy_particle_list(particle_list_copy, particle_list)

    ! Reorder the particles based on new_order
    do i = 1, size(particle_list)
      particle_list(i) = particle_list_copy(new_order(i))
    end do

    deallocate(particle_list_copy)
  end subroutine reorder_particles


  !> allocate an empty array of a specified particle type 
  subroutine copy_particle_list(particle_list_copy, particle_list_in)
    implicit none

    class(particle_base), allocatable, dimension(:), intent(in)  :: particle_list_in
    class(particle_base), allocatable, dimension(:), intent(out) :: particle_list_copy
    integer :: n_particles, j

    n_particles = size(particle_list_in)

    select type (particle_list_in)
    type is (particle_kinetic)
      allocate(particle_kinetic::particle_list_copy(n_particles))
    type is (particle_kinetic_leapfrog)
      allocate(particle_kinetic_leapfrog::particle_list_copy(n_particles))
    type is (particle_gc)
      allocate(particle_gc::particle_list_copy(n_particles))
    type is (particle_gc_vpar)
      allocate(particle_gc_vpar::particle_list_copy(n_particles))
    type is (particle_gc_Qin)
      allocate(particle_gc_Qin::particle_list_copy(n_particles))
    type is (particle_fieldline)
      allocate(particle_fieldline::particle_list_copy(n_particles))
    type is (particle_kinetic_relativistic)
      allocate(particle_kinetic_relativistic::particle_list_copy(n_particles))
    type is (particle_gc_relativistic)
      allocate(particle_gc_relativistic::particle_list_copy(n_particles))
    class default
      write(*,*) "ERROR: Unsupported particle type in copy_particle_list"
      stop
    end select

    do j=1,n_particles
      particle_list_copy(j) = particle_list_in(j)
    end do
  end subroutine copy_particle_list

  !> codify_single_particle_type returns a codified 
  !> particle type for a particle
  !> condification:
  !>   0 -> default
  !>   1 -> particle_fieldline
  !>   2 -> particle_gc
  !>   3 -> particle_gc_vpar
  !>   4 -> particle_gc_Qin
  !>   5 -> particle_kinetic
  !>   6 -> particle_kinetic_leapfrog
  !>   7 -> particle_kinetic_relativistic
  !>   8 -> particle_gc_relativistic
  function codify_single_particle_type(particle) result(p_type)
    implicit none
    class(particle_base),intent(in) :: particle
    integer :: p_type
    p_type = 0
    select type (p=>particle)
      type is (particle_fieldline)
      p_type = particle_fieldline_id
      type is (particle_gc)
      p_type = particle_gc_id
      type is (particle_gc_vpar)
      p_type = particle_gc_vpar_id
      type is (particle_gc_Qin)
      p_type = particle_gc_Qin_id
      type is (particle_kinetic)
      p_type = particle_kinetic_id
      type is (particle_kinetic_leapfrog)
      p_type = particle_kinetic_leapfrog_id
      type is (particle_kinetic_relativistic)
      p_type = particle_kinetic_relativistic_id
      type is (particle_gc_relativistic)
      p_type = particle_gc_relativistic_id
    end select
  end function codify_single_particle_type
 
  !> codify_particle_list_alloc_type returns a 
  !> codified particle type for a particle list
  !> condification:
  !>   1 -> particle_fieldline
  !>   2 -> particle_gc
  !>   3 -> particle_gc_vpar
  !>   4 -> particle_gc_Qin
  !>   5 -> particle_kinetic
  !>   6 -> particle_kinetic_leapfrog
  !>   7 -> particle_kinetic_relativistic
  !>   8 -> particle_gc_relativistic
  function codify_particle_list_alloc_type(particle_list) result(p_type)
    implicit none
    class(particle_base),dimension(:),allocatable,intent(in) :: particle_list
    integer :: p_type
    p_type = 0
    select type (p=>particle_list)
      type is (particle_fieldline)
      p_type = particle_fieldline_id
      type is (particle_gc)
      p_type = particle_gc_id
      type is (particle_gc_vpar)
      p_type = particle_gc_vpar_id
      type is (particle_gc_Qin)
      p_type = particle_gc_Qin_id
      type is (particle_kinetic)
      p_type = particle_kinetic_id
      type is (particle_kinetic_leapfrog)
      p_type = particle_kinetic_leapfrog_id
      type is (particle_kinetic_relativistic)
      p_type = particle_kinetic_relativistic_id
      type is (particle_gc_relativistic)
      p_type = particle_gc_relativistic_id
    end select
  end function codify_particle_list_alloc_type

  !> find_active_particle_id_type returns the number 
  !> and index of acitve particles withing a list for 
  !> are specific type of particle (encoded).
  !> Active particles are particles having i_elm>0.
  !> If particle code - particle type do not match
  !> returns 0 
  !> inputs:
  !>   particle_code: (integer) encoded particle type
  !>   n_particles:   (integer) number of particles
  !>   particle_list: (particle_base)(n_particles) particle list
  !> outputs:
  !>   n_active_particles: (integer) number of active particles
  !>   active_particle_id: (integer)(n_particles) active particle indices
  subroutine find_active_particle_id_type(particle_code,n_particles,&
  particle_list,n_active_particles,active_particle_id)
    implicit none
    !> inputs
    integer,intent(in) :: particle_code,n_particles
    class(particle_base),dimension(:),allocatable,intent(in) :: particle_list
    !> outputs
    integer,intent(out) :: n_active_particles
    integer,dimension(n_particles),intent(out) :: active_particle_id
    !> use select type as if condition, ok it is ugly ...
    n_active_particles = 0; active_particle_id = 0;
    select type (p_list=>particle_list)
      type is (particle_fieldline)
        if(particle_code.eq.particle_fieldline_id) &
          call find_active_particle_id_base(n_particles,particle_list,&
               n_active_particles,active_particle_id)
      type is (particle_gc)
        if(particle_code.eq.particle_gc_id) &
          call find_active_particle_id_base(n_particles,particle_list,&
               n_active_particles,active_particle_id)
      type is (particle_gc_vpar)
        if(particle_code.eq.particle_gc_vpar_id) &
          call find_active_particle_id_base(n_particles,particle_list,&
               n_active_particles,active_particle_id)
      type is (particle_gc_Qin)
        if(particle_code.eq.particle_gc_Qin_id) &
          call find_active_particle_id_base(n_particles,particle_list,&
               n_active_particles,active_particle_id)
      type is (particle_kinetic)
        if(particle_code.eq.particle_kinetic_id) &
          call find_active_particle_id_base(n_particles,particle_list,&
               n_active_particles,active_particle_id)
      type is (particle_kinetic_leapfrog)
        if(particle_code.eq.particle_kinetic_leapfrog_id) &
          call find_active_particle_id_base(n_particles,particle_list,&
               n_active_particles,active_particle_id)
      type is (particle_kinetic_relativistic)
        if(particle_code.eq.particle_kinetic_relativistic_id) &
          call find_active_particle_id_base(n_particles,particle_list,&
               n_active_particles,active_particle_id)
      type is (particle_gc_relativistic)
        if(particle_code.eq.particle_gc_relativistic_id) &
          call find_active_particle_id_base(n_particles,particle_list,&
               n_active_particles,active_particle_id)
    end select
  end subroutine find_active_particle_id_type

  !> find_active_particle_id_base returns the number 
  !> and index of acitve particles withing a list
  !> active particles are particles having i_elm>0
  !> interface for the sequential and openmp versions
  !> inputs:
  !>   n_particles:   (integer) number of particles
  !>   particle_list: (particle_base)(n_particles) particle list
  !>   n_particles_per_tile_in: (integer)(optional) number of
  !>                            particles per tile
  !> outputs:
  !>   n_active_particles: (integer) number of active particles
  !>   active_particle_id: (integer)(n_particles) active particle indices
  subroutine find_active_particle_id_base(n_particles,particle_list,&
  n_active_particles,active_particle_id)
    implicit none
    !> inputs
    integer,intent(in) :: n_particles
    class(particle_base),dimension(:),allocatable,intent(in) :: particle_list
    !> outputs
    integer,intent(out) :: n_active_particles
    integer,dimension(n_particles),intent(out) :: active_particle_id
#ifdef _OPENMP
    call find_active_particle_id_openmp(n_particles,particle_list,&
    n_active_particles,active_particle_id)
#else
    call find_active_particle_id_seq(n_particles,particle_list,&
    n_active_particles,active_particle_id)
#endif
  end subroutine find_active_particle_id_base

  !> find_active_particle_id_seq returns the number 
  !> and index of acitve particles withing a list
  !> active particles are particles having i_elm>0
  !> sequential version
  !> inputs:
  !>   n_particles:   (integer) number of particles
  !>   particle_list: (particle_base)(n_particles) particle list
  !> outputs:
  !>   n_active_particles: (integer) number of active particles
  !>   active_particle_id: (integer)(n_particles) active particle indices
  subroutine find_active_particle_id_seq(n_particles,particle_list,&
  n_active_particles,active_particle_id)
    implicit none
    !> inputs
    integer,intent(in) :: n_particles
    class(particle_base),dimension(:),allocatable,intent(in) :: particle_list
    !> outputs
    integer,intent(out) :: n_active_particles
    integer,dimension(n_particles),intent(out) :: active_particle_id
    !> variables
    integer :: ii
    n_active_particles = 0; active_particle_id = 0;
    do ii=1,n_particles
      if(particle_list(ii)%i_elm.lt.1) cycle !< skip invalid particle
       n_active_particles = n_active_particles + 1
       active_particle_id(n_active_particles) = ii
    enddo
  end subroutine find_active_particle_id_seq

  !> find_active_particle_id_openmp returns the number 
  !> and index of acitve particles withing a list
  !> active particles are particles having i_elm>0
  !> openmp enabled version.
  !> Note: the proposed openmp version is suboptimal and
  !>       hence, it must be improved in future
  !> inputs:
  !>   n_particles:   (integer) number of particles
  !>   particle_list: (particle_base)(n_particles) particle list
  !> outputs:
  !>   n_active_particles: (integer) number of active particles
  !>   active_particle_ids: (integer)(n_particles) active particle indices
  subroutine find_active_particle_id_openmp(n_particles,particle_list,&
  n_active_particles,active_particle_ids)
    !$ use omp_lib
    implicit none
    !> inputs
    integer,intent(in)          :: n_particles
    class(particle_base),dimension(:),allocatable,intent(in) :: particle_list
    !> outputs
    integer,intent(out) :: n_active_particles
    integer,dimension(n_particles),intent(out) :: active_particle_ids
    !> variables
    integer :: ii,jj,n_tiles
    integer :: n_particles_per_tile=25
    integer,dimension(:),allocatable :: n_active_particles_tile
    integer,dimension(:,:),allocatable :: particle_ids_tile

     !> initialisation
     n_active_particles=0; active_particle_ids=0;
     !> compute the number of tiles given an expected number of tiles
     n_tiles = 1+(n_particles/n_particles_per_tile)
     !> allocate arrays and initialise them if needed
     !allocate(start_id(n_tiles)); allocate(end_id(n_tiles));
     allocate(n_active_particles_tile(n_tiles)); n_active_particles_tile=0;
     allocate(particle_ids_tile(n_particles_per_tile,n_tiles)); particle_ids_tile=0;

    !> find active particles and store their index for each thread
    !$omp parallel do default(private) firstprivate(n_tiles,n_particles_per_tile) &
    !$omp shared(particle_list,n_active_particles_tile,particle_ids_tile) &
    !$omp schedule(dynamic)
    do ii=1,n_tiles-1
      do jj=n_particles_per_tile*(ii-1)+1,n_particles_per_tile*ii
        if(particle_list(jj)%i_elm.lt.1) cycle !< skip invalid particle
        n_active_particles_tile(ii) = n_active_particles_tile(ii) + 1
        particle_ids_tile(n_active_particles_tile(ii),ii) = jj
      enddo
    enddo
    !$omp end parallel do
    !> do remaining particles
    do jj=n_particles_per_tile*(n_tiles-1)+1,n_particles
      if(particle_list(jj)%i_elm.lt.1) cycle !< skip invalid particle
      n_active_particles_tile(n_tiles) = n_active_particles_tile(n_tiles) + 1
      particle_ids_tile(n_active_particles_tile(n_tiles),n_tiles) = jj
    enddo
    !> assemble the particle id array
    do ii=1,n_tiles
      active_particle_ids(n_active_particles+1:n_active_particles+n_active_particles_tile(ii)) = &
      particle_ids_tile(1:n_active_particles_tile(ii),ii)
      n_active_particles = n_active_particles + n_active_particles_tile(ii)
    enddo

    !> cleanup
    deallocate(n_active_particles_tile); deallocate(particle_ids_tile);
  end subroutine find_active_particle_id_openmp

  !> routines used for packing and unpacking particle types in MPI buffers
  !> pack the particle_base_type
  subroutine mpi_pack_particle_base(p_in,buffer,buff_position,ierr)
    use mpi
    implicit none
    !> inputs-outputs:
    integer,intent(inout) :: buff_position,ierr
    !> inputs:
    class(particle_base),intent(in) :: p_in
    !> outputs:
    character(len=particle_base_size),intent(out) :: buffer
    !> pack datatype
    call MPI_PACK(p_in%x,3,MPI_DOUBLE_PRECISION,buffer,particle_base_size,buff_position,MPI_COMM_WORLD,ierr)
    call MPI_PACK(p_in%st,2,MPI_DOUBLE_PRECISION,buffer,particle_base_size,buff_position,MPI_COMM_WORLD,ierr)
    call MPI_PACK(p_in%weight,1,MPI_DOUBLE_PRECISION,buffer,particle_base_size,buff_position,MPI_COMM_WORLD,ierr)
    call MPI_PACK(p_in%i_elm,1,MPI_INTEGER,buffer,particle_base_size,buff_position,MPI_COMM_WORLD,ierr)
    call MPI_PACK(p_in%i_life,1,MPI_INTEGER,buffer,particle_base_size,buff_position,MPI_COMM_WORLD,ierr)
    call MPI_PACK(p_in%t_birth,1,MPI_REAL,buffer,particle_base_size,buff_position,MPI_COMM_WORLD,ierr)
  end subroutine mpi_pack_particle_base

  !> unpack the particle base type
  subroutine mpi_unpack_particle_base(buffer,p_out,buff_position,ierr)
    use mpi
    implicit none
    !> inputs-outputs
    integer,intent(inout) :: buff_position,ierr
    !> input
    character(len=particle_base_size),intent(in) :: buffer
    !> outputs:
    class(particle_base),intent(out) :: p_out
    !> unpack datatype
    call MPI_UNPACK(buffer,particle_base_size,buff_position,p_out%x,3,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,ierr)
    call MPI_UNPACK(buffer,particle_base_size,buff_position,p_out%st,2,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,ierr)
    call MPI_UNPACK(buffer,particle_base_size,buff_position,p_out%weight,1,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,ierr)
    call MPI_UNPACK(buffer,particle_base_size,buff_position,p_out%i_elm,1,MPI_INTEGER,MPI_COMM_WORLD,ierr)
    call MPI_UNPACK(buffer,particle_base_size,buff_position,p_out%i_life,1,MPI_INTEGER,MPI_COMM_WORLD,ierr)
    call MPI_UNPACK(buffer,particle_base_size,buff_position,p_out%t_birth,1,MPI_REAL,MPI_COMM_WORLD,ierr)
  end subroutine mpi_unpack_particle_base

!> Allocate and re-order a particle list in arrays
subroutine particle_arrays_from_list(particle_list,n_particles,i_elm_arr,i_life_arr,&
q_arr,t_birth_arr,weight_arr,v_1d_arr,E_arr,mu_arr,vpar_arr,B_norm_arr,vpar_m_arr,&
st_arr,x_arr,B_hat_prev_arr,v_2d_arr,x_m_arr,Astar_m_arr,Astar_k_arr,&
Bn_k_arr,dBn_k_arr,Bnorm_k_arr,E_k_arr,dAstar_k_arr,particle_type_str)
  implicit none
  !> outputs:
  class(particle_base),dimension(:),allocatable,intent(in) :: particle_list
  integer,intent(out) :: n_particles
  integer*4,dimension(:),    allocatable,intent(out) :: i_elm_arr,i_life_arr
  integer*4,dimension(:),    allocatable,intent(out) :: q_arr
  real*4,   dimension(:),    allocatable,intent(out) :: t_birth_arr
  real*8,   dimension(:),    allocatable,intent(out) :: weight_arr,v_1d_arr
  real*8,   dimension(:),    allocatable,intent(out) :: E_arr,mu_arr,vpar_arr
  real*8,   dimension(:),    allocatable,intent(out) :: B_norm_arr,vpar_m_arr,Bn_k_arr
  real*8,   dimension(:,:),  allocatable,intent(out) :: st_arr,x_arr,B_hat_prev_arr,v_2d_arr
  real*8,   dimension(:,:),  allocatable,intent(out) :: x_m_arr,Astar_m_arr,Astar_k_arr
  real*8,   dimension(:,:),  allocatable,intent(out) :: dBn_k_arr,Bnorm_k_arr,E_k_arr
  real*8,   dimension(:,:,:),allocatable,intent(out) :: dAstar_k_arr
  character(len=:),          allocatable,intent(out) :: particle_type_str
  !> variables
  integer :: ii
  
  !> compute total number of particles
  n_particles = size(particle_list,1)
  allocate(x_arr(size(particle_list(1)%x,1)  ,n_particles))
  allocate(st_arr(size(particle_list(1)%st,1),n_particles))
  allocate(weight_arr(n_particles)); allocate(i_elm_arr(n_particles));
  allocate(i_life_arr(n_particles)); allocate(t_birth_arr(n_particles));
  select type(p=>particle_list(1))
  type is (particle_fieldline)
    allocate(v_1d_arr(n_particles))
    allocate(B_hat_prev_arr(size(p%B_hat_prev,1),n_particles))
    allocate(character(len=18)::particle_type_str); particle_type_str="particle_fieldline";
  type is (particle_gc)
    allocate(E_arr(n_particles)); allocate(mu_arr(n_particles));
    allocate(q_arr(n_particles)); allocate(character(len=11)::particle_type_str);
    particle_type_str="particle_gc";
  type is (particle_gc_vpar)
    allocate(vpar_arr(n_particles));   allocate(mu_arr(n_particles));
    allocate(B_norm_arr(n_particles)); allocate(q_arr(n_particles));
    allocate(character(len=16)::particle_type_str);
    particle_type_str="particle_gc_vpar";
  type is (particle_gc_Qin)
    allocate(vpar_arr(n_particles)); allocate(mu_arr(n_particles));
    allocate(B_norm_arr(n_particles)); allocate(q_arr(n_particles));
    allocate(x_m_arr(size(p%x_m,1),n_particles)); allocate(vpar_m_arr(n_particles));
    allocate(Astar_m_arr(size(p%Astar_m,1),n_particles));
    allocate(Astar_k_arr(size(p%Astar_k,1),n_particles));
    allocate(dAstar_k_arr(size(p%dAstar_k,1),size(p%dAstar_k,2),n_particles));
    allocate(Bn_k_arr(n_particles)); allocate(dBn_k_arr(size(p%dBn_k,1),n_particles));
    allocate(Bnorm_k_arr(size(p%Bnorm_k,1),n_particles));
    allocate(E_k_arr(size(p%E_k,1),n_particles));    
    allocate(character(len=15)::particle_type_str);
    particle_type_str = "particle_gc_Qin";
  type is (particle_kinetic)
    allocate(v_2d_arr(size(p%v,1),n_particles)); allocate(q_arr(n_particles));
    allocate(character(len=16)::particle_type_str);
    particle_type_str = "particle_kinetic";
  type is (particle_kinetic_leapfrog)
    allocate(v_2d_arr(size(p%v,1),n_particles)); allocate(q_arr(n_particles));
    allocate(character(len=25)::particle_type_str);
    particle_type_str = "particle_kinetic_leapfrog";
  type is (particle_kinetic_relativistic)
    allocate(v_2d_arr(size(p%p,1),n_particles)); allocate(q_arr(n_particles));
    allocate(character(len=29)::particle_type_str);
    particle_type_str = "particle_kinetic_relativistic";
  type is (particle_gc_relativistic)
    allocate(v_2d_arr(size(p%p,1),n_particles)); allocate(q_arr(n_particles));
    allocate(character(len=23)::particle_type_str);
    particle_type_str = "particle_gc_relativistic";
  end select
  !$omp parallel do default(none) private(ii) firstprivate(n_particles) & 
  !$omp shared(x_arr,st_arr,t_birth_arr,weight_arr,i_elm_arr,i_life_arr, particle_list)
  do ii=1,n_particles
    x_arr(:,ii)             = particle_list(ii)%x 
    st_arr(:,ii)            = particle_list(ii)%st
    t_birth_arr(ii)         = particle_list(ii)%t_birth
    weight_arr(ii)          = particle_list(ii)%weight 
    i_elm_arr(ii)           = particle_list(ii)%i_elm
    i_life_arr(ii)          = particle_list(ii)%i_life
  enddo
  !$omp end parallel do

  select type (p=>particle_list)
    
    type is (particle_fieldline)
    !$omp parallel do default(none) private(ii) firstprivate(n_particles) & 
    !$omp shared(v_1d_arr, B_hat_prev_arr, particle_list)
    do ii=1,n_particles
      v_1d_arr(ii)          = p(ii)%v
      B_hat_prev_arr(:,ii)  = p(ii)%B_hat_prev
    enddo
    !$omp end parallel do
    
    type is (particle_gc)
    !$omp parallel do default(none) private(ii) firstprivate(n_particles) & 
    !$omp shared(E_arr, mu_arr, q_arr, particle_list)
    do ii=1,n_particles
      E_arr(ii)             = p(ii)%E
      mu_arr(ii)            = p(ii)%mu
      q_arr(ii)             = int(p(ii)%q)
    enddo
    !$omp end parallel do
    
    type is (particle_gc_vpar)
    !$omp parallel do default(none) private(ii) firstprivate(n_particles) & 
    !$omp shared(vpar_arr, mu_arr, B_norm_arr, q_arr, particle_list)
    do ii=1,n_particles
      vpar_arr(ii)          = p(ii)%vpar 
      mu_arr(ii)            = p(ii)%mu 
      B_norm_arr(ii)        = p(ii)%B_norm
      q_arr(ii)             = int(p(ii)%q)
    enddo
    !$omp end parallel do

    type is (particle_gc_Qin)
    !$omp parallel do default(none) private(ii) firstprivate(n_particles)     & 
    !$omp shared(vpar_arr, mu_arr, q_arr, B_norm_arr, x_m_arr, vpar_m_arr,    &
    !$omp        Astar_m_arr, Astar_k_arr, dAstar_k_arr, Bn_k_arr, dBn_k_arr, &
    !$omp        Bnorm_k_arr, E_k_arr, particle_list)
    do ii=1,n_particles
      vpar_arr(ii)          = p(ii)%vpar 
      mu_arr(ii)            = p(ii)%mu
      B_norm_arr(ii)        = p(ii)%B_norm
      q_arr(ii)             = int(p(ii)%q)
      x_m_arr(:,ii)         = p(ii)%x_m
      vpar_m_arr(ii)        = p(ii)%vpar_m
      Astar_m_arr(:,ii)     = p(ii)%Astar_m
      Astar_k_arr(:,ii)     = p(ii)%Astar_k
      dAstar_k_arr(:,:,ii)  = p(ii)%dAstar_k
      Bn_k_arr(ii)          = p(ii)%Bn_k
      dBn_k_arr(:,ii)       = p(ii)%dBn_k
      Bnorm_k_arr(:,ii)     = p(ii)%Bnorm_k
      E_k_arr(:,ii)         = p(ii)%E_k
    enddo
    !$omp end parallel do

    type is (particle_kinetic)
    !$omp parallel do default(none) private(ii) firstprivate(n_particles) & 
    !$omp shared(v_2d_arr, q_arr, particle_list)
    do ii=1,n_particles
      v_2d_arr(:,ii)        = p(ii)%v
      q_arr(ii)             = int(p(ii)%q)
    enddo
    !$omp end parallel do

    type is (particle_kinetic_leapfrog)
    !$omp parallel do default(none) private(ii) firstprivate(n_particles) & 
    !$omp shared(v_2d_arr, q_arr, particle_list)
    do ii=1, n_particles
      v_2d_arr(:,ii)        = p(ii)%v
      q_arr(ii)             = int(p(ii)%q)
    enddo
    !$omp end parallel do

    type is (particle_kinetic_relativistic)
    !$omp parallel do default(none) private(ii) firstprivate(n_particles) & 
    !$omp shared(v_2d_arr, q_arr, particle_list)
    do ii=1, n_particles
      v_2d_arr(:,ii)        = p(ii)%p
      q_arr(ii)             = int(p(ii)%q)
    enddo
    !$omp end parallel do

    type is (particle_gc_relativistic)
    !$omp parallel do default(none) private(ii) firstprivate(n_particles) & 
    !$omp shared(v_2d_arr, q_arr, particle_list)
    do ii=1, n_particles
      v_2d_arr(:,ii)        = p(ii)%p
      q_arr(ii)             = int(p(ii)%q)
    enddo
    !$omp end parallel do

  end select

end subroutine particle_arrays_from_list

! initialise a particle list to 0
subroutine initialize_particle_list_to_zero(n_particles,particle_list,ierr)
  !> inputs:
  integer,intent(in) :: n_particles
  !> outputs:
  class(particle_base),dimension(:),allocatable,intent(inout)  :: particle_list
  !> inputs-outputs:
  integer,intent(inout) :: ierr
  !> variables:
  integer :: ii
  !> check if the particle list is allocated
  if(.not.allocated(particle_list)) then
    write(*,*) "WARNING: particle list not allocated: exit!"
    ierr = 1; return;
  endif
  !$omp parallel do default(none) private(ii),firstprivate(n_particles) &
  !$omp shared(particle_list)
  do ii=1,n_particles
    call initialize_particle_to_zero(particle_list(ii))
  enddo
  !$omp end parallel do
end subroutine initialize_particle_list_to_zero

!> initialize a single particle to 0
subroutine initialize_particle_to_zero(particle)
  class(particle_base), intent(inout) :: particle

  particle%i_elm=0;    particle%i_life=0; 
  particle%t_birth=0.; particle%weight=0d0;
  particle%st=0d0;     particle%x=0d0;
  select type (p=>particle)
    type is (particle_fieldline)
    p%v=0d0; p%B_hat_prev=0d0;
    type is (particle_gc)
    p%E=0d0; p%mu=0d0; p%q=int(0,kind=1);
    type is (particle_gc_vpar)
    p%vpar=0d0; p%mu=0d0; p%B_norm=0d0; p%q=int(0,kind=1);
    type is (particle_gc_Qin)
    p%vpar=0d0;    p%mu=0d0;      p%q=int(0,kind=1); p%B_norm=0d0; p%x_m=0d0; p%vpar_m=0d0;
    p%Astar_m=0d0; p%Astar_k=0d0; p%dAstar_k=0d0;    p%Bn_k=0d0;   p%dBn_k=0d0;
    p%Bnorm_k=0d0; p%E_k=0d0;
    type is (particle_kinetic)
    p%v=0d0; p%q=int(0,kind=1);
    type is (particle_kinetic_leapfrog) 
    p%v=0d0; p%q=int(0,kind=1);
    type is (particle_kinetic_relativistic)
    p%p=0d0; p%q=int(0,kind=1);
    type is (particle_gc_relativistic)
    p%p=0d0; p%q=int(0,kind=1);
  end select  
end subroutine initialize_particle_to_zero

! fille a particle list from arrays
subroutine particle_list_from_arrays(n_particles,particle_list,ierr,&
i_elm_arr,i_life_arr,t_birth_arr,weight_arr,x_arr,st_arr,q_arr,&
v_1d_arr,E_arr,mu_arr,vpar_arr,B_norm_arr,vpar_m_arr,B_hat_prev_arr,&
v_2d_arr,x_m_arr,Astar_m_arr,Astar_k_arr,Bn_k_arr,dBn_k_arr,&
Bnorm_k_arr,E_k_arr,dAstar_k_arr)
  implicit none
  !> inputs:
  integer, intent(in)                                          :: n_particles
  integer*4,dimension(:),    allocatable,intent(in),optional   :: i_elm_arr,i_life_arr
  real*4,   dimension(:),    allocatable,intent(in),optional   :: t_birth_arr
  real*8,   dimension(:),    allocatable,intent(in),optional   :: weight_arr
  real*8,   dimension(:,:),  allocatable,intent(in),optional   :: x_arr,st_arr
  integer*4,dimension(:),    allocatable,intent(in),optional   :: q_arr
  real*8,   dimension(:),    allocatable,intent(in),optional   :: v_1d_arr
  real*8,   dimension(:),    allocatable,intent(in),optional   :: E_arr,mu_arr,vpar_arr
  real*8,   dimension(:),    allocatable,intent(in),optional   :: B_norm_arr,vpar_m_arr,Bn_k_arr
  real*8,   dimension(:,:),  allocatable,intent(in),optional   :: B_hat_prev_arr,v_2d_arr
  real*8,   dimension(:,:),  allocatable,intent(in),optional   :: x_m_arr,Astar_m_arr,Astar_k_arr
  real*8,   dimension(:,:),  allocatable,intent(in),optional   :: dBn_k_arr,Bnorm_k_arr,E_k_arr
  real*8,   dimension(:,:,:),allocatable,intent(in),optional   :: dAstar_k_arr
  !> outputs:
  class(particle_base),dimension(:),allocatable,intent(inout)  :: particle_list
  !> inputs-outputs:
  integer,intent(inout) :: ierr
  !> variables:
  integer :: ii
  !> check if the particle list is allocated
  if(.not.allocated(particle_list)) then
    write(*,*) "WARNING: particle list not allocated: exit!"
    ierr = 1; return;
  endif 
  !> store particle base arrays
  !$omp parallel do default(none) private(ii) firstprivate(n_particles) &
  !$omp shared(particle_list,i_elm_arr,i_life_arr,t_birth_arr,weight_arr,&
  !$omp st_arr,x_arr,v_1d_arr,B_hat_prev_arr,E_arr,mu_arr,q_arr,vpar_arr,&
  !$omp B_norm_arr,x_m_arr,vpar_m_arr,Astar_m_arr,Astar_k_arr,dAstar_k_arr,&
  !$omp Bn_k_arr,dBn_k_arr,Bnorm_k_arr,E_k_arr,v_2d_arr)
  do ii=1,n_particles
    if(present(i_elm_arr))        then; if(allocated(i_elm_arr))   particle_list(ii)%i_elm    = i_elm_arr(ii);   endif;
    if(present(i_life_arr))       then; if(allocated(i_life_arr))  particle_list(ii)%i_life   = i_life_arr(ii);  endif;
    if(present(t_birth_arr))      then; if(allocated(t_birth_arr)) particle_list(ii)%t_birth  = t_birth_arr(ii); endif;
    if(present(weight_arr))       then; if(allocated(weight_arr))  particle_list(ii)%weight   = weight_arr(ii);  endif;
    if(present(st_arr))           then; if(allocated(st_arr))      particle_list(ii)%st       = st_arr(:,ii);    endif;
    if(present(x_arr))            then; if(allocated(x_arr))       particle_list(ii)%x        = x_arr(:,ii);     endif;
  enddo
  !$omp end parallel do

  select type (p=>particle_list)
    type is (particle_fieldline)
    !$omp parallel do default(none) private(ii) firstprivate(n_particles) & 
    !$omp shared(v_1d_arr, B_hat_prev_arr, particle_list)
    do ii=1, n_particles
      if(present(v_1d_arr))       then; if(allocated(v_1d_arr))       p(ii)%v          = v_1d_arr(ii);          endif;
      if(present(B_hat_prev_arr)) then; if(allocated(B_hat_prev_arr)) p(ii)%B_hat_prev = B_hat_prev_arr(:,ii);  endif;
    enddo
    !$omp end parallel do

    type is (particle_gc)
    !$omp parallel do default(none) private(ii) firstprivate(n_particles) & 
    !$omp shared(E_arr, mu_arr, q_arr, particle_list)
    do ii=1, n_particles
      if(present(E_arr))          then; if(allocated(E_arr))          p(ii)%E          = E_arr(ii);             endif;
      if(present(mu_arr))         then; if(allocated(mu_arr))         p(ii)%mu         = mu_arr(ii);            endif;
      if(present(q_arr))          then; if(allocated(q_arr))          p(ii)%q          = int(q_arr(ii),kind=1); endif;
    enddo
    !$omp end parallel do

    type is (particle_gc_vpar)
    !$omp parallel do default(none) private(ii) firstprivate(n_particles) & 
    !$omp shared(vpar_arr, mu_arr, q_arr, B_norm_arr, particle_list)
    do ii=1, n_particles
      if(present(vpar_arr))       then; if(allocated(vpar_arr))       p(ii)%vpar       = vpar_arr(ii);          endif;
      if(present(mu_arr))         then; if(allocated(mu_arr))         p(ii)%mu         = mu_arr(ii);            endif;
      if(present(q_arr))          then; if(allocated(q_arr))          p(ii)%q          = int(q_arr(ii),kind=1); endif;
      if(present(B_norm_arr))     then; if(allocated(B_norm_arr))     p(ii)%B_norm     = B_norm_arr(ii);        endif;
    enddo
    !$omp end parallel do

    type is (particle_gc_Qin)
    !$omp parallel do default(none) private(ii) firstprivate(n_particles)     & 
    !$omp shared(vpar_arr, mu_arr, q_arr, B_norm_arr, x_m_arr, vpar_m_arr,    &
    !$omp        Astar_m_arr, Astar_k_arr, dAstar_k_arr, Bn_k_arr, dBn_k_arr, &
    !$omp        Bnorm_k_arr, E_k_arr, particle_list)
    do ii=1, n_particles
      if(present(vpar_arr))       then; if(allocated(vpar_arr))       p(ii)%vpar       = vpar_arr(ii);          endif;
      if(present(mu_arr))         then; if(allocated(mu_arr))         p(ii)%mu         = mu_arr(ii);            endif;
      if(present(q_arr))          then; if(allocated(q_arr))          p(ii)%q          = int(q_arr(ii),kind=1); endif;
      if(present(B_norm_arr))     then; if(allocated(B_norm_arr))     p(ii)%B_norm     = B_norm_arr(ii);        endif;
      if(present(x_m_arr))        then; if(allocated(x_m_arr))        p(ii)%x_m        = x_m_arr(:,ii);         endif;
      if(present(vpar_m_arr))     then; if(allocated(vpar_m_arr))     p(ii)%vpar_m     = vpar_m_arr(ii);        endif; 
      if(present(Astar_m_arr))    then; if(allocated(Astar_m_arr))    p(ii)%Astar_m    = Astar_m_arr(:,ii);     endif;
      if(present(Astar_k_arr))    then; if(allocated(Astar_k_arr))    p(ii)%Astar_k    = Astar_k_arr(:,ii);     endif;
      if(present(dAstar_k_arr))   then; if(allocated(dAstar_k_arr))   p(ii)%dAstar_k   = dAstar_k_arr(:,:,ii);  endif;
      if(present(Bn_k_arr))       then; if(allocated(Bn_k_arr))       p(ii)%Bn_k       = Bn_k_arr(ii);          endif;
      if(present(dBn_k_arr))      then; if(allocated(dBn_k_arr))      p(ii)%dBn_k      = dBn_k_arr(:,ii);       endif;
      if(present(Bnorm_k_arr))    then; if(allocated(Bnorm_k_arr))    p(ii)%Bnorm_k    = Bnorm_k_arr(:,ii);     endif;
      if(present(E_k_arr))        then; if(allocated(E_k_arr))        p(ii)%E_k        = E_k_arr(:,ii);         endif;
    enddo
    !$omp end parallel do

    type is (particle_kinetic)
    !$omp parallel do default(none) private(ii) firstprivate(n_particles) & 
    !$omp shared(v_2d_arr, q_arr, particle_list)
    do ii=1, n_particles
      if(present(v_2d_arr))       then; if(allocated(v_2d_arr))       p(ii)%v          = v_2d_arr(:,ii);        endif;
      if(present(q_arr))          then; if(allocated(q_arr))          p(ii)%q          = int(q_arr(ii),kind=1); endif;
    enddo
    !$omp end parallel do

    type is (particle_kinetic_leapfrog) 
    !$omp parallel do default(none) private(ii) firstprivate(n_particles) & 
    !$omp shared(v_2d_arr, q_arr, particle_list)
    do ii=1, n_particles
      if(present(v_2d_arr))       then; if(allocated(v_2d_arr))       p(ii)%v          = v_2d_arr(:,ii);        endif;
      if(present(q_arr))          then; if(allocated(q_arr))          p(ii)%q          = int(q_arr(ii),kind=1); endif;
    enddo
    !$omp end parallel do

    type is (particle_kinetic_relativistic)
    !$omp parallel do default(none) private(ii) firstprivate(n_particles) & 
    !$omp shared(v_2d_arr, q_arr, particle_list)
    do ii=1, n_particles
      if(present(v_2d_arr))       then; if(allocated(v_2d_arr))       p(ii)%p          = v_2d_arr(:,ii);        endif;
      if(present(q_arr))          then; if(allocated(q_arr))          p(ii)%q          = int(q_arr(ii),kind=1); endif;
    enddo 
    !$omp end parallel do
      
    type is (particle_gc_relativistic)
    !$omp parallel do default(none) private(ii) firstprivate(n_particles) & 
    !$omp shared(v_2d_arr, q_arr, particle_list)
    do ii=1, n_particles
      if(present(v_2d_arr))       then; if(allocated(v_2d_arr))       p(ii)%p          = v_2d_arr(:,ii);        endif;
      if(present(q_arr))          then; if(allocated(q_arr))          p(ii)%q          = int(q_arr(ii),kind=1); endif;
    enddo
    !$omp end parallel do

    end select
  
end subroutine particle_list_from_arrays

!> deallocate all particle arrays
subroutine deallocate_particle_arrays(n_particles,i_elm_arr,i_life_arr,q_arr,&
t_birth_arr,weight_arr,v_1d_arr,E_arr,mu_arr,vpar_arr,B_norm_arr,vpar_m_arr,&
st_arr,x_arr,B_hat_prev_arr,v_2d_arr,x_m_arr,Astar_m_arr,Astar_k_arr,&
Bn_k_arr,dBn_k_arr,Bnorm_k_arr,E_k_arr,dAstar_k_arr)
  implicit none 
  !> inputs-outputs:
  integer,intent(inout)                                :: n_particles
  integer*4,dimension(:),    allocatable,intent(inout) :: i_elm_arr,i_life_arr
  integer*4,dimension(:),    allocatable,intent(inout) :: q_arr
  real*4,   dimension(:),    allocatable,intent(inout) :: t_birth_arr
  real*8,   dimension(:),    allocatable,intent(inout) :: weight_arr,v_1d_arr,E_arr,mu_arr
  real*8,   dimension(:),    allocatable,intent(inout) :: B_norm_arr,vpar_arr,vpar_m_arr
  real*8,   dimension(:),    allocatable,intent(inout) :: Bn_k_arr
  real*8,   dimension(:,:),  allocatable,intent(inout) :: st_arr,x_arr,B_hat_prev_arr
  real*8,   dimension(:,:),  allocatable,intent(inout) :: x_m_arr,Astar_m_arr,Astar_k_arr
  real*8,   dimension(:,:),  allocatable,intent(inout) :: dBn_k_arr,Bnorm_k_arr,E_k_arr,v_2d_arr
  real*8,   dimension(:,:,:),allocatable,intent(inout) :: dAstar_k_arr
  !> resets:
  n_particles = -1
  !> deallocates
  if(allocated(i_elm_arr))         deallocate(i_elm_arr)
  if(allocated(i_life_arr))        deallocate(i_life_arr)
  if(allocated(t_birth_arr))       deallocate(t_birth_arr)
  if(allocated(weight_arr))        deallocate(weight_arr)
  if(allocated(st_arr))            deallocate(st_arr)
  if(allocated(x_arr))             deallocate(x_arr)
  if(allocated(v_1d_arr))          deallocate(v_1d_arr)
  if(allocated(B_hat_prev_arr))    deallocate(B_hat_prev_arr)
  if(allocated(E_arr))             deallocate(E_arr)
  if(allocated(mu_arr))            deallocate(mu_arr)
  if(allocated(q_arr))             deallocate(q_arr)
  if(allocated(vpar_arr))          deallocate(vpar_arr)
  if(allocated(B_norm_arr))        deallocate(B_norm_arr)
  if(allocated(x_m_arr))           deallocate(x_m_arr)
  if(allocated(vpar_m_arr))        deallocate(vpar_m_arr)
  if(allocated(Astar_m_arr))       deallocate(Astar_m_arr)
  if(allocated(Astar_k_arr))       deallocate(Astar_k_arr)
  if(allocated(dAstar_k_arr))      deallocate(dAstar_k_arr)
  if(allocated(Bn_k_arr))          deallocate(Bn_k_arr)
  if(allocated(dBn_k_arr))         deallocate(dBn_k_arr)
  if(allocated(Bnorm_k_arr))       deallocate(Bnorm_k_arr)
  if(allocated(E_k_arr))           deallocate(E_k_arr)
  if(allocated(v_2d_arr))          deallocate(v_2d_arr)
end subroutine deallocate_particle_arrays

#if USE_GPU
! =====================================================================================
!  AoS <-> SoA conversion utilities for GPU interoperability
! =====================================================================================

  !> Convert an AoS array of particle_kinetic_relativistic into SoA buffers.
  !> The SoA arrays are allocated here; the caller must free them with dealloc_particle_SoA.
  !> Backing arrays (x_arr … tbirth_arr) are allocated here and returned to the caller.
  !> The caller must keep them alive until the GPU call completes and then deallocate them
  !> directly.  c_f_pointer + DEALLOCATE on a pointer recovered from c_loc is non-standard
  !> and raises Intel Fortran error 173.
  subroutine particles_AoS_to_SoA(particles, np, soa, &
      x_arr, p_arr, st_arr, w_arr, ielm_arr)
    use mod_settings, only: n_tor  ! just needed for parameter consistency
    implicit none
    type(particle_kinetic_relativistic), intent(in) :: particles(:)
    integer, intent(in) :: np  !< number of particles to convert
    type(particle_SoA_kinetic_relativistic_c), intent(out) :: soa
    real(c_double), allocatable, intent(out), target :: x_arr(:), p_arr(:), st_arr(:), w_arr(:)
    integer(c_int), allocatable, intent(out), target :: ielm_arr(:)
    integer :: j

    allocate(x_arr(3*np), p_arr(3*np), st_arr(2*np), w_arr(np))
    allocate(ielm_arr(np))

    !$omp parallel do default(none) shared(particles, np, x_arr, p_arr, st_arr, w_arr, ielm_arr) private(j)
    do j = 1, np
      x_arr(j)        = particles(j)%x(1)
      x_arr(j + np)   = particles(j)%x(2)
      x_arr(j + 2*np) = particles(j)%x(3)
      p_arr(j)        = particles(j)%p(1)
      p_arr(j + np)   = particles(j)%p(2)
      p_arr(j + 2*np) = particles(j)%p(3)
      st_arr(j)       = particles(j)%st(1)
      st_arr(j + np)  = particles(j)%st(2)
      w_arr(j)             = particles(j)%weight
      ielm_arr(j)          = int(particles(j)%i_elm, c_int)
    end do
    !$omp end parallel do

    soa%x       = c_loc(x_arr(1))
    soa%p       = c_loc(p_arr(1))
    soa%st      = c_loc(st_arr(1))
    soa%weight  = c_loc(w_arr(1))
    soa%i_elm   = c_loc(ielm_arr(1))
  end subroutine particles_AoS_to_SoA

  !> Copy SoA buffers back into an AoS array of particle_kinetic_relativistic.
  !> Only copies fields that the GPU kernel may have modified (x, p, st, i_elm).
  subroutine particles_SoA_to_AoS(soa, np, particles)
    implicit none
    type(particle_SoA_kinetic_relativistic_c), intent(in) :: soa
    integer, intent(in) :: np
    type(particle_kinetic_relativistic), intent(inout) :: particles(:)

    real(c_double), pointer :: x_arr(:), p_arr(:), st_arr(:)
    integer(c_int), pointer :: ielm_arr(:)
    integer :: j

    call c_f_pointer(soa%x,     x_arr,    [3*np])
    call c_f_pointer(soa%p,     p_arr,    [3*np])
    call c_f_pointer(soa%st,    st_arr,   [2*np])
    call c_f_pointer(soa%i_elm, ielm_arr, [np])

    !$omp parallel do default(none) shared(particles, np, x_arr, p_arr, st_arr, ielm_arr) private(j)
    do j = 1, np
      particles(j)%x(1)  = x_arr(j)
      particles(j)%x(2)  = x_arr(j + np)
      particles(j)%x(3)  = x_arr(j + 2*np)
      particles(j)%p(1)  = p_arr(j)
      particles(j)%p(2)  = p_arr(j + np)
      particles(j)%p(3)  = p_arr(j + 2*np)
      particles(j)%st(1) = st_arr(j)
      particles(j)%st(2) = st_arr(j + np)
      particles(j)%i_elm = int(ielm_arr(j), 4)
    end do
    !$omp end parallel do
  end subroutine particles_SoA_to_AoS

  !> Deallocate the SoA buffers allocated by particles_AoS_to_SoA
  !> Null out the c_ptr fields of a particle SoA struct.
  !> The backing memory is owned by the caller and must be freed there via DEALLOCATE.
  subroutine dealloc_particle_SoA(soa, np)
    implicit none
    type(particle_SoA_kinetic_relativistic_c), intent(inout) :: soa
    integer, intent(in) :: np
    soa%x       = c_null_ptr
    soa%p       = c_null_ptr
    soa%st      = c_null_ptr
    soa%weight  = c_null_ptr
    soa%i_elm   = c_null_ptr
  end subroutine dealloc_particle_SoA

  !> Convert the Fortran AoS type_node_list to a SoA node_list_SoA_c.
  !> The SoA arrays are allocated here; the caller must free them with dealloc_node_list_SoA.
  !> x_flat, val_flat, del_flat are allocated here and returned to the caller.
  !> The caller must keep them alive until the GPU call completes and then DEALLOCATE them.
  subroutine node_list_to_SoA(node_list, nl_soa, x_flat, val_flat, del_flat)
    use mod_settings, only: n_tor, n_degrees, n_dim, n_coord_tor
    use data_structure, only: type_node_list
    implicit none
    type(type_node_list), intent(in), target :: node_list
    type(node_list_SoA_c), intent(out) :: nl_soa
    real(c_double), allocatable, intent(out), target :: x_flat(:), val_flat(:), del_flat(:)
    integer :: i, nn
    integer :: idx, kc, kf, kd, kt

    nn = node_list%n_nodes

    nl_soa%n_nodes = int(nn, c_int)

    ! Allocate flat arrays in the order the C kernel expects (new optimal layout):
    !   x:      (n_dim, n_degrees, n_coord_tor, n_nodes)         — n_nodes slowest
    !   values: (2,     n_degrees, n_tor,       n_nodes)         — n_nodes slowest
    !   deltas: (2,     n_degrees, n_tor,       n_nodes)         — n_nodes slowest
    ! Only 2 field vars (P_par, P_perp) are stored; j_Phi is never read on GPU.
    allocate(x_flat(n_coord_tor * n_degrees * n_dim * nn))
    allocate(val_flat(n_tor * n_degrees * 2 * nn))
    allocate(del_flat(n_tor * n_degrees * 2 * nn))

    ! Copy node coordinates: node(i)%x(kc, kf, kd)
    ! C layout (new optimal): x_flat[ (kd-1) + n_dim*((kf-1) + n_degrees*((kc-1) + n_coord_tor*(i-1))) ]
    ! i.e. (n_dim, n_degrees, n_coord_tor, n_nodes) — n_nodes is the slowest dimension.
    !$omp parallel do default(none) shared(node_list, x_flat, nn) private(i, kc, kf, kd, idx) collapse(2)
    do i = 1, nn
      do kd = 1, n_dim
        do kf = 1, n_degrees
          do kc = 1, n_coord_tor
            idx = (kd-1) + n_dim * ((kf-1) + n_degrees * ((kc-1) + n_coord_tor * (i-1))) + 1
            x_flat(idx) = node_list%node(i)%x(kc, kf, kd)
          end do
        end do
      end do
    end do
    !$omp end parallel do

    ! Copy values/deltas: node(i)%values(kt, kf, kd), but only kd=1,2 (P_par, P_perp).
    ! C layout (new optimal): val_flat[ (kd-1) + 2*((kf-1) + n_degrees*((kt-1) + n_tor*(i-1))) ]
    ! i.e. (2, n_degrees, n_tor, n_nodes) — n_nodes is the slowest dimension.
    !$omp parallel do default(none) shared(node_list, val_flat, del_flat, nn) private(i, kt, kf, kd, idx) collapse(2)
    do i = 1, nn
      do kd = 1, 2
        do kf = 1, n_degrees
          do kt = 1, n_tor
            idx = (kd-1) + 2 * ((kf-1) + n_degrees * ((kt-1) + n_tor * (i-1))) + 1
            val_flat(idx) = node_list%node(i)%values(kt, kf, kd)
            del_flat(idx) = node_list%node(i)%deltas(kt, kf, kd)
          end do
        end do
      end do
    end do
    !$omp end parallel do

    nl_soa%x      = c_loc(x_flat(1))
    nl_soa%values = c_loc(val_flat(1))
    nl_soa%deltas = c_loc(del_flat(1))
  end subroutine node_list_to_SoA

  !> Deallocate the SoA buffers allocated by node_list_to_SoA
  !> Null out the c_ptr fields of a node_list SoA struct.
  !> The backing memory is owned by the caller and must be freed there via DEALLOCATE.
  subroutine dealloc_node_list_SoA(nl_soa)
    implicit none
    type(node_list_SoA_c), intent(inout) :: nl_soa
    nl_soa%x      = c_null_ptr
    nl_soa%values = c_null_ptr
    nl_soa%deltas = c_null_ptr
  end subroutine dealloc_node_list_SoA

  !> Convert the Fortran AoS type_element_list to SoA element_list_SoA_c.
  !> The SoA arrays are allocated here; the caller must free them with dealloc_element_list_SoA.
  !> vert_flat, neigh_flat, size_flat are allocated here and returned to the caller.
  !> The caller must keep them alive until the GPU call completes and then DEALLOCATE them.
  subroutine element_list_to_SoA(element_list, el_soa, vert_flat, neigh_flat, size_flat)
    use mod_settings, only: n_vertex_max, n_degrees
    use data_structure, only: type_element_list
    implicit none
    type(type_element_list), intent(in), target :: element_list
    type(element_list_SoA_c), intent(out) :: el_soa
    integer(c_int), allocatable, intent(out), target :: vert_flat(:), neigh_flat(:)
    real(c_double), allocatable, intent(out), target :: size_flat(:)
    integer :: ne, i, kv, kf, idx

    ne = element_list%n_elements
    el_soa%n_elements = int(ne, c_int)

    ! C layout (new optimal):
    !   vertex/neighbours: (n_vertex_max, n_elements)       — kv fastest, then i
    !   size:              (n_degrees, n_vertex_max, n_elements) — kf fastest, then kv, then i
    ! Warp threads on consecutive elements stride by n_vertex_max (or n_degrees*n_vertex_max)
    ! which is small and fits in a single cache line broadcast.
    allocate(vert_flat(n_vertex_max * ne))
    allocate(neigh_flat(n_vertex_max * ne))
    allocate(size_flat(n_degrees * n_vertex_max * ne))

    ! Fortran 1-based index formulas derived from C idx2(kv,ie,NV)=kv+NV*ie
    ! and idx3(kf,kv,ie,NDEG,NV)=kf+NDEG*(kv+NV*ie):
    !   vert/neigh:  idx = kv + n_vertex_max*(i-1)
    !   size:        idx = kf + n_degrees*((kv-1) + n_vertex_max*(i-1))
    !$omp parallel do default(none) shared(element_list, vert_flat, neigh_flat, size_flat, ne) private(i, kv, kf, idx) collapse(2)
    do i = 1, ne
      do kv = 1, n_vertex_max
        idx = kv + n_vertex_max*(i-1)
        vert_flat(idx)  = int(element_list%element(i)%vertex(kv), c_int)
        neigh_flat(idx) = int(element_list%element(i)%neighbours(kv), c_int)
        do kf = 1, n_degrees
          size_flat(kf + n_degrees*((kv-1) + n_vertex_max*(i-1))) = element_list%element(i)%size(kv, kf)
        end do
      end do
    end do
    !$omp end parallel do

    el_soa%vertex     = c_loc(vert_flat(1))
    el_soa%neighbours = c_loc(neigh_flat(1))
    el_soa%size       = c_loc(size_flat(1))
  end subroutine element_list_to_SoA

  !> Deallocate the SoA buffers allocated by element_list_to_SoA
  !> Null out the c_ptr fields of an element_list SoA struct.
  !> The backing memory is owned by the caller and must be freed there via DEALLOCATE.
  subroutine dealloc_element_list_SoA(el_soa)
    implicit none
    type(element_list_SoA_c), intent(inout) :: el_soa
    el_soa%vertex     = c_null_ptr
    el_soa%neighbours = c_null_ptr
    el_soa%size       = c_null_ptr
  end subroutine dealloc_element_list_SoA

#endif /* USE_GPU */

end module mod_particle_types
