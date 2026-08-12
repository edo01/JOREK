!> This modules contains tools needed by the different
!> particle pushers. Generally, they are mathematical 
!> stand-alone routines which can be used by multiple
!> independent algorithm
module mod_pusher_tools
implicit none
private
!> public procedures
public get_orthonormals
public cayley_transform, approximated_cayley_transform
public gc_position_to_particle, particle_position_to_gc
contains

!---------------------------------------------------------------------------

!> Get two vectors orthogonal to a given vector.
!> This is the RZPhi-version, the right-handed version will have different directions but will also work.
pure subroutine get_orthonormals(b, e1, e2)
  use mod_math_operators, only: cross_product
  implicit none

  real*8, dimension(3), intent(in)  :: b !< Does not need to be normalized
  real*8, dimension(3), intent(out) :: e1, e2

  ! Take r as a reference vector (this will fail therefore if B is purely in r direction)
  if (b(2) .eq. 0.d0 .and. b(3) .eq. 0.d0) then
    e1 = [0.d0, 1.d0, 0.d0]
    e2 = [0.d0, 0.d0, 1.d0]
  else
    e1 = cross_product(b, [1.d0, 0.d0, 0.d0])
    ! Normalize
    e1 = e1/norm2(e1)
    ! Obtain a second reference vector
    e2 = cross_product(b, e1)
    ! Normalize
    e2 = e2/norm2(e2)
  end if
end subroutine get_orthonormals

!---------------------------------------------------------------------------

!> This function computes the right-handed Cayley transform of a vector vec multiplied
!> by a scalar alpha. The Cayley transform is defined as:
!> cayley(alpha*B) = (I-alpha*B)^(-1) * (I+alpha*B)
!> where B is the vector product skew symmetric matrix of the vector vec
!> and I is the identity matrix.
!>
!> DUPLICATED IN C++ as pusher_tools::cayley_transform, in
!> particles/pushers/mod_pusher_tools/pusher_tools.h. Keep the two in step.
pure function cayley_transform(alpha,vec)
  ! defining input variables
  real(kind=8),intent(in) :: alpha !< multiplicative constant
  real(kind=8),dimension(3),intent(in) :: vec !< vector to be transformed
  ! defining output variables
  real(kind=8),dimension(3,3) :: cayley_transform !< Cayley transform of vec
  ! defining internal variables
  real(kind=8),dimension(3,3) :: A,B !< (I-alpha*B)^(-1) and (I+alpha*B)

! computing (I+alpha*B)
  B(1:3,1) = (/1.d0,-alpha*vec(3),alpha*vec(2)/)
  B(1:3,2) = (/alpha*vec(3),1.d0,-alpha*vec(1)/)
  B(1:3,3) = (/-alpha*vec(2),alpha*vec(1),1.d0/)

! computing (I-alpha*B)^(-1)
  A(1:3,1) = (/(1.0 + alpha*alpha*vec(1)*vec(1)),(alpha*(alpha*vec(1)*vec(2) - vec(3))),&
            (alpha*(alpha*vec(3)*vec(1) + vec(2)))/)
  A(1:3,2) = (/(alpha*(alpha*vec(2)*vec(1) + vec(3))),(1.0 + alpha*alpha*vec(2)*vec(2)),&
            (alpha*(alpha*vec(3)*vec(2) - vec(1)))/)
  A(1:3,3) = (/(alpha*(alpha*vec(3)*vec(1) - vec(2))),(alpha*(alpha*vec(2)*vec(3) + vec(1))),&
            (1.0 + alpha*alpha*vec(3)*vec(3))/)

! computing the Cayley transform
  cayley_transform = (matmul(A,B))/(1.d0 + alpha*alpha*(vec(1)*vec(1)+vec(2)*vec(2)+vec(3)*vec(3)))

end function cayley_transform

!---------------------------------------------------------------------------

!> This function computes the approximated Cayley transform a vector vec 
!> multiplied by a constant alpha as reported in R. Zhang et al., Phys. Plasmas 22 (2015) 044501.
pure function approximated_cayley_transform(alpha,vec)
  ! defining input variables
  real(kind=8),intent(in) :: alpha !< multiplicative constant
  real(kind=8),dimension(3),intent(in) :: vec !< vector to be transformed
  ! defining output variables
  real(kind=8),dimension(3,3) :: approximated_cayley_transform !< approximated Cayley transform of V
  ! internal variables
  real(kind=8) :: coefficient

  ! compute the approximated transfrom coefficient
  coefficient = (2.0*alpha)/(1.0 + alpha*alpha*(dot_product(vec,vec)))

  ! compute the approximated Cayley transform
  approximated_cayley_transform(1:3,1) = &
    (/1.0-coefficient*alpha*(vec(3)*vec(3) + vec(2)*vec(2)),&
    coefficient*(vec(3) + alpha*vec(2)*vec(1)),&
    coefficient*(alpha*vec(3)*vec(1) - vec(2))/)
  approximated_cayley_transform(1:3,2) = &
    (/coefficient*(alpha*vec(1)*vec(2) - vec(3)),&
    1.0-coefficient*alpha*(vec(3)*vec(3) + vec(1)*vec(1)),&
    coefficient*(alpha*vec(3)*vec(2) + vec(1))/)
  approximated_cayley_transform(1:3,3) = &
    (/coefficient*(alpha*vec(1)*vec(3) + vec(2)),&
    coefficient*(alpha*vec(2)*vec(3) - vec(1)),1.0-&
    coefficient*alpha*(vec(2)*vec(2) + vec(1)*vec(1))/)
end function approximated_cayley_transform

!---------------------------------------------------------------------------

!> This subroutine computes the guiding center position
!> from the particle position and momentum
subroutine particle_position_to_gc(node_list,element_list,&
  x_in,st_in,i_elm_in,p_in,q_in,B_hat,B_norm,x_gc_out,st_gc_out,i_elm_out)
  use data_structure
  use constants, only: ATOMIC_MASS_UNIT,EL_CHG
  use mod_math_operators, only: cross_product
  use mod_find_rz_nearby
  ! declare input variables
  type(type_node_list), intent(in)       :: node_list
  type(type_element_list), intent(in)    :: element_list
  integer(kind=1), intent(in)            :: q_in !< particle charge
  integer(kind=4), intent(in)            :: i_elm_in !< particle element
  real(kind=8), dimension(3), intent(in) :: x_in, p_in !< particle position and momentum
  real(kind=8), dimension(2), intent(in) :: st_in !< particle local coordinates
  real(kind=8), dimension(3), intent(in) :: B_hat !< Magnetic field direction B/B_norm
  real(kind=8), intent(in)               :: B_norm !< magnetic field intensity in [T]
  ! declare output variables
  integer(kind=4), intent(out)            :: i_elm_out !< gc element
  real(kind=8), dimension(2), intent(out) :: st_gc_out !< local gc position s,t
  real(kind=8), dimension(3), intent(out) :: x_gc_out  !< global position in R,Z,phi
  ! declare internal variables
  integer :: ifail

  ! compute the guiding center position in cartesian coordinates
  x_gc_out = x_in + (ATOMIC_MASS_UNIT*cross_product(p_in,B_hat))/(EL_CHG*real(q_in,8)*B_norm)  

  ! find the local coordinates
  call find_RZ_nearby(node_list,element_list,x_in(1),x_in(2), &
    st_in(1),st_in(2),i_elm_in,x_gc_out(1),x_gc_out(2),       &
    st_gc_out(1),st_gc_out(2),i_elm_out,ifail)
end subroutine particle_position_to_gc

!---------------------------------------------------------------------------
!> This subroutine computes the particle position from the guding center position
subroutine gc_position_to_particle(node_list,element_list, &
  x_gc_in,st_gc_in,i_elm_in,p_gc_in,q_gc_in,B_hat,B_norm,x_out,st_out,i_elm_out)
  use data_structure
  use constants, only: ATOMIC_MASS_UNIT,EL_CHG
  use mod_math_operators, only: cross_product
  use mod_find_rz_nearby
  ! declare input variables
  type(type_node_list), intent(in)       :: node_list
  type(type_element_list), intent(in)    :: element_list
  integer(kind=1), intent(in)            :: q_gc_in !< gc charge
  integer(kind=4), intent(in)            :: i_elm_in !< gc element
  real(kind=8), dimension(3), intent(in) :: x_gc_in,p_gc_in !< gc position and momentum
  real(kind=8), dimension(2), intent(in) :: st_gc_in !< gc local coordinates
  real(kind=8), dimension(3), intent(in) :: B_hat !< Magnetic field direction B/B_norm
  real(kind=8), intent(in)               :: B_norm !< magnetic field intensity in [T]
  ! declare output variables
  integer(kind=4), intent(out)            :: i_elm_out !< particle element
  real(kind=8), dimension(2), intent(out) :: st_out !< local particle position s,t
  real(kind=8), dimension(3), intent(out) :: x_out  !< global oarticle position in R, Z, phi
  ! declare internal variables
  integer :: ifail

  ! compute the particle position in cartesian coordinates
  x_out = x_gc_in + (ATOMIC_MASS_UNIT*cross_product(B_hat,p_gc_in))/(EL_CHG*real(q_gc_in,8)*B_norm)

  ! find the local coordinates
  call find_RZ_nearby(node_list,element_list,x_gc_in(1),x_gc_in(2), &
    st_gc_in(1),st_gc_in(2),i_elm_in,x_out(1),x_out(2),             &
    st_out(1),st_out(2),i_elm_out,ifail)
end subroutine gc_position_to_particle

!---------------------------------------------------------------------------

end module mod_pusher_tools
