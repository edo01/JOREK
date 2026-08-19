!> Registers particle_kinetic_relativistic's memory layout with jgx.
!>
!> The field list is jgx/jorek/records/particle_base_record.def followed by
!> particle_kin_rel_record.def, expanded here and by particle_set.h -- one list,
!> so nothing has to be kept in step by hand. The base list comes first, which is
!> what keeps the inherited components at ids 0..5 in every particle record.
!>
!> Inheritance is not visible to the registry: the parent's components are
!> ordinary fields of the concrete type, at offsets the compiler chose. They are
!> measured like all the others -- sizeof(particle_base) is not a valid stand-in
!> (ifx leaves 4 bytes of padding between t_birth and p).
!>
!> Offsets are measured from a local array of the record type, not from a
!> particle group, because they are properties of the type. The group is
!> polymorphic and only yields a concrete base pointer inside a select type;
!> that is a call-site concern, not a registration one.
module mod_jgx_particle_record
  use, intrinsic :: iso_c_binding
  use mod_particle_types, only: particle_kinetic_relativistic
  use mod_jgx_record_ids, only: JGX_REC_PARTICLE_KIN_REL
  use mod_jgx_record
  implicit none
  private
  public :: jgx_register_particle_kin_rel_record

contains

  subroutine jgx_register_particle_kin_rel_record()
    type(particle_kinetic_relativistic), target :: pk(2)
    integer(c_int32_t) :: f
    integer(c_size_t)  :: stride
    !> Short local so the expanded registration line stays inside the
    !> free-form 132-column limit.
    integer(c_int32_t), parameter :: rid = JGX_REC_PARTICLE_KIN_REL

    !> Counted from the same lists that register below, so begin() cannot
    !> disagree with what follows.
    f = 0
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES) f = f + 1
#include "jgx/jorek/records/particle_base_record.def"
#include "jgx/jorek/records/particle_kin_rel_record.def"
#undef JGX_FIELD

    stride = transfer(c_loc(pk(2)), 1_c_size_t) - transfer(c_loc(pk(1)), 1_c_size_t)
    call jgx_c_record_begin(rid, stride, f)

    f = 0
#define JGX_FIELD(TAG, comp, T, RANK, KIND, AXES) \
    call jgx_add_field(rid, f, c_loc(pk(1)%comp), c_loc(pk(1)), KIND, shape(pk(1)%comp)); f = f + 1
#include "jgx/jorek/records/particle_base_record.def"
#include "jgx/jorek/records/particle_kin_rel_record.def"
#undef JGX_FIELD

    call jgx_c_record_end(rid)
  end subroutine jgx_register_particle_kin_rel_record

end module mod_jgx_particle_record
