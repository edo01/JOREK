!> Registers every JOREK derived type that crosses the jgx seam.
module mod_jgx_jorek_records
  use mod_jgx_element_record,  only: jgx_register_element_record
  use mod_jgx_node_record,     only: jgx_register_node_record
  use mod_jgx_particle_record, only: jgx_register_particle_kin_rel_record
  implicit none
  private
  public :: jgx_register_jorek_records

contains

  subroutine jgx_register_jorek_records()
    call jgx_register_element_record()
    call jgx_register_node_record()
    call jgx_register_particle_kin_rel_record()
    ! add here the registration to your new structure
  end subroutine jgx_register_jorek_records

end module mod_jgx_jorek_records
