!> Module containing the particle side of recombination (fluid density --> neutral particle weight)
module mod_particle_recomb
  use mpi
  use mod_random_seed 
  use mod_atomic_elements
  use mod_interp
  use mod_particle_io
  use particle_tracer
  !$ use omp_lib

  implicit none

  contains

  subroutine do_1particle_recombination(element_list,node_list,target_group,jorek_stepper,rng,tstep_fluid_si)
    use mod_jorek_timestepping !< gives us access to sim?
    use mod_integrate_recomb, only : integrate_recombination
    use phys_module, only: use_manual_random_seed, central_density, central_mass, sqrt_mu0_over_rho0
  
    implicit none
  
    type(pcg32_rng), dimension(:), intent(inout)  :: rng
    type(jorek_timestep_action),target      :: jorek_stepper
    TYPE (type_node_list),     intent(in)   :: node_list
    TYPE (type_element_list),    intent(in)   :: element_list
    integer, intent(in)               :: target_group
    real*8, intent(in)              :: tstep_fluid_si
    
    !internal variables
    type (type_element)         :: element
    logical, allocatable, dimension(:) :: is_free
    integer, allocatable, dimension(:) :: i_free
    integer       :: Nrec_part, particles_per_element
    real*8        :: total_rec,total_rec_all ,total_volume,total_volume_all
    real*8        :: total_Erec_neutral,total_Erec_neutral_all, total_Erec_rad,total_Erec_rad_all
    integer       :: n_free, i, j, k,ielm,ife, i_rng, ierr
    real*8        :: s, t,R, Z, st_ran(2)
  
    !debug rec
    real*8                :: sanity_rec_local,total_sanity_rec
    !rec variables
    real*8, dimension(:), allocatable  :: rec_rate_local , rec_v_R, rec_v_Z, rec_v_phi 
    real*8, dimension(:), allocatable  :: volume_check, energy_neutrals, energy_radiation
  
    !Call mod_integrate_recombination
    call integrate_recombination(sim%my_id,sim%n_mpi, rec_rate_local, rec_v_R, rec_v_Z, rec_v_phi,volume_check, energy_neutrals, energy_radiation)

    sanity_rec_local = 0.d0
    !calculate total recombination per mpi proces
    total_volume = sum( volume_check(:) )
    total_Erec_neutral = sum( energy_neutrals(:) )
    total_Erec_rad = sum( energy_radiation(:) )
    total_rec = sum( rec_rate_local(:) )
    ! total recombination
    call MPI_REDUCE(total_rec, total_rec_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    call MPI_REDUCE(total_volume, total_volume_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    call MPI_REDUCE(total_Erec_neutral, total_Erec_neutral_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    call MPI_REDUCE(total_Erec_rad, total_Erec_rad_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    if (sim%my_id .eq. 0) then
      write(*,'(A30,2E16.8)') 'total recombination weight : ' , sim%time,total_rec_all* central_density* 1.d20 
      write(*,*) 'total energy to recombined neutrals [J] : ' , total_Erec_neutral_all *1.5d0 / MU_ZERO
      write(*,*) 'total energy lost to Prb [J]: ' , total_Erec_rad_all *1.5d0 / MU_ZERO
      write(*,*) 'total volume : ' , total_volume_all
      write(*,'(A30,2E16.8)') 'Recombination rate  [#/s] : ', sim%time, total_rec_all* central_density* 1.d20 /tstep_fluid_si
      write(*,*) 'total power to recombined neutrals [MW]: ' , total_Erec_neutral_all *1.5d0 / MU_ZERO/tstep_fluid_si /1.d6
      write(*,*) 'total power lost to Prb [MW]: ' , total_Erec_rad_all *1.5d0 / MU_ZERO/tstep_fluid_si /1.d6
      write(*,'(A15,6E14.6)') 'TOTAL RECOMB: ',sim%time, total_rec_all* central_density* 1.d20 , total_Erec_neutral_all *1.5d0 / MU_ZERO, total_Erec_neutral_all *1.5d0 / MU_ZERO/tstep_fluid_si /1.d6, &
                  total_Erec_rad_all *1.5d0 / MU_ZERO, total_Erec_rad_all *1.5d0 / MU_ZERO/tstep_fluid_si /1.d6
    endif
    !Nrec_part amount of particles needed for this amount of recombination
    Nrec_part = int( max(sim%groups(target_group)%n_particles * 1.d-2 ,total_rec/1.d14 ) )!< assumed average weight per particle (not necesarily the actual weight, as that depends on Srec)
    !< limited to 1% of the total initialized particles
  
  
    !============== Finding free particles !< make into a function?
    !> # is_free > n_elements * particles_per_element 
    if(use_manual_random_seed) then
      !$ call omp_set_schedule(omp_sched_static,100)
    else
      !$ call omp_set_schedule(omp_sched_dynamic,100)
    end if
  
    allocate(is_free(size(sim%groups(target_group)%particles,1))) 
    !$omp parallel do default(none) shared(sim, n_free, i_free, is_free, target_group) &
    !$omp private(j) schedule(runtime)
      do j=1,size(sim%groups(target_group)%particles,1) 
        is_free(j) = sim%groups(target_group)%particles(j)%i_elm .le. 0  !< array T/F is particle is free
      end do
    !$omp end parallel do
    !$omp barrier
    n_free = count(is_free)
    allocate(i_free(n_free))
    k = 1
    do j=1,size(is_free,1)
      if (is_free(j)) then
        i_free(k) = j !< i_free(k) has index of free particle in  sim%groups(target_group)%particles(j)
        k = k+1
      end if
    end do
    ! ==================
  
    ! loop over all elements
    k = 0 !< first free particle
    particles_per_element = 1  
    select type (particles => sim%groups(target_group)%particles)
    type is (particle_kinetic_leapfrog)
      if(use_manual_random_seed) then
        !$ call omp_set_schedule(omp_sched_static,10)
      else
        !$ call omp_set_schedule(omp_sched_dynamic,10)
      end if
      !omp
#ifdef __GFORTRAN__
      !$omp parallel do default(shared) & ! workaround for Error: �__vtab_mod_pcg32_rng_Pcg32_rng� not specified in enclosing �parallel�
#else
      !$omp parallel do default(shared) &
      !$omp shared(sim, particles,jorek_stepper, element_list, node_list, target_group, rec_v_R,rec_v_Z,rec_v_phi, &
      !$omp i_free,rng,rec_rate_local, &
      !$omp CENTRAL_DENSITY, CENTRAL_MASS,sqrt_mu0_over_rho0,particles_per_element ) &
#endif
      !$omp schedule(runtime)    &
      !$omp private(ife,ielm,k,i,element,s,t,R, Z , &
      !$omp st_ran, i_rng ) &
      !$omp reduction(+:sanity_rec_local)
        do ife = 1, size(rec_rate_local) ! loop over all local elements
  
          !if (isnan(rec_v_R(ife)) .or. isnan(rec_v_Z(ife)) .or. isnan(rec_v_phi(ife))) CYCLE !NaN check
          if (rec_v_R(ife) /= rec_v_R(ife) .or. &
              rec_v_Z(ife) /= rec_v_Z(ife) .or. &
              rec_v_phi(ife) /= rec_v_phi(ife)) CYCLE !NaN check
          if (rec_rate_local(ife) * central_density * 1.d20 .le. 1.d3) CYCLE
    
          !$ i_rng = omp_get_thread_num()+1
    
          k = ife !< every OMP thread gets different values
          !< every MPI process has it's own list of i_free.
    
          ! --- Get element
          ielm  = (sim%my_id+1) + sim%n_mpi*(ife - 1)
          element = element_list%element(ielm)
    
          ! initialise particle in the element with Position, Weight, Energy, Momentum
          do i = 1, particles_per_element
            k = k *i !< update free particle index ! at begin of loop as k is initialized at k =0
            particles(i_free(k))%weight = rec_rate_local(ife) / real(particles_per_element,8)* central_density* 1.d20 !< rec_rate = in jorek units?
            particles(i_free(k))%i_elm  = ielm  !x, i_elm, st
            particles(i_free(k))%q    = 0
                
            sanity_rec_local = sanity_rec_local + particles(i_free(k))%weight
                
            call rng(i_rng)%next(st_ran)
            !< sample random st combination
            particles(i_free(k))%st(1) = 0.5d0
            particles(i_free(k))%st(2) = 0.5d0
            
            s = particles(i_free(k))%st(1)
            t = particles(i_free(k))%st(2)
  
            !> uses i_elm and s,t to give us R,Z
            call interp_RZ(node_list,element_list,ielm,s,t,R,Z)
            particles(i_free(k))%x(1:2)  = [R, Z]!  = [R, Z, phi] no phi for axisymmetrix particles
            
            particles(i_free(k))%v(1)  = rec_v_R(ife)   / (particles(i_free(k))%weight * CENTRAL_MASS * ATOMIC_MASS_UNIT )/ sqrt_mu0_over_rho0 !m/s
            particles(i_free(k))%v(2)  = rec_v_Z(ife)   / (particles(i_free(k))%weight * CENTRAL_MASS * ATOMIC_MASS_UNIT )/ sqrt_mu0_over_rho0
            particles(i_free(k))%v(3)  = rec_v_phi(ife) / (particles(i_free(k))%weight * CENTRAL_MASS * ATOMIC_MASS_UNIT )/ sqrt_mu0_over_rho0
            !< v = momentum fluid lost to recombination / (mass of superparticle)
          end do ! parts_per_element
    
        enddo   !ife 
      !$omp end parallel do
    end select
    !end omp
  
    call MPI_REDUCE(sanity_rec_local, total_sanity_rec, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    if (sim%my_id .eq. 0) then
      write(*,*) 'SANITY recombination weight : ' , total_sanity_rec 
      write(*,*) 'SANITY Recombination rate  [#/s] : ' , total_sanity_rec /tstep_fluid_si
    endif  
  
  end subroutine !do_1particle_recombination

end module mod_particle_recomb