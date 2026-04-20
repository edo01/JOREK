module mod_re_avalanche
    use particle_tracer
    use mod_particle_io
    use mod_particle_diagnostics
    use mod_fields_linear   
    use mod_fields_hermite_birkhoff 
    use mod_kinetic_relativistic
    use mod_gc_relativistic
    use mod_sampling
    use mod_basisfunctions
    use mod_random_seed
    use mod_ccoll_relativistic
    use mod_pusher_tools
    implicit none
    private 

    public :: reorder_indices, Moller_scatt, do_binning_mpi

contains 

  ! Subroutine to reorder the particles 
  ! Let N_tot be the total amount of markers and N_active the amount of markers with i_elm>0
  ! Output of the subroutine is array emptyind which holds the indices of all the active 
  ! markers in emptyind(1:N_active) and all inactive markers in emptyind((N_active+1):N_tot)
  subroutine reorder_indices(sim, n_part_bound, emptyind)
    use mod_particle_io 
    use mod_kinetic_relativistic
    
    type(particle_sim)                  :: sim

    integer*4 :: n_part_bound   ! <-- size of sim%groups(1)%particles(:)
    integer*4 :: emptyind(:)    ! <-- output array 
    
    integer*4 :: i_empty, i_full, storedindex, j_empty, j_full
    real*8 :: start_time, end_time, tot_time(3)
    
    j_empty = 1
    j_full = n_part_bound
    i_empty = 1
    i_full = n_part_bound

    start_time = MPI_WTIME()

    ! Loop until the index of the inactive particle is less than the index of the active particle
    do while (i_empty .lt. i_full)

        ! Start from the beginning of the array (or from a previously found position)
        ! and search towards the right until an index is found which corresponds to an inactive marker
        do i_empty=j_empty,n_part_bound+1
            if ((sim%groups(1)%particles(emptyind(i_empty))%i_elm .le. 0) .or. (i_empty .ge. i_full))then 
              j_empty = i_empty
              exit
            end if 
        end do

        ! Start from the end of the array (or from a previously found position)
        ! and search towards the left until an index is found which corresponds to an active marker
        do i_full=j_full,1,-1
            if ((sim%groups(1)%particles(emptyind(i_full))%i_elm .gt. 0) .or. (i_empty .ge. i_full)) then 
              j_full = i_full
              exit
            end if 
        end do

        if (i_empty .ge. i_full) exit ! if inactive spot is to the right of the active spot, sorting is finished 

        storedindex = emptyind(i_full) ! store particle index corresponding to the active spot

        ! save the particle index corresponding to the inactive marker where the active marker used to be
        emptyind(i_full) = emptyind(i_empty)
        ! save the particle index corresponding to the active marker where the inactive marker used to be 
        emptyind(i_empty) = storedindex

    end do 

    end_time = MPI_WTIME()
    tot_time = mpi_minmeanmax(end_time-start_time)
    if (sim%my_id .eq. 0) then
        write(*,"(A,3f10.3,A)") , "Time taken for reordering finished in (min/mean/max): ", tot_time, " seconds"
    endif
  end subroutine reorder_indices

  subroutine Moller_scatt(sim, gamma_min, dt_coll, n_part, n_part_bound, n_part_old, emptyind)
    use mod_particle_io 
    use mod_kinetic_relativistic
    use constants, only: mu_zero, k_boltz, pi, eps_zero, el_chg, mass_proton, c_light
    use mod_random_seed
    
    type(particle_sim)                  :: sim
    type(particle_kinetic_relativistic) :: particle_kin_rel

    type(pcg32_rng), dimension(:), allocatable ::  rng1, rng2, rng3

    real*8 :: gamma_min, gamma, gamma_0, gamma_rand, theta, phi, sigma, re
    real*8 :: rand, Pgamma, gamma_prime, ne, Te, dt_coll
    real*8 :: tang1(3), tang2(3), pnorm(3), vector(3)
    integer*4 :: n_part, n_part_old, n_part_bound, emptyind(:), j !Nbin, test, 
    real*8 :: E(3), B(3), Bhat(3), psi, U, Ec
    real*8 :: rand_rng1(1), rand_rng2(1), rand_rng3(1)
    integer*4 :: counter
    integer*4 :: process_rank, process_rank_plus_one, ierror
    character(len=50) :: part_save
    character(len=10) :: file_id

    integer*4 :: amount_processes, i, ifail
    real*8 :: vel(3), prob, n1(3), n2(3)

    real*8 :: start_time, end_time, tot_time(3)

    start_time = MPI_WTIME()

    call reorder_indices(sim, n_part_bound, emptyind)

    call MPI_COMM_RANK(MPI_COMM_WORLD, process_rank, ierror)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, amount_processes, ierror)

    allocate(rng1(1))
    allocate(rng2(1))
    allocate(rng3(1))

    process_rank_plus_one = process_rank+1

    do i = 1, 1
      call rng1(i)%initialize(n_dims=1, seed=random_seed(), n_streams=amount_processes, i_stream=process_rank_plus_one, ierr=ifail)
      call rng2(i)%initialize(n_dims=1, seed=random_seed(), n_streams=amount_processes, i_stream=process_rank_plus_one, ierr=ifail)
      call rng3(i)%initialize(n_dims=1, seed=random_seed(), n_streams=amount_processes, i_stream=process_rank_plus_one, ierr=ifail)
    end do

    do j=1,n_part_old

      select type (pini=>sim%groups(1)%particles(emptyind(j)))
      type is (particle_kinetic_relativistic)

      if (pini%i_elm .le. 0) cycle

      call sim%fields%calc_EBpsiU(sim%time, pini%i_elm, pini%st, pini%x(3), E, B, psi, U)

      if (psi .gt. 1.d0) cycle  !!!!! -- double check/ correct -- !!!!

      ne = 1.d19

      vel = pini%p/sqrt(dot_product(pini%p, pini%p)/(c_light**2.d0) + sim%groups(1)%mass**2.d0)

      gamma_0 = 1/(sim%groups(1)%mass*c_light)*sqrt(dot_product(pini%p,pini%p) + sim%groups(1)%mass**2.d0*c_light**2.d0)

      if (gamma_0 .le. gamma_min) cycle
      if (0.5d0*(gamma_0 + 1) .lt. gamma_min) cycle

      re = el_chg**2.d0/(4.d0*pi*eps_zero*(9.109d-31)*c_light**2.d0) ! Classical electron radius
      sigma = (4.d0*pi*re**2.d0/(gamma_0**2.d0-1.d0))*(gamma_0**2.d0*(gamma_0-2.d0*gamma_min+1.d0)/((gamma_min-1.d0)*(gamma_0-gamma_min)) &
              + 0.5d0*(gamma_0+1.d0)- gamma_min -(2.d0*gamma_0-1.d0)/(gamma_0-1.d0)*log((gamma_0-gamma_min)/(gamma_min-1.d0)))
      prob = sigma*ne*sqrt(dot_product(vel,vel))*dt_coll/2.d0

      
      if ((prob .lt. 0.d0) .or. (prob .gt. 1.d0)) then
        write(*,*) 'probability', prob
        write(*,*) 'Something is wrong, sigma', sigma, 'vel', sqrt(dot_product(vel,vel)), 'ne', ne, 'dt_coll', dt_coll, 'gamma_0', gamma_0
      end if 

      counter = 0
      do !while(found .eq. .false.)
        call rng1(1)%next(rand_rng1)
        call rng2(1)%next(rand_rng2)
        rand = rand_rng1(1)
        gamma_rand = rand_rng2(1)
        gamma = gamma_rand*(1+gamma_0 - 2*gamma_min) + gamma_min
        rand = rand*((gamma_0/(gamma_min-1))**2 + (gamma_0/(gamma_0-gamma_min))**2 +1 - (2*gamma_0-1)/(gamma_0-1)*(1/(gamma_min-1)+1/(gamma_0-gamma_min)))/&
        (sigma/(2*pi*re**2/(gamma_0**2-1)))
        Pgamma = ((gamma_0/(gamma-1))**2 + (gamma_0/(gamma_0-gamma))**2 +1 - (2*gamma_0-1)/(gamma_0-1)*(1/(gamma-1)+1/(gamma_0-gamma)))/&
                (sigma/(2*pi*re**2/(gamma_0**2-1)))
        if (rand .le. Pgamma) exit
        counter = counter+1
      end do 

      theta = acos(sqrt((gamma-1)/(gamma+1))*sqrt((gamma_0+1)/(gamma_0-1)))
      gamma_prime = gamma_0 + 1 -gamma
      phi = asin(sqrt((gamma**2-1)/(gamma_prime**2-1))*sin(theta))

      if (cos(theta) .lt. 0) write(*,*) 'Apperently cosine theta can be negative?'
      if (cos(phi) .lt. 0) write(*,*) 'Apperently cosine phi can be negative?'

      if ((gamma .ge. gamma_0) .or. (gamma_prime .ge. gamma_0)) write(*,*) 'Particle after collision has higher energy than initial RE, gamma_0:', gamma_0, 'gamma:', gamma, 'gamma_prime', gamma_prime

      call rng3(1)%next(rand_rng3)

      rand = rand_rng3(1)

      !n1 = [0.d0, pini%p(3), - pini%p(2)]/sqrt(pini%p(2)**2 + pini%p(3)**2)
      !n2 = [-1/pini%p(1)*(pini%p(2)**2/pini%p(3)+pini%p(3))*(1/(pini%p(1))**2*(pini%p(2)**2/pini%p(3)+pini%p(3))**2+1+pini%p(2)**2/(pini%p(3)**2))**(-0.5d0), pini%p(2)/pini%p(3)*(1/(pini%p(1))**2*(pini%p(2)**2/pini%p(3)+pini%p(3))**2+1+pini%p(2)**2/(pini%p(3)**2))**(-0.5d0), (1/(pini%p(1))**2*(pini%p(2)**2/pini%p(3)+pini%p(3))**2+1+pini%p(2)**2/(pini%p(3)**2))**(-0.5d0)]
      call get_orthonormals(pini%p,n1,n2)
      vector = cos(rand*2*Pi)*n1 + sin(rand*2*Pi)*n2

      ! Select target and colliding markers
      select type (pc=>sim%groups(1)%particles(emptyind(n_part_old+1+(j-1)*2)))
      type is (particle_kinetic_relativistic)

      select type (pt=>sim%groups(1)%particles(emptyind(n_part_old+1+(j-1)*2 + 1)))
      type is (particle_kinetic_relativistic)

      pc = pini
      pt = pini

      pc%weight = prob*pini%weight
      pt%weight = prob*pini%weight
      pini%weight = (1-prob)*pini%weight

      pnorm = pini%p/sqrt(dot_product(pini%p,pini%p))
      !if (dot_product(pnorm, vector) .ne. 0.d0) write(*,*) 'vector not orthogonal to pnorm', dot_product(pnorm, vector), 'size vector2', dot_product(vector, vector)
      pt%p = sim%groups(1)%mass*c_light*sqrt(gamma**2-1)*(cos(theta)*pnorm - sin(theta)*vector)
      pc%p = sim%groups(1)%mass*c_light*sqrt(gamma_prime**2-1)*(cos(phi)*pnorm +sin(phi)*vector)

      n_part = n_part+2

      if ((pini%weight + pc%weight + pt%weight) .lt. pini%weight/(1-prob)) write(*,*) '!!!!!!!!!!!!!!!!!!! mistake in weights in scattering !!!!!!!!!!!!!!!!!!!!!!!!!', 'prob', prob, 'pini%weight', pini%weight, 'pc%weight', pc%weight, 'pt%weight', pt%weight

      end select
      end select
      end select

    end do

    end_time = MPI_WTIME()
    tot_time = mpi_minmeanmax(end_time-start_time)
    if (sim%my_id .eq. 0) then
        write(*,"(A,3f10.3,A)") , "Time taken for collisions finished in (min/mean/max): ", tot_time, " seconds"
    endif
  end subroutine Moller_scatt

  subroutine do_binning_mpi(sim, n_part, n_part_bound, n_limit, Nbin, Ntor, element_part_s, element_part_t, new_markers_per_element, emptyind, iterations)
    use mod_particle_io 
    use mod_kinetic_relativistic
    use constants, only: mu_zero, k_boltz, pi, eps_zero, el_chg, mass_proton, c_light
    use mod_sobseq_rng
    use mod_pcg32_rng
    use mod_random_seed
    use mod_coordinate_transforms, only: vector_cylindrical_to_cartesian
    use mod_interp, only: interp_RZ
    use mpi_f08
    
    type(particle_sim)                  :: sim
    type(particle_kinetic_relativistic) :: particle_kin_rel
    type(particle_gc_relativistic)      :: p_gc, p_gc_array(n_part)
    type(particle_gc_relativistic)      :: particle_gc_rel

    integer*4 :: n_part, n_part_old, n_part_bound, Ntor, index 
    integer*4 :: g, h, i, j, k, l, m, n
    real*8 :: weighttot, rand(1) 
    real*8, allocatable ::  cum_prob(:)
    real*8 :: B(3), B_hat(3), e1(3), e2(3), gyro_angle(1)
    integer*4, allocatable :: particles_in_volume(:,:,:,:), amount_markers_spatial_bin(:,:,:), next_empty(:,:,:)
    integer*4 :: countk, countbin
    integer*4 :: new_markers_per_element, emptyind(:)
    integer*4 :: check

    real*8  :: R, Z, phi, s, t, DUMMY_REAL
    real*8  :: R_s, R_t, Z_s, Z_t, f
    real*8  :: Rbox(2), Zbox(2)
    integer, dimension(n_vertex_max) :: vertices
    real*8 :: minR, minZ, minPhi, maxR, maxZ, maxPhi
    real*8 :: A_RZ

    real*8 :: p_para, mu 
    real*8, allocatable :: p_para_min(:,:,:,:), p_para_max(:,:,:,:), mu_min(:,:,:,:), mu_max(:,:,:,:)
    integer*4 :: Nbin, pk, i_elm_find
    real*8, allocatable :: p_para_mu_array(:), p_para_mu_array2(:,:,:,:,:,:), weighttot2(:,:,:,:)

    integer*4 :: element_part_s, element_part_t
    integer*4 :: iterations

    real*8 :: rand_sob(2), ran(4), rand_sob2(1), factor
    type(pcg32_rng), dimension(:), allocatable ::  rng1, rng, rng2

    real*8 :: B_val_old

    real*8 :: gc_st(2), gc_phi
    integer*4 :: gc_i_elm

    real*8 ::  pvec(3), ran_p_mu(2)

    integer*4 :: process_rank, amount_processes, ki, kj, process_rank_plus_one, counter, ierr

    integer*4 :: kii, kjj, n_limit, test, new_markers_per_element_loop, new_markers_loop_count, ifail

    real*8 ::  E(3), psi, U
    real*8 :: start_time, end_time, tot_time(3)

    call MPI_COMM_RANK(MPI_COMM_WORLD, process_rank)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, amount_processes)
    if (sim%my_id .eq. 0) write(*,*) 'Amount processes (compare with jobpart settings)', amount_processes, 'rank', process_rank

    
    allocate(rng(amount_processes))
    allocate(rng1(amount_processes))
    allocate(rng2(amount_processes))

    process_rank_plus_one = process_rank+1

    do i = 1, amount_processes
      call rng(i)%initialize(n_dims=4, seed=2, n_streams=amount_processes, i_stream=i, ierr=ifail)
      call rng1(i)%initialize(n_dims=1, seed=3, n_streams=amount_processes, i_stream=i, ierr=ifail)
      call rng2(i)%initialize(n_dims=2, seed=4, n_streams=amount_processes, i_stream=i, ierr=ifail)
    end do

    countk = 0
    countbin = 0

    factor = 1.d0
    
    allocate(p_para_min(Ntor,sim%fields%element_list%n_elements,element_part_s,element_part_t)) 
    allocate(p_para_max(Ntor,sim%fields%element_list%n_elements,element_part_s,element_part_t))
    allocate(mu_min(Ntor,sim%fields%element_list%n_elements,element_part_s,element_part_t))
    allocate(mu_max(Ntor,sim%fields%element_list%n_elements,element_part_s,element_part_t))

    p_para_min(:,:,:,:) = 1.d30
    p_para_max(:,:,:,:) = -1.d30
    mu_min(:,:,:,:) = 1.d30
    mu_max(:,:,:,:) = -1.d30

    if (sim%my_id .eq. 0) write(*,*), 1


    allocate(p_para_mu_array(Nbin**2), cum_prob(Nbin**2))
    allocate(p_para_mu_array2(Ntor, sim%fields%element_list%n_elements, element_part_s, element_part_t,Nbin**2, 3))
    allocate(weighttot2(Ntor, sim%fields%element_list%n_elements, element_part_s, element_part_t))

    weighttot2(:,:,:,:) = 0.d0

    p_para_mu_array(:) = 0.d0
    p_para_mu_array2(:,:,:,:,:,:) = 0.d0

    call reorder_indices(sim, n_part_bound, emptyind)

    allocate(amount_markers_spatial_bin(Ntor, sim%fields%element_list%n_elements, element_part_s*element_part_t))

    amount_markers_spatial_bin(:,:, :) = 0

    if (sim%my_id .eq. 0) write(*,*) 1.25d0

    index = 1

    weighttot = 0.d0

    start_time = MPI_WTIME()

    pk = 1
    do k = 1,n_part
        select type(p=>sim%groups(1)%particles(emptyind(k)))
        type is (particle_kinetic_relativistic)

          weighttot = weighttot + p%weight

          gc_i_elm = p%i_elm
          gc_st = p%st
          gc_phi = p%x(3)

          do i=1,iterations
              if (gc_i_elm .le. 0) exit
              call sim%fields%calc_EBpsiU(sim%time, gc_i_elm, gc_st, gc_phi, E, B, psi, U)
              p_gc = relativistic_kinetic_to_relativistic_gc(sim%fields%node_list, sim%fields%element_list, p, sim%groups(1)%mass, B)
              gc_i_elm = p_gc%i_elm
              gc_st = p_gc%st 
              gc_phi = p_gc%x(3)
          end do 

          if (p_gc%i_elm .gt. 0) then
            p_gc_array(pk) = p_gc

            if (p_gc_array(pk)%x(3) .lt. 0.d0) p_gc_array(pk)%x(3) = p_gc_array(pk)%x(3) + 2*Pi

            j = int(p_gc_array(pk)%x(3)/(2*Pi)*real(Ntor,8)) + 1
            if (j .eq. (Ntor + 1)) j = 1 
            h = int(p_gc_array(pk)%st(1)*real(element_part_s,8)) + 1
            if (h .eq. (element_part_s + 1)) h = element_part_s
            g = int(p_gc_array(pk)%st(2)*real(element_part_t,8)) + 1
            if (g .eq. (element_part_t + 1)) g = element_part_t
    
            amount_markers_spatial_bin(j, p_gc_array(pk)%i_elm, h+ (g-1)*element_part_s) = amount_markers_spatial_bin(j, p_gc_array(pk)%i_elm, h + (g-1)*element_part_s) + 1
            if (p_gc_array(pk)%p(1) .gt. p_para_max(j,p_gc_array(pk)%i_elm,h,g)) p_para_max(j,p_gc_array(pk)%i_elm,h,g) = p_gc_array(pk)%p(1)
            if (p_gc_array(pk)%p(1) .lt. p_para_min(j,p_gc_array(pk)%i_elm,h,g)) p_para_min(j,p_gc_array(pk)%i_elm,h,g) = p_gc_array(pk)%p(1)
            if (p_gc_array(pk)%p(2) .gt. mu_max(j,p_gc_array(pk)%i_elm,h,g)) mu_max(j,p_gc_array(pk)%i_elm,h,g) = p_gc_array(pk)%p(2)
            if (p_gc_array(pk)%p(2) .lt. mu_min(j,p_gc_array(pk)%i_elm,h,g)) mu_min(j,p_gc_array(pk)%i_elm,h,g) = p_gc_array(pk)%p(2)

            if (p_gc_array(pk)%p(2) .le. 0.d0) write(*,*) 'Marker with negative mu', p_gc_array(pk)%p(2)

            pk = pk +1

          end if
        end select
    end do 
    n_part = count(p_gc_array(:)%i_elm .gt. 0)
    if (sim%my_id .eq. 0) write(*,*) 'weighttot p_gc_array', weighttot
    weighttot = 0.d0

    if (sim%my_id .eq. 0) write(*,*) 1.5d0

    call MPI_ALLREDUCE(MPI_IN_PLACE, p_para_max, size(p_para_max), MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD) 
    call MPI_ALLREDUCE(MPI_IN_PLACE, p_para_min, size(p_para_min), MPI_DOUBLE_PRECISION, MPI_MIN, MPI_COMM_WORLD) 
    call MPI_ALLREDUCE(MPI_IN_PLACE, mu_max, size(mu_max), MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD) 
    call MPI_ALLREDUCE(MPI_IN_PLACE, mu_min, size(mu_min), MPI_DOUBLE_PRECISION, MPI_MIN, MPI_COMM_WORLD) 

    call MPI_ALLREDUCE(MPI_IN_PLACE, amount_markers_spatial_bin, size(amount_markers_spatial_bin), MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD) 

    p_para_max = p_para_max + 0.5d0*(p_para_max-p_para_min)/real(Nbin-1,8)!*1.0001d0
    p_para_min = p_para_min - 0.5d0*(p_para_max-p_para_min)/real(Nbin-1,8)!(:,:,:,:)*0.9999d0
    mu_max = mu_max + 0.5d0*(mu_max-mu_min)/real(Nbin-1,8)!(:,:,:,:)*1.0001d0
    mu_min = mu_min - 0.5d0*(mu_max-mu_min)/real(Nbin-1,8)!(:,:,:,:)*0.9999d0

    end_time = MPI_WTIME()
    tot_time = mpi_minmeanmax(end_time-start_time)
    if (sim%my_id .eq. 0) then
        write(*,"(A,3f10.3,A)") , "Time taken for converting all particles & MPI_ALLREDUCE finished in (min/mean/max): ", tot_time, " seconds"
    endif


    start_time = MPI_WTIME()

    allocate(particles_in_volume(Ntor, sim%fields%element_list%n_elements, element_part_s*element_part_t, maxval(amount_markers_spatial_bin(:,:, :))))

    allocate(next_empty(Ntor, sim%fields%element_list%n_elements, element_part_s*element_part_t))

    particles_in_volume(:,:,:, :) = 0
    next_empty(:,:, :) = 1

    if (sim%my_id .eq. 0) write(*,*) 2

    do k = 1, n_part

      j = int(p_gc_array(k)%x(3)/(2*Pi)*real(Ntor,8)) + 1
      if (j .eq. (Ntor + 1)) j = 1 
      h = int(p_gc_array(k)%st(1)*real(element_part_s,8)) + 1
      if (h .eq. (element_part_s + 1)) h = element_part_s
      g = int(p_gc_array(k)%st(2)*real(element_part_t,8)) + 1
      if (g .eq. (element_part_t + 1)) g = element_part_t

      particles_in_volume(j, p_gc_array(k)%i_elm, h+(g-1)*element_part_s, next_empty(j, p_gc_array(k)%i_elm, h+(g-1)*element_part_s)) = k
      next_empty(j, p_gc_array(k)%i_elm, h+(g-1)*element_part_s) = next_empty(j,p_gc_array(k)%i_elm, h+(g-1)*element_part_s) + 1

    end do

    end_time = MPI_WTIME()
    tot_time = mpi_minmeanmax(end_time-start_time)
    if (sim%my_id .eq. 0) then
        write(*,"(A,3f10.3,A)") , "Time taken for volume bining finished in (min/mean/max): ", tot_time, " seconds"
    endif

    if (sim%my_id .eq. 0) write(*,*) 3

    if (sim%my_id .eq. 0) write(*,*) 'amount of elements in which there is at least 1 marker', count(amount_markers_spatial_bin .ne. 0)

    start_time = MPI_WTIME()
    countk = 1
    do g = 1,element_part_t     
      do h=1,element_part_s   
        do i = 1, sim%fields%element_list%n_elements
          do j = 1, Ntor 

            if (mu_min(j,i,h,g) .le. 0.d0) mu_min(j,i,h,g) = 0.d0
            
            if (particles_in_volume(j, i, h+(g-1)*element_part_s, 1) .eq. 0) cycle 

            if (p_para_max(j,i,h,g) .lt. p_para_min(j,i,h,g)) write(*,*) 'p_para_max < p_para_min'
            if (mu_max(j,i,h,g) .lt. mu_min(j,i,h,g)) write(*,*) 'p_para_max < p_para_min'


            weighttot = 0.d0
            weighttot2(j,i,h,g) = 0.d0
            p_para_mu_array(:) = 0.d0
            p_para_mu_array2(j,i,h,g,:,1) = 0.d0

            if (p_para_max(j,i,h,g) .eq. p_para_min(j,i,h,g)) then 
               p_para_max(j,i,h,g) = p_para_max(j,i,h,g)*1.0001d0
               p_para_min(j,i,h,g) = p_para_min(j,i,h,g)*0.9999d0
            end if 
            if (mu_max(j,i,h,g) .eq. mu_min(j,i,h,g)) then
               mu_max(j,i,h,g) = mu_max(j,i,h,g)*1.0001d0
               mu_min(j,i,h,g) = mu_min(j,i,h,g)*0.9999d0
            end if
            if (mu_min(j,i,h,g) .lt. 0.d0) mu_min(j,i,h,g) = 0.d0
          

              do k = 1, size(particles_in_volume(j,i, h+(g-1)*element_part_s,:))
                  if (particles_in_volume(j, i, h+(g-1)*element_part_s, k) .eq. 0) exit
                  if (p_gc_array(particles_in_volume(j,i,h+(g-1)*element_part_s,k))%weight .lt. 0.d0) then
                    write(*,*) 'There is a particle with negative weight'
                    cycle
                  end if  

                  weighttot = weighttot + p_gc_array(particles_in_volume(j,i, h+(g-1)*element_part_s,k))%weight

                  if (p_gc_array(particles_in_volume(j,i, h+(g-1)*element_part_s, k))%i_elm .le. 0) then
                    write(*,*) '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Element number is less than zero', sim%groups(1)%particles(particles_in_volume(j,i,h+(g-1)*element_part_s,k))%i_elm
                  end if

                  kii = INT((p_gc_array(particles_in_volume(j,i,h+(g-1)*element_part_s,k))%p(1) - p_para_min(j,i,h,g))/(p_para_max(j,i,h,g) - p_para_min(j,i,h,g))*Nbin+1)
                  kjj = INT((p_gc_array(particles_in_volume(j,i,h+(g-1)*element_part_s,k))%p(2) - mu_min(j,i,h,g))/(mu_max(j,i,h,g) - mu_min(j,i,h,g))*Nbin+1)

                  p_para_mu_array(kjj+(kii-1)*Nbin) = p_para_mu_array(kjj+(kii-1)*Nbin) + p_gc_array(particles_in_volume(j,i,h+(g-1)*element_part_s,k))%weight

              end do
              
              p_para_mu_array2(j,i,h,g,:,1) = p_para_mu_array(:)
              weighttot2(j,i,h,g) = weighttot

            end do
            end do
          end do
        end do

        call MPI_ALLREDUCE(MPI_IN_PLACE, p_para_mu_array2(:,:,:,:,:,1), size(p_para_mu_array2(:,:,:,:,:,1)), MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD) 
        call MPI_ALLREDUCE(MPI_IN_PLACE, weighttot2, size(weighttot2), MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD) 

        end_time = MPI_WTIME()
        tot_time = mpi_minmeanmax(end_time-start_time)
        if (sim%my_id .eq. 0) then
            write(*,"(A,3f10.3,A)") , "Time taken for momentum binning finished in (min/mean/max): ", tot_time, " seconds"
        endif

        if (sim%my_id .eq. 0) write(*,*) 'Total weight in weighttot2', sum(weighttot2)


      start_time = MPI_WTIME()

      do g=1,element_part_t  
        do h=1,element_part_s  
          do i = 1, sim%fields%element_list%n_elements  
            do j = 1, Ntor
            test = 0

            weighttot = weighttot2(j,i,h,g)
            if (weighttot .eq. 0.d0) cycle

            if (amount_markers_spatial_bin(j,i,h+ (g-1)*element_part_s) .eq. 1) then
              weighttot = 0.d0
              if (particles_in_volume(j,i,h+(g-1)*element_part_s,1) .gt. 0) then 
                p_gc = p_gc_array(particles_in_volume(j,i,h+(g-1)*element_part_s,1))
                call sim%fields%calc_EBpsiU(sim%time, i, p_gc%st, p_gc%x(3), E, B, psi, U)
                call rng1(process_rank_plus_one)%next(gyro_angle)
                gyro_angle(1) = 0.d0
                select type(p=>sim%groups(1)%particles(index))
                type is (particle_kinetic_relativistic)
                  p = relativistic_gc_to_relativistic_kinetic(sim%fields%node_list, sim%fields%element_list, p_gc, sim%groups(1)%mass, B, gyro_angle(1))
                  weighttot = weighttot + p%weight
                end select
                if (sim%groups(1)%particles(index)%i_elm .le. 0) write(*,*) 'Single particle placed has been placed outside domain', sim%groups(1)%particles(index)%i_elm, p_gc%i_elm
                if (abs(sim%groups(1)%particles(index)%weight - weighttot) .gt. 1.d-6) write(*,*) 'Weights wrong when copying 1 marker', weighttot, sim%groups(1)%particles(index)%weight
                index = index +1
                test = 1
              end if 
              call MPI_ALLREDUCE(MPI_IN_PLACE, weighttot, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD) 
              if (abs(weighttot - weighttot2(j,i,h,g)) .gt. 1.d-10) write(*,*) 'Something wrong with total weight when 1 marker is placed', weighttot, weighttot2(j,i,h,g)
              cycle
            end if

            if (new_markers_per_element .gt. amount_markers_spatial_bin(j,i,h+(g-1)*element_part_s)) then
              weighttot = 0.d0
              do n = 1, amount_markers_spatial_bin(j,i,h+(g-1)*element_part_s)
                if (particles_in_volume(j,i,h+(g-1)*element_part_s,n) .eq. 0) exit
                p_gc = p_gc_array(particles_in_volume(j,i,h+(g-1)*element_part_s,n))
                call sim%fields%calc_EBpsiU(sim%time, i, p_gc%st, p_gc%x(3), E, B, psi, U)
                gyro_angle(1) = 0.d0
                select type(p=>sim%groups(1)%particles(index))
                type is (particle_kinetic_relativistic)
                  p = relativistic_gc_to_relativistic_kinetic(sim%fields%node_list, sim%fields%element_list, p_gc, sim%groups(1)%mass, B, gyro_angle(1))
                end select                
                weighttot = weighttot + sim%groups(1)%particles(index)%weight
                if (sim%groups(1)%particles(index)%i_elm .le. 0) write(*,*) 'Multiple but not resampled particle placed has been placed outside domain', sim%groups(1)%particles(index)%i_elm, p_gc%i_elm
                index = index +1
                test = 1
              end do
              call MPI_ALLREDUCE(MPI_IN_PLACE, weighttot, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD) 
              new_markers_per_element_loop = 0
              cycle
           else if (new_markers_per_element .lt. amount_processes) then
            new_markers_loop_count = 0
            new_markers_per_element_loop = 0
            do n=1,new_markers_per_element
               if (sim%my_id .eq. 0) call rng1(process_rank_plus_one)%next(rand)
               call MPI_BCAST(rand, 1, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD)
              if (sim%my_id .eq. int(rand(1)*real(amount_processes,8))) then
                
                 new_markers_per_element_loop = 1 + new_markers_per_element_loop
              end if 
            end do

            new_markers_loop_count = new_markers_per_element_loop

            call MPI_ALLREDUCE(MPI_IN_PLACE, new_markers_loop_count, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD)
            if (new_markers_loop_count .ne. new_markers_per_element) write(*,*) 'new_markers_loop_count: ', new_markers_loop_count, 'new_markers_per_element: ', new_markers_per_element, 'diff; ', new_markers_per_element - new_markers_loop_count 
           else
              new_markers_per_element_loop = int(real(new_markers_per_element, 8)/real(amount_processes, 8))
          end if


            if (test .eq. 1) write(*,*) 'The cycle is not working correctly'

            p_para_mu_array2(j,i,h,g,:,1) = p_para_mu_array2(j,i,h,g,:,1)/weighttot

            cum_prob = p_para_mu_array2(j,i,h,g,:,1)
            do m = 2, size(p_para_mu_array2(j,i,h,g,:,1))
              cum_prob(m) = cum_prob(m-1) + p_para_mu_array2(j,i,h,g,m,1)
            end do 
            if (abs(cum_prob(size(cum_prob)) - 1.d0) .gt. 1.d-6) write(*,*) 'last value cum_prob:', cum_prob(size(cum_prob))

            vertices = sim%fields%element_list%element(i)%vertex(:)
            minR = real(minval(sim%fields%node_list%node(vertices)%x(1,1,1)), 8)
            maxR = real(maxval(sim%fields%node_list%node(vertices)%x(1,1,1)), 8)
            minZ = real(minval(sim%fields%node_list%node(vertices)%x(1,1,2)), 8)
            maxZ = real(maxval(sim%fields%node_list%node(vertices)%x(1,1,2)), 8)
            Rbox = [minR - 0.02d0, maxR + 0.02d0]
            Zbox = [minZ - 0.02d0, maxZ + 0.02d0]

            A_RZ = (Rbox(2) - Rbox(1))*(Zbox(2)-Zbox(1))

            factor = 1.d0

            check = 0


            do n=1, new_markers_per_element_loop
              call rng1(process_rank_plus_one)%next(rand)

                call rng(process_rank_plus_one)%next(ran)
                if ((ran(2) .gt. 1) .or. (ran(2) .lt. 0) .or. (ran(3) .gt. 1) .or. (ran(3) .lt. 0)) then
                  write(*,*) 'ran2', ran(2), 'ran3', ran(3), 'process', process_rank
                  ran(2) = abs(ran(2))
                  ran(3) = abs(ran(3))
                  write(*,*) 'Took absolute values'
                end if 
                ran(2) = real(h-1, 8)/real(element_part_s,8) + ran(2)/real(element_part_s,8)
                ran(3) = real(g-1,8)/real(element_part_t,8) + ran(3)/real(element_part_t,8)
                if ((ran(2) .lt. real(h-1,8)/real(element_part_s,8)) .or. (ran(2) .gt. real(h,8)/real(element_part_s,8))) write(*,*) 'There is a mistake in ran(2)'
                if ((ran(3) .lt. real(g-1,8)/real(element_part_t,8)) .or. (ran(3) .gt. real(g,8)/real(element_part_t,8))) write(*,*) 'There is a mistake in ran(3)'

                l = minloc(cum_prob, dim=1, mask=((rand(1) .gt. [0.d0, cum_prob(1:(size(cum_prob)-1))]) .and. (rand(1) .le. cum_prob)))

                countbin = countbin+1

                  call rng2(process_rank_plus_one)%next(ran_p_mu)

                  p_para = (1/real(Nbin,8)*(real(l,8)-real(modulo(l,Nbin),8))+1.d0 -ran_p_mu(1))*(p_para_max(j,i,h,g)-p_para_min(j,i,h,g))/real(Nbin,8) + p_para_min(j,i,h,g)
                  if (modulo(l,Nbin) .eq. 0) p_para = (1.d0+1.d0-ran_p_mu(1))*(p_para_max(j,i,h,g)-p_para_min(j,i,h,g))/real(Nbin,8) + p_para_min(j,i,h,g)

                  mu = (real(modulo(l,Nbin), 8) -ran_p_mu(2))*(mu_max(j,i,h,g)-mu_min(j,i,h,g))/real(Nbin,8) + mu_min(j,i,h,g)
                  if (modulo(l,Nbin) .eq. 0) mu = (real(Nbin,8) -ran_p_mu(2))*(mu_max(j,i,h,g)-mu_min(j,i,h,g))/real(Nbin,8) + mu_min(j,i,h,g)


                  if (mu .le. 0.d0) write(*,*) 'mu is zero', mu, 'mu max', mu_max(j,i,h,g), 'mu_min', mu_min(j,i,h,g), 'weight', weighttot, 'p_para', p_para

                  if ((p_para .eq. 0.d0) .or. (mu .eq. 0.d0)) write(*,*) 'some particle is placed with zero, second case', p_para, mu, rand, l

                  select type(p=>sim%groups(1)%particles(index))
                  type is (particle_kinetic_relativistic)

                    call rng1(process_rank_plus_one)%next(gyro_angle)

                    gyro_angle(1) = gyro_angle(1)*2.d0*pi
                    !gyro_angle(1) = 0.d0

                    p%q = -1

                    if (new_markers_per_element .le. amount_processes) then 
                      p%weight = weighttot/real(new_markers_per_element,8)
                    else 
                      p%weight = weighttot/(real(new_markers_per_element/amount_processes,8)*real(amount_processes,8))
                    end if 

                    counter = 0
                    do 
                      counter = counter +1
                        call interp_RZ(sim%fields%node_list, sim%fields%element_list, i, ran(2), ran(3), R, R_s, R_t, Z, Z_s, Z_t)
                        f = abs(R_s*Z_t - R_t*Z_s)/A_RZ
                        if (f .gt. 1) then 
                            write(*,*) 'value distribution function', f
                            factor = factor*1.d-1
                            cycle
                        end if
                        if (ran(1) .le. f) then
                            call sim%fields%calc_EBpsiU(sim%time, i, [ran(2), ran(3)], 2.d0*Pi/Ntor*ran(4) + 2.d0*Pi/Ntor*(real(j,8)-1.d0), E, B, psi, U)
                            p_gc = relativistic_kinetic_to_relativistic_gc(sim%fields%node_list, sim%fields%element_list, p, sim%groups(1)%mass, B)                                

                            p_gc%x(1) = R 
                            p_gc%x(2) = Z
                            p_gc%st(1) = ran(2)
                            p_gc%st(2) = ran(3)
                            p_gc%x(3) = 2.d0*Pi/Ntor*ran(4) + 2.d0*Pi/Ntor*(real(j,8)-1.d0) 

                            call find_RZ(sim%fields%node_list,sim%fields%element_list,p_gc%x(1),p_gc%x(2),DUMMY_REAL,DUMMY_REAL,p_gc%i_elm,p_gc%st(1),p_gc%st(2),ifail)
                            if (p_gc%i_elm .ne. i) write(*,*) 'element numbers do not match'
                            exit 
                        end if 
                        call rng(process_rank_plus_one)%next(ran)
                        ran(2) = real(h-1, 8)/real(element_part_s,8) + ran(2)/real(element_part_s,8)
                        ran(3) = real(g-1,8)/real(element_part_t,8) + ran(3)/real(element_part_t,8)
                        if ((ran(2) .lt. real(h-1,8)/real(element_part_s,8)) .or. (ran(2) .gt. real(h,8)/real(element_part_s,8))) write(*,*) 'There is a mistake in ran(2)'
                        if ((ran(3) .lt. real(g-1,8)/real(element_part_t,8)) .or. (ran(3) .gt. real(g,8)/real(element_part_t,8))) write(*,*) 'There is a mistake in ran(3)'

                    end do 

                    p_gc%p(1) = p_para
                    p_gc%p(2) = mu

                    p_gc%q = -1

                    call sim%fields%calc_EBpsiU(sim%time, i, p_gc%st, p_gc%x(3), E, B, psi, U)
                    particle_kin_rel = relativistic_gc_to_relativistic_kinetic(sim%fields%node_list, sim%fields%element_list, p_gc, sim%groups(1)%mass, B, gyro_angle(1))

                    B_val_old = dot_product(B,B)

                    if (particle_kin_rel%i_elm .gt. 0) call sim%fields%calc_EBpsiU(sim%time, particle_kin_rel%i_elm, particle_kin_rel%st, particle_kin_rel%x(3), E, B, psi, U)
                    particle_gc_rel = relativistic_kinetic_to_relativistic_gc(sim%fields%node_list, &
                                      sim%fields%element_list, particle_kin_rel, sim%groups(1)%mass, B)

                    if (particle_kin_rel%i_elm .gt. 0) then
                      p = particle_kin_rel
                      if (p%i_elm .ne. particle_kin_rel%i_elm) write(*,*) 'something is going wrong with saving particle kin rel to a particle!!!'
                      index = index +1
                      if ((p%i_elm .gt. 0) .and. (p%i_elm .lt. 64)) countk = countk+1
                    else 
                      write(*,*) 'particle_kin_rel%i_elm <= 0, current element gc', i, p_gc%i_elm, p_gc%x(1), p_gc%x(2), p_gc%x(3), particle_kin_rel%x
                      call get_orthonormals(B/norm2(B),e1,e2)

                      pvec = p_gc%p(1)*B_hat + sqrt(2.d0*sim%groups(1)%mass*norm2(B)*p_gc%p(2))*(e1*cos(gyro_angle(1))+e2*sin(gyro_angle(1)))
                      write(*,*) 'xout', p_gc%x + (ATOMIC_MASS_UNIT*cross_product(B/norm2(B),pvec))/(EL_CHG*real(p_gc%q,8)*norm2(B))
                    end if
                end select

                check = check + 1

              if (check .eq. 0) then
                write(*,*) 'Hier gaat iets niet goed'
              end if
            end do 

            if (abs(check - new_markers_per_element_loop) .gt. 1.d-6) write(*,*) 'check komt ook niet overeen met de hoeveelheid markers die geplaatst zouden moeten worden', check, new_markers_per_element

          end do
          end do
        end do

    end do

    end_time = MPI_WTIME()
    tot_time = mpi_minmeanmax(end_time-start_time)
    if (sim%my_id .eq. 0) then
        write(*,"(A,3f10.3,A)") , "Time taken for last resampling loop finished in (min/mean/max): ", tot_time, " seconds"
    endif


    if (sim%my_id .eq. 0) write(*,*) 'aantal markers tussen 0 en 64:', countk-1


    if (sim%my_id .eq. 0) write(*,*) 'last index - first index', index - (n_part +1)

    n_part_old = n_part

    n_part = count(sim%groups(1)%particles(:)%i_elm .gt. 0)

    if (sim%my_id .eq. 0) write(*,*) 'npart', n_part

    if (sim%my_id .eq. 0) write(*,*) 'amount of elements in which there is at least 1 marker', count(amount_markers_spatial_bin .ne. 0)
    if (sim%my_id .eq. 0) write(*,*) 'countk', countk
    if (sim%my_id .eq. 0) write(*,*) 'countbin', countbin

    sim%groups(1)%particles(index:n_limit)%i_elm = 0
    sim%groups(1)%particles(index:n_limit)%weight = 0

    if (sim%my_id .eq. 0) write(*,*) 'Total weight at end do binning', sum(sim%groups(1)%particles(:)%weight)

    start_time = MPI_WTIME()

    emptyind = (/(l, l=1,n_limit, 1)/)

    end_time = MPI_WTIME()
    tot_time = mpi_minmeanmax(end_time-start_time)
    if (sim%my_id .eq. 0) then
        write(*,"(A,3f10.3,A)") , "Time taken for remaking emptyind finished in (min/mean/max): ", tot_time, " seconds"
    endif


    start_time = MPI_WTIME()
    
    deallocate(rng)
    deallocate(rng1)
    deallocate(rng2)

    deallocate(p_para_min)
    deallocate(p_para_max)
    deallocate(mu_min)
    deallocate(mu_max)

    deallocate(cum_prob)
    deallocate(p_para_mu_array)
    deallocate(p_para_mu_array2)
    deallocate(weighttot2)

    deallocate(amount_markers_spatial_bin)

    deallocate(particles_in_volume)
    deallocate(next_empty)

    end_time = MPI_WTIME()
    tot_time = mpi_minmeanmax(end_time-start_time)
    if (sim%my_id .eq. 0) then
        write(*,"(A,3f10.3,A)") , "Time taken for deallocating everythin finished in (min/mean/max): ", tot_time, " seconds"
    endif


  end subroutine do_binning_mpi


end module mod_re_avalanche
