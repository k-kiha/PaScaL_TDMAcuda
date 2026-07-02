program example_fortran_profile
    use mpi
    use cudafor
    use PaScaL_TDMA_cuda
    implicit none

    integer :: ierr, myrank, nprocs
    integer :: ngpu, gpurank
    integer :: n1, n2, n3, iterations
    integer :: nthread_modithomas, nthread_reduced
    integer :: n1sub, n2sub, n3sub, nsys
    integer :: ia, ib, iter
    integer :: nrow_min, nrow_max
    real(8) :: t0, t1, elapsed, elapsed_max, elapsed_sum

    real(8), allocatable, dimension(:,:,:) :: Aa, Ab, Ac, B
    real(8), allocatable, dimension(:,:,:), device :: Aa_d, Ab_d, Ac_d, B_d
    type(ptdma_plan_cuda) :: plan

    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)

    call parse_positive_arg(1, 64, n1, "n1", myrank)
    call parse_positive_arg(2, 64, n2, "n2", myrank)
    call parse_positive_arg(3, 2048, n3, "n3", myrank)
    call parse_positive_arg(4, 10, iterations, "iterations", myrank)
    call parse_positive_arg(5, 128, nthread_modithomas, "tdma_threads", myrank)
    call parse_positive_arg(6, 128, nthread_reduced, "reduced_threads", myrank)

    if (command_argument_count() > 6) then
        if (myrank == 0) then
            write(*,*) "usage: example_fortran_profile [n1] [n2] [n3]"
            write(*,*) "       [iterations] [tdma_threads] [reduced_threads]"
        endif
        call MPI_ABORT(MPI_COMM_WORLD, 1, ierr)
    endif

    ierr = cudaGetDeviceCount(ngpu)
    if (ngpu <= 0) then
        if (myrank == 0) write(*,*) "No CUDA device is visible."
        call MPI_ABORT(MPI_COMM_WORLD, 1, ierr)
    endif
    gpurank = mod(myrank, ngpu)
    ierr = cudaSetDevice(gpurank)
    ierr = cudaDeviceSynchronize()

    call para(0, n3 - 1, nprocs, myrank, ia, ib)
    n1sub = n1
    n2sub = n2
    n3sub = ib - ia + 1
    nsys = n1sub * n2sub

    call MPI_REDUCE(n3sub, nrow_min, 1, MPI_INTEGER, MPI_MIN, 0, MPI_COMM_WORLD, ierr)
    call MPI_REDUCE(n3sub, nrow_max, 1, MPI_INTEGER, MPI_MAX, 0, MPI_COMM_WORLD, ierr)

    allocate(Aa(0:n1sub-1,0:n2sub-1,0:n3sub-1), Aa_d(0:n1sub-1,0:n2sub-1,0:n3sub-1))
    allocate(Ab(0:n1sub-1,0:n2sub-1,0:n3sub-1), Ab_d(0:n1sub-1,0:n2sub-1,0:n3sub-1))
    allocate(Ac(0:n1sub-1,0:n2sub-1,0:n3sub-1), Ac_d(0:n1sub-1,0:n2sub-1,0:n3sub-1))
    allocate(B (0:n1sub-1,0:n2sub-1,0:n3sub-1), B_d (0:n1sub-1,0:n2sub-1,0:n3sub-1))

    Aa(:,:,:) = 1.0d0
    Ab(:,:,:) = -2.0d0
    Ac(:,:,:) = 1.0d0
    B (:,:,:) = 0.0d0
    if (myrank == 0) B(:,:,0) = -1.0d0
    if (myrank == nprocs - 1) B(:,:,n3sub-1) = -1.0d0

    Aa_d = Aa
    Ab_d = Ab
    Ac_d = Ac
    B_d  = B

    call pascal_plan_create(plan, nsys, MPI_COMM_WORLD, myrank, nprocs, &
                            nthread_modithomas, nthread_reduced)

    if (myrank == 0) then
        write(*,'(A)',advance='no') "solver,implementation,nranks,n1,n2,n3,nsys,"
        write(*,'(A)') "nrow_min,nrow_max,iter,iterations,mpi_mode,total_s_max,total_s_avg"
    endif

    do iter = 0, iterations - 1
        call MPI_BARRIER(MPI_COMM_WORLD, ierr)
        ierr = cudaDeviceSynchronize()
        t0 = MPI_WTIME()

        ! The profiling driver intentionally repeats only the solve call.
        ! Coefficients are not reinitialized between iterations.
        call pascal_solver(plan, Aa_d, Ab_d, Ac_d, B_d, nsys, n3sub)

        ierr = cudaDeviceSynchronize()
        t1 = MPI_WTIME()
        elapsed = t1 - t0

        call MPI_REDUCE(elapsed, elapsed_max, 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
        call MPI_REDUCE(elapsed, elapsed_sum, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

        if (myrank == 0) then
            call write_csv_row(nprocs, n1, n2, n3, nsys, nrow_min, nrow_max, &
                               iter, iterations, elapsed_max, elapsed_sum / dble(nprocs))
        endif
    enddo

    call pascal_plan_clean(plan)
    deallocate(Aa, Aa_d)
    deallocate(Ab, Ab_d)
    deallocate(Ac, Ac_d)
    deallocate(B, B_d)
    call MPI_FINALIZE(ierr)

contains

    subroutine parse_positive_arg(index, default_value, value, name, rank)
        integer, intent(in) :: index, default_value, rank
        integer, intent(out) :: value
        character(len=*), intent(in) :: name
        character(len=64) :: arg
        integer :: stat, ierr_local

        value = default_value
        if (command_argument_count() >= index) then
            call get_command_argument(index, arg)
            read(arg, *, iostat=stat) value
            if (stat /= 0 .or. value <= 0) then
                if (rank == 0) write(*,*) trim(name), " must be a positive integer."
                call MPI_ABORT(MPI_COMM_WORLD, 1, ierr_local)
            endif
        endif
    end subroutine parse_positive_arg

    subroutine write_csv_row(nprocs_arg, n1_arg, n2_arg, n3_arg, nsys_arg, &
                             nrow_min_arg, nrow_max_arg, iter_arg, iterations_arg, &
                             total_max_arg, total_avg_arg)
        integer, intent(in) :: nprocs_arg, n1_arg, n2_arg, n3_arg
        integer, intent(in) :: nsys_arg, nrow_min_arg, nrow_max_arg
        integer, intent(in) :: iter_arg, iterations_arg
        real(8), intent(in) :: total_max_arg, total_avg_arg

        write(*,'(A)',advance='no') "tdma,fortran-original,"
        write(*,'(I0,A)',advance='no') nprocs_arg, ","
        write(*,'(I0,A)',advance='no') n1_arg, ","
        write(*,'(I0,A)',advance='no') n2_arg, ","
        write(*,'(I0,A)',advance='no') n3_arg, ","
        write(*,'(I0,A)',advance='no') nsys_arg, ","
        write(*,'(I0,A)',advance='no') nrow_min_arg, ","
        write(*,'(I0,A)',advance='no') nrow_max_arg, ","
        write(*,'(I0,A)',advance='no') iter_arg, ","
        write(*,'(I0,A)',advance='no') iterations_arg, ","
        write(*,'(A)',advance='no') "device,"
        write(*,'(ES24.16,A)',advance='no') total_max_arg, ","
        write(*,'(ES24.16)') total_avg_arg
    end subroutine write_csv_row

end program example_fortran_profile
