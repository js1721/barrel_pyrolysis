!===================================================================================================
!
! CVODE solver module.
!
! Date:     Programmer:     Changes:
! =====     ===========     ========
! 29-01-26  J Salter        Original
!===================================================================================================

module mod_cvode_wrapper 

    use, intrinsic :: iso_c_binding
    use fsundials_core_mod
    use fcvode_mod
    use fnvector_serial_mod
    use fsunlinsol_spgmr_mod
    use fsunmatrix_dense_mod
    use fsunlinsol_dense_mod
    use fsunmatrix_sparse_mod

    use mod_pk
     
    implicit none 

    ! type :: input_data 
    !     real(KIND=8)    :: tstart, tend, dtout
    !     real(KIND=8)    :: rtol, atol 
    !     integer(KIND=8) :: neq 
    !     real(KIND=8)    :: ics(:)
    !     type(kinetics_params) :: user_data
    ! end type 

contains 


    subroutine check_flag(ierr, msg)
        integer(c_int), intent(in) :: ierr
        character(*),   intent(in) :: msg 
        if (ierr /= 0_c_int) then 
            print *, trim(msg), "failed with ierr = ", ierr
            stop 1 
        end if 
    end subroutine check_flag


    subroutine cvode_solver(tstart, tend, dtout, rtol, atol, neq, ics, userdata)

        !Inputs
        real(KIND=8), intent(in)    :: tstart, tend, dtout  
        real(KIND=8), intent(in)    :: rtol, atol
        integer(KIND=8), intent(in) :: neq
        real(KIND=8), intent(in)    :: ics(:)
        type(kinetics_params), intent(in), target    :: userdata

        !Locals
        real(KIND=8)   :: tout
        integer(c_int) :: nout
        integer(c_int) :: outstep 

        type(c_ptr)     :: ctx     ! SUNDIALS context for sim
        real(c_double)  :: tcur(1) ! current time
        integer(c_int)  :: ierr    ! error flag from C functions 
        
        type(N_Vector), pointer        :: sunvec_u   ! SUNDIALS vector
        type(SUNLinearSolver), pointer :: sunls     ! SUNDIALS linear solver
        type(SUNMatrix), pointer       :: sunmat     ! SUNDIALS matrix
        type(c_ptr)                    :: cvode_mem ! CVODE memory
        real(c_double), pointer        :: uvec(:) 

        integer(c_int), parameter :: pretype = SUN_PREC_NONE 
        integer(c_int), parameter :: maxl = 10

        integer :: out_unit, ii
        character(len=*), parameter :: out_file = "cvode_output.dat"
        
        
        print*, "Starting CVODE Solver..."

        !-------------------------------------!
        ! Create context                      
        !-------------------------------------!

        ierr = FSUNContext_create(SUN_COMM_NULL, ctx)
        call check_flag(ierr, "FSUNContext_create")

        print*, "SUNDIALS context created."

        !-------------------------------------!
        ! Initialize ODE                   
        !-------------------------------------!

        tcur  = tstart
        tout  = tstart
        nout  = ceiling((tend-tstart)/dtout)

        !-------------------------------------!                                                 
        ! Set initial values vector                      
        !-------------------------------------!
        
        sunvec_u => FN_VNew_Serial(neq, ctx)
        if (.not. associated(sunvec_u)) then 
            print*, "ERROR: sunvec = NULL"
            stop 1 
        end if

        print*, "SUNDIALS N_Vector created." 

        uvec => FN_VGetArrayPointer(sunvec_u)
        uvec(1:neq) = ics(1:neq)
       
        !-------------------------------------!
        ! Create CVODE object and initialize                    
        !-------------------------------------!

        cvode_mem = FCVodeCreate(CV_BDF, ctx)
        if (.not. c_associated(cvode_mem)) then 
            print*, "ERROR: cvode_mem = NULL"
            stop 1 
        end if
        
        ierr = FCVodeInit(cvode_mem, c_funloc(RhsFn), tstart, sunvec_u)
        call check_flag(ierr, "FCVodeInit")

        print*, "CVODE object created and initialized."

        !-------------------------------------!
        ! Specify tolerances and limits                   
        !-------------------------------------!

        ierr = FCVodeSStolerances(cvode_mem, rtol, atol)
        call check_flag(ierr, "FCVodeSStolerances")

        ! ierr = FCVodeSetMaxNumSteps(cvode_mem, int(max_steps, kind=8))
        ! call check_flag(ierr, "FCVodeSetMaxNumSteps")

        print*, "Tolerances and max steps set."

        !-------------------------------------!
        ! Create matrix and linsol objects                     
        !-------------------------------------!

        sunmat => FSUNDenseMatrix(neq, neq, ctx)
        if (.not. associated(sunmat)) then 
            print *, 'ERROR: sunmat = NULL'
            stop 1
        end if 

        sunls => FSUNLinSol_SPGMR(sunvec_u, pretype, maxl, ctx)
        if (.not. associated(sunls)) then 
            print*, "ERROR: sunls = NULL"
            stop 1 
        end if

        ierr = FCVodeSetLinearSolver(cvode_mem, sunls, sunmat)
        call check_flag(ierr, "FCVodeSetLinearSolver")

        print*, "Matrix and linear solver objects created."

        !-------------------------------------!
        ! Set optional inputs                       
        !-------------------------------------!

        ierr = FCVodeSetUserData(cvode_mem, c_loc(userdata))
        call check_flag(ierr, "FCVodeSetUserData")

        print*, "User data set."

        !-------------------------------------!
        ! Open output file and write header                
        !-------------------------------------!

        out_unit = 20
        open(unit=out_unit, file=out_file, status='replace', action='write', &
            form='formatted')
        write(out_unit,'(A)', advance='no') '# t '
        do ii = 1, neq
            write(out_unit,'(A)', advance='no') 'y'//trim(adjustl(to_string(ii)))//' '
        end do
        write(out_unit,*)

        !-------------------------------------!
        ! Advance problem in time                     
        !-------------------------------------!

        print *, '   '
        print *, 'Finished initialization, starting time steps'
        write(out_unit,'(es24.16,1x,*(es24.16,1x))') tcur, uvec(1:neq)
        do outstep = 1, nout
            
            tout = min(tout + dtout, tend)
            ierr = FCVode(cvode_mem, tout, sunvec_u, tcur, CV_NORMAL)
            call check_flag(ierr, "FCVode")
            write(out_unit,'(es24.16,1x,*(es24.16,1x))') tcur, uvec(1:neq)

        end do 

        !-------------------------------------!
        ! Get optional outputs                       
        !-------------------------------------!

        call CVodeStats(cvode_mem)
       
        !-------------------------------------!
        ! Cleanup                   
        !-------------------------------------!!
        
        call FCVodeFree(cvode_mem)
        call FN_VDestroy(sunvec_u)
        call FSUNMatDestroy(sunmat)
        ierr = FSUNLinSolFree(sunls)
        ierr = FSUNContext_Free(ctx)

    end subroutine cvode_solver


    subroutine CVodeStats(cvode_mem)

        type(c_ptr), intent(in) :: cvode_mem ! solver memory structure

        integer(c_int)  :: ierr          ! error flag

        integer(c_long) :: nsteps(1)     ! num steps
        integer(c_long) :: nfe(1)        ! num function evals
        integer(c_long) :: netfails(1)   ! num error test fails
        integer(c_long) :: nniters(1)    ! nonlinear solver iterations
        integer(c_long) :: nliters(1)    ! linear solver iterations
        integer(c_long) :: ncf(1)        ! num convergence failures nonlinear
        integer(c_long) :: ncfl(1)       ! num convergence failures linear

        !======= Internals ============

        ierr = FCVodeGetNumSteps(cvode_mem, nsteps)
        if (ierr /= 0) then
            print *, 'Error in FCVodeGetNumSteps, ierr = ', ierr, '; halting'
            stop 1
        end if

        ierr = FCVodeGetNumRhsEvals(cvode_mem, nfe)
        if (ierr /= 0) then
            print *, 'Error in FCVodeGetNumRhsEvals, ierr = ', ierr, '; halting'
            stop 1
        end if

        ierr = FCVodeGetNumErrTestFails(cvode_mem, netfails)
        if (ierr /= 0) then
            print *, 'Error in FCVodeGetNumErrTestFails, ierr = ', ierr, '; halting'
            stop 1
        end if

        ierr = FCVodeGetNumNonlinSolvIters(cvode_mem, nniters)
        if (ierr /= 0) then
            print *, 'Error in FCVodeGetNumNonlinSolvIters, ierr = ', ierr, '; halting'
            stop 1
        end if

        ierr = FCVodeGetNumLinIters(cvode_mem, nliters)
        if (ierr /= 0) then
            print *, 'Error in FCVodeGetNumLinIters, ierr = ', ierr, '; halting'
            stop 1
        end if

        ierr = FCVodeGetNumLinConvFails(cvode_mem, ncfl)
        if (ierr /= 0) then
            print *, 'Error in FCVodeGetNumLinConvFails, ierr = ', ierr, '; halting'
            stop 1
        end if

        ierr = FCVodeGetNumNonlinSolvConvFails(cvode_mem, ncf)
        if (ierr /= 0) then
            print *, 'Error in FCVodeGetNumNonlinSolvConvFails, ierr = ', ierr, '; halting'
            stop 1
        end if

        print *, ' '
        print *, ' General Solver Stats:'
        print '(4x,A,i9)', 'Total internal steps taken      =', nsteps
        print '(4x,A,i9)', 'Total rhs function call         =', nfe
        print '(4x,A,i9)', 'Num error test failures         =', netfails
        print '(4x,A,i9)', 'Num nonlinear solver iters      =', nniters
        print '(4x,A,i9)', 'Num linear solver iters         =', nliters
        print '(4x,A,i9)', 'Num nonlinear solver fails      =', ncf
        print '(4x,A,i9)', 'Num linear solver fails         =', ncfl
        print *, ' '

        

    end subroutine CVodeStats


    function to_string(ii) result(str)
        integer, intent(in) :: ii
        character(len=16) :: str
        write(str,'(I0)') ii
    end function to_string


end module mod_cvode_wrapper