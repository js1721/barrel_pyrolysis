!===================================================================================================
!
! Point kinetics solver module.
!
! Date:     Programmer:     Changes:
! =====     ===========     ========
! 11-02-26  J Salter        Original
!===================================================================================================

module mod_pk_solver 
    
    use mod_cvode_wrapper 
    use mod_pk
    implicit none 


contains 

    subroutine make_kinetics_params(userdata, G, rho, lambda, beta, lambdai)
        
        type(kinetics_params), intent(inout) :: userdata
        integer(KIND=8), intent(in)          :: G
        real(KIND=8), intent(in)             :: rho, lambda
        real(KIND=8), intent(in)             :: beta(:), lambdai(:)

        if (size(beta) /= G) then 
            print*, "Size of beta /= G."
            stop 
        end if 

        if (size(lambdai) /= G) then 
            print*, "Size of lambdai /= G."
            stop 
        end if 

        userdata%G = G 
        userdata%rho = rho; userdata%lambda = lambda 
        userdata%beta = beta; userdata%lambdai = lambdai

    end subroutine make_kinetics_params

    subroutine pk_solver(G, rho, lambda, beta, lambdai, ics, tstart, tend, dtout, rtol, atol)

        integer(KIND=8), intent(in)       :: G
        real(KIND=8), intent(in)          :: rho, lambda
        real(KIND=8), intent(in)          :: beta(:), lambdai(:)
        real(kind=8), intent(in)          :: ics(:)
        real(kind=8), intent(in)          :: tstart, tend, dtout, rtol, atol 

        integer(kind=8) :: neq 
        type(kinetics_params) :: userdata
        call make_kinetics_params(userdata, G, rho, lambda, beta, lambdai)

        neq = G + 1 

        if (size(ics) /= neq) then 
            print*, "Number of initial conditions and equations don't match."
            stop 
        end if 

        call cvode_solver(tstart, tend, dtout, rtol, atol, neq, ics, userdata)

    end subroutine pk_solver

end module mod_pk_solver