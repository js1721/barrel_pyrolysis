!===================================================================================================
!
! Point kinetics module. Contains template PK equations.
!
! Date:     Programmer:     Changes:
! =====     ===========     ========
! 11-02-26  J Salter        Original
!===================================================================================================

module mod_pk

    use, intrinsic :: iso_c_binding
    use fsundials_core_mod 
    implicit none 


    type :: kinetics_params
        integer(c_int)        :: G          ! number of delayed groups
        real(c_double)        :: rho        ! reactivity
        real(c_double)        :: lambda     ! neutron generation time
        ! Arrays sized at compile time for simplicity; adjust as needed
        real(c_double), allocatable        :: beta(:)   ! delayed fractions per group
        real(c_double), allocatable        :: lambdai(:) ! decay constants per group
    end type kinetics_params

contains



    integer(c_int) function RhsFn(tn, sunvec_u, sunvec_f, user_data) bind(C, name="RhsFn") result(ierr)

        use, intrinsic :: iso_c_binding 
        implicit none

        real(c_double), value :: tn 
        type(N_Vector)        :: sunvec_u 
        type(N_Vector)        :: sunvec_f 
        type(c_ptr), value    :: user_data 
          
        real(c_double), pointer :: uvec(:)
        real(c_double), pointer :: fvec(:)

        type(kinetics_params), pointer :: p

        integer(c_int) :: ii, G 
        real(c_double) :: rho,  lambda 
        real(c_double), allocatable :: lambdai(:), beta(:)
        real(c_double) :: beta_tot

        uvec => FN_VGetArrayPointer(sunvec_u)
        fvec => FN_VGetArrayPointer(sunvec_f)

        call c_f_pointer(user_data, p)
        G = p%G
        rho = p%rho  ; lambda = p%lambda 
        lambdai = p%lambdai ; beta = p%beta
        beta_tot = sum(beta)


        fvec(1) = ((rho-beta_tot)/lambda)*uvec(1) + dot_product(lambdai, uvec(2:(G+1)))

        do ii = 2, G+1 
            fvec(ii) = (beta(ii-1)/lambda)*uvec(1) - lambdai(ii-1) * uvec(ii)
        end do 

        ierr = 0
        !return

    end function RhsFn

end module mod_pk