module bc_module
    use precision
    implicit none

    type, abstract :: BoundaryCondition                 !base class 
    contains                                            
        procedure(apply_bc), deferred :: apply 
    end type BoundaryCondition

    type, extends(BoundaryCondition) :: RobinBC
        real(dp) :: alpha_left, beta_left, gamma_left
        real(dp) :: alpha_right, beta_right, gamma_right
        real(dp) :: dx
    contains
        procedure :: apply => apply_robin
    end type RobinBC

    type, extends(BoundaryCondition) :: PeriodicBC 
    contains 
        procedure :: apply => apply_periodic
    end type PeriodicBC

    abstract interface                              !template for deferred procedure 
        subroutine apply_bc(this, a, b, c, rhs)
            import :: BoundaryCondition, dp             !interface sits in own area
            class(BoundaryCondition), intent(in) :: this
            real(dp), intent(inout) :: a(:), b(:), c(:), rhs(:)
        end subroutine apply_bc
    end interface

contains 

        subroutine apply_robin(this, a, b, c, rhs)
            class(RobinBC), intent(in) :: this 
            real(dp), intent(inout) :: a(:), b(:), c(:), rhs(:)
            integer :: N 
            N = size(b)

            !LH boundary 
            b(1) = this%alpha_left - this%beta_left/this%dx 
            c(1) = this%beta_left/this%dx
            rhs(1) = this%gamma_left

            !RH boundary
            b(N) = this%alpha_right+this%beta_right/this%dx 
            a(N-1) = -this%beta_right/this%dx 
            rhs(N) = this%gamma_right
        end subroutine apply_robin

        subroutine apply_periodic(this, a, b, c, rhs)
            class(PeriodicBC), intent(in) :: this 
            real(dp), intent(inout) :: a(:), b(:), c(:), rhs(:)
            integer :: N 
            N = size(b) 

            !LH boundary
            c(1) = 0
            b(1) = 1
            rhs(1) = 0

            !RH boundary 
            a(N-1) = 0
            b(N) = 1
            rhs(N) = 0
        
        end subroutine apply_periodic
 
        function make_robin_bc(alphaL, betaL, gammaL, alphaR, betaR, gammaR, dx) result(bc)
            real(dp), intent(in) :: alphaL, betaL, gammaL
            real(dp), intent(in) :: alphaR, betaR, gammaR
            real(dp), intent(in) :: dx
            type(RobinBC) :: bc

            bc%alpha_left  = alphaL
            bc%beta_left   = betaL
            bc%gamma_left  = gammaL
            bc%alpha_right = alphaR
            bc%beta_right  = betaR
            bc%gamma_right = gammaR
            bc%dx          = dx
        end function make_robin_bc

!BCs are of form alpha*phi + beta*dphi/dx = gamma 
        function make_ZeroFluxBC(dx) result(bc)
            real(dp), intent(in) :: dx
            type(RobinBC) :: bc
            bc%alpha_left  = 1.0_dp
            bc%beta_left   = 0.0_dp
            bc%gamma_left  = 0.0_dp
            bc%alpha_right = 1.0_dp
            bc%beta_right  = 0.0_dp
            bc%gamma_right = 0.0_dp
            bc%dx          = dx
        end function make_ZeroFluxBC

        function make_ReflectiveBC(dx) result(bc)
            real(dp), intent(in) :: dx
            type(RobinBC) :: bc
            bc%alpha_left  = 0.0_dp
            bc%beta_left   = 1.0_dp
            bc%gamma_left  = 0.0_dp
            bc%alpha_right = 0.0_dp
            bc%beta_right  = 1.0_dp
            bc%gamma_right = 0.0_dp
            bc%dx          = dx
        end function make_ReflectiveBC

        function make_VacuumBC(dx, D_left, D_right) result(bc)
            real(dp), intent(in) :: dx, D_left, D_right
            type(RobinBC) :: bc
            bc%alpha_left  = 1.0_dp
            bc%beta_left   = 0.7104_dp*D_left
            bc%gamma_left  = 0.0_dp
            bc%alpha_right = 1.0_dp
            bc%beta_right  = 0.7104_dp*D_right
            bc%gamma_right = 0.0_dp
            bc%dx          = dx
        end function make_VacuumBC

        function make_AlbedoBC(dx, albedo_left, albedo_right, D_left, D_right) result(bc)
            real(dp), intent(in) :: dx, albedo_left, albedo_right, D_left, D_right
            type(RobinBC) :: bc
            bc%alpha_left  = 0.25_dp *(albedo_left-1)
            bc%beta_left   = 0.5_dp *D_left*(albedo_left+1)
            bc%gamma_left  = 0.0_dp
            bc%alpha_right = 0.25_dp *(albedo_right-1)
            bc%beta_right  = -0.5_dp * D_right*(albedo_right+1)
            bc%gamma_right = 0_dp
            bc%dx          = dx
        end function make_AlbedoBC

        function make_SurfaceSourceBC(dx, surface_left, surface_right, D_left, D_right) result(bc)
            real(dp), intent(in) :: dx, surface_left, surface_right, D_left, D_right
            type(RobinBC) :: bc
            bc%alpha_left  = 0.25_dp
            bc%beta_left   = -0.5_dp *D_left
            bc%gamma_left  = surface_left
            bc%alpha_right = 0.25_dp
            bc%beta_right  = 0.5_dp *D_right
            bc%gamma_right = surface_right
            bc%dx          = dx
        end function make_SurfaceSourceBC

        function make_PeriodicBC() result(bc)
            type(PeriodicBC) :: bc 
        end function make_PeriodicBC



end module bc_module