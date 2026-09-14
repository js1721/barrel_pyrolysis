module solver_module
    use precision
    use mesh_module 
    use power_iteration_module
    use bc_module
    use TA_module 
    use tridiag_module
    implicit none 

contains

    function solver_fixed_source(mymesh, tri, bc) result(phi)
        type(mesh), intent(in) :: mymesh
        class(tridiag), intent(in) :: tri
        class(BoundaryCondition), intent(in) :: bc
        real(dp), allocatable :: phi(:)

        real(dp), allocatable :: a(:), b(:), c(:), rhs(:)
        integer :: N

        N = mymesh%N 
       
        allocate(a(size(tri%a)))
        allocate(b(size(tri%b)))
        allocate(c(size(tri%c)))
        allocate(rhs(size(mymesh%S)))

        a = tri%a 
        b = tri%b
        c = tri%c
        
        rhs = mymesh%S
        call bc%apply(a, b, c, rhs)

        select type(tri)
        type is (cyclic_tridiag) 
            allocate(phi(N))
            phi = 0.0_dp
            call cyclic_algorithm(a, b, c, tri%alpha, tri%beta, rhs, phi)      
        class default 
            phi = thomas_algorithm(a, b, c, rhs)
        end select 
    end function solver_fixed_source  


   subroutine solver_fission_source(matrix, bc, phi, k, nusigma_f, dx)
        !type(mesh), intent(in) :: mymesh 
        !type(tridiag), intent(in) :: tri 
        class(BoundaryCondition), intent(in) :: bc 
        real(dp), intent(inout) :: phi(:)
        real(dp), intent(inout) :: k 
        real(dp), intent(in) :: matrix(:,:)
        real(dp), intent(in) :: nusigma_f
        real(dp), intent(in) :: dx
        integer :: N_iterations = 100 
        integer :: N, ii
        real(dp) :: total_fission

        !matrix = convert_tridiag_to_matrix(tri)
        phi = power_iteration_vector(matrix, phi, N_iterations)
        k = 1/(nusigma_f*power_iteration_eigenvalue(matrix, phi))

        N = size(phi)
        total_fission = 0.0_dp
        do ii = 1, N 
            total_fission = total_fission + nusigma_F * phi(ii) * dx
        end do 
        phi = phi/total_fission

    
    end subroutine solver_fission_source

    function solver_eigenvalue_tridiag(mymesh, tri, bc, N_iter, nusigma_f, k) result(phi)
        !use TA_module
       
      
        type(mesh), intent(in) :: mymesh
        class(tridiag), intent(in) :: tri
        class(BoundaryCondition), intent(in) :: bc
        integer, intent(in) :: N_iter
        real(dp), intent(in) :: nusigma_f
     
        real(dp), allocatable :: phi(:)
        real(dp), intent(out) :: k

        integer :: N, ii
        real(dp), allocatable :: a(:), b(:), c(:), rhs(:)
        real(dp), allocatable :: phi_old(:), phi_new(:)

        N = mymesh%N
        allocate(a(size(tri%a)), b(size(tri%b)), c(size(tri%c)), rhs(N))
        a = tri%a
        b = tri%b
        c = tri%c

        ! Initial guess
        allocate(phi_old(N))
        phi_old = 1.0_dp

        ! Power iteration loop
        do ii = 1, N_iter
            ! Compute fission source for current flux: S_f = nu*Sigma_f * phi_old
            rhs = nusigma_f * phi_old

            ! Solve A * phi_new = rhs using tridiagonal solver
            select type(tri)
            type is (cyclic_tridiag)
                allocate(phi_new(N))
                call cyclic_algorithm(a, b, c, tri%alpha, tri%beta, rhs, phi_new)
            class default
                phi_new = thomas_algorithm(a, b, c, rhs)
            end select

            ! Normalize flux
            phi_new = phi_new / sum(phi_new*mymesh%dx)

            ! Update old flux
            phi_old = phi_new
        end do

        phi = phi_new

        ! Rayleigh quotient for eigenvalue
        rhs = nusigma_f * phi
        k = sum(phi*mymesh%dx) / sum(rhs*mymesh%dx)

    end function solver_eigenvalue_tridiag

    function solver_mixed_source(mymesh, tri, bc) result(phi)

    

    end function solver_mixed_source



end module solver_module