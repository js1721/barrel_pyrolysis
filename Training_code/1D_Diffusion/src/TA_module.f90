module TA_module 
    use precision 
    implicit none 

contains 

    function thomas_algorithm(a, b, c, rhs) result(x)
        real(dp), intent(in) :: a(:), b(:), c(:), rhs(:)
        real(dp), allocatable :: x(:)
        real(dp), allocatable :: cprime(:), dprime(:)
        integer :: N, ii 

        N = size(b)
        allocate(x(N), cprime(N-1), dprime(N))

        !forward sweep
        cprime(1) = c(1) / b(1)
        dprime(1) = rhs(1) / b(1)
        do ii = 2, N-1
            cprime(ii) = c(ii) / (b(ii) - a(ii-1)*cprime(ii-1))
        end do 
        do ii = 2, N 
            dprime(ii) = (rhs(ii) - a(ii-1)*dprime(ii-1)) / (b(ii) - a(ii-1)*cprime(ii-1))
        end do 

        !back sub
        x(N) = dprime(N)
        do ii = N-1, 1, -1
            x(ii) = dprime(ii) - cprime(ii)*x(ii+1)
        end do 

    end function thomas_algorithm

    subroutine cyclic_algorithm(a, b, c, alpha, beta, rhs, x)
        real(dp), intent(inout) :: a(:), b(:), c(:)       
        real(dp), intent(in) :: alpha, beta
        real(dp), intent(in) :: rhs(:)
        real(dp), intent(inout) :: x(:)

        real(dp), allocatable :: u(:), v(:), z(:), w(:)
        real(dp), allocatable :: correction_matrix(:,:)
        real(dp) :: lambda
        real(dp) :: gamma
        integer :: N , ii, jj
        N = size(b) 
        gamma = -b(1)      !Avoids loss of precision, allegedly 

        !Modify initial tridiag slightly 
        b(1) = b(1) - gamma 
        b(N) = b(N) - alpha * beta /gamma 

        !Set up correction vectors 
        allocate(u(N), v(N)) 
        u(1) = gamma 
        u(N) = alpha 
        v(1) = 1.0
        v(N) = beta/gamma 
        do ii = 2, N-1
            u(ii) = 0.0 
            v(ii) = 0.0 
        end do 

        !Set up correction matrix 
        z = thomas_algorithm(a, b, c, u)
        w = thomas_algorithm(c, b, a, v)
        lambda = dot_product(v, z)

        allocate(correction_matrix(N,N))
        do ii = 1, N 
            do jj = 1, N 
                correction_matrix(ii,jj) = z(ii) * w(jj)
            end do 
        end do 
        correction_matrix = (1/(1+lambda))* correction_matrix

        !Solve 
        x = thomas_algorithm(a, b, c, rhs) - matmul(correction_matrix, rhs)


    end subroutine cyclic_algorithm



end module TA_module