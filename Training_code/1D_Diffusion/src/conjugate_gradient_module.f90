module conjugate_gradient_module 
    use precision
    use sparse_matrix_module
    implicit none 



contains 
    

    function cjm(sprs, b, initial_guess, N_iterations) result(x)
        !Solves Ax=b, where we are taking A as a  sprse matrix;this is actually biconj gradient method 
        type(sparse_matrix), intent(in) :: sprs 
        real(dp), intent(in) :: b(:), initial_guess(:)
        real(dp), allocatable :: x(:)
        integer, intent(in) :: N_iterations

        real(dp), allocatable :: r(:), rbar(:), p(:), pbar(:)
        real(dp) :: alpha, beta, rbardotrprev
        integer :: ii 

        x = initial_guess
        r = b - sparse_times_vector(sprs, initial_guess)
        rbar = r 
        p = r 
        pbar = rbar 


        do ii = 1, N_iterations
            alpha = dot_product(rbar, r)/(dot_product(pbar, sparse_times_vector(sprs, p)))
            x = x+ alpha*p           !most certainly not optimally placed 
            if (ii.eq.N_iterations) then 
                exit
            end if  
            rbardotrprev = dot_product(rbar, r)
            r = r - alpha*sparse_times_vector(sprs, p)
            rbar = rbar - alpha *sparseTrans_times_vector(sprs,pbar)     !Should really be sprs transpose
            beta = dot_product(rbar, r)/rbardotrprev 
            p = r + beta*p 
            pbar = rbar + beta*pbar 
        end do 

    end function cjm

    !function pcjm() result()

    
   ! end function pcjm



end module conjugate_gradient_module