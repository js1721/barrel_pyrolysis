!===================================================================================================
!
! Sparse Matrix module
!
! Date:     Programmer:     Changes:
! =====     ===========     ========
! 20-11-25  J Salter        Original
!===================================================================================================


module mod_sprs 
    use mod_constants
    implicit none 

    type :: sparse_matrix 
        real(dp), allocatable :: sa(:)
        integer, allocatable :: ija(:)

        !!! RULES: 
        !!! - Represents NxN matrix, mostly zeros.
        !!! - sa(<=N) store diagonal entries, even if 0.
        !!! - ija(<=N) store the index of sa that contains the first
        !!!   off-diag element of corresponding row in matrix - if 
        !!!   there are no off-diag elements for a row, value is one 
        !!!   more than index in sa of most recently stored element 
        !!!   of a previous row.
        !!! - ija(1) = N+2, always.
        !!! - ija(N+1) = index in sa of last off diag element + 1.
        !!! - sa(N+1) not used, can be set arbitrarily.
        !!! - sa(>=N+2) contain off-diag values, ordered by rows, and 
        !!!   within rows by columns.
        !!! - ija(>=N+2) contain column number of corresponding element in sa.
    
    end type sparse_matrix



contains 

    function get_sparse_N(sprs) result(N)
        type(sparse_matrix), intent(in) :: sprs 
        integer :: N 
        N = sprs%ija(1) - 2
    end function get_sparse_N


    subroutine convert_to_sparse(matrix, sprs, thresh)
        real(dp), intent(in) :: matrix(:,:)
        type(sparse_matrix), intent(inout) :: sprs 
        real(dp), intent(in) :: thresh !Will discard values below this 
        integer :: nmax !dimensions of sa and ija, number of matrix entries shouldnt exceed this
        integer :: ii, jj, kk 
        integer ::  N, n_off

        N = size(matrix, dim=2)
        n_off = 0
        do ii = 1, N
            do jj = 1, N
                if (ii /= jj .and. abs(matrix(ii,jj)) >= thresh) n_off = n_off + 1
            end do
        end do
   
        nmax = n_off + N + 1

        allocate(sprs%sa(nmax), sprs%ija(nmax))
        sprs%sa = 0.0_dp 
        sprs%ija = 0
        N = size(matrix, dim=2)
        

        do ii = 1, N            ! Store diag values
            sprs%sa(ii) = matrix(ii,ii)  
        end do 
        
        sprs%ija(1) = N+2
        kk = N+1 

        do ii = 1, N 
            do jj = 1, N 
                if (abs(matrix(ii,jj)).ge.thresh) then 
                    if (ii.ne.jj) then 
                        kk = kk +1
                        if (kk.gt.nmax) stop "nmax too small in sprs"
                        sprs%sa(kk) = matrix(ii,jj)
                        sprs%ija(kk) = jj 
                    end if 
                end if 
            end do 
            sprs%ija(ii+1) = kk + 1
        end do 

    end subroutine convert_to_sparse

    function sparse_times_vector(sprs, x) result(y)
        type(sparse_matrix), intent(in) :: sprs 
        real(dp), intent(in) :: x(:)
        real(dp), allocatable :: y(:)
        integer :: ii, jj, N 

        N = size(x)
        if (sprs%ija(1) /= N+2) stop "mismatched vector and matrix in sparse_times_vector"
        allocate(y(N))

        do ii = 1, N
            y(ii) = sprs%sa(ii) * x(ii)
            do jj = sprs%ija(ii), sprs%ija(ii+1)-1
                y(ii) = y(ii) + sprs%sa(jj)*x(sprs%ija(jj))
            end do 
        end do 
    end function sparse_times_vector

    function sparseTrans_times_vector(sprs, x) result(y)
        type(sparse_matrix), intent(in) :: sprs
        real(dp), intent(in) :: x(:)
        real(dp), allocatable :: y(:)
        integer :: ii, jj, kk, N 

        N = size(x)
        if (sprs%ija(1) /= N+2) stop "mismatched matrix and vector in sparseTrans_times_vector"
        allocate(y(N))

        do ii = 1, N 
            y(ii) = sprs%sa(ii) * x(ii)
        end do 

        do ii = 1, N 
            do jj = sprs%ija(ii), sprs%ija(ii+1)-1
                kk = sprs%ija(jj)
                y(kk) = y(kk) + sprs%sa(jj)*x(ii)
            end do 
        end do 

    end function sparseTrans_times_vector

end module mod_sprs