module power_iteration_module
    use precision
    implicit none


contains

    function power_iteration_vector(matrix, guess_vector, N_iterations) result(eigenvector)
        real(dp), intent(in) :: matrix(:,:)
        real(dp), allocatable :: eigenvector(:)
        real(dp), intent(in) :: guess_vector(:)
        integer, intent(in) :: N_iterations
        integer :: ii
        integer :: size_matrix

        eigenvector = guess_vector
        size_matrix = size(matrix, dim=2)
        eigenvector = eigenvector/sqrt(sum(eigenvector**2))
        do ii = 1, N_iterations
            eigenvector = matmul(matrix, eigenvector)
            eigenvector = eigenvector/sqrt(sum(eigenvector**2))
        end do
        

    end function power_iteration_vector

    function power_iteration_eigenvalue(matrix, eigenvector) result(eigenvalue)
        real(dp), intent(in) :: matrix(:,:)
        real(dp), intent(in) :: eigenvector(:)
        real(dp) :: eigenvalue
        eigenvalue = dot_product(eigenvector, matmul(matrix, eigenvector))
    end function power_iteration_eigenvalue

end module power_iteration_module