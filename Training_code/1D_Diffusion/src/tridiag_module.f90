module tridiag_module 
    use precision 
    use mesh_module
    implicit none 

    type :: tridiag 
        real(dp), allocatable :: a(:), b(:), c(:)
    end type tridiag 

    type, extends(tridiag) :: cyclic_tridiag 
        real(dp) :: alpha = 0.0_dp
        real(dp) :: beta = 0.0_dp
    end type cyclic_tridiag


contains 

    function make_tridiag(mymesh) result(tri)
        type(mesh) :: mymesh
        type(tridiag) :: tri

        real(dp), allocatable :: sigma_a(:), D(:), S(:)
        real(dp) :: dx
        integer :: N

        integer :: ii

        sigma_a = mymesh%sigma_a
        D = mymesh%D 
        S = mymesh%S
        dx = mymesh%dx 
        N = mymesh%N

        allocate(tri%a(N-1), tri%b(N), tri%c(N-1))

        tri%c(1) = -D(1)/(dx*dx)
        tri%b(1) = 2.0_dp * D(1)/(dx*dx) + sigma_a(1)
        tri%a(N-1) = -D(N)/(dx**2)
        tri%b(N) = 2.0_dp * D(N)/(dx*dx) + sigma_a(N)

        !interior points 
        do ii = 2, N-1 
            tri%a(ii-1) = -D(ii)/(dx**2)
            tri%b(ii) = 2.0_dp*D(ii)/(dx**2) + sigma_a(ii)
            tri%c(ii) = -D(ii)/(dx**2)
        end do 
    
    end function make_tridiag

    function make_cyclic_tridiag(mymesh) result(tri)
        type(mesh) :: mymesh
        type(cyclic_tridiag) :: tri

        real(dp), allocatable :: sigma_a(:), D(:), S(:)
        real(dp) :: dx
        integer :: N

        integer :: ii

        sigma_a = mymesh%sigma_a
        D = mymesh%D 
        S = mymesh%S
        dx = mymesh%dx 
        N = mymesh%N

        allocate(tri%a(N-1), tri%b(N), tri%c(N-1))

        tri%c(1) = -D(1)/(dx*dx)
        tri%b(1) = 2.0_dp * D(1)/(dx*dx) + sigma_a(1)
        tri%a(N-1) = -D(N)/(dx**2)
        tri%b(N) = 2.0_dp * D(N)/(dx*dx) + sigma_a(N)

        !interior points 
        do ii = 2, N-1 
            tri%a(ii-1) = -D(ii)/(dx**2)
            tri%b(ii) = 2.0_dp*D(ii)/(dx**2) + sigma_a(ii)
            tri%c(ii) = -D(ii)/(dx**2)
        end do 
    
    end function make_cyclic_tridiag

    function convert_tridiag_to_matrix(tri) result (matrix)
        integer :: size_matrix
        integer :: ii

        type(tridiag), intent(in) :: tri
        real(dp), allocatable :: a(:), b(:), c(:)
        real(dp), allocatable :: matrix(:,:)

        a = tri%a 
        b = tri%b 
        c = tri%c

        size_matrix = size(b)
        allocate(matrix(size_matrix,size_matrix))


        matrix(1,1) = b(1)
        matrix(1,2) = c(1)
        do ii = 2, size_matrix -1
            matrix(ii, ii) = b(ii)
            matrix(ii,ii-1) = a(ii-1)
            matrix(ii,ii+1) = c(ii)
        end do
        matrix(size_matrix,size_matrix) = b(size_matrix)
        matrix(size_matrix,size_matrix-1) = a(size_matrix-1)
    end function convert_tridiag_to_matrix

    function convert_cyctri_to_matrix(cyctri) result (matrix)
        integer :: size_matrix
        integer :: ii

        type(cyclic_tridiag), intent(in) :: cyctri
        real(dp), allocatable :: a(:), b(:), c(:)
        real(dp), allocatable :: matrix(:,:)
        real(dp) :: alpha, beta

        a = cyctri%a 
        b = cyctri%b 
        c = cyctri%c

        alpha = cyctri%alpha 
        beta = cyctri%beta

        size_matrix = size(b)
        allocate(matrix(size_matrix,size_matrix))


        matrix(1,1) = b(1)
        matrix(1,2) = c(1)
        do ii = 2, size_matrix -1
            matrix(ii, ii) = b(ii)
            matrix(ii,ii-1) = a(ii-1)
            matrix(ii,ii+1) = c(ii)
        end do
        matrix(size_matrix,size_matrix) = b(size_matrix)
        matrix(size_matrix,size_matrix-1) = a(size_matrix-1)
        matrix(1, size_matrix) = beta           !unsure if correct order here
        matrix(size_matrix, 1) = alpha  
    end function convert_cyctri_to_matrix

end module tridiag_module