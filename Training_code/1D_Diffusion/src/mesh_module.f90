module mesh_module                           !Mesh for 1d slab, finite volume  
    use precision                         
    implicit none
    

    type :: mesh
        integer :: N                        !Number of cells
        real(dp) :: L, dx                   !Slab length, cell width
        real(dp), allocatable :: x(:)            
        real(dp), allocatable :: S(:), D(:), sigma_a(:)   
    end type

contains  

    function make_mesh(L, N, S, D, sigma_a) result(m)
        real(dp), intent(in) :: L
        integer, intent(in) :: N
        real(dp), intent(in) :: S(:), D(:), sigma_a(:)
        type(mesh) :: m 

        integer :: ii 

        m%N = N 
        m%L = L 
        m%dx = L/real(N,dp) 
        m%S = S
        m%D = D 
        m%sigma_a = sigma_a
        

        allocate(m%x(N))
        do ii = 1, N
            m%x(ii) = (ii-0.5_dp) * m%dx
        end do
    
    end function make_mesh

end module mesh_module 