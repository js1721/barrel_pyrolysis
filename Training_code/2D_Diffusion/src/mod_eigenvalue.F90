!===================================================================================================
!
! Eigenvalue module
!
! Date:     Programmer:     Changes:
! =====     ===========     ========
! 26-11-25  J Salter        Original
!===================================================================================================

module mod_eigenvalue 

    use mod_constants
    use mod_types
    use mod_sprs
    use mod_map
    use mod_materials
    use mod_solver_pcg 
    implicit none


contains 

    subroutine solver_eigenvalue(A, N_iter, material_list, mesh, initial_guess, k, S)
        
        type(sparse_matrix), intent(in) :: A 
        type(mesh_xy), intent(in) :: mesh
        real(dp), intent(out), allocatable :: S(:,:)
        real(dp), intent(in) :: initial_guess(:)
        real(dp), allocatable :: flat_S(:), flat_phi(:), flat_S_old(:)
        real(dp), intent(inout) :: k
        integer, intent(in) :: N_iter 
        type(material), intent(in) :: material_list(:)
        real(dp) :: k_old

        real(dp), allocatable :: nusigma_f(:,:), flat_nusigma_f(:)

        integer :: nx, ny 
        integer :: ii, jj
        real(dp) :: dx, dy

        !real(dp) :: norm
        real(dp), allocatable :: rhs(:)

        call assemble_nuSigma_f_xy(mesh, nusigma_f, material_list)


        

        nx = mesh%nx ; ny = mesh%ny 
        dx = mesh%dx ; dy = mesh%dy
        allocate(flat_nusigma_f(nx*ny))
        allocate(flat_S(nx*ny))

        do ii = 1, nx
            do jj = 1, ny 
                flat_nusigma_f(map(ii,jj,nx))=nusigma_f(ii,jj)
            end do 
        end do 

    !    flat_S = flat_nusigma_f * initial_guallocate(flat_phi(nx*ny))
    !    flat_phi = cjm(B, rhs, initial_guess, N_iterations)
       
        flat_S = flat_nusigma_f * initial_guess

        do ii = 1, N_iter

            k_old = k
            flat_S_old = flat_S

            ! Solve A φ = S_old / k_old
            rhs = flat_S_old / k_old
            flat_phi = cjm(A, rhs, initial_guess, 50)

            ! Normalize φ
            ! norm = sum(flat_phi * dx * dy)
            !norm = sqrt(sum(flat_phi * flat_phi))
            !flat_phi = flat_phi / norm

            ! Update S
            flat_S = flat_nusigma_f * flat_phi

            ! Update eigenvalue
            k = k_old * sum(flat_S * dx * dy) / sum(flat_S_old * dx * dy)
            !print*, k

            if (abs((k-k_old)/k) < 1e-5) exit

        end do

        

        allocate(S(nx,ny))
        do ii = 1, nx 
            do jj = 1, ny 
                S(ii,jj) = flat_S(map(ii,jj,nx))
            end do 
        end do 



    end subroutine solver_eigenvalue

    subroutine solver_eigenvalue_2(A, N_iter, initial_guess, material_list, mesh, k, flat_phi)

        type(sparse_matrix), intent(in) :: A 
        type(mesh_xy), intent(in) :: mesh
       
        real(dp), intent(in) :: initial_guess(:)
        real(dp), allocatable, intent(out) ::  flat_phi(:)
        real(dp), intent(inout) :: k
        integer, intent(in) :: N_iter 
        type(material), intent(in) :: material_list(:)
        
        real(dp), allocatable :: nusigma_f(:,:), flat_nusigma_f(:)

        integer :: nx, ny 
        integer :: ii, jj
        real(dp) :: dx, dy

        call assemble_nuSigma_f_xy(mesh, nusigma_f, material_list)

        nx = mesh%nx ; ny = mesh%ny 
        dx = mesh%dx ; dy = mesh%dy
        allocate(flat_nusigma_f(nx*ny))
       

        do ii = 1, nx
            do jj = 1, ny 
                flat_nusigma_f(map(ii,jj,nx))=nusigma_f(ii,jj)
            end do 
        end do 
        flat_phi = initial_guess
        do ii =1, N_iter 
            flat_phi = sparse_times_vector(A, flat_phi)
            flat_phi = flat_phi/sqrt(sum(flat_phi**2))
            k = dot_product(flat_phi, sparse_times_vector(A, flat_phi))
            print*, k
        end do 
        k = dot_product(flat_phi, sparse_times_vector(A, flat_phi))
        
    
    end subroutine solver_eigenvalue_2


end module mod_eigenvalue