!===================================================================================================
!
! Multigroup module
!
! Date:     Programmer:     Changes:
! =====     ===========     ========
! 09-12-25  J Salter        Original
!===================================================================================================

module mod_mg 

    use mod_constants
    use mod_solver_pcg
    use mod_types
    use mod_sprs
    use mod_map 
    use mod_diffusion_matrix
    use mod_params
    implicit none 


contains 

    subroutine solver_mg(initial_guess, k, mg_phi)

        real(dp), intent(in) :: initial_guess(:) ! Guess for source
        real(dp), intent(inout) :: k
      
        real(dp), intent(inout), allocatable :: mg_phi(:,:,:)
     

        type(sparse_matrix), allocatable :: A
        real(dp), allocatable :: rhs(:), flat_phi(:,:), S(:)
        !real(dp), allocatable :: nusigma_f(:), sigma_s(:,:)
        real(dp) :: k_old
        real(dp), allocatable :: S_old(:)
        
        integer :: ii, jj, kk

        call init_params()
        allocate(S(nx*ny))
        S = initial_guess
        allocate(rhs(nx*ny))
        allocate(flat_phi(nx*ny,G))
        do ii = 1, N_iterations
            k_old = k
            S_old = S
            do jj = 1, G
                rhs = chi(jj) * S/k 
                do kk = 1, jj-1
                    rhs = rhs + sigma_s(kk, jj) * flat_phi(:,kk)
                end do 
                call assemble_diffusion_xy(A, mesh, material_list, bc_L, bc_R, bc_B, bc_T, rhs)
                flat_phi(:, jj) = cjm(A, rhs, initial_guess, 300)
            end do
            S = 0.0_dp
            do kk = 1, G
                S = S + nusigma_f(kk) * flat_phi(:,kk)
            end do 

            k = k_old * sum(S * mesh%dx * mesh%dy) / sum(S_old * mesh%dx * mesh%dy)
            if (abs((k-k_old)/k) < 1e-5) exit
        end do 



    end subroutine solver_mg

    

    ! function solve_multigroup_fixedsource_2(bc, mg_sigmaS, mg_sigmaA, S, G, L, N, N_iter) result(mg_phi)
    !     type(mesh) :: mymesh
    !     class(tridiag), allocatable :: tri
    !     class(BoundaryCondition), intent(in) :: bc
    !     real(dp), intent(in) :: S(:,:), mg_sigmaS(:,:), mg_sigmaA(:,:)
    !     real(dp), allocatable :: mg_phi(:,:)
    !     integer, intent(in) :: G
    !     real(dp), intent(in) :: L
    !     integer, intent(in) :: N, N_iter
    !     real(dp), allocatable :: rhs(:)


    !     integer :: ii, jj, kk
    !     !allocate(mg_phi(G,N), S(G,N), mg_sigmaS(G))
    !     allocate(mg_phi(G,N))
    !     do ii = 1, G 
    !         mg_phi(ii,:) = 0.0_dp
    !     end do 
        
    !     do kk = 1, N_iter
    !         do ii = 1, G
    !             rhs = s(ii,:)
    !             do jj = 1, G
    !                 if (ii /= jj) then 
    !                     rhs = rhs + mg_sigmaS(jj, ii)*mg_phi(jj,:)
    !                 end if 
    !             end do 
    !             mymesh = make_mesh(L, N, rhs, 1/(3*mg_sigmaA(ii,:)), mg_sigmaA(ii,:))
    !             tri = make_tridiag(mymesh)
    !             mg_phi(ii,:) = solver_fixed_source(mymesh, make_tridiag(mymesh), bc)
    !         end do
    !     end do
    ! end function solve_multigroup_fixedsource_2



end module mod_mg 