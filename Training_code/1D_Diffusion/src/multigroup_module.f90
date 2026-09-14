module multigroup_module 
    use precision
    use mesh_module
    use tridiag_module
    use bc_module
    use solver_module
    implicit none

contains

    function solve_multigroup_fixedsource(bc, mg_sigmaS, mg_sigmaA, S, G, L, N) result(mg_phi)
        type(mesh) :: mymesh
        class(tridiag), allocatable :: tri
        class(BoundaryCondition), intent(in) :: bc
        real(dp), intent(in) :: S(:,:), mg_sigmaS(:,:), mg_sigmaA(:,:)
        real(dp), allocatable :: mg_phi(:,:)
        integer, intent(in) :: G
        real(dp), intent(in) :: L
        integer, intent(in) :: N    
        real(dp), allocatable :: rhs(:)


        integer :: ii, jj
        !allocate(mg_phi(G,N), S(G,N), mg_sigmaS(G))
        allocate(mg_phi(G,N))

        mymesh = make_mesh(L, N, S(1,:), 1/(3*mg_sigmaA(1,:)), mg_sigmaA(1,:))
        tri = make_tridiag(mymesh)
        mg_phi(1,:) = solver_fixed_source(mymesh, tri, bc)

        do ii = 2, G
            rhs = s(ii,:)
            do jj = 2, ii
                rhs = rhs + mg_sigmaS(ii-1, jj)*mg_phi(jj-1,:)
            end do 
            mymesh = make_mesh(L, N, rhs, 1/(3*mg_sigmaA(ii,:)), mg_sigmaA(ii,:))
            tri = make_tridiag(mymesh)
            mg_phi(ii,:) = solver_fixed_source(mymesh, make_tridiag(mymesh), bc)
        end do

    end function solve_multigroup_fixedsource

    function solve_multigroup_fixedsource_2(bc, mg_sigmaS, mg_sigmaA, S, G, L, N, N_iter) result(mg_phi)
        type(mesh) :: mymesh
        class(tridiag), allocatable :: tri
        class(BoundaryCondition), intent(in) :: bc
        real(dp), intent(in) :: S(:,:), mg_sigmaS(:,:), mg_sigmaA(:,:)
        real(dp), allocatable :: mg_phi(:,:)
        integer, intent(in) :: G
        real(dp), intent(in) :: L
        integer, intent(in) :: N, N_iter
        real(dp), allocatable :: rhs(:)


        integer :: ii, jj, kk
        !allocate(mg_phi(G,N), S(G,N), mg_sigmaS(G))
        allocate(mg_phi(G,N))
        do ii = 1, G 
            mg_phi(ii,:) = 0.0_dp
        end do 
        
        do kk = 1, N_iter
            do ii = 1, G
                rhs = s(ii,:)
                do jj = 1, G
                    if (ii /= jj) then 
                        rhs = rhs + mg_sigmaS(jj, ii)*mg_phi(jj,:)
                    end if 
                end do 
                mymesh = make_mesh(L, N, rhs, 1/(3*mg_sigmaA(ii,:)), mg_sigmaA(ii,:))
                tri = make_tridiag(mymesh)
                mg_phi(ii,:) = solver_fixed_source(mymesh, make_tridiag(mymesh), bc)
            end do
        end do
    end function solve_multigroup_fixedsource_2

    subroutine mg_fission() 


    end subroutine mg_fission



end module multigroup_module