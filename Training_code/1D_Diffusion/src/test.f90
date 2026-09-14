program test 
    use precision
    use mesh_module
    use power_iteration_module
    use bc_module
    use tridiag_module
    use TA_module 
    use solver_module
    use multigroup_module
    use write_to_csv_module
    implicit none  

    class(BoundaryCondition), allocatable :: bc 
    real(dp), allocatable :: mg_S(:,:), mg_sigmaA(:,:), mg_sigmaS(:), mg_phi(:,:)
    real(dp) :: L, dx
    integer :: ii, G, N 

    G=5
    N=100
    L=10.0_dp

    dx = L/N
    allocate(mg_phi(G,N), mg_sigmaA(G,N), mg_sigmaS(G), mg_S(G,N))

    !bc, mg_sigmaS, mg_sigmaA, S, G, L, N

    bc = make_ZeroFluxBC(dx)
    do ii =1, G 
        mg_S(ii,:) = ii 
        mg_sigmaA(ii,:) = 1.0_dp 
        mg_sigmaS(ii) = 1.0_dp
    end do 



    mg_phi = solve_multigroup_fixedsource(bc, mg_sigmaS, mg_sigmaA, mg_S, G, L, N)

    !print*, mg_phi(1,:)
    call write_to_csv_multigroup("solution_multigroup.csv", [( (ii-0.5_dp)*dx, ii=1,N )], mg_phi)



end program test 