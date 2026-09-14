program main
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
   

    type(mesh) :: mymesh
    class(BoundaryCondition), allocatable :: bc
    class(tridiag), allocatable :: tri 
    real(dp), allocatable :: phi(:)
    real(dp), allocatable :: mg_phi(:,:)


    real(dp), allocatable :: D(:), sigma_a(:), S(:)      !Diffusion length, absorption cross-sec, source
    real(dp), allocatable :: mg_S(:,:), mg_sigmaA(:,:), mg_sigmaS(:,:)
    real(dp) :: L, dx                  !Mesh length, mesh spacing 
    integer :: N, ii, G, jj                     !Number of cells , dummy, number of groups 
    real(dp) :: k, nusigma_f                !evalue, fission cross sec * <neutrons per fission> 
    real(dp) :: albedo_left, albedo_right, D_left, D_right      !self explanatory

    !Initialize params
    N = 100 
    allocate(D(N), sigma_a(N), S(N))
    D(1:50) = 0.8_dp
    D(51:100) = 3.8_dp

    sigma_a = 1/(3*D)

    D_left = D(1)
    D_right = D(N)

    albedo_left = 0.3_dp 
    albedo_right = 0.3_dp 
           
    L = 10.0_dp
    dx = L/N

    S = 1.0_dp
    !rhs = S

    print*,"hello"


    mymesh = make_mesh(L, N, S, D, sigma_a)
    

    !bc  = make_ZeroFluxBC(dx)
    bc = make_VacuumBC(dx, D_left, D_right)
    !bc = make_AlbedoBC(dx, albedo_left, albedo_right, D_left, D_right)
    !bc = make_SurfaceSourceBC(dx, surface_left, surface_right, D_left, D_right)
    !bc = make_ReflectiveBC(dx)
    !bc = make_PeriodicBC()

    tri = make_tridiag(mymesh)
    !tri = make_cyclic_tridiag(mymesh)
    
   ! phi = solver_fixed_source(mymesh, tri, bc)
    k = 1.0_dp
    nusigma_f = 5.0_dp
    phi = solver_eigenvalue_tridiag(mymesh, tri, bc, 5, nusigma_f, k)
   ! print*, k 

    !phi = 1.0_dp
    !k = 0.0_dp 
    !nusigma_f = 5.0_dp
    !allocate(matrix(N,N))
    !matrix = convert_tridiag_to_matrix(tri)
    !call solver_fission_source(matrix, bc, phi, k, nusigma_f, dx)
    !call write_to_csv("solution.csv", mymesh%x, phi)

!==============================Multigroup====================================================!
  
    G=5
    N=100
    L=10.0_dp

    dx = L/N
    allocate(mg_phi(G,N), mg_sigmaA(G,N), mg_sigmaS(G,G), mg_S(G,N))

    bc = make_ZeroFluxBC(dx)
    do ii =1, G 
        mg_S(ii,:) = 1.0_dp
        mg_sigmaA(ii,1:50) = 1.0_dp 
        mg_sigmaA(ii,51:100) = 5.0_dp
        !mg_sigmaS(ii) = 1.0_dp
    end do 

    !do ii =1,G 
    !    do jj = 1, G 
    !        mg_SigmaS(ii,jj) = 1.0_dp
    !    end do 
    !end do 

    do ii = 1, G
        do jj = 1, G
            if (ii == jj) then
                ! Self-scatter: largest contribution
                mg_SigmaS(ii,jj) = 0.7_dp
            else if (jj == ii-1) then
                ! Downscatter from higher to lower energy
                mg_SigmaS(ii,jj) = 0.2_dp
            else if (jj == ii+1) then
                ! Upscatter (small but nonzero)
                mg_SigmaS(ii,jj) = 0.05_dp
            else
                ! Very small coupling to distant groups
                mg_SigmaS(ii,jj) = 0.0_dp
            end if
        end do
    end do



    !mg_phi = solve_multigroup_fixedsource(bc, mg_sigmaS, mg_sigmaA, mg_S, G, L, N)
    mg_phi = solve_multigroup_fixedsource_2(bc, mg_sigmaS, mg_sigmaA, mg_S, G, L, N, 5)

    call write_to_csv_multigroup("solution_multigroup.csv", [( (ii-0.5_dp)*dx, ii=1,N )], mg_phi)



!==========================================================================================!



end program main