program test 

    use mod_pk_solver
    implicit none 

    real(KIND=8) :: tstart, tend, dtout, rtol, atol
    real(KIND=8),allocatable :: ics(:)
    integer(KIND=8)          :: G
    real(KIND=8)             :: rho, lambda
    real(KIND=8), allocatable             :: beta(:), lambdai(:)
    
    tstart = 0.0d0 ; tend = 1.0d0 
    rtol = 1.0e-13 ; atol = 1.0e-13
    dtout = 0.001d0

 
    ics=[1.0d0,1.0d0]


    G = 1
    rho = -0.05d0 
    lambda = 6.5e-5
    beta = [0.0075e-3]
    lambdai = [0.08d0]

    call pk_solver(G, rho, lambda, beta, lambdai, ics, tstart, tend, dtout, rtol, atol)




end program test 