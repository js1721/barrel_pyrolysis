


module mod_diffusion_matrix
    use mod_petsc_wrapper
    use mod_map
    implicit none 


contains 

    subroutine diffusion_matrix(A, mesh, D, sigmaR, sigmaS, G)

        !Arguments
        Mat, intent(inout) :: A
        type(mesh_xy), intent(in) :: mesh
        real(dp), intent(in) :: D(:,:,:), sigmaR(:,:,:), sigmaS(:,:,:,:)
        integer, intent(in) :: G
     
        
        !Locals
        real(dp) :: dx, dy
        integer :: ii, jj, gg
        integer :: N 

        real(dp) :: DE, DW, DN, DS !diff coeff values at faces
        real(dp) :: aE, aW, aN, aS, aP !discretised equation coeffs
        integer :: kk, kkE, kkW, kkN, kkS, gp, kkP !Mapped points
        integer :: ierr

        dx = mesh%dx; dy = mesh%dy
      
        N = (mesh%nx) * (mesh%ny)
       
        do ii = 2, mesh%nx -1

            do jj = 2, mesh%ny -1

                do gg = 1, G 

                    kk = map(ii, jj, gg, G, mesh%nx)
                    aP = 0.0_dp

                    !West point
                    kkW = map(ii-1, jj, gg, G, mesh%nx)
                    DW = 2.0_dp * D(ii,jj,gg)*D(ii-1,jj,gg) / (D(ii,jj,gg)+D(ii-1,jj,gg))
                    aW = -DW * mesh%dy / mesh%dx
                    aP = aP - aW
                    call MatSetValue(A, kk, kkW, aW, ADD_VALUES, ierr)

                    !East point
                    kkE = map(ii+1, jj, gg, G, mesh%nx)
                    DE = 2.0_dp * D(ii,jj,gg)*D(ii+1,jj,gg) / (D(ii,jj,gg)+D(ii+1,jj,gg))
                    aE = -DE * mesh%dy / mesh%dx
                    aP = aP - aE
                    call MatSetValue(A, kk, kkE, aE, ADD_VALUES, ierr)

                    !North point
                    kkN = map(ii, jj-1, gg, G, mesh%nx)
                    DN = 2.0_dp * D(ii,jj,gg)*D(ii,jj-1,gg) / (D(ii,jj,gg)+D(ii,jj-1,gg))
                    aN = -DN * mesh%dx / mesh%dy 
                    aP = aP - aN
                    call MatSetValue(A, kk, kkN, aN, ADD_VALUES, ierr)

                    !South point
                    kkS = map(ii, jj+1, gg, G, mesh%nx)
                    DS = 2.0_dp * D(ii,jj,gg)*D(ii,jj+1,gg) / (D(ii,jj,gg)+D(ii,jj+1,gg))
                    aS = -DS * mesh%dx / mesh%dy
                    aP = aP - aS
                    call MatSetValue(A, kk, kkS, aS, ADD_VALUES, ierr)

                
                    aP = aP + sigmaR(ii,jj,gg) *mesh%dx *mesh%dy

                    do gp = 1, G
                        if (gp /= gg) then
                            kkP = map(ii, jj, gp, G, mesh%nx)
                            ! Forward operator: -Sigma_s(gp -> g)
                            call MatSetValue(A, kk, kkP, &
                                -SigmaS(ii,jj,gp,gg) * mesh%dx * mesh%dy, &
                                ADD_VALUES, ierr)
                        end if
                    end do

                    call MatSetValue(A, kk, kk, aP, ADD_VALUES, ierr)
                end do
            end do 
        end do 
        
    end subroutine diffusion_matrix

    subroutine build_rhs(rhs, mesh, source, G)

        !Arguments
        Vec, intent(inout) :: rhs
        type(mesh_xy), intent(in) :: mesh
        real(dp), intent(in) :: source(:,:,:)
        integer, intent(in) :: G 

        !Locals 
        integer :: ii, jj, kk, gg, ierr
        real(dp) :: dx, dy, val 

        dx = mesh%dx; dy = mesh%dy

        do ii = 1, mesh%nx 
            do jj = 1, mesh%ny
                do gg = 1, G 
                    kk = map(ii, jj, gg, G, mesh%nx)
                    val = source(ii, jj, gg) * dx * dy 
                    call VecSetValue(rhs, kk, val, INSERT_VALUES, ierr)
                end do 
            end do 
        end do


        call VecAssemblyBegin(rhs, ierr)
        call VecAssemblyEnd(rhs, ierr)


    end subroutine build_rhs



end module mod_diffusion_matrix