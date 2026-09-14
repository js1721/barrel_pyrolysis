!===================================================================================================
!
! Diffusion Matrix module
!
! Date:     Programmer:     Changes:
! =====     ===========     ========
! 20-11-25  J Salter        Original
!===================================================================================================


module mod_diffusion_matrix

  use mod_types
  use mod_materials
  use mod_sprs
  use mod_bc
  use mod_map
  use mod_output
  implicit none

contains

    subroutine assemble_diffusion_xy(A, mesh, material_list, bc_L, bc_R, bc_B, bc_T, rhs)
        
        type(sparse_matrix), intent(inout) :: A
        type(material), intent(in) :: material_list(:)
        class(bc), intent(in) :: bc_L, bc_R, bc_B, bc_T
        real(dp), intent(inout) :: rhs(:)
 
        type(mesh_xy), intent(in) :: mesh
        real(dp) :: dx, dy

        real(dp), allocatable :: M(:,:)   !Temp matrix to be converted to sprs
        real(dp), allocatable :: D(:,:), sigmaA(:,:)
        integer :: ii, jj
        integer :: N 

        real(dp) :: DE, DW, DN, DS !diff coeff values at faces
        real(dp) :: aE, aW, aN, aS, aP !discretised equation coeffs
        integer :: kk, kkE, kkW, kkN, kkS !Mapped points

        dx = mesh%dx; dy = mesh%dy
        call assign_diffusion_xy(mesh, D, material_list)
        call assign_sigmaA_xy(mesh, sigmaA, material_list)

        N = (mesh%nx) * (mesh%ny)
        allocate(M(N,N))
        M = 0.0_dp
       
        do ii = 2, mesh%nx -1

            do jj = 2, mesh%ny -1
                kk = map(ii, jj, mesh%nx)
                aP = 0.0_dp

                !West point
                kkW = map(ii-1, jj, mesh%nx)
                DW = 2.0_dp * D(ii,jj)*D(ii-1,jj) / (D(ii,jj)+D(ii-1,jj))
                aW = -DW * mesh%dy / mesh%dx
                aP = aP - aW
                M(kk, kkW) = aW
           

                !East point
                kkE = map(ii+1, jj, mesh%nx)
                DE = 2.0_dp * D(ii,jj)*D(ii+1,jj) / (D(ii,jj)+D(ii+1,jj))
                aE = -DE * mesh%dy / mesh%dx
                aP = aP - aE
                M(kk, kkE) = aE
                
                !North point
                kkN = map(ii, jj-1, mesh%nx)
                DN = 2.0_dp * D(ii,jj)*D(ii,jj-1) / (D(ii,jj)+D(ii,jj-1))
                aN = -DN * mesh%dx / mesh%dy 
                aP = aP - aN
                M(kk, kkN) = aN
            

                !South point
                kkS = map(ii, jj+1, mesh%nx)
                DS = 2.0_dp * D(ii,jj)*D(ii,jj+1) / (D(ii,jj)+D(ii,jj+1))
                aS = -DS * mesh%dx / mesh%dy
                aP = aP - aS
                M(kk, kkS) = aS
            

                aP = aP + sigmaA(ii,jj) *mesh%dx *mesh%dy
                
                M(kk, kk) = aP

            end do 
        end do 
      
        rhs = rhs *mesh%dx *mesh%dy
        
        call apply_bc_xy(bc_L, bc_R, bc_B, bc_T, M, mesh, rhs, D)
        !print*, M - transpose(M)
        !rhs = -rhs *mesh%dx *mesh%dy

  
        
        ! do ii = 1, size(M(1,:))
        !     write(*,'(100(1X,F12.6))') (M(ii,jj), jj=1,size(M(1,:)))
        ! end do
        ! print*, rhs

        !call write_matrix_csv("diff_matrix.csv", M)
      

        call convert_to_sparse(M, A, 1.0e-8_dp)

    end subroutine assemble_diffusion_xy 

    subroutine assemble_diffusion_eigenvalue(A, mesh, material_list)

        type(sparse_matrix), intent(inout) :: A
        type(material), intent(in) :: material_list(:)
        type(mesh_xy), intent(in) :: mesh
        real(dp) :: dx, dy
        integer :: nx, ny 

        real(dp), allocatable :: M(:,:)   !Temp matrix to be converted to sprs
        real(dp), allocatable :: D(:,:), sigmaA(:,:)
        integer :: ii, jj
        integer :: N 

        real(dp) :: DE, DW, DN, DS !diff coeff values at faces
        real(dp) :: aE, aW, aN, aS, aP !discretised equation coeffs
        integer :: kk, kkE, kkW, kkN, kkS !Mapped points

        dx = mesh%dx; dy = mesh%dy
        call assign_diffusion_xy(mesh, D, material_list)
        call assign_sigmaA_xy(mesh, sigmaA, material_list)

        nx = mesh%nx ; ny = mesh%ny
        N = (mesh%nx) * (mesh%ny)
        allocate(M(N,N))
        M = 0.0_dp
       
        do ii = 1, mesh%nx 

            do jj = 1, mesh%ny 
                kk = map(ii, jj, mesh%nx)
                aP = 0.0_dp

                !West point
                if (ii/=1) then
                    kkW = map(ii-1, jj, mesh%nx)
                    DW = 2.0_dp * D(ii,jj)*D(ii-1,jj) / (D(ii,jj)+D(ii-1,jj))
                    aW = -DW * mesh%dy / mesh%dx
                    M(kk, kkW) = aW
                else 
                    DW = D(ii,jj)
                    aW = -DW * mesh%dy / mesh%dx
                end if 
                aP = aP - aW
          
              

                !East point
                if (ii/=nx) then
                    kkE = map(ii+1, jj, mesh%nx)
                    DE = 2.0_dp * D(ii,jj)*D(ii+1,jj) / (D(ii,jj)+D(ii+1,jj))
                    aE = -DE * mesh%dy / mesh%dx
                    M(kk, kkE) = aE
                else 
                    DE = D(ii,jj)
                    aE = -DE * mesh%dy / mesh%dx
                end if 
                aP = aP - aE
               
                !North point
                if (jj/=1) then
                    kkN = map(ii, jj-1, mesh%nx)
                    DN = 2.0_dp * D(ii,jj)*D(ii,jj-1) / (D(ii,jj)+D(ii,jj-1))
                    aN = -DN * mesh%dx / mesh%dy
                    M(kk, kkN) = aN
                else 
                    DN = D(ii,jj)
                    aN = -DN * mesh%dx / mesh%dy
                end if 
                aP = aP - aN
                

                !South point
                if (jj/=ny) then
                    kkS = map(ii, jj+1, mesh%nx)
                    DS = 2.0_dp * D(ii,jj)*D(ii,jj+1) / (D(ii,jj)+D(ii,jj+1))
                    aS = -DS * mesh%dx / mesh%dy
                    M(kk, kkS) = aS
                else 
                    DS = D(ii,jj)
                    aS = -DS * mesh%dx / mesh%dy
                end if 
                aP = aP - aS

                aP = aP + sigmaA(ii,jj) *mesh%dx *mesh%dy
                
                M(kk, kk) = aP

            end do 
        end do 

        ! do ii = 1, size(M(1,:))
        !     write(*,'(100(1X,F12.6))') (M(ii,jj), jj=1,size(M(1,:)))
        ! end do
     
      

        call convert_to_sparse(M, A, 1.0e-8_dp)

    end subroutine assemble_diffusion_eigenvalue



end module mod_diffusion_matrix
