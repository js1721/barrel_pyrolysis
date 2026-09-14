!===================================================================================================
!
! Boundary Conditions module
!
! Date:     Programmer:     Changes:
! =====     ===========     ========
! 20-11-25  J Salter        Original
!===================================================================================================


module mod_bc

    use mod_constants
    use mod_types
    use mod_map
    implicit none 


contains 

    subroutine apply_bc_xy(bc_L, bc_R, bc_B, bc_T, M, mesh, rhs, D)
        class(bc), intent(in) :: bc_L, bc_R, bc_B, bc_T
        type(mesh_xy), intent(in) :: mesh
        real(dp), intent(inout) :: M(:,:)
        real(dp), intent(inout) :: rhs(:)
        real(dp), intent(in) :: D(:,:)

        integer :: ii, jj, kk
        real(dp) :: dx, dy !coeff, dd

        integer :: kkL, kkR, kkE, kkW, kkN, kkS
        real(dp) :: DE, a

        integer :: kkB, kkT 
        real(dp) :: DN

        integer :: nx, ny

        nx = mesh%nx ; ny = mesh%ny

        dx = mesh%dx
        dy = mesh%dy

        

        !Left and right boundaries
        select type(bc_L)
            type is (periodicBC)
                
                do jj = 1, ny
                    kkL = map(1, jj, nx)
                    kkR = map(nx, jj, nx)
                    if (D(1,jj) + D(nx,jj) == 0.0_dp) then
                        DE = 0.0_dp
                    else
                        DE = 2.0_dp* D(1,jj) * D(nx,jj) / (D(1,jj) + D(nx,jj))
                    end if
                    a = - DE * mesh%dy / mesh%dx
                    M(kkL, kkR) = M(kkL, kkR) + a
                    M(kkL, kkL) = M(kkL, kkL) - a
                    M(kkR, kkL) = M(kkR, kkL) + a
                    M(kkR, kkR) = M(kkR, kkR) - a
                end do
            
            class default
    
                
                do jj = 2, mesh%ny -1
                    kk = map(1, jj, nx)
                    kkE = map(2,jj,nx)
                    
                    M(kk,kk) = M(kk,kk)  + bc_L%alpha -bc_L%beta /dx
                    M(kk,kkE) = M(kk,kkE) + bc_L%beta /dx
                    M(kk, kk) = M(kk,kk) 
                    M(kk, kkE) = M(kk,kkE) 
                    rhs(kk) =  +bc_L%gamma
                    
                end do

                do jj = 2, mesh%ny -1
                    kk = map(nx, jj, nx)
                    kkW = map(nx -1, jj, nx)
                    
                    M(kk,kk) = M(kk,kk)  + bc_R%alpha +bc_R%beta /dx
                    M(kk,kkW) = M(kk,kkW) - bc_R%beta /dx
                    rhs(kk) = bc_R%gamma

                    M(kk, kk) = M(kk,kk) 
                    M(kk, kkW) = M(kk,kkW) 
                

                end do

                
        end select
        
        !B and T boundaries 
        select type(bc_B)
            type is (periodicBC)
                
                do ii = 1, nx
                    kkB = map(ii, 1, nx)
                    kkT = map(ii, ny, nx)
                    if (D(ii,1) + D(ii,ny) == 0.0_dp) then
                        DN = 0.0_dp
                    else
                        DN = 2.0_dp* D(ii,1) * D(ii,ny) / (D(ii,1) + D(ii,ny))
                    end if
                
                    a = - DN * mesh%dx / mesh%dy
                    M(kkB, kkT) = M(kkB, kkT) + a
                    M(kkB, kkB) = M(kkB, kkB) - a
                    M(kkT, kkB) = M(kkT, kkB) + a
                    M(kkT, kkT) = M(kkT, kkT) - a
                end do
            

            class default 


                do ii = 2, mesh%nx -1
                    kk = map(ii, 1, nx)
                    kkN = map(ii, 2, nx)

                    M(kk,kk) = M(kk,kk)  + bc_B%alpha -bc_B%beta /dy
                    M(kk,kkN) = M(kk,kkN) + bc_B%beta /dy

                    M(kk, kk) = M(kk,kk) 
                    M(kk, kkN) = M(kk,kkN) 
                    rhs(kk) =  +bc_B%gamma

                   
                end do

                do ii = 2, mesh%nx -1
                    kk = map(ii, ny, nx)
                    kkS = map(ii, ny-1, nx)

                    M(kk,kk) = M(kk,kk)  + bc_T%alpha +bc_T%beta /dy
                    M(kk,kkS) = M(kk,kkS) - bc_T%beta /dy

                    M(kk, kk) = M(kk,kk) 
                    M(kk, kkS) = M(kk,kkS) 
                    rhs(kk) =  +bc_T%gamma
                   
                end do

                kk = map(1,1,nx)
                M(kk,kk) = 0.1
                kk = map(1,ny,nx)
                M(kk,kk) = 0.1
                kk = map(nx,1,nx)
                M(kk,kk) = 0.1
                kk = map(nx,ny,nx)
                M(kk,kk) = 0.1


        end select

    end subroutine apply_bc_xy

    
    


 




end module mod_bc