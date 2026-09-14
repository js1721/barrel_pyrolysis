!===================================================================================================
!
! Source module
!
! Date:     Programmer:     Changes:
! =====     ===========     ========
! 26-11-25  J Salter        Original
!===================================================================================================

module mod_source 

    use mod_constants
    use mod_types 
    use mod_map
    use mod_params
    use mod_eigenvalue
    implicit none 

    !real(dp), allocatable :: rhs(:)


contains 

    subroutine assemble_fixed_source(rhs, value)

        real(dp), intent(in) :: value
        real(dp), intent(inout), allocatable :: rhs(:)
    
        allocate(rhs((nx)*(ny)))
        rhs = value 
    
    end subroutine assemble_fixed_source

    subroutine assemble_mms_source(rhs)

      
        real(dp), allocatable :: S(:,:)
        integer :: ii, jj
        real(dp), intent(out), allocatable :: rhs(:)
        
        allocate(S(nx,ny))
        allocate(rhs(nx*ny))
        do ii = 1, nx 
            do jj =1, ny 
                S(ii,jj) = (2.0_dp * 3.14_dp * 3.14_dp) * cos(3.14_dp*mesh%x(ii)) * cos(3.14_dp*mesh%y(jj))
            end do 
        end do 
        do ii =1, nx
            do jj =1, ny
                rhs(map(ii,jj,nx))=S(ii,jj)
            end do 
        end do

    end subroutine assemble_mms_source

    subroutine assemble_fission_source(rhs, A)
         
        real(dp), intent(inout), allocatable :: rhs(:)
        real(dp), allocatable :: S(:,:)
        integer :: ii, jj 
        type(sparse_matrix), intent(in) :: A
        
        call solver_eigenvalue(A, N_iterations, material_list, mesh, initial_guess, k, S)
        allocate(rhs(nx*ny))
        do ii = 1, nx
            do jj = 1, ny 
                rhs(map(ii,jj,nx)) = S(ii,jj)
            end do 
        end do 


    end subroutine assemble_fission_source
        


end module mod_source

        