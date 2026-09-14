!===================================================================================================
!
! Postprocess module
!
! Date:     Programmer:     Changes:
! =====     ===========     ========
! 09-12-25  J Salter        Original
!===================================================================================================

module mod_postprocess 

    use mod_constants
    use mod_map 
    use mod_types
    use mod_output

    implicit none 

contains 

    subroutine plotter(phi, flat_phi, mesh)

        real(dp), intent(inout), allocatable :: phi(:,:)
        real(dp), intent(in) :: flat_phi(:)
        type(mesh_xy), intent(in) :: mesh
        integer :: ii, jj 

        allocate(phi(mesh%nx,mesh%ny))
        do ii = 1, mesh%nx 
            do jj = 1, mesh%ny
                phi(ii,jj) = flat_phi(map(ii,jj, mesh%nx))
            end do 
        end do
    
        call write_xy_flux("flux.txt",mesh%x, mesh%y,phi)

    end subroutine plotter

end module mod_postprocess