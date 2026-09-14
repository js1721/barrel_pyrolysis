!===================================================================================================
!
! Geometry module
!
! Date:     Programmer:     Changes:
! =====     ===========     ========
! 20-11-25  J Salter        Original
!===================================================================================================


module mod_geometry 

    use mod_constants
    use mod_types
    implicit none 

contains 

    subroutine build_mesh_xy(mesh, nx, ny, Lx, Ly)

        type(mesh_xy), intent(out) :: mesh
        integer, intent(in) :: nx, ny 
        real(dp), intent(in) :: Lx, Ly 
        integer :: ii 

        mesh%nx = nx; mesh%ny = ny
        mesh%Lx = Lx; mesh%Ly = Ly 
        mesh%dx = Lx/(nx-1); mesh%dy = Ly/(ny-1)

        allocate(mesh%x(nx), mesh%y(ny))

        do ii = 1, nx
            mesh%x(ii) = (ii-1) *mesh%dx
        end do 

        do ii = 1, ny 
            mesh%y(ii) = (ii-1) *mesh%dy
        end do 

        

    end subroutine build_mesh_xy 


end module mod_geometry