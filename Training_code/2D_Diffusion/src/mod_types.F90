!===================================================================================================
!
! Types module
!
! Date:     Programmer:     Changes:
! =====     ===========     ========
! 19-11-25  J Salter        Original
!===================================================================================================


module mod_types 

    use mod_constants
    implicit none 

    type :: mesh_xy 
        integer :: nx, ny               !Number of cells
        real(dp) :: dx, dy, Lx, Ly
        real(dp), allocatable :: x(:), y(:) 
        integer, allocatable :: region(:,:)     !x, y 
    end type 


    type :: material 
        real(dp) :: D, sigma_a, nuSigma_f, chi
        real(dp), allocatable :: sigma_s(:)
    end type

    type :: mg_flux 
        real(dp), allocatable :: phi(:,:,:), source(:,:,:)   !x, y, G
    end type

    type :: bc 
        real(dp) :: alpha, beta, gamma 
    end type 

    type, extends(bc) :: periodicBC
    end type 

end module mod_types