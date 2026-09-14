!===================================================================================================
!
! Parameters module
!
! Date:     Programmer:     Changes:
! =====     ===========     ========
! 19-11-25  J Salter        Original
!===================================================================================================

module mod_params 

    use mod_constants 
    use mod_types 
    use mod_geometry
    use mod_materials
    use mod_materials_data
    implicit none 

    !Mesh related
    type(mesh_xy) :: mesh 
    real(dp) :: Lx = 1.0_dp, Ly = 1.0_dp
    integer :: nx = 50, ny = 50
 

    !BC related
    type(BC) :: bc_L, bc_R
    type(bc) :: bc_B, bc_T 
   

    !Material related
    type(material), allocatable :: material_list(:)
    integer, allocatable :: materials(:), region_start_x(:), region_end_x(:)
    real(dp) :: fixed_source
    real(dp), allocatable :: sigma_s(:,:), nusigma_f(:)
    real(dp), allocatable :: chi(:)

    !Computationally related 
    integer :: N_iterations 
    real(dp), allocatable :: initial_guess(:)
    real(dp) :: k
    integer :: G

    

contains 

    
    subroutine init_params()

        !Mesh
        call build_mesh_xy(mesh, nx, ny, Lx, Ly)
        
        
        
        
        !BCs                                                
        bc_L%alpha = 1.0_dp; bc_L%beta=0.0_dp; bc_L%gamma =0.0_dp
        bc_R%alpha = 1.0_dp; bc_R%beta=0.0_dp; bc_R%gamma =0.0_dp
        bc_B%alpha = 1.0_dp; bc_B%beta=0.0_dp; bc_B%gamma =0.0_dp
        bc_T%alpha = 1.0_dp; bc_T%beta=0.0_dp; bc_T%gamma =0.0_dp


        !Materials 
        allocate(materials(1), region_start_x(1), region_end_x(1))
        allocate(material_list(1))
        materials = [1]
        region_start_x = [1] ; region_end_x = [nx]
        call assign_material_regions_xy(mesh, materials, region_start_x, region_end_x)
        call init_materials()
        material_list = [material_test]
        fixed_source = 100.0_dp

        !Computationally related 
        N_iterations = 500
        allocate(initial_guess(nx*ny))
        initial_guess = 1.0_dp
        k = 1.0_dp
        G = 2

        allocate(nusigma_f(2), sigma_s(2,2))
        allocate(chi(2))
        nusigma_f = [1.0_dp,2.0_dp]
        !sigma_s = [[0.0_dp,0.0_dp],[1.0_dp,0.0_dp]]
        sigma_s(1,1) = 0.0_dp ; sigma_s(1,2) = 0.0_dp 
        sigma_s(2,1) = 1.0_dp ; sigma_s(2,2) = 0.0_dp
        chi = [0.5_dp,0.5_dp]
 
    end subroutine init_params

end module mod_params