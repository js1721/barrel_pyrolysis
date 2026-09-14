!===================================================================================================
!
! Materials module
!
! Date:     Programmer:     Changes:
! =====     ===========     ========
! 20-11-25  J Salter        Original
!===================================================================================================



module mod_materials

    use mod_constants
    use mod_types
    implicit none 

    !integer, allocatable :: region_start_x(:), region_end_x(:)
    !nteger, allocatable :: materials(:)
    !e.g materials(1) =  water, materials(2) = fuel
    !region_start_x = [1, 51]
    !region_end_x = [50, 100]
    !materials = [1,2]


contains  

    subroutine assign_material_regions_xy(mesh, materials, region_start_x, region_end_x)
        type(mesh_xy), intent(inout) :: mesh 
        integer, intent(in) :: materials(:) 
        integer :: ii
        integer, intent(in) :: region_start_x(:), region_end_x(:)
        !Mesh has region(:,:) that needs allocating here;
        !i.e region(x,y) =integer where integer references a specific
        !material in materials(:). Will also assign materials(integer)=(some specific material)

        allocate(mesh%region(mesh%nx, mesh%ny))
        do ii = 1, size(region_start_x)
            mesh%region(region_start_x(ii):region_end_x(ii), :) = materials(ii)
        end do 

        
    
    end subroutine assign_material_regions_xy

    subroutine assign_diffusion_xy(mesh, D, material_list)
        type(mesh_xy), intent(in) :: mesh 
        type(material), intent(in) :: material_list(:)
        real(dp), intent(inout), allocatable :: D(:,:)
        integer :: ii 
        allocate(D(size(mesh%x), size(mesh%y)))
        do ii = 1, size(mesh%x)
            D(ii,:)= material_list(mesh%region(ii,1))%D
        end do 
         
    end subroutine assign_diffusion_xy
    
    
    
    subroutine assign_sigmaA_xy(mesh, sigmaA, material_list)
        type(mesh_xy), intent(in) :: mesh 
        type(material), intent(in) :: material_list(:)
        real(dp), intent(inout), allocatable :: sigmaA(:,:)
        integer :: ii 
        allocate(sigmaA(size(mesh%x), size(mesh%y)))
        do ii = 1, size(mesh%x)
            sigmaA(ii,:)= material_list(mesh%region(ii,1))%sigma_a
        end do 
         
    end subroutine assign_sigmaA_xy

    subroutine assemble_nuSigma_f_xy(mesh, nusigma_f, material_list)
        type(mesh_xy), intent(in) :: mesh 
        type(material), intent(in) :: material_list(:)
        real(dp), intent(inout), allocatable :: nusigma_f(:,:)
        integer :: ii 
        allocate(nusigma_f(size(mesh%x), size(mesh%y)))
        do ii = 1, size(mesh%x)
            nusigma_f(ii,:)= material_list(mesh%region(ii,1))%nuSigma_f
        end do 
    end subroutine assemble_nuSigma_f_xy
    


end module mod_materials
