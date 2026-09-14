!===================================================================================================
!
! Solver module
!
! Date:     Programmer:     Changes:
! =====     ===========     ========
! 09-12-25  J Salter        Original
!===================================================================================================

module mod_solver 

    use mod_constants
    use mod_types
    use mod_solver_pcg
    use mod_source
    use mod_diffusion_matrix

    implicit none 


contains 


    subroutine solver_fixed_source(flat_phi)

        real(dp), intent(inout), allocatable :: flat_phi(:)
      
        real(dp), allocatable :: rhs(:)
        type(sparse_matrix) :: A

        call assemble_fixed_source(rhs, fixed_source)
        call assemble_diffusion_xy(A, mesh, material_list, bc_L, bc_R, bc_B, bc_T, rhs)
        allocate(flat_phi(nx*ny))
        flat_phi = cjm(A, rhs, initial_guess, N_iterations)


    end subroutine solver_fixed_source

    subroutine solver_mms_source(flat_phi)

        real(dp), intent(inout), allocatable :: flat_phi(:)
        type(sparse_matrix) :: A
        real(dp), allocatable :: rhs(:)

        call assemble_mms_source(rhs) 
        call assemble_diffusion_xy(A, mesh, material_list, bc_L, bc_R, bc_B, bc_T, rhs)
        allocate(flat_phi(nx*ny))
        flat_phi = cjm(A, rhs, initial_guess, N_iterations)
        

    end subroutine solver_mms_source

    subroutine solver_fission_source(flat_phi)

        real(dp), intent(inout), allocatable :: flat_phi(:)
        type(sparse_matrix) :: A, B 
        real(dp), allocatable :: rhs(:)

        call assemble_diffusion_eigenvalue(A, mesh, material_list)
        call assemble_fission_source(rhs, A)
        call assemble_diffusion_xy(B, mesh, material_list, bc_L, bc_R, bc_B, bc_T, rhs)
        allocate(flat_phi(nx*ny))
        flat_phi = cjm(B, rhs, initial_guess, N_iterations)

    end subroutine solver_fission_source

end module mod_solver