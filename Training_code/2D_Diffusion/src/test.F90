program test 
#include <petsc/finclude/petscvec.h>
! #include "$PETSC_DIR/include/petsc/finclude/petscvec.h"

    use petsc
    use petscsys
    use petscvec
    use mod_constants
    use mod_types
    use mod_params 
    use mod_source 
    use mod_solver 
    use mod_postprocess
    use mod_mg

    implicit none 

    
    PetscErrorCode :: ierr

    real(dp), allocatable :: flat_phi(:)
    real(dp), allocatable :: phi(:,:)
    real(dp), allocatable :: mg_phi(:,:,:)
   
    call PetscInitialize(ierr)
  

    call init_params()

    call solver_fixed_source(flat_phi)
    !call solver_mms_source(flat_phi)
    !call solver_fission_source(flat_phi)

    
    call plotter(phi, flat_phi, mesh)


    !call solver_mg(initial_guess,k,mg_phi)
    call PetscFinalize(ierr)

end program test