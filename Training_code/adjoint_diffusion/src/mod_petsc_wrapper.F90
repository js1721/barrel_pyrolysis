!===================================================================================================
!
! PETSc wrapper module
!
! Date:     Programmer:     Changes:
! =====     ===========     ========
! 12-01-26  J Salter        Original
!===================================================================================================

module mod_petsc_wrapper
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscvec.h>
#include <petsc/finclude/petscmat.h>
#include <petsc/finclude/petscksp.h>

    use petscsys
    use petscvec
    use petscmat
    use petscksp 

    implicit none
    private 

    ! === Public types === 
    public :: petsc_init, petsc_finalize

contains 


    subroutine petsc_init()
        PetscErrorCode :: ierr 
        call PetscInitialize(ierr)
        if (ierr /= 0) then 
            write(*,*) "Error in PetscInitialize, ierr=", ierr
            stop 
        end if 
    end subroutine petsc_init

    subroutine petsc_finalize() 
        PetscErrorCode :: ierr 
        call PetscFinalize(ierr)
        if (ierr /= 0) then 
            write(*,*) "Error in PetscFinalize, ierr=", ierr
        end if 
    end subroutine petsc_finalize




end module mod_petsc_wrapper