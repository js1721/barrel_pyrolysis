!===================================================================================================
!
! Map module
!
! Date:     Programmer:     Changes:
! =====     ===========     ========
! 25-11-25  J Salter        Original
!===================================================================================================



module mod_map 

    implicit none 

contains 


    integer function map(ii, jj, gg, G, nx) result (kk)
        integer, intent(in) :: ii, jj, gg, G, nx
        !kk = (jj-1) *nx +ii
        !PETSc indexes starting at 0
        kk = (gg-1) + G*((ii-1) + nx*(jj-1))

    end function map

end module mod_map