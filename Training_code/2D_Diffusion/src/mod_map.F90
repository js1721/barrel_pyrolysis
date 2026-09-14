!===================================================================================================
!
! Map module
!
! Date:     Programmer:     Changes:
! =====     ===========     ========
! 25-11-25  J Salter        Original
!===================================================================================================



module mod_map 

    use mod_constants
    implicit none 

contains 


    integer function map(ii, jj, nx) result (kk)
        integer, intent(in) :: ii, jj, nx
        kk = (jj-1) *nx +ii
    end function map

end module mod_map