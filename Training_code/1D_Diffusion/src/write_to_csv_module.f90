module write_to_csv_module 
    use precision 
    implicit none 

contains 

    subroutine write_to_csv(filename, x, phi)
        

        character(len=*), intent(in) :: filename
        real(dp), intent(in) :: x(:), phi(:)
        integer :: i, unit

        ! Choose a safe unit number
        unit = 12  

        open(unit=unit, file=filename, status="replace", action="write")

        do i = 1, size(x)
            write(unit, '(F12.6, ",", F12.6)') x(i), phi(i)
        end do

        close(unit)
    end subroutine write_to_csv


    subroutine write_to_csv_multigroup(filename, x, phi)
        character(len=*), intent(in) :: filename
        real(dp), intent(in) :: x(:)               ! mesh positions
        real(dp), intent(in) :: phi(:,:)           ! (G,N): group, cell
        integer :: i, gg, N, G
        integer :: unit

        N = size(x)
        G = size(phi, dim=1)

        open(newunit=unit, file=filename, status='replace', action='write')

        ! --- Write header ---
        write(unit,'(A)',advance='no') "x"
        do gg = 1, G
            write(unit,'(A,I0)',advance='no') ",phi_group"//trim(adjustl(to_string(gg)))
        end do
        write(unit,*)

        ! --- Write rows ---
        do i = 1, N
            write(unit,'(ES16.8)',advance='no') x(i)
            do gg = 1, G
                write(unit,'(",",ES16.8)',advance='no') phi(gg,i)
            end do
            write(unit,*)
        end do

        close(unit)
    end subroutine write_to_csv_multigroup

    ! helper to convert integer to string
    pure function to_string(i) result(str)
        integer, intent(in) :: i
        character(len=20) :: str
        write(str,'(I0)') i
    end function to_string



end module 