!===================================================================================================
!
! Output module
!
! Date:     Programmer:     Changes:
! =====     ===========     ========
! 25-11-25  J Salter        Original
!===================================================================================================




module mod_output
    implicit none
    private
    public :: write_matrix_text, write_matrix_binary, write_xy_flux, write_matrix_csv, write_vector_csv

contains

    !==============================================================
    ! Write a real 2D array to a human-readable text file
    !==============================================================
    subroutine write_matrix_text(filename, A)
        character(*), intent(in) :: filename
        real(kind=8), intent(in) :: A(:,:)
        integer :: i, unit

        open(newunit=unit, file=filename, status="replace", action="write")
        do i = 1, size(A,1)
            write(unit,'(*(E24.16))') A(i,:)   ! write row i nicely formatted
        end do
        close(unit)

    end subroutine write_matrix_text

    !==============================================================
    ! Write a real 2D array to a binary file (numpy-readable)
    !==============================================================
    subroutine write_matrix_binary(filename, A)
        character(*), intent(in) :: filename
        real(kind=8), intent(in) :: A(:,:)
        integer :: unit

        open(newunit=unit, file=filename, access="stream", form="unformatted", status="replace")
        write(unit) A   ! writes data in column-major order
        close(unit)

    end subroutine write_matrix_binary

    subroutine write_xy_flux(filename, x, y, A)
        character(*), intent(in) :: filename
        real(8), intent(in) :: x(:), y(:)
        real(8), intent(in) :: A(:,:)
        integer :: i, j, unit

        open(newunit=unit, file=filename, status="replace", action="write")

        do i = 1, size(x)
            do j = 1, size(y)
                write(unit,'(3E24.16)') x(i), y(j), A(i,j)
            end do
        end do

        close(unit)
    end subroutine write_xy_flux

    subroutine write_matrix_csv(filename, A)
    
        character(len=*), intent(in) :: filename
        real(kind=8), intent(in)     :: A(:,:)

        integer :: i, j, unit

        ! Get a free logical unit number
        inquire (iolength=unit)
        open(newunit=unit, file=filename, status='replace', action='write')

        do i = 1, size(A, 1)
            do j = 1, size(A, 2)
                if (j < size(A, 2)) then
                    write(unit, '(F20.10, ",")', advance='no') A(i, j)
                else
                    write(unit, '(F20.10)') A(i, j)
                end if
            end do
        end do

        close(unit)
    end subroutine write_matrix_csv

    subroutine write_vector_csv(filename, v)
        implicit none
        character(len=*), intent(in) :: filename
        real(kind=8), intent(in)     :: v(:)

        integer :: i, unit

        ! Get a free unit
        open(newunit=unit, file=filename, status='replace', action='write')

        do i = 1, size(v)
            if (i < size(v)) then
                write(unit, '(F20.10, ",")', advance='no') v(i)
            else
                write(unit, '(F20.10)') v(i)
            end if
        end do

        close(unit)
    end subroutine write_vector_csv




end module mod_output
