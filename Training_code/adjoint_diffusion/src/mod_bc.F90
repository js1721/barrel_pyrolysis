module mod_bc
  use mod_types
  use mod_map          
  use mod_petsc_wrapper
  implicit none
contains

  subroutine apply_bc_xy_multigroup(bc_L, bc_R, bc_B, bc_T, &
                                    A, b, mesh, D, G)
    ! Arguments
    class(bc),    intent(in)    :: bc_L, bc_R, bc_B, bc_T
    type(mesh_xy),intent(in)    :: mesh
    Mat,          intent(inout) :: A
    Vec,          intent(inout) :: b
    real(dp),     intent(in)    :: D(:,:)
    integer,      intent(in)    :: G   ! number of groups

    ! Locals
    integer :: ii, jj, gg
    integer :: nx, ny
    real(dp) :: dx, dy
    integer :: kk, kkL, kkR, kkE, kkW, kkN, kkS, kkB, kkT
    real(dp) :: DE, DN, a
    integer :: ierr

    nx = mesh%nx
    ny = mesh%ny
    dx = mesh%dx
    dy = mesh%dy

    !----------------------------------------
    ! LEFT and RIGHT boundaries (x = 1, nx)
    !----------------------------------------
    select type(bc_L)
    type is (periodicBC)

      ! Periodic left-right for all rows and all groups
      do jj = 1, ny
        do gg = 1, G
          kkL = map(1,  jj, gg, G, nx)
          kkR = map(nx, jj, gg, G, nx)

          if (D(1,jj) + D(nx,jj) == 0.0_dp) then
            DE = 0.0_dp
          else
            DE = 2.0_dp * D(1,jj) * D(nx,jj) / (D(1,jj) + D(nx,jj))
          end if
          a = - DE * dy / dx

          call MatSetValue(A, kkL, kkR, a, ADD_VALUES, ierr)
          call MatSetValue(A, kkL, kkL, -a, ADD_VALUES, ierr)
          call MatSetValue(A, kkR, kkL, a, ADD_VALUES, ierr)
          call MatSetValue(A, kkR, kkR, -a, ADD_VALUES, ierr)
        end do
      end do

    class default

      ! Left boundary: j = 2..ny-1
      do jj = 2, ny-1
        do gg = 1, G
          kk  = map(1, jj, gg, G, nx)   ! boundary cell
          kkE = map(2, jj, gg, G, nx)   ! interior neighbor

          call MatSetValue(A, kk, kk,  bc_L%alpha - bc_L%beta/dx, ADD_VALUES, ierr)
          call MatSetValue(A, kk, kkE, bc_L%beta/dx,             ADD_VALUES, ierr)
          call VecSetValue(b, kk, bc_L%gamma, INSERT_VALUES, ierr)
        end do
      end do

      ! Right boundary: j = 2..ny-1
      do jj = 2, ny-1
        do gg = 1, G
          kk  = map(nx,   jj, gg, G, nx)
          kkW = map(nx-1, jj, gg, G, nx)

          call MatSetValue(A, kk, kk,  bc_R%alpha + bc_R%beta/dx, ADD_VALUES, ierr)
          call MatSetValue(A, kk, kkW, -bc_R%beta/dx,             ADD_VALUES, ierr)
          call VecSetValue(b, kk, bc_R%gamma, INSERT_VALUES, ierr)
        end do
      end do

    end select

    !----------------------------------------
    ! BOTTOM and TOP boundaries (y = 1, ny)
    !----------------------------------------
    select type(bc_B)
    type is (periodicBC)

      ! Periodic bottom-top for all columns and all groups
      do ii = 1, nx
        do gg = 1, G
          kkB = map(ii, 1,  gg, G, nx)
          kkT = map(ii, ny, gg, G, nx)

          if (D(ii,1) + D(ii,ny) == 0.0_dp) then
            DN = 0.0_dp
          else
            DN = 2.0_dp * D(ii,1) * D(ii,ny) / (D(ii,1) + D(ii,ny))
          end if

          a = - DN * dx / dy
          call MatSetValue(A, kkB, kkT, a,  ADD_VALUES, ierr)
          call MatSetValue(A, kkB, kkB, -a, ADD_VALUES, ierr)
          call MatSetValue(A, kkT, kkB, a,  ADD_VALUES, ierr)
          call MatSetValue(A, kkT, kkT, -a, ADD_VALUES, ierr)
        end do
      end do

    class default

      ! Bottom boundary: i = 2..nx-1
      do ii = 2, nx-1
        do gg = 1, G
          kk  = map(ii, 1, gg, G, nx)
          kkN = map(ii, 2, gg, G, nx)

          call MatSetValue(A, kk, kk,  bc_B%alpha - bc_B%beta/dy, ADD_VALUES, ierr)
          call MatSetValue(A, kk, kkN, bc_B%beta/dy,             ADD_VALUES, ierr)
          call VecSetValue(b, kk, bc_B%gamma, INSERT_VALUES, ierr)
        end do
      end do

      ! Top boundary: i = 2..nx-1
      do ii = 2, nx-1
        do gg = 1, G
          kk  = map(ii, ny,   gg, G, nx)
          kkS = map(ii, ny-1, gg, G, nx)

          call MatSetValue(A, kk, kk,  bc_T%alpha + bc_T%beta/dy, ADD_VALUES, ierr)
          call MatSetValue(A, kk, kkS, -bc_T%beta/dy,             ADD_VALUES, ierr)
          call VecSetValue(b, kk, bc_T%gamma, INSERT_VALUES, ierr)
        end do
      end do

      ! Corners: simple diagonal regularization (per group)
      do gg = 1, G
        kk = map(1,   1,   gg, G, nx); call MatSetValue(A, kk, kk, 0.1_dp, INSERT_VALUES, ierr)
        kk = map(1,   ny,  gg, G, nx); call MatSetValue(A, kk, kk, 0.1_dp, INSERT_VALUES, ierr)
        kk = map(nx,  1,   gg, G, nx); call MatSetValue(A, kk, kk, 0.1_dp, INSERT_VALUES, ierr)
        kk = map(nx,  ny,  gg, G, nx); call MatSetValue(A, kk, kk, 0.1_dp, INSERT_VALUES, ierr)
      end do

    end select

    ! NOTE: we do not call MatAssemblyBegin/End or VecAssemblyBegin/End here;
    ! those are done once in the calling code after all interior + BC entries
    ! have been set.

  end subroutine apply_bc_xy_multigroup

end module mod_bc