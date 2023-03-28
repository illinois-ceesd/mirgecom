subroutine gas_get_index_table(bounds, T, Ng, idx)

    implicit none
    integer i, j
    integer, intent(in) :: Ng
    real(kind=8), intent(in) :: bounds(152)
    real(kind=8), intent(in) :: T(Ng)
    integer, intent(out) :: idx(Ng)

    idx(:) = 0
    do i = 1,Ng
        do j = 1,152-1
            if (bounds(j) <= T(i) .and. T(i) < bounds(j+1)) then
                idx(i) = j-1
                exit
            endif
        enddo
    enddo

    return

end subroutine


subroutine gas_enthalpy(idx, T, table, N, val)

    implicit none
    integer, intent(in) :: N
    integer, intent(in) :: idx(N)
    real(kind=8), intent(in) :: T(N)
    real(kind=8), intent(in) :: table(152,7)
    real(kind=8), intent(out) :: val(N)
    real(kind=8) :: dx(N), dy(N)
    integer :: aux(N)

    aux = idx + 1

    dx(:) = table(aux(:)+1, 1) - table(aux(:), 1)
    dy(:) = table(aux(:)+1, 5) - table(aux(:), 5)
    val(:) = (table(aux(:), 5) + dy(:)/dx(:)*(T(:) - table(aux(:), 1)))*1000.0d0

    return    

end subroutine

subroutine gas_heat_capacity(idx, T, table, N, val)

    implicit none
    integer, intent(in) :: N
    integer, intent(in) :: idx(N)
    real(kind=8), intent(in) :: T(N)
    real(kind=8), intent(in) :: table(152,7)
    real(kind=8), intent(out) :: val(N)
    real(kind=8) :: dx(N), dy(N)
    integer :: i
    integer :: aux(N)

    aux = idx + 1

    dx(:) = table(aux(:)+1, 1) - table(aux(:), 1)
    dy(:) = table(aux(:)+1, 3) - table(aux(:), 3)
    val(:) = (table(aux(:), 3) + dy(:)/dx(:)*(T(:) - table(aux(:), 1)))*1000.0d0

    return    

end subroutine

subroutine gas_molar_mass(idx, T, table, N, val)

    implicit none
    integer, intent(in) :: N
    integer, intent(in) :: idx(N)
    real(kind=8), intent(in) :: T(N)
    real(kind=8), intent(in) :: table(152,7)
    real(kind=8), intent(out) :: val(N)
    real(kind=8) :: dx(N), dy(N)
    integer :: aux(N)

    aux = idx + 1

    dx(:) = table(aux(:)+1, 1) - table(aux(:), 1)
    dy(:) = table(aux(:)+1, 2) - table(aux(:), 2)
    val(:) = (table(aux(:), 2) + dy(:)/dx(:)*(T(:) - table(aux(:), 1)))

    return    

end subroutine

subroutine gas_dMdT(idx, T, table, N, val)

    implicit none
    integer, intent(in) :: N
    integer, intent(in) :: idx(N)
    real(kind=8), intent(in) :: T(N)
    real(kind=8), intent(in) :: table(152,7)
    real(kind=8), intent(out) :: val(N)
    real(kind=8) :: dx(N), dy(N)
    integer :: aux(N)

    aux = idx + 1

    dx(:) = table(aux(:)+1, 1) - table(aux(:), 1)
    dy(:) = table(aux(:)+1, 2) - table(aux(:), 2)
    val(:) = dy/dx

    return    

end subroutine

subroutine gas_viscosity(idx, T, table, N, val)

    implicit none
    integer, intent(in) :: N
    integer, intent(in) :: idx(N)
    real(kind=8), intent(in) :: T(N)
    real(kind=8), intent(in) :: table(152,7)
    real(kind=8), intent(out) :: val(N)
    real(kind=8) :: dx(N), dy(N)
    integer :: aux(N)

    aux = idx + 1

    dx(:) = table(aux(:)+1, 1) - table(aux(:), 1)
    dy(:) = table(aux(:)+1, 6) - table(aux(:), 6)
    val(:) = (table(aux(:), 6) + dy(:)/dx(:)*(T(:) - table(aux(:), 1)))*1.0d-4

    return    

end subroutine
