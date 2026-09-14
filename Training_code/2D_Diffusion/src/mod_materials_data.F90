!===================================================================================================
!
! Material data module
!
! Date:     Programmer:     Changes:
! =====     ===========     ========
! 19-11-25  J Salter        Original
!===================================================================================================


module mod_materials_data
    use mod_constants
    use mod_types
    implicit none

    type(material) :: material_UO2
    type(material) :: material_MOX
    type(material) :: material_H2O
    type(material) :: material_graphite
    type(material) :: material_steel
    type(material) :: material_vacuum
    type(material) :: material_test

contains

    subroutine init_materials()
        !----------------------------------
        ! UO2 Fuel
        material_UO2%D         = 1.3_dp
        material_UO2%sigma_a   = 0.0083_dp
        material_UO2%nuSigma_f = 0.0048_dp
        material_UO2%chi       = 0.98_dp
        !material_UO2%sigma_s   = [0.25_dp, 0.03_dp]

        !----------------------------------
        ! MOX Fuel
        material_MOX%D         = 1.1_dp
        material_MOX%sigma_a   = 0.0095_dp
        material_MOX%nuSigma_f = 0.0062_dp
        material_MOX%chi       = 0.97_dp
        !material_MOX%sigma_s   = [0.22_dp, 0.025_dp]

        !----------------------------------
        ! Water
        material_H2O%D         = 1.5_dp
        material_H2O%sigma_a   = 0.0002_dp
        material_H2O%nuSigma_f = 0.0_dp
        material_H2O%chi       = 0.0_dp
        !material_H2O%sigma_s   = [0.35_dp, 0.32_dp]

        !----------------------------------
        ! Graphite
        material_graphite%D         = 1.8_dp
        material_graphite%sigma_a   = 0.00004_dp
        material_graphite%nuSigma_f = 0.0_dp
        material_graphite%chi       = 0.0_dp
        !material_graphite%sigma_s   = [0.43_dp, 0.40_dp]

        !----------------------------------
        ! Stainless Steel
        material_steel%D         = 0.9_dp
        material_steel%sigma_a   = 0.02_dp
        material_steel%nuSigma_f = 0.0_dp
        material_steel%chi       = 0.0_dp
        !material_steel%sigma_s   = [0.08_dp, 0.01_dp]

        !----------------------------------
        ! Vacuum
        material_vacuum%D         = 1.0_dp
        material_vacuum%sigma_a   = 1.0e6_dp
        material_vacuum%nuSigma_f = 0.0_dp
        material_vacuum%chi       = 0.0_dp
        !material_vacuum%sigma_s   = [0.0_dp, 0.0_dp]


        !---------------------------------
        !Test
        material_test%D = 1.0_dp
        material_test%sigma_a = 0.01_dp 
        material_test%nuSigma_f = 0.02_dp 


    end subroutine init_materials

end module mod_materials_data
