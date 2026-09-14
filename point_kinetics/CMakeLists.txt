cmake_minimum_required(VERSION 3.6) 
include(FindPkgConfig REQUIRED)

# set(CMAKE_Fortran_COMPILER ifx)

# Set our project name and Fortran as only language we compile 
# (C and C++ are also available) and should be added if we interop 
project(neutron_code VERSION 0.1 
    DESCRIPTION "Code for solving Point Kinetics"
    LANGUAGES Fortran C
    )

# we default to Release build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()


# Compiler flags (GNU Fortran)

if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    # Common flags
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -ffree-line-length-none")

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_Fortran_FLAGS_DEBUG
            "-O0 -g -fcheck=all -fdec -DDEBUG -fopenmp \
             -finit-real=snan -finit-integer=-9999 \
             -ffpe-trap=invalid,zero,overflow -fbacktrace"
        )
    else()
        # Release (or others)
        set(CMAKE_Fortran_FLAGS_RELEASE
            "-O2 -march=native -DNDEBUG -fopenmp"
        )
    endif()
endif()

set(CMAKE_Fortran_PREPROCESS ON)

# Put runtime binaries in build/.. (one level up)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod)
include_directories(${CMAKE_Fortran_MODULE_DIRECTORY})

#External Packages 
find_package(PkgConfig REQUIRED)
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)


# PETSc config


set(PETSC $ENV{PETSC_ARCH})
set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)
set(PETSC_INCLUDES ${PETSC}/include)
set(PETSC_LIBRARIES ${PETSC}/lib/libpetsc.so)


pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)

message(STATUS "Found PETSc via pkg-config")
message(STATUS "  Include dirs: ${PETSC_INCLUDE_DIRS}")
message(STATUS "  Libraries:    ${PETSC_LINK_LIBRARIES}")
message(STATUS "  Version:      ${PETSC_VERSION}")


# SUNDIALS config 

# Specify the path to SUNDIALS Fortran modules
set(SUNDIALS_INCLUDE_DIR /Users/joe/Documents/sundials-install/fortran/ CACHE PATH "Location of SUNDIALS Fortran module files")
# Specify the path to SUNDIALS libraries
set(SUNDIALS_LIBRARY_DIR /Users/joe/Documents/sundials-install/lib/ CACHE PATH "Location of SUNDIALS libraries")


message(STATUS "Found SUNDIALS")
message(STATUS "  SUNDIALS_DIR: ${SUNDIALS_INCLUDE_DIR}")

find_library(SUNDIALS_CORE_LIB sundials_core ${SUNDIALS_LIBRARY_DIR} DOC "SUNDIALS core library")

find_library(SUNDIALS_FCORE_LIB sundials_fcore_mod ${SUNDIALS_LIBRARY_DIR} DOC "SUNDIALS Fortran core library")

find_library(SUNDIALS_SOLVER_LIB sundials_cvode ${SUNDIALS_LIBRARY_DIR} DOC "CVODE library")

find_library(SUNDIALS_SOLVER_FLIB sundials_fcvode_mod ${SUNDIALS_LIBRARY_DIR} DOC "CVODE Fortran-C library")

find_library(SUNDIALS_NVEC_LIB sundials_nvecserial ${SUNDIALS_LIBRARY_DIR} DOC "NVECTOR library")

find_library(SUNDIALS_NVEC_FLIB sundials_fnvecserial_mod ${SUNDIALS_LIBRARY_DIR} DOC "NVECTOR library")

find_library(SUNDIALS_SUNLS_LIB sundials_sunlinsolspgmr ${SUNDIALS_LIBRARY_DIR} DOC "SUNDIALS linear solver library")

find_library(SUNDIALS_SUNLS_FLIB sundials_fsunlinsolspgmr_mod ${SUNDIALS_LIBRARY_DIR} DOC "SUNDIALS Fortran-C linear solver library")

find_library(SUNDIALS_BANDM_LIB sundials_sunmatrixband ${SUNDIALS_LIBRARY_DIR} DOC "SUNDIALS band matrix library")

find_library(SUNDIALS_BANDM_FLIB sundials_fsunmatrixband_mod ${SUNDIALS_LIBRARY_DIR} DOC "SUNDIALS Fortran-C band matrix library")



# List of SUNDIALS libraries
set(SUNDIALS_LIBRARIES
-L${SUNDIALS_LIBRARY_DIR}
${SUNDIALS_SOLVER_FLIB}
${SUNDIALS_SOLVER_LIB}
${SUNDIALS_NVEC_FLIB}
${SUNDIALS_NVEC_LIB}
${SUNDIALS_FCORE_LIB}
${SUNDIALS_CORE_LIB}
${SUNDIALS_SUNLS_LIB}
${SUNDIALS_SUNLS_FLIB}
${SUNDIALS_BANDM_LIB}
${SUNDIALS_BANDM_FLIB}
${SUNDIALS_BANDM_HELPER_LIB})


#Targets

add_library(src

        #Point kinetics
        src/pk/mod_cvode_wrapper.F90
        src/pk/mod_pk.F90
        src/pk/mod_pk_solver.F90

        #Diffusion 
        src/diffusion/mod_mesh.F90
)

add_executable(out src/test.F90)

target_include_directories(src PRIVATE ${PETSC_INCLUDE_DIRS})




target_link_libraries(out
    PRIVATE
        src
        PkgConfig::PETSC

)
target_link_libraries(out
    PRIVATE
        src
        ${SUNDIALS_LIBRARIES}

)

message(STATUS "SUNDIALS_LIBRARIES: ${SUNDIALS_LIBRARIES}")
# Include Fortran modules directory for both src library and executable
target_include_directories(src PRIVATE ${SUNDIALS_INCLUDE_DIR})
target_include_directories(out PRIVATE ${SUNDIALS_INCLUDE_DIR})


