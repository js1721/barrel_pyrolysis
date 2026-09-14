cmake_minimum_required(VERSION 3.6) 
include(FindPkgConfig REQUIRED)

# set(CMAKE_Fortran_COMPILER ifx)

# Set our project name and Fortran as only language we compile 
# (C and C++ are also available) and should be added if we interop 
project(neutron_code VERSION 0.1 
    DESCRIPTION "Code for solving neutron diffusion"
    LANGUAGES Fortran C
    )

# we default to Release build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# if(CMAKE_Fortran_COMPILER_ID MATCHES Intel)
#     set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -Wall")
#     set(CMAKE_Fortran_FLAGS_DEBUG   "-g -traceback")
#     set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -ip -xHOST")
# endif()

# petsc SHIT

if(APPLE)
    set(PETSC $ENV{PETSC_ARCH})
    set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)
    set(PETSC_INCLUDES ${PETSC}/include)
    set(PETSC_LIBRARIES ${PETSC}/lib/libpetsc.so)
else()
    #set(PETSC_DIR "/home/joe/petsc")
    # set(PETSC_DIR "$ENV{PETSC_DIR}")
    # set(PETSC_INCLUDES "${PETSC_DIR}/include")
    # set(PETSC_LIBRARIES "${PETSC_DIR}/lib/libpetsc.so")
    # set(PETSC_LIBRARIES "${PETSC_DIR}/lib/libpetsc.so")
    # set(ENV{PKG_CONFIG_PATH} ${PETSC_DIR}/lib)
    set(PETSC $ENV{PETSC_ARCH})
    set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)
    set(PETSC_INCLUDES ${PETSC}/include)
    set(PETSC_LIBRARIES ${PETSC}/lib/libpetsc.so)
endif()


if(APPLE)
    if(CMAKE_Fortran_COMPILER_ID MATCHES GNU)
        set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -ffree-line-length-none ")
        #set(CMAKE_Fortran_FLAGS_DEBUG   "-O0 -fcheck=all -fdec -DDEBUG -fopenmp -finit-real=snan -finit-integer=-9999 -ffpe-trap=invalid,zero,overflow -pedantic -I${PETSC_DIR}/include")
        set(CMAKE_Fortran_FLAGS_DEBUG   "-O0 -fcheck=all -fdec -DDEBUG -fopenmp -finit-real=snan -finit-integer=-9999 -ffpe-trap=invalid,zero,overflow -pedantic ")
        # set(CMAKE_Fortran_FLAGS_RELEASE "-Ofast -DNDEBUG -fopenmp ")
        #set(CMAKE_Fortran_FLAGS_RELEASE "-O2 -march=native -DNDEBUG -fopenmp -I${PETSC_DIR}/include")
        set(CMAKE_Fortran_FLAGS_RELEASE "-O2 -march=native -DNDEBUG -fopenmp ")
    endif()
else()
    if(CMAKE_Fortran_COMPILER_ID MATCHES GNU)
        set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -Wall -ffree-line-length-none ")
        #set(CMAKE_Fortran_FLAGS_DEBUG   "-O0 -fcheck=all -fdec -DDEBUG -pg -fopenmp -finit-real=snan -finit-integer=-9999 -ffpe-trap=invalid,zero,overflow -I${PETSC_DIR}/include")
        set(CMAKE_Fortran_FLAGS_DEBUG   "-O0 -fcheck=all -fdec -DDEBUG -pg -fopenmp -finit-real=snan -finit-integer=-9999 -ffpe-trap=invalid,zero,overflow ")
        # set(CMAKE_Fortran_FLAGS_RELEASE "-Ofast -DNDEBUG -fopenmp ")
        #set(CMAKE_Fortran_FLAGS_RELEASE "-O2 -march=native -DNDEBUG -fopenmp -pg -I${PETSC_DIR}/include")
        set(CMAKE_Fortran_FLAGS_RELEASE "-O2 -march=native -DNDEBUG -fopenmp -pg ")
        set(CMAKE_Fortran_FLAGS_DEBUG "-O0 -g -fcheck=all -fbacktrace -finit-real=snan -finit-integer=-9999")
    endif()
endif()


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)

# set(CMAKE_OSX_ARCHITECTURES x86_64)

# set(CMAKE_VERBOSE_MAKEFILE ON)

# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgcrt1.o")

# EXTERNAL PACKAGES
# Fortran Standard Library:

# find_package(LAPACK REQUIRED)
# find_package(BLAS REQUIRED)
# find_package(PkgConfig REQUIRED)
# set(CMAKE_PREFIX_PATH "$ENV{PETSC_DIR}" ${CMAKE_PREFIX_PATH})
# find_package(PETSc REQUIRED)


#PETSC SHIT
message(STATUS "Found PETSc: $ENV{PETSC_DIR}")
message(STATUS "PETSc Includes: ${PETSC_INCLUDES}")
message(STATUS "PETSc Libraries: ${PETSC_LIBRARIES}")

#find_package(fortran_stdlib REQUIRED)

# To use this please have stdlib built with CMake already 

#NB this should be the actual build e.g. the ARCH folder I think
# (I am using a ./configure --prefix <....> build)
#set(MY_PETSC_PATH "/usr/lib/cmake/PETSc") 
#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${MY_PETSC_PATH}) 

# the `pkg_check_modules` function is created with this call
#find_package(PkgConfig REQUIRED)
#pkg_check_modules(PETSC REQUIRED PETSc)

# Add include directories
#INCLUDE_DIRECTORIES(${PETSC_INCLUDE_DIRS})

# Add our source files
# All .F90 files in lib/ will be added to the library as sources
# Below command finds all .F90 and adds them to the ${SOURCES} variable

# file(GLOB SOURCES "stdlib/*.F90" "stdlib/*.F90" main.F90)
# add_executable(output ${SOURCES})

add_library(src
        src/mod_constants.F90
        src/mod_types.F90
        src/mod_materials.F90
        src/mod_materials_data.F90
        src/mod_geometry.F90
        src/mod_params.F90
        src/mod_map.F90
        src/mod_source.F90
        src/mod_bc.F90
        src/mod_sprs.F90
        src/mod_output.F90
        src/mod_diffusion_matrix.F90
        src/mod_eigenvalue.F90
        src/mod_solver_pcg.F90
        src/mod_solver.F90
        src/mod_postprocess.F90
        src/mod_mg.F90
)

# add_subdirectory(stdlib)
add_executable(out src/test.F90)
find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)
target_include_directories(out PRIVATE ${PETSC_INCLUDES})
target_link_libraries(out src PkgConfig::PETSC)


#NON PETSC 
# target_link_libraries(Bantam.exe src)
# target_link_libraries(Bantam.exe LAPACK::LAPACK BLAS::BLAS)

#PETSC


#TARGET_LINK_LIBRARIES(Output ${PETSC_LINK_LIBRARIES} fortran_stdlib::fortran_stdlib)