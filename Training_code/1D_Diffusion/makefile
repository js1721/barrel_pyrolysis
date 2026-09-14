#===================================================================================================
#
# makefile
#
# Date:     Programmer:     Changes:
# =====     ===========     ========
# 19-10-23  B Wheeler       Original
#===================================================================================================

.PHONY: clean
.PHONY: debug
# .PHONY: prof
# .PHONY: parallel

# set compilers and linker
FC = gfortran
CC = gcc
LD = gfortran

# SUNDIALs directory
#SUN_DIR = /home/barnaby/3TB/Programs/cvode_parallel/INSTALL/

# PETSc directory
#PETSc_DIR = /home/barnaby/3TB/Programs/petsc/INSTALL/double_precision_basic/
# PETSc_DIR = /home/barnaby/Documents/PhD/Imperial/Programs/PETSc/Builds/opt_hyp/

# SUNDIALs lib flags
#S_INCLUDE = -I$(SUN_DIR)include
#S_MOD = -I$(SUN_DIR)fortran
#S_LINK = -L$(SUN_DIR)lib
#S_LOAD = -Wl,-rpath=$(SUN_DIR)lib
#S_LIB = -lsundials_cvode -lsundials_fcvode_mod -lsundials_core -lsundials_fcore_mod -lsundials_nvecopenmp -lsundials_fnvecopenmp_mod

# PETSc libs
#P_INCLUDE = -I$(PETSc_DIR)include
#P_LINK = -L$(PETSc_DIR)lib
#P_LOAD = -Wl,-rpath=$(PETSc_DIR)lib
#P_LIB = -lpetsc

# set executable name
EXEC = out

all: $(EXEC)
debug: $(EXEC)
# prof: $(EXEC)
# parallel: $(EXEC)

# set path variables
SRC = src
MOD = mod_files
INCLUDE = $(S_INCLUDE) $(S_MOD) $(P_INCLUDE)
LINK = $(S_LINK) $(P_LINK)
LOAD = $(S_LOAD) $(P_LOAD)
LIB = $(S_LIB) $(P_LIB)

# set compiler options
FC_flags_debug = -ffree-line-length-none -Wall --pedantic -fcheck=all -O0 -g -cpp -J$(MOD) -c -o
# FC_flags_prof = -ffree-line-length-none -fopenmp -O0 -DPROFILING -DNDEBUG -g -pg -cpp -J$(MOD) -c -o
# FC_flags_parallel = -ffree-line-length-none -fopenmp -O3 -DNDEBUG -cpp -J$(MOD) -c -o
FC_flags = -ffree-line-length-none -O3 -DNDEBUG -cpp -J$(MOD) -c -o

# set linker options
LD_flags_debug = -O0 -g -o
# LD_flags_prof = -O0 -g -fopenmp -pg -o
# LD_flags_parallel = -O3 -g -fopenmp -o
LD_flags = -O3 -o

# if debugging, alter flags
debug: FC_flags := $(FC_flags_debug)
debug: LD_flags := $(LD_flags_debug)
# prof: FC_flags := $(FC_flags_prof)
# prof: LD_flags := $(LD_flags_prof)
#prof: SUN_DIR := /home/barnaby/3TB/Programs/cvode_parallel/PROFILE/
# parallel: FC_flags := $(FC_flags_parallel)
# parallel: LD_flags := $(LD_flags_parallel)
#parallel: PETSc_DIR := /home/barnaby/3TB/Programs/petsc/INSTALL/double_precision_OMP/

# set makefile variable to update all files if it is changed
MAKE = makefile

# set the object files
OBJS =  $(SRC)/precision.o \
		$(SRC)/mesh_module.o \
		$(SRC)/tridiag_module.o \
		$(SRC)/power_iteration_module.o \
		$(SRC)/bc_module.o \
		$(SRC)/TA_module.o \
		$(SRC)/solver_module.o \
		$(SRC)/multigroup_module.o \
	   	$(SRC)/write_to_csv_module.o \
		$(SRC)/sparse_matrix_module.o \
		$(SRC)/conjugate_gradient_module.o \
		$(SRC)/main.o
		

files = $(OBJS)

# variable for the .mod files generated
MODS= $(files:.o=.mod)

# if makefile changes, update the objects
%.o : $(MAKE)

# if .F03 files change, update the object
%.o : %.f03
	$(FC) $(FC_flags) $@ $< $(INCLUDE) $(LINK)
#	$(FC) $(FC_flags) $@ $< $(INCLUDE) $(LINK) $(LIB)

# if .F95 files change, update the object
%.o : %.f95
	$(FC) $(FC_flags) $@ $< $(INCLUDE) $(LINK)
#	$(FC) $(FC_flags) $@ $< $(INCLUDE) $(LINK) $(LIB)

# if .F90 files change, update the object
%.o : %.f90
	$(FC) $(FC_flags) $@ $< $(INCLUDE) $(LINK)
#	$(FC) $(FC_flags) $@ $< $(INCLUDE) $(LINK) $(LIB)

# if .F files change, update the object
%.o : %.f
	$(FC) $(FC_flags) $@ $< $(INCLUDE) $(LINK)
#	$(FC) $(FC_flags) $@ $< $(INCLUDE) $(LINK) $(LIB)

# if any object file changes, update the executable
$(EXEC) : $(files)
	$(LD) $(LD_flags) $@ $? $(LOAD) $(LINK) $(LIB)

# to clean the directory remove all compiled files
clean:
	rm -fv $(EXEC) $(files) $(MODS) *.o *.mod *.out $(MOD)/*