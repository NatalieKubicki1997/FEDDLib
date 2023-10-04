INCLUDE(TribitsCopyFilesToBinaryDir)
INCLUDE(TribitsAddExecutableAndTest)
INCLUDE(AddSubdirectories)
include_directories(${Trilinos_INCLUDE_DIRS})
include_directories(${Trilinos_LIBRARY_DIRS})

SET(Trilinos_PACKAGES_AND_DIRS
Teuchos teuchos
RTOp rtop
Epetra epetra
Triutils triutils
EpetraExt epetraext
Thyra thyra
)

#include_directories(${Trilinos_INCLUDE_DIRS})
### neumannBC
TRIBITS_ADD_EXECUTABLE(
neumannBC
SOURCES neumannBC.cpp
)

TRIBITS_ADD_TEST(
neumannBC
NAME neumannBC
ARGS
NUM_MPI_PROCS 4
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_bc_2dSquare
	SOURCE_DIR ${CMAKE_SOURCE_DIR}/meshes
	SOURCE_FILES square.mesh
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES square.mesh
	EXEDEPS neumannBC
)


### pybind11_test
#TRIBITS_ADD_EXECUTABLE(
#pybind11_test
#SOURCES pybind11_test.cpp
#)

#TRIBITS_ADD_TEST(
#pybind11_test
#NAME pybind11_test
#ARGS
#COMM serial
#)


#cmake_minimum_required(VERSION 3.4)

# Set the C++ standard to C++11 or higher

# Find the Pybind11 package
#find_package(pybind11 REQUIRED)
#find_package(Trilinos REQUIRED)
##TRILINOS_DIR=/home/user/Programme/opt/ntrilinos_newestVersion/

# Add your source files to the project
#IF (HAVE_pybind11)

#pybind11_add_module(example example.cpp)
#target_link_libraries(example PUBLIC ${Trilinos_LIBRARY_DIRS} Trilinos_PACKAGES_AND_DIRS)

#ENDIF (HAVE_pybind11)

#target_link_libraries(example PRIVATE )

#link_directories(/home/user/Programme/opt/ntrilinos_newestVersion/lib)
#target_link_libraries(example PRIVATE teuchos)

#target_link_libraries(example PRIVATE /home/user/Programme/opt/ntrilinos_newestVersion/lib/libteuchos.so)


### sacado
###TRIBITS_ADD_EXECUTABLE(
###sacado_example
###SOURCES sacado_example.cpp
###)

###TRIBITS_ADD_TEST(
###sacado_example
###NAME sacado_example
###ARGS
###NUM_MPI_PROCS 4
###)

###TRIBITS_COPY_FILES_TO_BINARY_DIR(data_bc_2dSquare2
###	SOURCE_DIR ${CMAKE_SOURCE_DIR}/meshes
###	SOURCE_FILES square.mesh
###	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
###	DEST_FILES square.mesh
###	EXEDEPS sacado_example
###)
