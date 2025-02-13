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
PyTrilinos2 pytrilinos2
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

#cmake_minimum_required(VERSION 3.4)
#find_package(pybind11 REQUIRED)
#IF (HAVE_pybind11)
#pybind11_add_module(parameterList_Python_example parameterList_Python_example.cpp)
#########target_link_libraries(example PUBLIC ${Trilinos_LIBRARY_DIRS} Trilinos_PACKAGES_AND_DIRS)
#ENDIF (HAVE_pybind11)

### python parameterList
#TRIBITS_ADD_EXECUTABLE(
#parameterList_Python_example
#SOURCES parameterList_Python_example.cpp
#)

#TRIBITS_ADD_TEST(
#parameterList_Python_example
#NAME parameterList_Python_example
#ARGS
#COMM serial
#)

## Python script test
add_subdirectory(PyTrilinos_PythonTests)
add_subdirectory(PyTrilinos_C++Tests)






#cmake_minimum_required(VERSION 3.4)
#project(example2)

# Set the C++ standard to C++11 or higher
# Find the Pybind11 package
#find_package(pybind11 REQUIRED)
#pybind11_add_module(example2 example.cpp)
#target_link_libraries(example PUBLIC ${Trilinos_LIBRARY_DIRS} Trilinos_PACKAGES_AND_DIRS)

#target_link_libraries(example2 PRIVATE )

#link_directories(/home/user/Programme/opt/ntrilinos_newestVersion/lib)
#target_link_libraries(example PRIVATE teuchos)

#target_link_libraries(example PRIVATE /home/user/Programme/opt/ntrilinos_newestVersion/lib/libteuchos.so)

