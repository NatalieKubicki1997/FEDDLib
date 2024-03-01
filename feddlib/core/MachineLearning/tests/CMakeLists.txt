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

### MultiVector
TRIBITS_ADD_EXECUTABLE(
MatrixVectorProduct
SOURCES MatrixVectorProduct.cpp
)


TRIBITS_ADD_TEST(
MatrixVectorProduct
NAME MatrixVectorProduct
ARGS
NUM_MPI_PROCS 1
)

### MultiVector
TRIBITS_ADD_EXECUTABLE(
ReadMatrixFromFile
SOURCES ReadMatrixFromFile.cpp
)


TRIBITS_ADD_TEST(
    ReadMatrixFromFile
NAME ReadMatrixFromFile
ARGS
NUM_MPI_PROCS 1
)
