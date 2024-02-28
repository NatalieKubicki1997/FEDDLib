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