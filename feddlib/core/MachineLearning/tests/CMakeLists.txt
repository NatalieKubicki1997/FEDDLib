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


### MultiVector
TRIBITS_ADD_EXECUTABLE(
Evaluate_DNN_linear_activation
SOURCES Evaluate_DNN_linear_activation.cpp
)


TRIBITS_ADD_TEST(
Evaluate_DNN_linear_activation
NAME Evaluate_DNN_linear_activation
ARGS
NUM_MPI_PROCS 1
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_test  
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Test_Data_Linear_Activation_Function
	SOURCE_FILES biases_layer_1.mtx biases_layer_2.mtx biases_layer_3.mtx biases_layer_4.mtx weight_matrix_layer_1.mtx weight_matrix_layer_2.mtx weight_matrix_layer_3.mtx weight_matrix_layer_4.mtx 
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/Test_Data_Linear_Activation_Function
	DEST_FILES biases_layer_1.mtx biases_layer_2.mtx biases_layer_3.mtx biases_layer_4.mtx weight_matrix_layer_1.mtx weight_matrix_layer_2.mtx weight_matrix_layer_3.mtx weight_matrix_layer_4.mtx 
	EXEDEPS data_test 
)

### MultiVector
TRIBITS_ADD_EXECUTABLE(
Evaluate_DNN_nonlinear_activation
SOURCES Evaluate_DNN_nonlinear_activation.cpp
)


TRIBITS_ADD_TEST(
Evaluate_DNN_nonlinear_activation
NAME Evaluate_DNN_nonlinear_activation
ARGS
NUM_MPI_PROCS 1
)


TRIBITS_COPY_FILES_TO_BINARY_DIR(data_test2  
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Test_Data_Nonlinear_Activation_Function
	SOURCE_FILES biases_layer_1.mtx biases_layer_2.mtx biases_layer_3.mtx biases_layer_4.mtx weight_matrix_layer_1.mtx weight_matrix_layer_2.mtx weight_matrix_layer_3.mtx weight_matrix_layer_4.mtx 
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/Test_Data_Nonlinear_Activation_Function
	DEST_FILES biases_layer_1.mtx biases_layer_2.mtx biases_layer_3.mtx biases_layer_4.mtx weight_matrix_layer_1.mtx weight_matrix_layer_2.mtx weight_matrix_layer_3.mtx weight_matrix_layer_4.mtx 
	EXEDEPS data_test 
)

### MultiVector
TRIBITS_ADD_EXECUTABLE(
ReadAndWriteMultiVectorsFromFile
SOURCES ReadAndWriteMultiVectorsFromFile.cpp
)


TRIBITS_ADD_TEST(
ReadAndWriteMultiVectorsFromFile
NAME ReadAndWriteMultiVectorsFromFile
ARGS
NUM_MPI_PROCS 1
)
