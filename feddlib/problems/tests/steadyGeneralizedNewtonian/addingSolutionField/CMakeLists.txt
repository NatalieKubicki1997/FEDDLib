TRIBITS_ADD_EXECUTABLE(
	addingSolutionField
	SOURCES main.cpp
)

TRIBITS_ADD_TEST(
	addingSolutionField 
	NAME addingSolutionField 
	ARGS
	NUM_MPI_PROCS 4
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_addingSolutionField  
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES parametersPrec.xml parametersProblem.xml parametersSolver.xml 
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES parametersPrec.xml parametersProblem.xml parametersSolver.xml
	EXEDEPS addingSolutionField 
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(mesh_addingSolutionField  
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES rectangle_400.mesh predictions_density.h5
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES rectangle_400.mesh predictions_density.h5
	EXEDEPS addingSolutionField 
)

TRIBITS_ADD_EXECUTABLE(
	addingSolutionField_saveAssembleFEElement
	SOURCES main_addToAssembleFEElement.cpp
)

TRIBITS_ADD_TEST(
	addingSolutionField_saveAssembleFEElement 
	NAME addingSolutionField_saveAssembleFEElement 
	ARGS
	NUM_MPI_PROCS 4
)


