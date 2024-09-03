TRIBITS_ADD_EXECUTABLE(
	addingSolutionField2
	SOURCES main.cpp
)

TRIBITS_ADD_TEST(
	addingSolutionField2 
	NAME addingSolutionField2 
	ARGS
	NUM_MPI_PROCS 4
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_addingSolutionField2  
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES parametersPrec.xml parametersProblem.xml parametersSolver.xml 
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES parametersPrec.xml parametersProblem.xml parametersSolver.xml
	EXEDEPS addingSolutionField2 
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(mesh_addingSolutionField2  
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES rectangle_400.mesh predictions_density.h5
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES rectangle_400.mesh predictions_density.h5
	EXEDEPS addingSolutionField2 
)

TRIBITS_ADD_EXECUTABLE(
	addingSolutionField2_saveAssembleFEElement
	SOURCES main_addToAssembleFEElement.cpp
)

TRIBITS_ADD_TEST(
	addingSolutionField2_saveAssembleFEElement 
	NAME addingSolutionField2_saveAssembleFEElement 
	ARGS
	NUM_MPI_PROCS 4
)


