TRIBITS_ADD_EXECUTABLE(
	generateTrainingData
	SOURCES main_generateTrainingData.cpp
)

TRIBITS_ADD_TEST(
	generateTrainingData 
	NAME generateTrainingData 
	ARGS
	NUM_MPI_PROCS 4
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_generateTrainingData  
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES parametersPrec.xml parametersProblem.xml parametersSolver.xml 
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES parametersPrec.xml parametersProblem.xml parametersSolver.xml
	EXEDEPS generateTrainingData 
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(mesh_generateTrainingData  
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES predictions_density.h5 dummyMaterialLaw.m rectangle_H20_L_200.0Nele_250.0ref2.0.mesh
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES  predictions_density.h5 dummyMaterialLaw.m rectangle_H20_L_200.0Nele_250.0ref2.0.mesh
	EXEDEPS generateTrainingData 
)

