INCLUDE(TribitsCopyFilesToBinaryDir)
INCLUDE(TribitsAddExecutableAndTest)
INCLUDE(AddSubdirectories)

TRIBITS_ADD_EXECUTABLE(
	steadyGeneralizedNewtonian_PatchBasedCNN_TrainingData 
	SOURCES main.cpp
)

TRIBITS_ADD_TEST(
	steadyGeneralizedNewtonian_PatchBasedCNN_TrainingData  
	NAME steadyGeneralizedNewtonian_PatchBasedCNN_TrainingData  
	ARGS
	NUM_MPI_PROCS 4
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_steadyGeneralizedNewtonian_PatchBasedCNN_TrainingData   
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES parametersPrec.xml parametersProblem.xml parametersSolver.xml 
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES parametersPrec.xml parametersProblem.xml parametersSolver.xml 
	EXEDEPS steadyGeneralizedNewtonian_PatchBasedCNN_TrainingData  
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(mesh_steadyGeneralizedNewtonian_PatchBasedCNN_TrainingData   
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES rectangle_200.mesh run_simulation.sh run_simulation_loop.sh check_simulation_files.sh checkNewtonIteriations.sh 2D_artery_number_2.mesh
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES   rectangle_200.mesh run_simulation.sh run_simulation_loop.sh check_simulation_files.sh checkNewtonIteriations.sh 2D_artery_number_2.mesh
	EXEDEPS steadyGeneralizedNewtonian_PatchBasedCNN_TrainingData  
)



