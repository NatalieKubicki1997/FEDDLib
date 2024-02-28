TRIBITS_ADD_EXECUTABLE(
	steadyStokesFlow_Newtonian_NonNewtonian_Generation_Window_Data 
	SOURCES main.cpp
)

TRIBITS_ADD_TEST(
	steadyStokesFlow_Newtonian_NonNewtonian_Generation_Window_Data 
	NAME steadyStokesFlow_Newtonian_NonNewtonian_Generation_Window_Data 
	ARGS
#--precfile=parametersPrecMumps.xml
	NUM_MPI_PROCS 4
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_steadyStokesFlow_Newtonian_NonNewtonian_Generation_Window_Data  
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES parametersPrec.xml parametersProblem.xml parametersProblem_Newtonian.xml  parametersSolver.xml parametersTeko.xml
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES parametersPrec.xml parametersProblem.xml  parametersProblem_Newtonian.xml parametersSolver.xml parametersTeko.xml
	EXEDEPS steadyStokesFlow_Newtonian_NonNewtonian_Generation_Window_Data 
)

#TRIBITS_COPY_FILES_TO_BINARY_DIR(mesh_steadyCreepingMotionNonNewtonianAssFE 
#	SOURCE_DIR ${CMAKE_SOURCE_DIR}/meshes
	#SOURCE_FILES #dfg_fsi_fluid_h008.mesh DFG3DCylinder_1k.mesh
	#DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	#DEST_FILES #dfg_fsi_fluid_h008.mesh DFG3DCylinder_1k.mesh
	#EXEDEPS steadyCreepingMotionNonNewtonianAssFE 
#)




