TRIBITS_ADD_EXECUTABLE(
	steadyGeneralizedNewtonian_BFS_Scalability 
	SOURCES main.cpp
)

TRIBITS_ADD_TEST(
	steadyGeneralizedNewtonian_BFS_Scalability  
	NAME steadyGeneralizedNewtonian_BFS_Scalability  
	ARGS
#--precfile=parametersPrecMumps.xml
	NUM_MPI_PROCS 4
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_steadyGeneralizedNewtonian_BFS_Scalability 
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES parametersPrec.xml parametersProblem.xml parametersProblem_structured_P2_3D.xml parametersSolver.xml parametersTeko.xml
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES parametersPrec.xml parametersProblem.xml parametersProblem_structured_P2_3D.xml parametersSolver.xml parametersTeko.xml
	EXEDEPS steadyGeneralizedNewtonian_BFS_Scalability  
)

#TRIBITS_COPY_FILES_TO_BINARY_DIR(mesh_steadyGeneralizedNewtonian_BFS_Scalability
#	SOURCE_DIR ${CMAKE_SOURCE_DIR}/meshes
	#SOURCE_FILES #dfg_fsi_fluid_h008.mesh DFG3DCylinder_1k.mesh
	#DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	#DEST_FILES #dfg_fsi_fluid_h008.mesh DFG3DCylinder_1k.mesh
	#EXEDEPS steadyGeneralizedNewtonian_BFS_Scalability
#)


TRIBITS_ADD_EXECUTABLE(
	steadyGeneralizedNewtonian_BFS_Scalability_BC 
	SOURCES main_BC.cpp
)

TRIBITS_ADD_TEST(
	steadyGeneralizedNewtonian_BFS_Scalability_BC  
	NAME steadyGeneralizedNewtonian_BFS_Scalability_BC
	ARGS
#--precfile=parametersPrecMumps.xml
	NUM_MPI_PROCS 4
)

