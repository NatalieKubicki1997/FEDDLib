TRIBITS_ADD_EXECUTABLE(
	steadyGeneralizedNonNewtonianAssFE_Scalability 
	SOURCES main.cpp
)

TRIBITS_ADD_TEST(
	steadyGeneralizedNonNewtonianAssFE_Scalability  
	NAME steadyGeneralizedNonNewtonianAssFE_Scalability  
	ARGS
#--precfile=parametersPrecMumps.xml
	NUM_MPI_PROCS 4
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_steadyGeneralizedNonNewtonianAssFE_Scalability 
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES parametersPrec.xml parametersProblem.xml parametersProblem_structured_P2_3D.xml parametersProblem_Model2.xml parametersSolver.xml parametersTeko.xml
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES parametersPrec.xml parametersProblem.xml parametersProblem_structured_P2_3D.xml parametersProblem_Model2.xml parametersSolver.xml parametersTeko.xml
	EXEDEPS steadyGeneralizedNonNewtonianAssFE_Scalability  
)

#TRIBITS_COPY_FILES_TO_BINARY_DIR(mesh_steadyCreepingMotionNonNewtonianAssFE 
#	SOURCE_DIR ${CMAKE_SOURCE_DIR}/meshes
	#SOURCE_FILES #dfg_fsi_fluid_h008.mesh DFG3DCylinder_1k.mesh
	#DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	#DEST_FILES #dfg_fsi_fluid_h008.mesh DFG3DCylinder_1k.mesh
	#EXEDEPS steadyCreepingMotionNonNewtonianAssFE 
#)




