TRIBITS_ADD_EXECUTABLE(
	steadyGeneralizedNewtonianAssFE_BFS_Scalability 
	SOURCES main.cpp
)

TRIBITS_ADD_TEST(
	steadyGeneralizedNewtonianAssFE_BFS_Scalability  
	NAME steadyGeneralizedNewtonianAssFE_BFS_Scalability  
	ARGS
#--precfile=parametersPrecMumps.xml
	NUM_MPI_PROCS 4
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_steadyGeneralizedNewtonianAssFE_BFS_Scalability 
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES parametersPrec.xml parametersProblem.xml parametersProblem_structured_P2_3D.xml parametersSolver.xml parametersTeko.xml sbatch4608MonoIPOU__Paper_MOS_2025_table_6_5.sh
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR} 
	DEST_FILES parametersPrec.xml parametersProblem.xml parametersProblem_structured_P2_3D.xml parametersSolver.xml parametersTeko.xml sbatch4608MonoIPOU__Paper_MOS_2025_table_6_5.sh
	EXEDEPS steadyGeneralizedNewtonianAssFE_BFS_Scalability  
)

TRIBITS_ADD_EXECUTABLE(
	steadyGeneralizedNewtonianAssFE_BFS_Scalability_GlobalAssembly 
	SOURCES main_GlobalAssembly.cpp
)

TRIBITS_ADD_TEST(
	steadyGeneralizedNewtonianAssFE_BFS_Scalability_GlobalAssembly  
	NAME steadyGeneralizedNewtonianAssFE_BFS_Scalability_GlobalAssembly  
	ARGS
	NUM_MPI_PROCS 4
)