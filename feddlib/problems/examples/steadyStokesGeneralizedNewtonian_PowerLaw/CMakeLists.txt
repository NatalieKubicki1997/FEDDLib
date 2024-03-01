TRIBITS_ADD_EXECUTABLE(
	steadyStokesGeneralizedNonNewtonianAssFE_PowerLaw 
	SOURCES main.cpp
)

TRIBITS_ADD_TEST(
	steadyStokesGeneralizedNonNewtonianAssFE_PowerLaw  
	NAME steadyStokesGeneralizedNonNewtonianAssFE_PowerLaw  
	ARGS
#--precfile=parametersPrecMumps.xml
	NUM_MPI_PROCS 4
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_steadyStokesGeneralizedNonNewtonianAssFE_PowerLaw   
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES parametersPrec.xml parametersProblem.xml parametersProblem_Newtonian.xml parametersSolver.xml parametersTeko.xml
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES parametersPrec.xml parametersProblem.xml parametersProblem_Newtonian.xml parametersSolver.xml parametersTeko.xml
	EXEDEPS steadyStokesGeneralizedNonNewtonianAssFE_PowerLaw  
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(mesh_steadyStokesGeneralizedNonNewtonianAssFE_PowerLaw   
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES rectangle_200.mesh 
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES rectangle_200.mesh 
	EXEDEPS steadyStokesGeneralizedNonNewtonianAssFE_PowerLaw  
)




