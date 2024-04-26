TRIBITS_ADD_EXECUTABLE(
	steadyGeneralizedNewtonian 
	SOURCES main.cpp
)

TRIBITS_ADD_TEST(
	steadyGeneralizedNewtonian  
	NAME steadyGeneralizedNewtonian  
	ARGS
#--precfile=parametersPrecMumps.xml
	NUM_MPI_PROCS 4
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_steadyGeneralizedNewtonian   
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES parametersPrec.xml parametersProblem.xml parametersSolver.xml parametersTeko.xml
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES parametersPrec.xml parametersProblem.xml parametersSolver.xml parametersTeko.xml
	EXEDEPS steadyGeneralizedNewtonian  
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(mesh_steadyGeneralizedNewtonian   
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES rectangle_200.mesh Simple_3D_Mesh.mesh
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES rectangle_200.mesh Simple_3D_Mesh.mesh
	EXEDEPS steadyGeneralizedNewtonian  
)





TRIBITS_ADD_EXECUTABLE(
	steadyGeneralizedNonNewtonianAssFE_wBI 
	SOURCES main_BC.cpp
)

TRIBITS_ADD_TEST(
	steadyGeneralizedNonNewtonianAssFE_wBI  
	NAME steadyGeneralizedNonNewtonianAssFE_wBI  
	ARGS
#--precfile=parametersPrecMumps.xml
	NUM_MPI_PROCS 4
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_steadyCreepingMotionNonNewtonianAssFE_wBI 
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES parametersPrec.xml rectangle_200.mesh parametersProblem.xml parametersSolver.xml parametersTeko.xml
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES parametersPrec.xml rectangle_200.mesh  parametersProblem.xml parametersSolver.xml parametersTeko.xml
	EXEDEPS steadyGeneralizedNonNewtonianAssFE_wBI  
)
