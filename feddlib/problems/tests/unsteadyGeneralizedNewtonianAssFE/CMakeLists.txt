tribits_add_executable(
	unsteadyGeneralizedNewtonianAssFE
	SOURCES main.cpp
)

tribits_add_test(
	unsteadyGeneralizedNewtonianAssFE
	NAME unsteadyGeneralizedNewtonianAssFE
	ARGS
	#--precfile=parametersPrecMumps.xml
	NUM_MPI_PROCS 4
)

tribits_copy_files_to_binary_dir(data_unsteadyGeneralizedNewtonianAssFE
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES parametersPrec.xml parametersProblem.xml parametersSolver.xml parametersTeko.xml rectangle_200.mesh run_single.sh
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES parametersPrec.xml parametersProblem.xml parametersSolver.xml parametersTeko.xml rectangle_200.mesh run_single.sh
	EXEDEPS unsteadyGeneralizedNewtonianAssFE
)

