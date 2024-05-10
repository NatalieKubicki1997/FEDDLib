### mesh_nodeFlagsP2
TRIBITS_ADD_EXECUTABLE(
    refinePreprocessExport_uniform
    SOURCES refine.cpp
)

TRIBITS_ADD_TEST(
	refinePreprocessExport_uniform
    NAME refinePreprocessExportsTestUniform
    ARGS 
	--problemfile=parametersProblem.xml
    NUM_MPI_PROCS 4
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_refinePreprocessExport
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES parametersProblem.xml
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES parametersProblem.xml
	EXEDEPS refinePreprocessExport_uniform
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(mesh_refinePreprocessExport
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/meshes
    SOURCE_FILES meshNodeTestP23D_2.mesh
    DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
    DEST_FILES meshNodeTestP23D_2.mesh
    EXEDEPS refinePreprocessExport_uniform
)

### 
