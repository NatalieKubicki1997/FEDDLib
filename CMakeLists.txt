#
# A) Define the project
#

# To be safe, define your minimum CMake version.  This may be newer than the
# min required by TriBITS.

CMAKE_MINIMUM_REQUIRED(VERSION 3.17.0 FATAL_ERROR)

INCLUDE(${CMAKE_SOURCE_DIR}/ProjectName.cmake)
PROJECT(${PROJECT_NAME})

# The version number.
SET(FEDDLIB_VERSION_MAJOR 1)
SET(FEDDLIB_VERSION_MINOR 0)

SET(${PROJECT_NAME}_ENABLE_Fortran ON CACHE BOOL "Enable Fortran compiler")

# Set default C++ standard to C++17
SET(FEDDlib_ENABLE_CXX17_DEFAULT ON)


# set up a Profiling build type
SET(CMAKE_CXX_FLAGS_PROFILING "-O3 -pg -g" CACHE STRING "CXX profiling flags")
SET(CMAKE_C_FLAGS_PROFILING   "-O3 -pg -g" CACHE STRING "C profiling flags")

#SET(GCC_COVERAGE_LINK_FLAGS    "-lm")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}" )


SET(CMAKE_MACOSX_RPATH 1)



#
# B) Pull in the TriBITS system and execute
#


INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tribits/TriBITS.cmake)

# CMake requires this be in the top file and not in an include file :-(
CMAKE_MINIMUM_REQUIRED(VERSION ${TRIBITS_CMAKE_MINIMUM_REQUIRED})

# Add documentation
#ADD_SUBDIRECTORY(doc)

# Add source dir to includes

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
MESSAGE(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")


# Do all of the processing for this Tribits project - all packages are found etc
TRIBITS_PROJECT()



#SET_TARGET_PROPERTIES(PROJECTLANGUAGE PROPERTIES LINKER_LANGUAGE C11) #CH 14.12.18:can we get rid of this?

FOREACH(ENABLED_PACKAGE ${FEDDlib_ENABLED_PACKAGES})
	STRING(TOUPPER ${ENABLED_PACKAGE} UPPER_PACKAGE)
	SET(FEDD_HAVE_${UPPER_PACKAGE} ON)
ENDFOREACH()


### SET IMPORTANT OUTPUTS FOR USER
#START
# Retrieve the Pybind11 version
EXECUTE_PROCESS(COMMAND
  ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.__version__)"
  OUTPUT_VARIABLE PYBIND11_VERSION
  ERROR_VARIABLE  PYBIND11_VERSION_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )

IF(NOT PYBIND11_VERSION_ERROR)
  MESSAGE("  -- Pybind11 Enabled.")
ELSE()
  MESSAGE(FATAL_ERROR "PYBIND11_VERSION_ERROR is defined; the python executable cannot access pybind11.")
ENDIF()

# Retrieve the Mpi4Py version
EXECUTE_PROCESS(COMMAND
  ${PYTHON_EXECUTABLE} -c "import mpi4py; print(mpi4py.__version__)"
  OUTPUT_VARIABLE Mpi4Py_VERSION
  ERROR_VARIABLE  Mpi4Py_VERSION_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )

IF(NOT Mpi4Py_VERSION_ERROR)
  MESSAGE("  -- Mpi4Py Enabled.")
ELSE()
  MESSAGE(FATAL_ERROR "Mpi4Py_VERSION_ERROR is defined; the python executable cannot access mpi4py.")
ENDIF()

# Retrieve the Numpy version
EXECUTE_PROCESS(COMMAND
  ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.__version__)"
  OUTPUT_VARIABLE NUMPY_VERSION
  ERROR_VARIABLE  NUMPY_VERSION_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )

IF(NOT NUMPY_VERSION_ERROR)
  MESSAGE("  -- Numpy Enabled.")
ELSE()
  MESSAGE(FATAL_ERROR "NUMPY_VERSION_ERROR is defined; the python executable cannot access numpy.")
ENDIF()

MESSAGE("-- FEDDLib uses pybind11 ${PYBIND11_VERSION}, mpi4py ${Mpi4Py_VERSION}, and numpy ${NUMPY_VERSION}.")
### END


CONFIGURE_FILE(	${CMAKE_SOURCE_DIR}/${PROJECT_NAME}_Config.h.in
				${CMAKE_BINARY_DIR}/${PROJECT_NAME}_Config.h)
# We need the next line to install feddlib_config.h.in too
INSTALL(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}_config.h DESTINATION include)

INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/feddlib/debug.areas
DESTINATION share/feddlib)



#find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
#FIND_PACKAGE(PythonInterp 3)
#FIND_PACKAGE(PythonLibs   3)
#find_package(pybind11 REQUIRED)  # or `add_subdirectory(pybind11)`
#add_subdirectory(/home/user/Programme/opt/Anaconda/anaconda3/envs/pyfeddlib/lib/python3.11/site-packages/pybind11/include/pybind11 build)



