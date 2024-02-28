SET(General_HEADERS
    MachineLearning/TrainedMLModelClass.hpp
	MachineLearning/TrainedMLModelClass_decl.hpp
	MachineLearning/TrainedMLModelClass_def.hpp

CACHE INTERNAL "")

SET(General_SOURCES
MachineLearning/TrainedMLModelClass.cpp
CACHE INTERNAL "")


SET(LOCAL_HEADERS)
FOREACH(INC ${General_HEADERS})
  STRING(REPLACE "General/" "" LOCAL_INC ${INC})
  SET(LOCAL_HEADERS ${LOCAL_HEADERS} ${LOCAL_INC})
ENDFOREACH()

INSTALL(
  FILES ${LOCAL_HEADERS}
  DESTINATION "${${PROJECT_NAME}_INSTALL_INCLUDE_DIR}/feddlib/core/MachineLearning"
  COMPONENT ${PACKAGE_NAME}
)
