SET(MachineLearningSpecific_HEADERS
	MachineLearning/specific/MultiLayerPerceptron_decl.hpp
	MachineLearning/specific/MultiLayerPerceptron_def.hpp
    MachineLearning/specific/MultiLayerPerceptron.hpp
CACHE INTERNAL "")

SET(MachineLearningSpecific_SOURCES
    MachineLearning/specific/MultiLayerPerceptron.cpp

CACHE INTERNAL "")


SET(LOCAL_HEADERS)
FOREACH(INC ${MachineLearningSpecific_HEADERS})
  STRING(REPLACE "specific/" "" LOCAL_INC ${INC})
  SET(LOCAL_HEADERS ${LOCAL_HEADERS} ${LOCAL_INC})
ENDFOREACH()

INSTALL(
  FILES ${LOCAL_HEADERS}
  DESTINATION "${${PROJECT_NAME}_INSTALL_INCLUDE_DIR}/feddlib/core/MachineLearning/specific"
  COMPONENT ${PACKAGE_NAME}
)
