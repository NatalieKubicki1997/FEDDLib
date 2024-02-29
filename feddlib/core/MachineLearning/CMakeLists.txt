SET(MachineLearning_HEADERS
  MachineLearning/TrainedMLModelClass.hpp
	MachineLearning/TrainedMLModelClass_decl.hpp
	MachineLearning/TrainedMLModelClass_def.hpp
  MachineLearning/ActivationFunctions/Tanh_decl.hpp
  MachineLearning/ActivationFunctions/Tanh_def.hpp
  MachineLearning/ActivationFunctions/Tanh.hpp
  MachineLearning/FeedForwardNeuronalNetworkClass.hpp
	MachineLearning/FeedForwardNeuronalNetworkClass_decl.hpp
	MachineLearning/FeedForwardNeuronalNetworkClass_def.hpp


CACHE INTERNAL "")

SET(MachineLearning_SOURCES
MachineLearning/TrainedMLModelClass.cpp
MachineLearning/ActivationFunctions/Tanh.cpp
MachineLearning/FeedForwardNeuronalNetworkClass.cpp


CACHE INTERNAL "")


SET(LOCAL_HEADERS)
FOREACH(INC ${MachineLearning_HEADERS})
  STRING(REPLACE "MachineLearning/" "" LOCAL_INC ${INC})
  SET(LOCAL_HEADERS ${LOCAL_HEADERS} ${LOCAL_INC})
ENDFOREACH()

INSTALL(
  FILES ${LOCAL_HEADERS}
  DESTINATION "${${PROJECT_NAME}_INSTALL_INCLUDE_DIR}/feddlib/core/MachineLearning"
  COMPONENT ${PACKAGE_NAME}
)
