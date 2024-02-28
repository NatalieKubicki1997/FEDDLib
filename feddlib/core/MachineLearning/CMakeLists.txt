SET(General_HEADERS
     /BCBuilder.hpp
	General/BCBuilder_decl.hpp
	General/BCBuilder_def.hpp

CACHE INTERNAL "")

SET(General_SOURCES
	General/BCBuilder.cpp
    General/ExporterParaView.cpp
    General/ExporterTxt.cpp
	General/DifferentiableFuncClass.cpp
	General/InputToOutputMappingClass.cpp
CACHE INTERNAL "")


SET(LOCAL_HEADERS)
FOREACH(INC ${General_HEADERS})
  STRING(REPLACE "General/" "" LOCAL_INC ${INC})
  SET(LOCAL_HEADERS ${LOCAL_HEADERS} ${LOCAL_INC})
ENDFOREACH()

INSTALL(
  FILES ${LOCAL_HEADERS}
  DESTINATION "${${PROJECT_NAME}_INSTALL_INCLUDE_DIR}/feddlib/core/General"
  COMPONENT ${PACKAGE_NAME}
)
