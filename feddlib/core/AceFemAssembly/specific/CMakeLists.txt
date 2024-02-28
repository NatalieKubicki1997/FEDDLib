SET(AceFemAssemblySpecific_HEADERS
	AceFemAssembly/specific/AssembleFEAceLaplace_decl.hpp
	AceFemAssembly/specific/AssembleFEAceLaplace_def.hpp
	AceFemAssembly/specific/AssembleFEAceLaplace.hpp
	AceFemAssembly/specific/AssembleFENavierStokes_decl.hpp
	AceFemAssembly/specific/AssembleFENavierStokes_def.hpp
	AceFemAssembly/specific/AssembleFENavierStokes.hpp
	AceFemAssembly/specific/AssembleFEStokes_decl.hpp
	AceFemAssembly/specific/AssembleFEStokes_def.hpp
	AceFemAssembly/specific/AssembleFEStokes.hpp
	AceFemAssembly/specific/AssembleFENavierStokesNonNewtonian_decl.hpp
	AceFemAssembly/specific/AssembleFENavierStokesNonNewtonian_def.hpp
	AceFemAssembly/specific/AssembleFENavierStokesNonNewtonian.hpp
	AceFemAssembly/specific/AssembleFEStokesNonNewtonian_decl.hpp
	AceFemAssembly/specific/AssembleFEStokesNonNewtonian_def.hpp
	AceFemAssembly/specific/AssembleFEStokesNonNewtonian.hpp
	AceFemAssembly/specific/AssembleFEAceLinElas_decl.hpp
	AceFemAssembly/specific/AssembleFEAceLinElas_def.hpp
	AceFemAssembly/specific/AssembleFEAceLinElas.hpp
	AceFemAssembly/specific/AssembleFEAceNonLinElas_decl.hpp
	AceFemAssembly/specific/AssembleFEAceNonLinElas_def.hpp
	AceFemAssembly/specific/AssembleFEAceNonLinElas.hpp
	AceFemAssembly/specific/GeneralizedNewtonianModels/CarreauYasuda_decl.hpp
	AceFemAssembly/specific/GeneralizedNewtonianModels/CarreauYasuda_def.hpp
	AceFemAssembly/specific/GeneralizedNewtonianModels/CarreauYasuda.hpp
	AceFemAssembly/specific/GeneralizedNewtonianModels/PowerLaw_decl.hpp
	AceFemAssembly/specific/GeneralizedNewtonianModels/PowerLaw_def.hpp
	AceFemAssembly/specific/GeneralizedNewtonianModels/PowerLaw.hpp
	AceFemAssembly/specific/GeneralizedNewtonianModels/Dimless_Carreau_decl.hpp
	AceFemAssembly/specific/GeneralizedNewtonianModels/Dimless_Carreau_def.hpp
	AceFemAssembly/specific/GeneralizedNewtonianModels/Dimless_Carreau.hpp
CACHE INTERNAL "")

SET(AceFemAssemblySpecific_SOURCES
	AceFemAssembly/specific/AssembleFEAceLaplace.cpp
	AceFemAssembly/specific/AssembleFENavierStokes.cpp
	AceFemAssembly/specific/AssembleFEStokes.cpp
	AceFemAssembly/specific/AssembleFENavierStokesNonNewtonian.cpp
	AceFemAssembly/specific/AssembleFEStokesNonNewtonian.cpp
	AceFemAssembly/specific/AssembleFEAceLinElas.cpp
	AceFemAssembly/specific/AssembleFEAceNonLinElas.cpp
	AceFemAssembly/specific/GeneralizedNewtonianModels/CarreauYasuda.cpp
	AceFemAssembly/specific/GeneralizedNewtonianModels/PowerLaw.cpp
	AceFemAssembly/specific/GeneralizedNewtonianModels/Dimless_Carreau.cpp
CACHE INTERNAL "")


SET(LOCAL_HEADERS)
FOREACH(INC ${AceFemAssemblySpecific_HEADERS})
  STRING(REPLACE "specific/" "" LOCAL_INC ${INC})
  SET(LOCAL_HEADERS ${LOCAL_HEADERS} ${LOCAL_INC})
ENDFOREACH()

INSTALL(
  FILES ${LOCAL_HEADERS}
  DESTINATION "${${PROJECT_NAME}_INSTALL_INCLUDE_DIR}/feddlib/core/AceFemAssembly/specific"
  COMPONENT ${PACKAGE_NAME}
)
