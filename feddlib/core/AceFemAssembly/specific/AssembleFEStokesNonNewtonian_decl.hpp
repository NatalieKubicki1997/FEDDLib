#ifndef ASSEMBLEFESTOKESNEWTONIAN_DECL_hpp
#define ASSEMBLEFESTOKESNEWTONIAN_DECL_hpp

#include "feddlib/core/AceFemAssembly/specific/AssembleFENavierStokesNonNewtonian.hpp"
#include "feddlib/core/AceFemAssembly/Helper.hpp"
#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/LinearAlgebra/Matrix.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector.hpp"
#include "feddlib/core/General/DifferentiableFuncClass.hpp"
#include "feddlib/core/AceFemAssembly/specific/GeneralizedNewtonianModels/CarreauYasuda.hpp"
#include "feddlib/core/AceFemAssembly/specific/GeneralizedNewtonianModels/PowerLaw.hpp"
#include "feddlib/core/AceFemAssembly/specific/GeneralizedNewtonianModels/Dimless_Carreau.hpp"

// Specific Assembly class for Stokes elements derived from Navier-Stokes Non-Newtonian
// elements assembly where all important functions are implemented
// here we just neglect the convective part in the momentum equation but still consider a generalized-Newtonian model

namespace FEDD {

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class AssembleFEStokesNonNewtonian : public AssembleFENavierStokesNonNewtonian<SC,LO,GO,NO> {
  public:

    typedef Matrix<SC,LO,GO,NO> Matrix_Type;
    typedef Teuchos::RCP<Matrix_Type> MatrixPtr_Type;

	typedef SmallMatrix<SC> SmallMatrix_Type;
    typedef Teuchos::RCP<SmallMatrix_Type> SmallMatrixPtr_Type;

	typedef MultiVector<SC,LO,GO,NO> MultiVector_Type;
    typedef Teuchos::RCP<MultiVector_Type> MultiVectorPtr_Type;

	typedef AssembleFE<SC,LO,GO,NO> AssembleFE_Type;

	typedef DifferentiableFuncClass<SC,LO,GO,NO>  DifferentiableFuncClass_Type;
	typedef Teuchos::RCP<DifferentiableFuncClass_Type> DifferentiableFuncClassPtr_Type;
    // smart pointer inside we need a type

	/*!
	 \brief Assemble the element Jacobian matrix.
	*/
	virtual void assembleJacobian();

	/*!
	 \brief Assemble the element right hand side vector.
	*/
	virtual void assembleRHS();


   protected:
 
   
   private:

	/*!

	 \brief Constructor for AssembleFEAceNavierStokes

	@param[in] flag Flag of element
	@param[in] nodesRefConfig Nodes of element in reference configuration
	@param[in] params Parameterlist for current problem
	@param[in] tuple vector of element information tuples. 
	*/
	AssembleFEStokesNonNewtonian(int flag, vec2D_dbl_Type nodesRefConfig, ParameterListPtr_Type parameters,tuple_disk_vec_ptr_Type tuple); 



    friend class AssembleFEFactory<SC,LO,GO,NO>; // Must have for specfic classes

	// bool* switchToNewton_; maybe it is better to proivde a pointer to the entry of problem.getParameterList()->sublist("General").get("SwitchToNewton",true); 
	//friend class CarreauYasuda<SC,LO,GO,NO>; the other way around carrea-yasuda has to give access rights for this class
 };

}
#endif

