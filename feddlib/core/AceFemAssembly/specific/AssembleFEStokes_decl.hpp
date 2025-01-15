#ifndef ASSEMBLEFESTOKES_DECL_hpp
#define ASSEMBLEFESTOKES_DECL_hpp

#include "feddlib/core/AceFemAssembly/AssembleFE.hpp"
#include "feddlib/core/AceFemAssembly/specific/AssembleFENavierStokes.hpp"
#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/FE/Helper.hpp"
#include "feddlib/core/LinearAlgebra/Matrix.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector.hpp"

namespace FEDD {

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class AssembleFEStokes : public AssembleFENavierStokes<SC,LO,GO,NO> {
  public:

    typedef Matrix<SC,LO,GO,NO> Matrix_Type;
    typedef Teuchos::RCP<Matrix_Type> MatrixPtr_Type;

	typedef SmallMatrix<SC> SmallMatrix_Type;
    typedef Teuchos::RCP<SmallMatrix_Type> SmallMatrixPtr_Type;

	typedef MultiVector<SC,LO,GO,NO> MultiVector_Type;
    typedef Teuchos::RCP<MultiVector_Type> MultiVectorPtr_Type;

	typedef AssembleFE<SC,LO,GO,NO> AssembleFE_Type;


	/*!
	 \brief Assemble the element Jacobian matrix.
	*/
	void assembleJacobian() override;

	/*!
	 \brief Assemble the element right hand side vector.
	*/
	void assembleRHS() override;

	/*!
		\brief Assemble the element Jacobian matrix.
		@param[in] block ID i
	*/
	void assembleJacobianBlock(LO i) override {};
	

	/*! 
	\brief Assembly of FixedPoint- Matrix (System Matrix K with current u) 
	*/
	void assembleFixedPoint();


   protected:

	/*!

	 \brief Constructor for AssembleFEAceNavierStokes

	@param[in] flag Flag of element
	@param[in] nodesRefConfig Nodes of element in reference configuration
	@param[in] params Parameterlist for current problem
	@param[in] tuple vector of element information tuples. 
	*/
	AssembleFEStokes(int flag, vec2D_dbl_Type nodesRefConfig, ParameterListPtr_Type parameters,tuple_disk_vec_ptr_Type tuple); 



    friend class AssembleFEFactory<SC,LO,GO,NO>; // Must have for specfic classes

	// Inherit everything from parent class

   private:

	
 };

}
#endif

