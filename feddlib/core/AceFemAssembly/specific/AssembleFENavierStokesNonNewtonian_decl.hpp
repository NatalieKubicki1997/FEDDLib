#ifndef ASSEMBLEFENAVIERSTOKESNEWTONIAN_DECL_hpp
#define ASSEMBLEFENAVIERSTOKESNEWTONIAN_DECL_hpp

#include "feddlib/core/AceFemAssembly/AssembleFE.hpp"
#include "feddlib/core/AceFemAssembly/specific/AssembleFENavierStokes.hpp"
#include "feddlib/core/AceFemAssembly/Helper.hpp"
#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/LinearAlgebra/Matrix.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector.hpp"
#include "feddlib/core/General/DifferentiableFuncClass.hpp"
#include "feddlib/core/AceFemAssembly/specific/GeneralizedNewtonianModels/CarreauYasuda.hpp"

namespace FEDD {

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class AssembleFENavierStokesNonNewtonian : public AssembleFENavierStokes<SC,LO,GO,NO> {
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

    // With have to add here class of specific material model or have we?
	//typedef CarreauYasuda<SC,LO,GO,NO>  CarreauYasuda_Type;
	//typedef Teuchos::RCP<CarreauYasuda> CarreauYasudaPtr_Type;

	/*!
	 \brief Assemble the element Jacobian matrix.
	*/
	virtual void assembleJacobian();

	/*!
	 \brief Assemble the element right hand side vector.
	*/
	virtual void assembleRHS();

	//SmallMatrixPtr_Type getFixedPointMatrix(){return ANB_;};

	//void setCoeff(SmallMatrix_Type coeff);
   protected:
 
   std::string shearThinningModel;

   private:

	/*!

	 \brief Constructor for AssembleFEAceNavierStokes

	@param[in] flag Flag of element
	@param[in] nodesRefConfig Nodes of element in reference configuration
	@param[in] params Parameterlist for current problem
	@param[in] tuple vector of element information tuples. 
	*/
	AssembleFENavierStokesNonNewtonian(int flag, vec2D_dbl_Type nodesRefConfig, ParameterListPtr_Type parameters,tuple_disk_vec_ptr_Type tuple); 

	/*!

	 \brief Assembly function for extra stress tensor which includes viscosity function \f$ \int_T \nabla v : (2\eta(\Dot{\gamma}(u))) D(u) ~dx\f$, which is a highly nonlinear Term 
	@param[in] &elementMatrix

	*/
	void assemblyStress(SmallMatrixPtr_Type &elementMatrix);

	/*!

	/*!

	 \brief Assembly function for extra derivative of extra stress tensor  resulting of applying the Gateaux-derivative
	@param[in] &elementMatrix

	*/
	void assemblyStressDev(SmallMatrixPtr_Type &elementMatrix);

	/*!

	 \brief Assembly advection vector field \f$ \int_T \nabla v \cdot u(\nabla u) ~dx\f$ 
	@param[in] &elementMatrix

	*/
	//void assemblyAdvection(SmallMatrixPtr_Type &elementMatrix);
	
	/*!
	 \brief Assembly advection vector field in u  
	@param[in] &elementMatrix

	*/
	//void assemblyAdvectionInU(SmallMatrixPtr_Type &elementMatrix); 

	/*!

	 \brief Assembly \f$ \int_T  div(v) p ~dx\f$ / \f$ \int_T  div(u) q ~dx\f$
	@param[in] &elementMatrix

	*/
	//void assemblyDivAndDivT(SmallMatrixPtr_Type &elementMatrix);



    friend class AssembleFEFactory<SC,LO,GO,NO>; // Must have for specfic classes

	void buildTransformation(SmallMatrix<SC>& B);

	void applyBTinv(vec3D_dbl_ptr_Type& dPhiIn,
		            vec3D_dbl_Type& dPhiOut,
		            SmallMatrix<SC>& Binv);

	//tuple_disk_vec_ptr_Type returnTuple(); /// @todo return tuple in case or check tuple

    DifferentiableFuncClassPtr_Type materialModel;

	//friend class CarreauYasuda<SC,LO,GO,NO>; the other way around carrea-yasuda has to give access rights for this class
 };

}
#endif

