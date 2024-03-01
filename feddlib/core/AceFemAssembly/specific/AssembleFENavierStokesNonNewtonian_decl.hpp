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
#include "feddlib/core/AceFemAssembly/specific/GeneralizedNewtonianModels/PowerLaw.hpp"
#include "feddlib/core/AceFemAssembly/specific/GeneralizedNewtonianModels/Dimless_Carreau.hpp"


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


    typedef InputToOutputMappingClass<SC,LO,GO,NO>  InputToOutputMappingClass_Type;    
	typedef Teuchos::RCP<InputToOutputMappingClass_Type> InputToOutputMappingClassPtr_Type;
    // smart pointer inside we need a type

	/*!
	 \brief Assemble the element Jacobian matrix.
	*/
	virtual void assembleJacobian();

	/*!
	 \brief Assemble the element right hand side vector.
	*/
	virtual void assembleRHS();

	  /*!
		\brief Assemble the element Jacobian matrix.
		@param[in] block ID i
	*/
	virtual void assembleJacobianBlock(LO i) {};

		/*!
	 \brief Compute the viscosity for an element depending on the knwon velocity solution.
	*/
	virtual void computeLocalViscosity();

   protected:
 
   std::string shearThinningModel;
   int dofsElementViscosity_;
   vec_dbl_Type solutionViscosity;

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

	 \brief Assembly function for neumann boundary term

	*/
	void assemblyNeumannBoundaryTerm(SmallMatrixPtr_Type &elementMatrix);

	/*!

	/*!

	 \brief Assembly function for neumann boundary term

	*/
	void assemblyNeumannBoundaryTermDev(SmallMatrixPtr_Type &elementMatrix);

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

	

    friend class AssembleFEFactory<SC,LO,GO,NO>; // Must have for specfic classes

	void buildTransformation(SmallMatrix<SC>& B);


	void applyBTinv(vec3D_dbl_ptr_Type& dPhiIn,
		            vec3D_dbl_Type& dPhiOut,
		            SmallMatrix<SC>& Binv);

	void computeShearRate(vec3D_dbl_Type dPhiTrans, vec_dbl_ptr_Type& gammaDot, int dim);

	virtual void set_LinearizationToNewton();
	// bool* switchToNewton_; maybe it is better to proivde a pointer to the entry of problem.getParameterList()->sublist("General").get("SwitchToNewton",true); 

    //DifferentiableFuncClassPtr_Type viscosityModel  ;   // Make it more general such that the viscosity material Model can be any Input to Output Mapping 
    InputToOutputMappingClassPtr_Type viscosityModel;
	
	 private:
	//friend class CarreauYasuda<SC,LO,GO,NO>; the other way around carrea-yasuda has to give access rights for this class
 };

}
#endif

