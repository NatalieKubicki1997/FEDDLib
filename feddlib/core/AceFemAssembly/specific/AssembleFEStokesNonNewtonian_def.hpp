#ifndef ASSEMBLEFESTOKESNONNEWTONIAN_DEF_hpp
#define ASSEMBLEFESTOKESNONNEWTONIAN_DEF_hpp

#include "AssembleFENavierStokes_decl.hpp"

namespace FEDD {
// // All important things are so far defined in AssembleFENavierStokes and AssembleFENavierStokesNonNewtonian . Please check there.
// Here is the class for generalized Newtonian fluids with neglected CONVECTIVE Term
/* *******************************************************************************
This class is for Non-Newtonian fluids where we consider in specifc generalized Newtonian models where the viscosityis non-constant
- Because the viscosity is no longer constant the conventional formulation with the laplacian term can not be considered
(but there is a generalized laplacian version of the equation see "On outflow boundary conditions in finite element simulations of non- Newtonian internal flow" 2021)
Instead we use the stress-divergence formulation of the momentum equation and derive from that the element-wise entries
******************************************************************************* */
template <class SC, class LO, class GO, class NO>
AssembleFEStokesNonNewtonian<SC,LO,GO,NO>::AssembleFEStokesNonNewtonian(int flag, vec2D_dbl_Type nodesRefConfig, ParameterListPtr_Type params,tuple_disk_vec_ptr_Type tuple):
AssembleFENavierStokesNonNewtonian<SC,LO,GO,NO>(flag, nodesRefConfig, params,tuple)
{

    ////******************* If we want to save viscosity - it is also possible to compute in paraview**********************************
	this->dofsElementViscosity_ = this->dofsPressure_*this->numNodesVelocity_; // So it is a scalar quantity but as it depend on the velocity it is defined at the nodes of the velocity
	this->solutionViscosity_ = vec_dbl_Type(this->dofsElementViscosity_ );    ////**********************************************************************************
    
    // Reading through parameterlist
    this->shearThinningModel= params->sublist("Material").get("ShearThinningModel","");
    // New: We have to check which material model we use
	if(this->shearThinningModel == "Carreau-Yasuda"){
		Teuchos::RCP<CarreauYasuda<SC,LO,GO,NO>> viscosityModelSpecific(new CarreauYasuda<SC,LO,GO,NO>(params) );
		this->viscosityModel= viscosityModelSpecific;
	}
	else if(this->shearThinningModel == "Power-Law"){
		Teuchos::RCP<PowerLaw<SC,LO,GO,NO>> viscosityModelSpecific(new PowerLaw<SC,LO,GO,NO>(params) );
		this->viscosityModel= viscosityModelSpecific;
	}
    else if(this->shearThinningModel == "Dimless-Carreau"){
		Teuchos::RCP<Dimless_Carreau<SC,LO,GO,NO>> viscosityModelSpecific(new Dimless_Carreau<SC,LO,GO,NO>(params) );
		this->viscosityModel= viscosityModelSpecific;
	}
	else
    		TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "No specific implementation for your material model request. Valid are:Carreau-Yasuda, Power-Law, Dimless-Carreau");


    // @TODO es wäre doch viel besser ein Pointer auf den Eintrag zu setzen der sich dann verändert??
    // switchToNewton_ = this->params_->sublist("General").getPtr<bool*>("SwitchToNewton");
    // this->params_->sublist("General").getPtr<bool*>("SwitchToNewton");
    // switchToNewton_ = *(this->params_->sublist("General").getEntryRCP("SwitchToNewton"));
}


template <class SC, class LO, class GO, class NO>
void AssembleFEStokesNonNewtonian<SC,LO,GO,NO>::assembleJacobian() 
{

    // For nonlinear generalized newtonian stress tensor part
	SmallMatrixPtr_Type elementMatrixN =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));
	SmallMatrixPtr_Type elementMatrixW =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));

    // For nonlinear convection
    SmallMatrixPtr_Type elementMatrixNC =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));
	SmallMatrixPtr_Type elementMatrixWC =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));
    

    // In the first iteration step we initialize the constant matrices
    // So in the case of a newtonian fluid we would have the matrix A with the contributions of the Laplacian term
    // and the matrix B with the mixed-pressure terms. Latter one exists also in the non-newtonian case
	if(this->newtonStep_ ==0){
		SmallMatrixPtr_Type elementMatrixA =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));
		SmallMatrixPtr_Type elementMatrixB =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));

		this->constantMatrix_.reset(new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));
        // Construct the matrix B from FE formulation - as it is equivalent to Newtonian case we call the same function
	    this->assemblyDivAndDivT(elementMatrixB); // For Matrix B
		elementMatrixB->scale(-1.);
		this->constantMatrix_->add( (*elementMatrixB),(*this->constantMatrix_));

    }
    
    // The other element matrices are not constant so we have to update them in each step
    // As the stress tensor term, considering a non-newtonian constitutive equation, is nonlinear we add its contribution here
  
    // ANB is the FixedPoint Formulation which was named for newtonian fluids. 
    // Matrix A (Laplacian term (here not occuring)), Matrix B for div-Pressure Part, Matrix N for nonlinear parts -

	this->ANB_.reset(new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_)); // A + B + N
	this->ANB_->add( ((*this->constantMatrix_)),((*this->ANB_)));


    // For a non-newtonian fluid we add additional element matrix and fill it with specific contribution
    // Remember that this term is based on the stress-divergence formulation of the momentum equation
    // \nabla \dot \tau with \tau=\eta(\gammaDot)(\nabla u + (\nabla u)^T)
    //    // As this class is derived from NavierStokesNonNewtonian class we can call already implemented function
    //*************** STRESS TENSOR************************
    this->assemblyStress(elementMatrixN);
	this->ANB_->add( (*elementMatrixN),((*this->ANB_)));


    this->jacobian_.reset(new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));
	this->jacobian_->add((*this->ANB_),(*this->jacobian_));


    // If linearization is not FixdPoint (so NOX or Newton) we add the derivative to the Jacobian matrix. Otherwise the FixedPoint formulation becomes the jacobian.
    if(this->linearization_ != "FixedPoint"){

	    this->assemblyStressDev(elementMatrixW);     // stress tensor
        //this->assemblyAdvectionInU(elementMatrixWC); // convection
	    //elementMatrixWC->scale(this->density_);
        
        this->jacobian_->add((*elementMatrixW),(*this->jacobian_)); 
        //this->jacobian_->add((*elementMatrixWC),(*this->jacobian_));  // int add(SmallMatrix<T> &bMat, SmallMatrix<T> &cMat); //this+B=C elementMatrix + constantMatrix_;
    }
	
    //**************** BOUNDARY TERM *******************************
    // Only reasonable for unstructured Mesh elements (triangles)
    // Because we have stress-divergence form of Navier-Stokes equations in the non-newtonian case
    // we have to add a extra boundary term to get the same outflow boundary condition as in the conventional formulation with
    // the laplacian operator in the equations due to the fact that in the stress-divergence formulation the
    // natural boundary condition is different 
    // We have to check whether it is an element which has edges (2D) / surfaces (3D) corresponding to an Outflow Neumann boundary
    // Then we have to compute contribution 
    if (this->surfaceElement == true) // Only if we consider element with neumann edge
    {
      SmallMatrixPtr_Type elementMatrixNB =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));
      this->assemblyNeumannBoundaryTerm(elementMatrixNB);
      this->ANB_->add( (*elementMatrixNB),((*this->ANB_)));
 
      // Newton converges also if unabled and also in same steps so we can also comment that out
      // If linearization is not FixdPoint (so NOX or Newton) we add the derivative to the Jacobian matrix. Otherwise the FixedPoint formulation becomes the jacobian.
      if(this->linearization_ != "FixedPoint")
      {
       SmallMatrixPtr_Type elementMatrixNBW =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));
	   this->assemblyNeumannBoundaryTermDev(elementMatrixNBW); //
       this->jacobian_->add((*elementMatrixNBW),(*this->jacobian_));  
      }
      
   
   
   
    }

}


// "Fixpunkt"- Matrix without jacobian for calculating Ax 
// Here update please to unlinearized System Matrix accordingly.
template <class SC, class LO, class GO, class NO>
void AssembleFEStokesNonNewtonian<SC,LO,GO,NO>::assembleRHS(){

	SmallMatrixPtr_Type elementMatrixN =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));
    SmallMatrixPtr_Type elementMatrixNC =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));

	this->ANB_.reset(new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_)); // A + B + N
	this->ANB_->add( (*this->constantMatrix_),(*this->ANB_));

    // Nonlinear stress tensor *******************************
	this->assemblyStress(elementMatrixN);
	this->ANB_->add( (*elementMatrixN),(*this->ANB_));

    // If boundary element - nonlinear boundar term *
    if (this->surfaceElement == true)
    {
      SmallMatrixPtr_Type elementMatrixNB =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));
      this->assemblyNeumannBoundaryTerm(elementMatrixNB);
      this->ANB_->add( (*elementMatrixNB),((*this->ANB_)));
    }


	this->rhsVec_.reset( new vec_dbl_Type ( this->dofsElement_,0.) );
	// Multiplying ANB_ * solution // ANB Matrix without nonlinear part.
	int s=0,t=0;
	for(int i=0 ; i< this->ANB_->size();i++){
		if (i >= this->dofsElementVelocity_)
			s=1;
		for(int j=0; j < this->ANB_->size(); j++){
			if(j >= this->dofsElementVelocity_)
				t=1;
			(*this->rhsVec_)[i] += (*this->ANB_)[i][j]*(*this->solution_)[j]*this->coeff_[s][t];
		}
		t=0;
	}
}



}
#endif

