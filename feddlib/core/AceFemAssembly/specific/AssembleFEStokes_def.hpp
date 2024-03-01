#ifndef ASSEMBLEFESTOKES_DEF_hpp
#define ASSEMBLEFESTOKES_DEF_hpp

#include "AssembleFEStokes_decl.hpp"

namespace FEDD {

template <class SC, class LO, class GO, class NO>
AssembleFEStokes<SC,LO,GO,NO>::AssembleFEStokes(int flag, vec2D_dbl_Type nodesRefConfig, ParameterListPtr_Type params,tuple_disk_vec_ptr_Type tuple):
AssembleFENavierStokes<SC,LO,GO,NO>(flag, nodesRefConfig, params,tuple)
{
	int locVelocity=0;
	int locPressure=0;		
	if(std::get<0>(this->diskTuple_->at(0))=="Velocity"){
		locVelocity=0;
		locPressure=1;
	}
	else if(std::get<0>(this->diskTuple_->at(1))=="Velocity"){
		locVelocity=1;
		locPressure=0;
	}
	else
    	TEUCHOS_TEST_FOR_EXCEPTION( true, std::logic_error, "No discretisation Information for Velocity in Navier Stokes Element." );
		
    //AssembleFEStokes
	/// Tupel construction follows follwing pattern:
	/// string: Physical Entity (i.e. Velocity) , string: Discretisation (i.e. "P2"), int: Degrees of Freedom per Node, int: Number of Nodes per element)
	this->FETypeVelocity_ = std::get<1>(this->diskTuple_->at(locVelocity));
	this->FETypePressure_ =std::get<1>(this->diskTuple_->at(locPressure));

	this->dofsVelocity_ = std::get<2>(this->diskTuple_->at(locVelocity));
	this->dofsPressure_ =std::get<2>(this->diskTuple_->at(locPressure));

	this->numNodesVelocity_ = std::get<3>(this->diskTuple_->at(locVelocity));
	this->numNodesPressure_=std::get<3>(this->diskTuple_->at(locPressure));

	this->dofsElementVelocity_ = this->dofsVelocity_*this->numNodesVelocity_;
	this->dofsElementPressure_  = this->dofsPressure_*this->numNodesPressure_;	

	//this->solution_ = vec_dbl_Type(this->dofsElementVelocity_); //this->dofsElementPressure_+
	this->solutionVelocity_ = vec_dbl_Type(this->dofsElementVelocity_);
	this->solutionPressure_ = vec_dbl_Type(this->dofsElementPressure_);

 	this->viscosity_ = this->params_->sublist("Parameter").get("Viscosity",1.);
    this->density_  = this->params_->sublist("Parameter").get("Density",1.);

	this->dofsElement_ = this->dofsElementVelocity_+ this->dofsElementPressure_;

	SmallMatrix_Type coeff(2);
	coeff[0][0]=1.; coeff[0][1] = 1.; coeff[1][0] = 1.; coeff[1][1] = 1.; // we keep it constant like this for now. For BDF time disc. okay.
	this->coeff_ = coeff;


    this->linearization_= this->params_->sublist("General").get("Linearization","FixedPoint"); // Information to assemble Jacobian accordingly


}


template <class SC, class LO, class GO, class NO>
void AssembleFEStokes<SC,LO,GO,NO>::assembleJacobian() {

	SmallMatrixPtr_Type elementMatrixN =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));
	SmallMatrixPtr_Type elementMatrixW =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));

	if(this->newtonStep_ ==0){
		SmallMatrixPtr_Type elementMatrixA =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));
		SmallMatrixPtr_Type elementMatrixB =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));

		this->constantMatrix_.reset(new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));

		this->assemblyLaplacian(elementMatrixA);

		elementMatrixA->scale(this->viscosity_);
		elementMatrixA->scale(this->density_);

		this->constantMatrix_->add( (*elementMatrixA),(*this->constantMatrix_));

		this->assemblyDivAndDivT(elementMatrixB); // For Matrix B

		elementMatrixB->scale(-1.);

		this->constantMatrix_->add( (*elementMatrixB),(*this->constantMatrix_));
    }

	this->ANB_.reset(new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_)); // A + B + N
	this->ANB_->add( (*this->constantMatrix_),(*this->ANB_));

	//elementMatrix->add((*this->constantMatrix_),(*elementMatrix));
	this->jacobian_.reset(new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));

	this->jacobian_->add((*this->ANB_),(*this->jacobian_));
}





// Assemble RHS with updated solution coming from Fixed Point Iter or der Newton.
template <class SC, class LO, class GO, class NO>
void AssembleFEStokes<SC,LO,GO,NO>::assembleRHS(){

	SmallMatrixPtr_Type elementMatrixN =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));

	this->ANB_.reset(new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_)); // A + B + N
	this->ANB_->add( (*this->constantMatrix_),(*this->ANB_));

	this->rhsVec_.reset( new vec_dbl_Type ( this->dofsElement_,0.) );
	// Multiplying this->ANB_ * solution // ANB Matrix without nonlinear part.
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

