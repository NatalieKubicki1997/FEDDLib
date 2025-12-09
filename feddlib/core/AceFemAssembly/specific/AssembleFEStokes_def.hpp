#ifndef ASSEMBLEFESTOKES_DEF_hpp
#define ASSEMBLEFESTOKES_DEF_hpp

namespace FEDD {

template <class SC, class LO, class GO, class NO>
AssembleFEStokes<SC,LO,GO,NO>::AssembleFEStokes(int flag, vec2D_dbl_Type nodesRefConfig, ParameterListPtr_Type params, tuple_disk_vec_ptr_Type tuple) : AssembleFENavierStokes<SC, LO, GO, NO>(flag, nodesRefConfig, params, tuple)
{
   // Everything handled in Navier Stokes Constructor
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

	//elementMatrix->add((*constantMatrix_),(*elementMatrix));
	this->jacobian_.reset(new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));
	this->jacobian_->add((*this->ANB_),(*this->jacobian_));

}

template <class SC, class LO, class GO, class NO>
void AssembleFEStokes<SC,LO,GO,NO>::assembleFixedPoint() {

	SmallMatrixPtr_Type elementMatrixN =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));

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

}



// Assemble RHS with updated solution coming from Fixed Point Iter or der Newton.
template <class SC, class LO, class GO, class NO>
void AssembleFEStokes<SC,LO,GO,NO>::assembleRHS(){

	SmallMatrixPtr_Type elementMatrixN =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));

	this->ANB_.reset(new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_)); // A + B + N
	this->ANB_->add( (*this->constantMatrix_),(*this->ANB_));

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

