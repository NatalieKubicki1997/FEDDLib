#ifndef ASSEMBLEFENAVIERSTOKESNONNEWTONIAN_DEF_hpp
#define ASSEMBLEFENAVIERSTOKESNONNEWTONIAN_DEF_hpp

#include "AssembleFENavierStokes_decl.hpp"

namespace FEDD {
// All important things are so far defined in AssembleFENavierStokes. Please check there.
template <class SC, class LO, class GO, class NO>
AssembleFENavierStokesNonNewtonian<SC,LO,GO,NO>::AssembleFENavierStokesNonNewtonian(int flag, vec2D_dbl_Type nodesRefConfig, ParameterListPtr_Type params,tuple_disk_vec_ptr_Type tuple):
AssembleFENavierStokes<SC,LO,GO,NO>(flag, nodesRefConfig, params,tuple)
{

    ////******************* IF we want to save viscosity**********************************
	dofsElementViscosity_ = this->dofsPressure_*this->numNodesVelocity_; // So it is a scalar quantity but as it depend on the velocity it is defined at the nodes of the velocity
	this->solutionViscosity_ = vec_dbl_Type(dofsElementViscosity_ );
    //this->solutionViscosity_ = vec_dbl_Type(1 );
    ////**********************************************************************************
    
    // Reading through parameterlist
    shearThinningModel= params->sublist("Material").get("ShearThinningModel","");
    // New: We have to check which material model we use 
	if(shearThinningModel == "Carreau-Yasuda"){
		Teuchos::RCP<CarreauYasuda<SC,LO,GO,NO>> materialModelSpecific(new CarreauYasuda<SC,LO,GO,NO>(params) );
		materialModel = materialModelSpecific;
	}
	else if(shearThinningModel == "Power-Law"){
		Teuchos::RCP<PowerLaw<SC,LO,GO,NO>> materialModelSpecific(new PowerLaw<SC,LO,GO,NO>(params) );
		materialModel = materialModelSpecific;
	}
	else
    		TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "No specific implementation for your material model request.");
}






template <class SC, class LO, class GO, class NO>
void AssembleFENavierStokesNonNewtonian<SC,LO,GO,NO>::assembleJacobian() {

	SmallMatrixPtr_Type elementMatrixN =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));
	SmallMatrixPtr_Type elementMatrixW =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));

    // In the first iteration step we initialize the constant matrices
    // So in the case of a newtonian fluid we would have the matrix A with the contributions of the Laplacian term
    // and the matrix B with the mixed-pressure terms. Latter one exists also in the non-newtonian case
	if(this->newtonStep_ ==0){
		SmallMatrixPtr_Type elementMatrixA =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));
		SmallMatrixPtr_Type elementMatrixB =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));

		this->constantMatrix_.reset(new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));

        // See AssemebleFENavierStokes_def.hpp to see implementation of Laplacian term

        // Construct the matrix B from FE formulation - as it is equivalent to Newtonian case we call the same function
	    this->assemblyDivAndDivT(elementMatrixB); // For Matrix B
		elementMatrixB->scale(-1.);
		this->constantMatrix_->add( (*elementMatrixB),(*this->constantMatrix_));

    }
    
    // The other element matrices are not constant so we have to update them in each step
    // As the stress tensor term, considering a non-newtonian constitutive equation, is nonlinear we add its contribution here
    // ZUDEM: Da dies ein nichtlinearen Term ist packen wir ihn zu dem Advektionsterm unten

    // ANB is the FixedPoint Formulation which was named for newtonian fluids. 
    // Matrix A (Laplacian term (here not occuring))
    // Matrix B for div-Pressure Part
    // Matrix N for nonlinear advection part - We neglect it in the first step but we can easily add it by uncommenting coming lines

	this->ANB_.reset(new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_)); // A + B + N
	this->ANB_->add( ((*this->constantMatrix_)),((*this->ANB_)));

    // In order to consider nonlinear advection term \rho (u \cdot \nabla) u uncomment following lines:
    // as this class is derived from NavierStokes class we can call this function
	//this->assemblyAdvection(elementMatrixN);
	//elementMatrixN->scale(this->density_);
	//this->ANB_->add( (*elementMatrixN),((*this->ANB_)));


    // For a non-newtonian fluid we add additional element matrix and fill it with specific contribution
    this->assemblyStress(elementMatrixN);
	// elementMatrixN->scale(this->density_); 
    // elementMatrixN->scale(-1.0);
	this->ANB_->add( (*elementMatrixN),((*this->ANB_)));


    // If linearization is not FixdPoint (so NOX or Newton) we add the derivative to the Jacobian matrix. Otherwise the FixedPoint formulation becomes the jacobian.
    if(this->linearization_ != "FixedPoint"){
	    this->assemblyStressDev(elementMatrixW);
	    // elementMatrixW->scale(this->density_);
    }

	//elementMatrix->add((*constantMatrix_),(*elementMatrix));
	this->jacobian_.reset(new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));
	this->jacobian_->add((*this->ANB_),(*this->jacobian_));
    // If the linearization is Newtons Method  or NOX we need to add W-Matrix
    if(this->linearization_ != "FixedPoint"){
    	this->jacobian_->add((*elementMatrixW),(*this->jacobian_));  // int add(SmallMatrix<T> &bMat, SmallMatrix<T> &cMat); //this+B=C elementMatrix + constantMatrix_;
    }
}

// Extra stress term resulting from chosen non-newtonian constitutive model  - Compute element matrix entries
template <class SC, class LO, class GO, class NO>
void AssembleFENavierStokesNonNewtonian<SC,LO,GO,NO>::assemblyStress(SmallMatrixPtr_Type &elementMatrix) {

	int dim = this->getDim();
	int numNodes= this->numNodesVelocity_;
	int Grad =2; // Needs to be fixed	
	string FEType = this->FETypeVelocity_;
	int dofs = this->dofsVelocity_; // For pressure it would be 1 

    vec3D_dbl_ptr_Type 	dPhi;
    vec_dbl_ptr_Type weights = Teuchos::rcp(new vec_dbl_Type(0));
    
    UN deg = Helper::determineDegree(dim,FEType,Grad); // for P1 3
    Helper::getDPhi(dPhi, weights, dim, FEType, deg); // for deg 5 we get weight vector with 7 entries weights->at(7)
    // Example Values: dPhi->size() = 7 so number of quadrature points, dPhi->at(0).size() = 3 number of local element points, dPhi->at(0).at(0).size() = 2 as we have dim 2 therefore we have 2 derivatives (xi/eta in natural coordinates)
    //so phi is defined on the reference element

    SC detB;
    SC absDetB;
    SmallMatrix<SC> B(dim);
    SmallMatrix<SC> Binv(dim);
  
    buildTransformation(B);
    detB = B.computeInverse(Binv); // The function computeInverse returns a double value corrsponding to determinant of B
    //B.print();
    absDetB = std::fabs(detB); // absolute value of B


    // dPhiTrans are the transorfmed basisfunctions, so B^(-T) * \grad_phi bzw. \grad_phi^T * B^(-1)
    // Corresponds to \hat{grad_phi}.
    vec3D_dbl_Type dPhiTrans( dPhi->size(), vec2D_dbl_Type( dPhi->at(0).size(), vec_dbl_Type(dim,0.) ) );
    applyBTinv( dPhi, dPhiTrans, Binv ); // so dPhiTrans corresponds now to our basisfunction in natural coordinates
    //dPhiTrans.size() = 7 so number of quadrature points
    //dPhiTrans[0].size() = 3 number local element points
    //dPhiTrans[0][0].size() = 2 as we have in dim=2 case two derivatives




    TEUCHOS_TEST_FOR_EXCEPTION(dim == 1,std::logic_error, "AssemblyStress Not implemented for dim=1");
    //***************************************************************************
    //***************************************************************************
    if (dim == 2)
    {
   
    //************************************
    // Compute shear rate gammaDot, which is a vector because it is evaluated at a gaussian quadrature point 
    // for that compute velocity gradient
    vec_dbl_ptr_Type gammaDot(new vec_dbl_Type(weights->size(),0.0)); //gammaDot->at(j) j=0...weights 
    computeShearRate( dPhiTrans, gammaDot, dim); // updates gammaDot using velocity solution 
    //*******************************
    // Compute entries    
    // Initialize some helper vectors/matrices
    double v11, v12, v21, v22, value1_j, value2_j , value1_i, value2_i, viscosity_atw;
        SmallMatrix<double> e1i(dim);
        SmallMatrix<double> e2i(dim);
        SmallMatrix<double> e1j(dim);
        SmallMatrix<double> e2j(dim);

        viscosity_atw = 0.;
    
     

    // Construct element matrices 
    for (UN i=0; i < numNodes; i++) {
       // Teuchos::Array<SC> value(dPhiTrans[0].size(), 0. ); // dPhiTrans[0].size() is 3        
      
        for (UN j=0; j < numNodes; j++) {
        // Reset values
        v11 = 0.0;v12 = 0.0;v21 = 0.0;v22 = 0.0;

            // So in general compute the components of eta*[ dPhiTrans_i : ( dPhiTrans_j + (dPhiTrans_j)^T )]
            for (UN w=0; w<dPhiTrans.size(); w++) {

                 value1_j = dPhiTrans[w][j][0]; // so this corresponds to d\phi_j/dx
                 value2_j = dPhiTrans[w][j][1]; // so this corresponds to d\phi_j/dy

                 value1_i = dPhiTrans[w][i][0]; // so this corresponds to d\phi_i/dx
                 value2_i = dPhiTrans[w][i][1]; // so this corresponds to d\phi_i/dy

                // Now we compute the helper matrices which we need that we later can multiply the different values of dphi_j dphi_i together to get the single components of element matrice entry
                 e1j[0][0] = 2.*value1_j;
                 e1j[0][1] = value2_j;
                 e1j[1][0] = value2_j;
                 // e1j = [ 2 dphi_j/dx  ,  dphi_j/dy ; dphi_j/dy , 0]

                 e1i[0][0] = value1_i;
                 e1i[0][1] = value2_i;
                // e1i = [ dphi_i/dx  ,  dphi_i/dy ; 0 , 0]

                 e2j[1][0] = value1_j;
                 e2j[0][1] = value1_j;
                 e2j[1][1] = 2*value2_j;
                // e2j = [0 ,  dphi_j/dx ; dphi_j/dx , 2  dphi_j/dy ]

                 e2i[1][0] = value1_i;
                 e2i[1][1] = value2_i;
                // e2i = [ 0, 0 ; dphi_i/dx  ,  dphi_i/dy ]

                // viscosity function evaluated where we consider the dynamic viscosity!!
                this->materialModel->evaluateFunction(this->params_,  gammaDot->at(w), viscosity_atw);
                //if (i==0 && j==0) viscosity_averageOverT += viscosity_atw; // see below ### 

                // Construct entries - we go over all quadrature points and if j is updated we set v11 etc. again to zero
                 v11 = v11 + viscosity_atw * weights->at(w) * e1i.innerProduct(e1j); // xx contribution: 2 *dphi_i/dx *dphi_j/dx + dphi_i/dy* dphi_j/dy
                 v12 = v12 + viscosity_atw *  weights->at(w) * e1i.innerProduct(e2j); // xy contribution:  dphi_i/dy* dphi_j/dx
                 v21 = v21 + viscosity_atw *  weights->at(w) * e2i.innerProduct(e1j); // yx contribution:  dphi_i/dx* dphi_j/dy
                 v22 = v22 + viscosity_atw *  weights->at(w) * e2i.innerProduct(e2j); // yy contribution: 2 *dphi_i/dy *dphi_j/dy + dphi_i/dx* dphi_j/dx

            
            } // loop end quadrature points
            
            //multiply determinant from transformation
           // if (i==0 && j==0)  viscosity_averageOverT *= absDetB; 
            v11 *= absDetB; 
            v12 *= absDetB;
            v21 *= absDetB;
            v22 *= absDetB;
            
            // Put values on the right position in element matrix - d=2 because we are in two dimensional case
            // [v11  v12  ]
            // [v21  v22  ]
            (*elementMatrix)[i*dofs][j*dofs]   = v11; // d=0, first dimension
            (*elementMatrix)[i*dofs][j*dofs+1] = v12;
            (*elementMatrix)[i*dofs+1][j*dofs] = v21;
            (*elementMatrix)[i*dofs +1][j*dofs+1] =v22; //d=1, second dimension

        } // loop end over j node 

    } // loop end over i node


    } // end if dim 2
    //***************************************************************************
    else if (dim == 3)
    {
          //************************************#

    // Compute shear rate gammaDot, which is a vector because it is evaluated at a gaussian quadrature point 
    // for that compute velocity gradient
    vec_dbl_ptr_Type gammaDot(new vec_dbl_Type(weights->size(),0.0)); //gammaDot->at(j) j=0...weights 
    computeShearRate( dPhiTrans, gammaDot, dim); // updates gammaDot using velcoity solution 

         // Initialize some helper vectors/matrices
        double v11, v12, v13, v21, v22, v23, v31, v32, v33, value1_j, value2_j, value3_j , value1_i, value2_i, value3_i, viscosity_atw;
        SmallMatrix<double> e1i(dim);
        SmallMatrix<double> e2i(dim);
        SmallMatrix<double> e3i(dim);
        SmallMatrix<double> e1j(dim);
        SmallMatrix<double> e2j(dim);
        SmallMatrix<double> e3j(dim);

        viscosity_atw = 0.;

    // Construct element matrices 
     for (UN i=0; i < numNodes; i++) {
       // Teuchos::Array<SC> value(dPhiTrans[0].size(), 0. ); // dPhiTrans[0].size() is 3        
      
        for (UN j=0; j < numNodes; j++) {
        // Reset values
        v11 = 0.0;v12 = 0.0;v13=0.0; v21 = 0.0;v22 = 0.0;v23=0.0;v31=0.0;v32=0.0;v33=0.0;

            // So in general compute the components of eta*[ dPhiTrans_i : ( dPhiTrans_j + (dPhiTrans_j)^T )]
            for (UN w=0; w<dPhiTrans.size(); w++) {

                        value1_j = dPhiTrans.at(w).at(j).at(0); // so this corresponds to d\phi_j/dx
                        value2_j = dPhiTrans.at(w).at(j).at(1); // so this corresponds to d\phi_j/dy
                        value3_j = dPhiTrans.at(w).at(j).at(2); // so this corresponds to d\phi_j/dz


                        value1_i = dPhiTrans.at(w).at(i).at(0); // so this corresponds to d\phi_i/dx
                        value2_i = dPhiTrans.at(w).at(i).at(1); // so this corresponds to d\phi_i/dy
                        value3_i = dPhiTrans.at(w).at(i).at(2); // so this corresponds to d\phi_i/dz

                        // In an analagous way as in 2D
                        e1j[0][0] = 2.*value1_j;
                        e1j[0][1] = value2_j;
                        e1j[0][2] = value3_j;
                        e1j[1][0] = value2_j;
                        e1j[2][0] = value3_j;

                        e1i[0][0] = value1_i;
                        e1i[0][1] = value2_i;
                        e1i[0][2] = value3_i;


                        e2j[1][0] = value1_j;
                        e2j[1][1] = 2.*value2_j;
                        e2j[1][2] = value3_j;
                        e2j[0][1] = value1_j;
                        e2j[2][1] = value3_j;

                        e2i[1][0] = value1_i;
                        e2i[1][1] = value2_i;
                        e2i[1][2] = value3_i;

                        e3j[2][0] = value1_j;
                        e3j[2][1] = value2_j;
                        e3j[2][2] = 2.*value3_j;
                        e3j[0][2] = value1_j;
                        e3j[1][2] = value2_j;

                        e3i[2][0] = value1_i;
                        e3i[2][1] = value2_i;
                        e3i[2][2] = value3_i;

                       this->materialModel->evaluateFunction(this->params_,  gammaDot->at(w), viscosity_atw);

                    
                       // Construct entries - we go over all quadrature points and if j is updated we set v11 etc. again to zero
                        v11 = v11 + viscosity_atw *  weights->at(w) * e1i.innerProduct(e1j);
                        v12 = v12 + viscosity_atw *  weights->at(w) * e1i.innerProduct(e2j);
                        v13 = v13 + viscosity_atw *  weights->at(w) * e1i.innerProduct(e3j);

                        v21 = v21 + viscosity_atw *  weights->at(w) * e2i.innerProduct(e1j);
                        v22 = v22 + viscosity_atw *  weights->at(w) * e2i.innerProduct(e2j);
                        v23 = v23 + viscosity_atw *  weights->at(w) * e2i.innerProduct(e3j);

                        v31 = v31 + viscosity_atw *  weights->at(w) * e3i.innerProduct(e1j);
                        v32 = v32 + viscosity_atw *  weights->at(w) * e3i.innerProduct(e2j);
                        v33 = v33 + viscosity_atw *  weights->at(w) * e3i.innerProduct(e3j);

                    }// loop end quadrature points
            
                   //multiply determinant from transformation
                    v11 *= absDetB ;
                    v12 *= absDetB ;
                    v13 *= absDetB ;
                    v21 *= absDetB ;
                    v22 *= absDetB ;
                    v23 *= absDetB ;
                    v31 *= absDetB ;
                    v32 *= absDetB ;
                    v33 *= absDetB ;

                   // Put values on the right position in element matrix - d=2 because we are in two dimensional case
                   // [v11  v12  v13]
                   // [v21  v22  v23]
                   // [v31  v32  v33]
            (*elementMatrix)[i*dofs][j*dofs]   = v11; // d=0, first dimension
            (*elementMatrix)[i*dofs][j*dofs+1] = v12;
            (*elementMatrix)[i*dofs][j*dofs+2] = v13;
            (*elementMatrix)[i*dofs+1][j*dofs] = v21;
            (*elementMatrix)[i*dofs +1][j*dofs+1] =v22; //d=1, second dimension
            (*elementMatrix)[i*dofs +1][j*dofs+2] =v22; //d=1, second dimension
            (*elementMatrix)[i*dofs+2][j*dofs] = v31;
            (*elementMatrix)[i*dofs +2][j*dofs+1] =v32; //d=2, third dimension
            (*elementMatrix)[i*dofs +2][j*dofs+2] =v33; //d=2, third dimension

                }// loop end over j node 
            }// loop end over i node 
        }// end if dim==3

}





//Extra Derivative of Extra stress term resulting from chosen nonlinear non-newtonian model  -----
//Same structure and functions as in assemblyStress 
template <class SC, class LO, class GO, class NO>
void AssembleFENavierStokesNonNewtonian<SC,LO,GO,NO>::assemblyStressDev(SmallMatrixPtr_Type &elementMatrix) {

	int dim = this->getDim();
	int numNodes= this->numNodesVelocity_;
	int Grad =2; // Needs to be fixed	
	string FEType = this->FETypeVelocity_;
	int dofs = this->dofsVelocity_; // for pressure it would be 1 

    vec3D_dbl_ptr_Type 	dPhi;
    vec_dbl_ptr_Type weights = Teuchos::rcp(new vec_dbl_Type(0));
    
    UN deg = Helper::determineDegree(dim,FEType,Grad); 
    Helper::getDPhi(dPhi, weights, dim, FEType, deg); 
    

    SC detB;
    SC absDetB;
    SmallMatrix<SC> B(dim);
    SmallMatrix<SC> Binv(dim);
  
    buildTransformation(B);
    detB = B.computeInverse(Binv); 
    absDetB = std::fabs(detB); 

    vec3D_dbl_Type dPhiTrans( dPhi->size(), vec2D_dbl_Type( dPhi->at(0).size(), vec_dbl_Type(dim,0.) ) );
    applyBTinv( dPhi, dPhiTrans, Binv ); 

    if (dim == 2)
    {
    //************************************
    //************************************
    // Compute shear rate gammaDot, which is a vector because it is evaluated at a gaussian quadrature point 
    // for that compute velocity gradient
    // Due to the extra term related to the Gaetaeux-derivative there arise prefactors (a, b, fab) which depend on the velocity gradients 
    // which therefore also have to be computed here
    vec2D_dbl_Type u11(1, vec_dbl_Type(weights->size(), -1.)); // should correspond to du/dx
    vec2D_dbl_Type u12(1, vec_dbl_Type(weights->size(), -1.)); // should correspond to du/dy
    vec2D_dbl_Type u21(1, vec_dbl_Type(weights->size(), -1.)); // should correspond to dv/dx
    vec2D_dbl_Type u22(1, vec_dbl_Type(weights->size(), -1.)); // should correspond to dv/dy
    vec_dbl_ptr_Type 	gammaDot(new vec_dbl_Type(weights->size(),0.0)); //gammaDot->at(j) j=0...weights 

   vec_dbl_ptr_Type 	a1(new vec_dbl_Type(weights->size(),0.0)); // prefactor a= 4*(du/dx)^2 + (du/dy+dv/dx)^2
   vec_dbl_ptr_Type 	b1(new vec_dbl_Type(weights->size(),0.0)); // prefactor b= 4*(dv/dy)^2 + (du/dy+dv/dx)^2
   vec_dbl_ptr_Type 	fab1(new vec_dbl_Type(weights->size(),0.0)); // prefactor fab = 2*(du/dy+dv/dx)*(du/dx+dv/dy)
    for (UN w=0; w<dPhiTrans.size(); w++){ //quads points
      // set again to zero 
      u11[0][w] = 0.0;
      u12[0][w] = 0.0;
      u21[0][w] = 0.0;
      u22[0][w] = 0.0;
      
            for (UN i=0; i < dPhiTrans[0].size(); i++) {
                LO index1 = dim * i + 0; // x
                LO index2 = dim * i + 1; //y 
               // uLoc[d][w] += this->solution_[index] * phi->at(w).at(i); 
                u11[0][w] += this->solution_[index1] * dPhiTrans[w][i][0];
                u12[0][w] += this->solution_[index1] * dPhiTrans[w][i][1]; // because we are in 3D , 0 and 1 
                u21[0][w] += this->solution_[index2] * dPhiTrans[w][i][0];
                u22[0][w] += this->solution_[index2] * dPhiTrans[w][i][1];
                
            }
            gammaDot->at(w) = sqrt(2.0*u11[0][w]*u11[0][w]+ 2.0*u22[0][w]*u22[0][w] + (u12[0][w]+u21[0][w])*(u12[0][w]+u21[0][w]));

            a1->at(w)= 4.0*(u11[0][w]*u11[0][w]) + (u12[0][w]+u21[0][w])*(u12[0][w]+u21[0][w]);
            b1->at(w)= 4.0*(u22[0][w]*u22[0][w]) + (u12[0][w]+u21[0][w])*(u12[0][w]+u21[0][w]);
            fab1->at(w) = 2.0*(u12[0][w]+u21[0][w])*(u11[0][w]+u22[0][w]);

    }
    //*******************************
        
    // Initialize some helper vectors/matrices
    double v11, v12, v21, v22, value1_j, value2_j , value1_i, value2_i, deta_dgamma_dgamma_dtau ;
    
        SmallMatrix<double> tmpRes1(dim);
        SmallMatrix<double> tmpRes2(dim);
        SmallMatrix<double> e1i(dim);
        SmallMatrix<double> e2i(dim);
        SmallMatrix<double> e1j(dim);
        SmallMatrix<double> e2j(dim);
        deta_dgamma_dgamma_dtau =0.;

    // Construct element matrices 
    for (UN i=0; i < numNodes; i++) {
       // Teuchos::Array<SC> value(dPhiTrans[0].size(), 0. ); // dPhiTrans[0].size() is 3        
      
        for (UN j=0; j < numNodes; j++) {
        // Reset values
        v11 = 0.0;v12 = 0.0;v21 = 0.0;v22 = 0.0;

            // So in general compute the components of (dPhiTrans_i : [-1/4 * deta/dgammaDot * dgammaDot/dTau * (dv^k + (dvh^k)^T)^2 * ( dPhiTrans_j + (dPhiTrans_j)^T)    ]
            // Only the part  deta/dgammaDot is different for all shear thinning models (because we make the assumption of incompressibility)? NO ALSO DIFFERENT FOR EXAMPLE FOR CASSON SO CONSIDERING YIELD STRESS
            // but we put the two terms together because then we can multiply them together and get e.g. for carreau yasuda  : gammaDot^{a-2.0} which is for a=2.0 equals 0 and we do not have to worry about the problem what if gammaDot = 0.0
            for (UN w=0; w<dPhiTrans.size(); w++) {

                 value1_j = dPhiTrans[w][j][0]; // so this corresponds to d\phi_j/dx
                 value2_j = dPhiTrans[w][j][1]; // so this corresponds to d\phi_j/dy

                 value1_i = dPhiTrans[w][i][0]; // so this corresponds to d\phi_i/dx
                 value2_i = dPhiTrans[w][i][1]; // so this corresponds to d\phi_i/dy

                // Now we compute the helper matrices which we need that we later can multiply the different values of dphi_j dphi_i together to get the single components of element matrice entry
                 e1j[0][0] = 2.*value1_j;
                 e1j[0][1] = value2_j;
                 e1j[1][0] = value2_j;
                 // e1j = [ 2 dphi_j/dx  ,  dphi_j/dy ; dphi_j/dy , 0]

                // We get prefactors because due to the term  (dv^k + (dvh^k)^T)^2 
                 e1i[0][0] = value1_i*a1->at(w);
                 e1i[0][1] = value2_i*a1->at(w);
                 e1i[1][0] = value1_i*fab1->at(w);
                 e1i[1][1] = value2_i*fab1->at(w);
                // e1i = [ dphi_i/dx*a1  ,  dphi_i/dy*a1 ; dphi_i/dx*f , dphi_i/dy*f]

                e2j[1][0] = value1_j;
                e2j[0][1] = value1_j;
                e2j[1][1] = 2*value2_j;
                // e2j = [0 ,  dphi_j/dx ; dphi_j/dx , 2  dphi_j/dy ]

                 e2i[0][0] = value1_i*fab1->at(w);
                 e2i[0][1] = value2_i*fab1->at(w);
                 e2i[1][0] = value1_i*b1->at(w);
                 e2i[1][1] = value2_i*b1->at(w);
                // e2i = [  dphi_i/dx*f  , dphi_i/dy*f ; dphi_i/dx*b1  ,  dphi_i/dy*b1 ]


                this->materialModel->evaluateDerivative(this->params_,  gammaDot->at(w), deta_dgamma_dgamma_dtau);
	
                 v11 = v11 + (-0.25)*deta_dgamma_dgamma_dtau  * weights->at(w) * (e1i.innerProduct(e1j) ); // xx contribution: 2 *dphi_i/dx *dphi_j/dx*a1 + dphi_i/dy* dphi_j/dy*a1 + dphi_i/dx *dphi_j/dy*f
                 v12 = v12 + (-0.25)*deta_dgamma_dgamma_dtau  * weights->at(w) * (e1i.innerProduct(e2j) ); // xy contribution:  dphi_i/dy* dphi_j/dx*a1 + 2 *dphi_i/dy *dphi_j/dy*f + dphi_i/dx* dphi_j/dx*f
                 v21 = v21 + (-0.25)*deta_dgamma_dgamma_dtau  * weights->at(w) * (e2i.innerProduct(e1j) ); // yx contribution:  dphi_i/dx* dphi_j/dy*b1 + 2 *dphi_i/dx *dphi_j/dx*f + dphi_i/dy* dphi_j/dy*f
                 v22 = v22 + (-0.25)*deta_dgamma_dgamma_dtau  * weights->at(w) * (e2i.innerProduct(e2j) ); // yy contribution: 2 *dphi_i/dy *dphi_j/dy*b1 + dphi_i/dx* dphi_j/dx*b1 +  dphi_i/dy* dphi_j/dx*f

            } // loop end quadrature points
            
            //multiply determinant from transformation
            v11 *= absDetB; 
            v12 *= absDetB;
            v21 *= absDetB;
            v22 *= absDetB;
            
            // Put values on the right position in element matrix - d=2 because we are in two dimensional case
            // [v11  v12  ]
            // [v21  v22  ]
            (*elementMatrix)[i*dofs][j*dofs]   = v11; // d=0, first dimension
            (*elementMatrix)[i*dofs][j*dofs+1] = v12;
            (*elementMatrix)[i*dofs+1][j*dofs] = v21;
            (*elementMatrix)[i*dofs +1][j*dofs+1] =v22; //d=1, second dimension

        } // loop end over j node 

    } // loop end over i node

    } // end if dim 2
    //***************************************************************************
    //***************************************************************************
    else if (dim == 3)
    {
          //************************************
    // Compute shear rate gammaDot, which is a vector because it is evaluated at a gaussian quadrature point 
    // for that compute velocity gradient
    vec2D_dbl_Type u11(1, vec_dbl_Type(weights->size(), -1.)); // should correspond to du/dx at each quadrature point
    vec2D_dbl_Type u12(1, vec_dbl_Type(weights->size(), -1.)); // should correspond to du/dy at each quadrature point
    vec2D_dbl_Type u13(1, vec_dbl_Type(weights->size(), -1.)); // should correspond to du/dz at each quadrature point
    vec2D_dbl_Type u21(1, vec_dbl_Type(weights->size(), -1.)); // should correspond to dv/dx at each quadrature point
    vec2D_dbl_Type u22(1, vec_dbl_Type(weights->size(), -1.)); // should correspond to dv/dy at each quadrature point
    vec2D_dbl_Type u23(1, vec_dbl_Type(weights->size(), -1.)); // should correspond to dv/dz at each quadrature point
    vec2D_dbl_Type u31(1, vec_dbl_Type(weights->size(), -1.)); // should correspond to dw/dx at each quadrature point
    vec2D_dbl_Type u32(1, vec_dbl_Type(weights->size(), -1.)); // should correspond to dw/dy at each quadrature point
    vec2D_dbl_Type u33(1, vec_dbl_Type(weights->size(), -1.)); // should correspond to dw/dz at each quadrature point

    vec_dbl_ptr_Type gammaDot(new vec_dbl_Type(weights->size(),0.0)); //gammaDot->at(j) j=0...weights 

    vec_dbl_ptr_Type 	a1(new vec_dbl_Type(weights->size(),0.0)); // prefactor a= 4*(du/dx)^2 + (du/dy+dv/dx)^2 +  (dw/dx+du/dz)^2
    vec_dbl_ptr_Type 	b1(new vec_dbl_Type(weights->size(),0.0)); // prefactor b= 4*(dv/dy)^2 + (du/dy+dv/dx)^2 +  (dw/dy+dv/dz)^2
    vec_dbl_ptr_Type 	c1(new vec_dbl_Type(weights->size(),0.0)); // prefactor c= 4*(dw/dz)^2 + (dw/dx+du/dz)^2 +  (dw/dy+dv/dz)^2
    vec_dbl_ptr_Type 	d1(new vec_dbl_Type(weights->size(),0.0)); // prefactor d= 2*(du/dy+dv/dx)*(du/dx+dv/dy) + (dw/dx+du/dz)*(dw/dy+dv/dz)
    vec_dbl_ptr_Type 	e1(new vec_dbl_Type(weights->size(),0.0)); // prefactor e= 2*(dw/dx+du/dz)*(du/dx+dw/dz) + (du/dy+dv/dx)*(dw/dy+dv/dz)
    vec_dbl_ptr_Type 	g1(new vec_dbl_Type(weights->size(),0.0)); // prefactor g= 2*(dw/dy+dv/dz)*(dv/dy+dw/dz) + (du/dy+dv/dx)*(dw/dx+du/dz)
   
   for (UN w=0; w<dPhiTrans.size(); w++){ //quads points
   
      // set again to zero 
      u11[0][w] = 0.0;
      u12[0][w] = 0.0;
      u13[0][w] = 0.0;
      u21[0][w] = 0.0;
      u22[0][w] = 0.0;
      u23[0][w] = 0.0;
      u31[0][w] = 0.0;
      u32[0][w] = 0.0;
      u33[0][w] = 0.0;
      
            for (UN i=0; i < dPhiTrans[0].size(); i++) {
                LO index1 = dim * i + 0; //x
                LO index2 = dim * i + 1; //y 
                LO index3 = dim * i + 2; //z
               // uLoc[d][w] += this->solution_[index] * phi->at(w).at(i);
                u11[0][w] += this->solution_[index1] * dPhiTrans[w][i][0]; // u*dphi_dx
                u12[0][w] += this->solution_[index1] * dPhiTrans[w][i][1]; // because we are in 3D , 0 and 1, 2 
                u13[0][w] += this->solution_[index1] * dPhiTrans[w][i][2]; 
                u21[0][w] += this->solution_[index2] * dPhiTrans[w][i][0]; // v*dphi_dx
                u22[0][w] += this->solution_[index2] * dPhiTrans[w][i][1];
                u23[0][w] += this->solution_[index2] * dPhiTrans[w][i][2];
                u31[0][w] += this->solution_[index3] * dPhiTrans[w][i][0]; // w*dphi_dx
                u32[0][w] += this->solution_[index3] * dPhiTrans[w][i][1];
                u33[0][w] += this->solution_[index3] * dPhiTrans[w][i][2];
                
            }
            gammaDot->at(w) = sqrt(2.0*u11[0][w]*u11[0][w]+ 2.0*u22[0][w]*u22[0][w] + 2.0*u33[0][w]*u33[0][w] +  (u12[0][w]+u21[0][w])*(u12[0][w]+u21[0][w])   + (u13[0][w]+u31[0][w])*(u13[0][w]+u31[0][w]) + (u23[0][w]+u32[0][w])*(u23[0][w]+u32[0][w]) );
            a1->at(w)= 4.0*(u11[0][w]*u11[0][w]) + (u12[0][w]+u21[0][w])*(u12[0][w]+u21[0][w])+(u31[0][w]+u13[0][w])*(u31[0][w]+u13[0][w]);
            b1->at(w)= 4.0*(u22[0][w]*u22[0][w]) + (u12[0][w]+u21[0][w])*(u12[0][w]+u21[0][w])+(u32[0][w]+u23[0][w])*(u32[0][w]+u23[0][w]);
            c1->at(w)= 4.0*(u33[0][w]*u33[0][w]) + (u31[0][w]+u13[0][w])*(u31[0][w]+u13[0][w])+(u32[0][w]+u23[0][w])*(u32[0][w]+u23[0][w]);
            d1->at(w)= 2.0*(u12[0][w]+u21[0][w])*(u11[0][w]+u22[0][w])+(u31[0][w]+u13[0][w])*(u32[0][w]+u23[0][w]);
            e1->at(w)= 2.0*(u31[0][w]+u13[0][w])*(u11[0][w]+u33[0][w])+(u12[0][w]+u21[0][w])*(u32[0][w]+u23[0][w]);
            g1->at(w)= 2.0*(u32[0][w]+u23[0][w])*(u22[0][w]+u33[0][w])+(u12[0][w]+u21[0][w])*(u31[0][w]+u13[0][w]);


    }


      // Initialize some helper vectors/matrices
        double v11, v12, v13, v21, v22, v23, v31, v32, v33, value1_j, value2_j, value3_j , value1_i, value2_i, value3_i,deta_dgamma_dgamma_dtau;
        SmallMatrix<double> e1i(dim);
        SmallMatrix<double> e2i(dim);
        SmallMatrix<double> e3i(dim);
        SmallMatrix<double> e1j(dim);
        SmallMatrix<double> e2j(dim);
        SmallMatrix<double> e3j(dim);

         deta_dgamma_dgamma_dtau =0.;

    // Construct element matrices 
     for (UN i=0; i < numNodes; i++) {
       // Teuchos::Array<SC> value(dPhiTrans[0].size(), 0. ); // dPhiTrans[0].size() is 3        
      
        for (UN j=0; j < numNodes; j++) {
        // Reset values
        v11 = 0.0;v12 = 0.0;v13=0.0; v21 = 0.0;v22 = 0.0;v23=0.0;v31=0.0;v32=0.0;v33=0.0;

            // So in general compute the components of eta*[ dPhiTrans_i : ( dPhiTrans_j + (dPhiTrans_j)^T )]
            for (UN w=0; w<dPhiTrans.size(); w++) {

                        value1_j = dPhiTrans.at(w).at(j).at(0); // so this corresponds to d\phi_j/dx
                        value2_j = dPhiTrans.at(w).at(j).at(1); // so this corresponds to d\phi_j/dy
                        value3_j = dPhiTrans.at(w).at(j).at(2); // so this corresponds to d\phi_j/dz

                        value1_i = dPhiTrans.at(w).at(i).at(0); // so this corresponds to d\phi_i/dx
                        value2_i = dPhiTrans.at(w).at(i).at(1); // so this corresponds to d\phi_i/dy
                        value3_i = dPhiTrans.at(w).at(i).at(2); // so this corresponds to d\phi_i/dz

                        e1j[0][0] = 2.*value1_j;
                        e1j[0][1] = value2_j;
                        e1j[0][2] = value3_j;
                        e1j[1][0] = value2_j;
                        e1j[2][0] = value3_j;

                        e1i[0][0] = value1_i*a1->at(w);
                        e1i[0][1] = value2_i*a1->at(w);
                        e1i[0][2] = value3_i*a1->at(w);
                        e1i[1][0] = value1_i*d1->at(w);
                        e1i[1][1] = value2_i*d1->at(w);
                        e1i[1][2] = value3_i*d1->at(w);
                        e1i[2][0] = value1_i*e1->at(w);
                        e1i[2][1] = value2_i*e1->at(w);
                        e1i[2][2] = value3_i*e1->at(w);
                        
                        e2j[1][0] = value1_j;
                        e2j[1][1] = 2.*value2_j;
                        e2j[1][2] = value3_j;
                        e2j[0][1] = value1_j;
                        e2j[2][1] = value3_j;

                        e2i[0][0] = value1_i*d1->at(w);
                        e2i[0][1] = value2_i*d1->at(w);
                        e2i[0][2] = value3_i*d1->at(w);
                        e2i[1][0] = value1_i*b1->at(w);
                        e2i[1][1] = value2_i*b1->at(w);
                        e2i[1][2] = value3_i*b1->at(w);
                        e2i[2][0] = value1_i*g1->at(w);
                        e2i[2][1] = value2_i*g1->at(w);
                        e2i[2][2] = value3_i*g1->at(w);

                        e3j[2][0] = value1_j;
                        e3j[2][1] = value2_j;
                        e3j[2][2] = 2.*value3_j;
                        e3j[0][2] = value1_j;
                        e3j[1][2] = value2_j;

                        e3i[0][0] = value1_i*e1->at(w);
                        e3i[0][1] = value2_i*e1->at(w);
                        e3i[0][2] = value3_i*e1->at(w);
                        e3i[1][0] = value1_i*g1->at(w);
                        e3i[1][1] = value2_i*g1->at(w);
                        e3i[1][2] = value3_i*g1->at(w);
                        e3i[2][0] = value1_i*c1->at(w);
                        e3i[2][1] = value2_i*c1->at(w);
                        e3i[2][2] = value3_i*c1->at(w);

                       this->materialModel->evaluateDerivative(this->params_,  gammaDot->at(w), deta_dgamma_dgamma_dtau);
	
                       // Construct entries - we go over all quadrature points and if j is updated we set v11 etc. again to zero
                        v11 = v11 + (-0.25)*deta_dgamma_dgamma_dtau  *  weights->at(w) * e1i.innerProduct(e1j);
                        v12 = v12 + (-0.25)*deta_dgamma_dgamma_dtau  *  weights->at(w) * e1i.innerProduct(e2j);
                        v13 = v13 + (-0.25)*deta_dgamma_dgamma_dtau  *  weights->at(w) * e1i.innerProduct(e3j);

                        v21 = v21 + (-0.25)*deta_dgamma_dgamma_dtau  *  weights->at(w) * e2i.innerProduct(e1j);
                        v22 = v22 + (-0.25)*deta_dgamma_dgamma_dtau  *  weights->at(w) * e2i.innerProduct(e2j);
                        v23 = v23 + (-0.25)*deta_dgamma_dgamma_dtau  *  weights->at(w) * e2i.innerProduct(e3j);

                        v31 = v31 + (-0.25)*deta_dgamma_dgamma_dtau  *  weights->at(w) * e3i.innerProduct(e1j);
                        v32 = v32 + (-0.25)*deta_dgamma_dgamma_dtau  *  weights->at(w) * e3i.innerProduct(e2j);
                        v33 = v33 + (-0.25)*deta_dgamma_dgamma_dtau  *  weights->at(w) * e3i.innerProduct(e3j);

                    }// loop end quadrature points
            
                   //multiply determinant from transformation
                    v11 *= absDetB ;
                    v12 *= absDetB ;
                    v13 *= absDetB ;
                    v21 *= absDetB ;
                    v22 *= absDetB ;
                    v23 *= absDetB ;
                    v31 *= absDetB ;
                    v32 *= absDetB ;
                    v33 *= absDetB ;

                   // Put values on the right position in element matrix - d=2 because we are in two dimensional case
                   // [v11  v12  v13]
                   // [v21  v22  v23]
                   // [v31  v32  v33]
            (*elementMatrix)[i*dofs][j*dofs]      = v11; // d=0, first dimension
            (*elementMatrix)[i*dofs][j*dofs+1]    = v12;
            (*elementMatrix)[i*dofs][j*dofs+2]    = v13;
            (*elementMatrix)[i*dofs+1][j*dofs]    = v21;
            (*elementMatrix)[i*dofs +1][j*dofs+1] = v22; //d=1, second dimension
            (*elementMatrix)[i*dofs +1][j*dofs+2] = v22; //d=1, second dimension
            (*elementMatrix)[i*dofs+2][j*dofs]    = v31;
            (*elementMatrix)[i*dofs +2][j*dofs+1] = v32; //d=2, third dimension
            (*elementMatrix)[i*dofs +2][j*dofs+2] = v33; //d=2, third dimension

                }// loop end over j node 
            }// loop end over i node 

    }// end if dim = 3

}



// "Fixpunkt"- Matrix without jacobian for calculating Ax 
// Here update please to unlinearized System Matrix accordingly.
template <class SC, class LO, class GO, class NO>
void AssembleFENavierStokesNonNewtonian<SC,LO,GO,NO>::assembleRHS(){

	SmallMatrixPtr_Type elementMatrixN =Teuchos::rcp( new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_));

	this->ANB_.reset(new SmallMatrix_Type( this->dofsElementVelocity_+this->numNodesPressure_)); // A + B + N
	this->ANB_->add( (*this->constantMatrix_),(*this->ANB_));

	this->assemblyStress(elementMatrixN);
	//elementMatrixN->scale(-1.0);
	this->ANB_->add( (*elementMatrixN),(*this->ANB_));

	this->rhsVec_ = vec_dbl_Type(this->dofsElement_,0);
	// Multiplying ANB_ * solution // System Matrix times solution
	int s=0,t=0;
	for(int i=0 ; i< this->ANB_->size();i++){
		if (i >= this->dofsElementVelocity_)
			s=1;
		for(int j=0; j < this->ANB_->size(); j++){
			if(j >= this->dofsElementVelocity_)
				t=1;
			this->rhsVec_[i] += (*this->ANB_)[i][j]*this->solution_[j]*this->coeff_[s][t];
			//cout <<"Solution["<<j <<"]" << this->solution_[i] << endl;
		}
		t=0;
		//cout <<"RHS["<<i <<"]" << this->rhsVec_[i] << endl;
	}
}


/*!

 \brief Building Transformation
@param[in] &B
*/
/* In 2D B=[ nodesRefConfig(1).at(0)-nodesRefConfig(0).at(0)    nodesRefConfig(2).at(0)-nodesRefConfig(0).at(0)    ]
           [ nodesRefConfig(1).at(1)-nodesRefConfig(0).at(1)    nodesRefConfig(2).at(1)-nodesRefConfig(0).at(1)   ]
/*!
    - Triangle numbering

                    2
                  * *
                *   *
              4	    5
            *       *
          *         *
        1 * * 3 * * 0

    In 2D B=[ 1(x)-0(x)      2(x)-0(x)  ]
            [ 1(y)-0(y)    2(y)-0(y)]
*/
template <class SC, class LO, class GO, class NO>
void AssembleFENavierStokesNonNewtonian<SC,LO,GO,NO>::buildTransformation(SmallMatrix<SC>& B){

    TEUCHOS_TEST_FOR_EXCEPTION( (B.size()<2 || B.size()>3), std::logic_error, "Initialize SmallMatrix for transformation.");
    UN index;
    UN index0 = 0;
    for (UN j=0; j<B.size(); j++) {
        index = j+1;
        for (UN i=0; i<B.size(); i++) {
            B[i][j] = this->nodesRefConfig_.at(index).at(i) - this->nodesRefConfig_.at(index0).at(i);
        }
    }

}

template <class SC, class LO, class GO, class NO>
void AssembleFENavierStokesNonNewtonian<SC,LO,GO,NO>::applyBTinv( vec3D_dbl_ptr_Type& dPhiIn,
                                    vec3D_dbl_Type& dPhiOut,
                                    SmallMatrix<SC>& Binv){
    UN dim = Binv.size();
    for (UN w=0; w<dPhiIn->size(); w++){  // quadrature point
        for (UN i=0; i < dPhiIn->at(w).size(); i++) { // basisfunction iterator 
            for (UN d1=0; d1<dim; d1++) {   // dimension of problem
                for (UN d2=0; d2<dim; d2++) { // dimension of problem
                    dPhiOut[w][i][d1] += dPhiIn->at(w).at(i).at(d2) * Binv[d2][d1];
                }
            }
        }
    }
}


// Compute Shear Rate dependent on quadrature points
template <class SC, class LO, class GO, class NO>
void AssembleFENavierStokesNonNewtonian<SC,LO,GO,NO>::computeShearRate(  vec3D_dbl_Type dPhiTrans,
                                    vec_dbl_ptr_Type& gammaDot, int dim){

//****************** TWO DIMENSIONAL *********************************
    if (dim == 2)    
    {                           
    vec2D_dbl_Type u11(1, vec_dbl_Type(dPhiTrans.size(), -1.)); // should correspond to du/dx at each quadrature point
    vec2D_dbl_Type u12(1, vec_dbl_Type(dPhiTrans.size(), -1.)); // should correspond to du/dy at each quadrature point
    vec2D_dbl_Type u21(1, vec_dbl_Type(dPhiTrans.size(), -1.)); // should correspond to dv/dx at each quadrature point
    vec2D_dbl_Type u22(1, vec_dbl_Type(dPhiTrans.size(), -1.)); // should correspond to dv/dy at each quadrature point
    for (UN w=0; w<dPhiTrans.size(); w++){ //quads points
      // set again to zero 
      u11[0][w] = 0.0;
      u12[0][w] = 0.0;
      u21[0][w] = 0.0;
      u22[0][w] = 0.0;
      
            for (UN i=0; i < dPhiTrans[0].size(); i++) {
                LO index1 = dim * i + 0; // x
                LO index2 = dim * i + 1; //y 
               // uLoc[d][w] += this->solution_[index] * phi->at(w).at(i);
                u11[0][w] += this->solution_[index1] * dPhiTrans[w][i][0]; // u*dphi_dx
                u12[0][w] += this->solution_[index1] * dPhiTrans[w][i][1]; // because we are in 2D , 0 and 1 
                u21[0][w] += this->solution_[index2] * dPhiTrans[w][i][0];
                u22[0][w] += this->solution_[index2] * dPhiTrans[w][i][1];
                
            }
            gammaDot->at(w) = sqrt(2.0*u11[0][w]*u11[0][w]+ 2.0*u22[0][w]*u22[0][w] + (u12[0][w]+u21[0][w])*(u12[0][w]+u21[0][w]));
    }
    }// end if dim == 2
    //****************** THREE DIMENSIONAL *********************************
    else if (dim == 3)
    {
    vec2D_dbl_Type u11(1, vec_dbl_Type(dPhiTrans.size(), -1.)); // should correspond to du/dx at each quadrature point
    vec2D_dbl_Type u12(1, vec_dbl_Type(dPhiTrans.size(), -1.)); // should correspond to du/dy at each quadrature point
    vec2D_dbl_Type u13(1, vec_dbl_Type(dPhiTrans.size(), -1.)); // should correspond to du/dz at each quadrature point
    vec2D_dbl_Type u21(1, vec_dbl_Type(dPhiTrans.size(), -1.)); // should correspond to dv/dx at each quadrature point
    vec2D_dbl_Type u22(1, vec_dbl_Type(dPhiTrans.size(), -1.)); // should correspond to dv/dy at each quadrature point
    vec2D_dbl_Type u23(1, vec_dbl_Type(dPhiTrans.size(), -1.)); // should correspond to dv/dz at each quadrature point
    vec2D_dbl_Type u31(1, vec_dbl_Type(dPhiTrans.size(), -1.)); // should correspond to dw/dx at each quadrature point
    vec2D_dbl_Type u32(1, vec_dbl_Type(dPhiTrans.size(), -1.)); // should correspond to dw/dy at each quadrature point
    vec2D_dbl_Type u33(1, vec_dbl_Type(dPhiTrans.size(), -1.)); // should correspond to dw/dz at each quadrature point

    for (UN w=0; w<dPhiTrans.size(); w++){ //quads points
      // set again to zero 
      u11[0][w] = 0.0;
      u12[0][w] = 0.0;
      u13[0][w] = 0.0;
      u21[0][w] = 0.0;
      u22[0][w] = 0.0;
      u23[0][w] = 0.0;
      u31[0][w] = 0.0;
      u32[0][w] = 0.0;
      u33[0][w] = 0.0;
      
            for (UN i=0; i < dPhiTrans[0].size(); i++) {
                LO index1 = dim * i + 0; //x
                LO index2 = dim * i + 1; //y 
                LO index3 = dim * i + 2; //z
               // uLoc[d][w] += this->solution_[index] * phi->at(w).at(i);
                u11[0][w] += this->solution_[index1] * dPhiTrans[w][i][0]; // u*dphi_dx
                u12[0][w] += this->solution_[index1] * dPhiTrans[w][i][1]; // because we are in 3D , 0 and 1, 2 
                u13[0][w] += this->solution_[index1] * dPhiTrans[w][i][2]; 
                u21[0][w] += this->solution_[index2] * dPhiTrans[w][i][0]; // v*dphi_dx
                u22[0][w] += this->solution_[index2] * dPhiTrans[w][i][1];
                u23[0][w] += this->solution_[index2] * dPhiTrans[w][i][2];
                u31[0][w] += this->solution_[index3] * dPhiTrans[w][i][0]; // w*dphi_dx
                u32[0][w] += this->solution_[index3] * dPhiTrans[w][i][1];
                u33[0][w] += this->solution_[index3] * dPhiTrans[w][i][2];
                
            }
            gammaDot->at(w) = sqrt(2.0*u11[0][w]*u11[0][w]+ 2.0*u22[0][w]*u22[0][w] + 2.0*u33[0][w]*u33[0][w] +  (u12[0][w]+u21[0][w])*(u12[0][w]+u21[0][w])   + (u13[0][w]+u31[0][w])*(u13[0][w]+u31[0][w]) + (u23[0][w]+u32[0][w])*(u23[0][w]+u32[0][w]) );
    }
    } // end if dim == 3
    

}





// So based on the previous solution we can compute viscosity at the nodal values
// So what we need are the coordinates of an local reference element which are always the same
// DOES WORK

template <class SC, class LO, class GO, class NO>
void AssembleFENavierStokesNonNewtonian<SC,LO,GO,NO>::computeLocalViscosity_AtNodes()
{
	int dim = this->getDim();
	int numNodes= this->numNodesVelocity_;
	int Grad =2; // Needs to be fixed	
	string FEType = this->FETypeVelocity_;
	int dofs = this->dofsVelocity_; // for pressure it would be 1 
   
    SC detB;
    SC absDetB;
    SmallMatrix<SC> B(dim);
    SmallMatrix<SC> Binv(dim);
  
    buildTransformation(B);
    detB = B.computeInverse(Binv); 
    absDetB = std::fabs(detB); 

    vec3D_dbl_ptr_Type 	dPhiAtNodes;
    vec2D_dbl_Type RefNodesCoor(numNodes,  vec_dbl_Type(dim,0.) ); 

    Helper::getCoorNodesEle(RefNodesCoor, dim, FEType );
    Helper::getDPhiAtNodes(dPhiAtNodes, dim, FEType, RefNodesCoor); // these are the original coordinates of the element
    vec3D_dbl_Type dPhiTransAtNodes( dPhiAtNodes->size(), vec2D_dbl_Type( dPhiAtNodes->at(0).size(), vec_dbl_Type(dim,0.) ) );
    applyBTinv( dPhiAtNodes, dPhiTransAtNodes, Binv ); 

    vec_dbl_ptr_Type gammaDoti(new vec_dbl_Type( dPhiAtNodes->size(),0.0)); //gammaDot->at(i) i=0...number Nodal values
    computeShearRate(dPhiTransAtNodes, gammaDoti, dim); // updates gammaDot using velcoity solution 
     for (UN i=0; i<dPhiAtNodes->size(); i++){ // for each node
        this->materialModel->evaluateFunction(this->params_,  gammaDoti->at(i), this->solutionViscosity_.at(i));
    }
   

}

// So based on the previous solution we can compute viscosity at the nodal values
// So what we need are the coordinates of an local reference element which are always the same

template <class SC, class LO, class GO, class NO>
void AssembleFENavierStokesNonNewtonian<SC,LO,GO,NO>::computeLocalViscosity()
{
	int dim = this->getDim();
	int numNodes= this->numNodesVelocity_;
	int Grad =2; // Needs to be fixed	
	string FEType = this->FETypeVelocity_;
	int dofs = this->dofsVelocity_; // for pressure it would be 1 
   
    SC detB;
    SC absDetB;
    SmallMatrix<SC> B(dim);
    SmallMatrix<SC> Binv(dim);
  
    buildTransformation(B);
    detB = B.computeInverse(Binv); 
    absDetB = std::fabs(detB); 

    vec3D_dbl_ptr_Type 	dPhiAtCM;
    vec_dbl_Type 	CM(dim,0.0); // center of mass - so we want to compute the viscosity value in the middle of the element
    if (dim == 2) // center of mass of reference triangle (xi-eta coordinates)
    {
    CM[0]=1.0/3.0;
    CM[1]=1.0/3.0;
    }
    else if(dim == 3) // center of mass of reference tetrahedor (xi - eta - omega coordinates)
    {
    CM[0]=1.0/4.0;
    CM[1]=1.0/4.0;
    CM[2]=1.0/4.0;
    }

    Helper::getDPhiAtCM(dPhiAtCM, dim, FEType, CM); // these are the original coordinates of the reference element
    vec3D_dbl_Type dPhiTransAtCM( dPhiAtCM->size(), vec2D_dbl_Type( dPhiAtCM->at(0).size(), vec_dbl_Type(dim,0.) ) );
    applyBTinv( dPhiAtCM, dPhiTransAtCM, Binv ); 

    vec_dbl_ptr_Type gammaDoti(new vec_dbl_Type( dPhiAtCM->size(),0.0)); //only one value because size is one
    computeShearRate(dPhiTransAtCM, gammaDoti, dim); // updates gammaDot using velcoity solution 
    this->materialModel->evaluateFunction(this->params_,  gammaDoti->at(0), this->solutionViscosity_.at(0));
    

}

}
#endif

