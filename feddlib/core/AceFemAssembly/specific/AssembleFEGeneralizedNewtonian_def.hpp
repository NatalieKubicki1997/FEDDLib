#ifndef AssembleFEGeneralizedNewtonian_DEF_hpp
#define AssembleFEGeneralizedNewtonian_DEF_hpp

#include "AssembleFENavierStokes_decl.hpp"

namespace FEDD
{
    // All important things are so far defined in AssembleFENavierStokes. Please check there.
    /* Interesting paper to generalized Newtonian fluids
    @article{Poole2023,
    author = {Poole, Robert J.},
    doi = {10.1016/j.jnnfm.2023.105106},
    issn = {03770257},
    journal = {Journal of Non-Newtonian Fluid Mechanics},
    keywords = {Constitutive modelling,Flow-type,Generalised Newtonian fluids,Inelastic},
    number = {August},
    pages = {105106},
    publisher = {Elsevier B.V.},
    title = {{Inelastic and flow-type parameter models for non-Newtonian fluids}},
    url = {https://doi.org/10.1016/j.jnnfm.2023.105106},
    volume = {320},
    year = {2023}
    }

    */

    /* *******************************************************************************
    This class is for Generalized-Newtonian fluids, where we consider that the viscosity is non-constant. Because the viscosity is no longer constant, the conventional formulation with the Laplacian term cannot be considered 
    (although there is a generalized Laplacian version of the equation, see "On outflow boundary conditions in finite element simulations of non-Newtonian internal flow" 2021). 
    Instead, we use the stress-divergence formulation of the momentum equation and derive from that the element-wise entrie
    ******************************************************************************* */
    template <class SC, class LO, class GO, class NO>
    AssembleFEGeneralizedNewtonian<SC, LO, GO, NO>::AssembleFEGeneralizedNewtonian(int flag, vec2D_dbl_Type nodesRefConfig, ParameterListPtr_Type params, tuple_disk_vec_ptr_Type tuple) : AssembleFENavierStokes<SC, LO, GO, NO>(flag, nodesRefConfig, params, tuple)
    {

        ////******************* If we have an analytical formula we could also just use Paraview Postprocessing tools to compute the viscosity **********************************
        dofsElementViscosity_ = this->dofsPressure_ * this->numNodesVelocity_; // So it is a scalar quantity but as it depend on the velocity it is defined at the nodes of the velocity
        this->constOutputField_ = vec_dbl_Type(dofsElementViscosity_);         ////**********************************************************************************

        // Reading through parameterlist
        shearThinningModel = params->sublist("Material").get("ShearThinningModel", "");
        // New: We have to check which material model we use
        if (shearThinningModel == "Carreau-Yasuda")
        {
            Teuchos::RCP<CarreauYasuda<SC, LO, GO, NO>> viscosityModelSpecific(new CarreauYasuda<SC, LO, GO, NO>(params));
            viscosityModel = viscosityModelSpecific;
        }
        else if (shearThinningModel == "Power-Law")
        {
            Teuchos::RCP<PowerLaw<SC, LO, GO, NO>> viscosityModelSpecific(new PowerLaw<SC, LO, GO, NO>(params));
            viscosityModel = viscosityModelSpecific;
        }
        else if (shearThinningModel == "Dimless-Carreau")
        {
            Teuchos::RCP<Dimless_Carreau<SC, LO, GO, NO>> viscosityModelSpecific(new Dimless_Carreau<SC, LO, GO, NO>(params));
            viscosityModel = viscosityModelSpecific;
        }
        else
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "No specific implementation for your material model request. Valid are:Carreau-Yasuda, Power-Law, Dimless-Carreau");

    }

    template <class SC, class LO, class GO, class NO>
    void AssembleFEGeneralizedNewtonian<SC, LO, GO, NO>::assembleJacobian()
    {

        // For nonlinear generalized newtonian stress tensor part
        SmallMatrixPtr_Type elementMatrixN = Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));
        SmallMatrixPtr_Type elementMatrixW = Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));

        // For nonlinear convection
        SmallMatrixPtr_Type elementMatrixNC = Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));
        SmallMatrixPtr_Type elementMatrixWC = Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));

        // In the first iteration step we initialize the constant matrices
        // So in the case of a newtonian fluid we would have the matrix A with the contributions of the Laplacian term
        // and the matrix B with the mixed-pressure terms. Latter one exists also in the generlized-newtonian case
        if (this->newtonStep_ == 0)
        {
            SmallMatrixPtr_Type elementMatrixA = Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));
            SmallMatrixPtr_Type elementMatrixB = Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));

            this->constantMatrix_.reset(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));
            // Construct the matrix B from FE formulation - as it is equivalent to Newtonian case we call the same function
            this->assemblyDivAndDivT(elementMatrixB); // For Matrix B
            elementMatrixB->scale(-1.);
            this->constantMatrix_->add((*elementMatrixB), (*this->constantMatrix_));
        }

        // The other element matrices are not constant so we have to update them in each step
        // As the stress tensor term, considering a generalized-newtonian constitutive equation, is nonlinear we add its contribution here

        // ANB is the FixedPoint Formulation which was named for newtonian fluids.
        // Matrix A (Laplacian term (here not occuring)), Matrix B for div-Pressure Part, Matrix N for nonlinear parts -

        this->ANB_.reset(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_)); // A + B + N
        this->ANB_->add(((*this->constantMatrix_)), ((*this->ANB_)));

        // Nonlinear advection term \rho (u \cdot \nabla) u
        // As this class is derived from NavierStokes class we can call already implemented function
        //*************** ADVECTION************************
        this->assemblyAdvection(elementMatrixNC);
        elementMatrixNC->scale(this->density_);
        this->ANB_->add((*elementMatrixNC), ((*this->ANB_)));

        // For a generalized-newtonian fluid we add additional element matrix and fill it with specific contribution
        // Remember that this term is based on the stress-divergence formulation of the momentum equation
        // \nabla \dot \tau with \tau=\eta(\gammaDot)(\nabla u + (\nabla u)^T)
        //*************** STRESS TENSOR************************
        this->assemblyStress(elementMatrixN);
        this->ANB_->add((*elementMatrixN), ((*this->ANB_)));

        this->jacobian_.reset(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));
        this->jacobian_->add((*this->ANB_), (*this->jacobian_));

        // If linearization is not FixdPoint (so NOX or Newton) we add the derivative to the Jacobian matrix. Otherwise the FixedPoint formulation becomes the jacobian.
        if (this->linearization_ != "FixedPoint")
        {

            this->assemblyStressDev(elementMatrixW);     // shear stress tensor
            this->assemblyAdvectionInU(elementMatrixWC); // convection
            elementMatrixWC->scale(this->density_);

            this->jacobian_->add((*elementMatrixW), (*this->jacobian_));
            this->jacobian_->add((*elementMatrixWC), (*this->jacobian_)); // int add(SmallMatrix<T> &bMat, SmallMatrix<T> &cMat); //this+B=C elementMatrix + constantMatrix_;
        }

        //*************** BOUNDARY TERM *******************************
        /* Because we have stress-divergence form of Navier-Stokes equations in the non-newtonian case
         we have to add a extra boundary term to get the same outflow boundary condition as in the conventional formulation with
         the laplacian operator in the equations due to the fact that in the stress-divergence formulation the
         natural boundary condition is different
         We have to check whether it is an element which has edges (2D) / surfaces (3D) corresponding to an Outflow Neumann boundary
         Then we have to compute contribution
         !! Side fact: If we have convection dominant flow the effects get less important there are more important for shear flows so Stokes flow
         */ 
        if (this->FEObject_->getNeumannBCElement() == true) // Our corresponding FE Elements corresponds to element on the outer boundary where we want to assign a special outflow boundary condition
        {
            // @ToDo: If we have different flags, so i.e. we want to assigne different boundary conditions on different flags we should check here for the case that we are an element in the corner with two corrsponding boundaries
            SmallMatrixPtr_Type elementMatrixNB = Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));
            this->assemblyOutflowNeumannBoundaryTerm(elementMatrixNB);
            this->ANB_->add((*elementMatrixNB), ((*this->ANB_)));

            // Newton converges also if unabled
            // If linearization is not FixdPoint (so NOX or Newton) we add the derivative to the Jacobian matrix. Otherwise the FixedPoint formulation becomes the jacobian.
            if (this->linearization_ != "FixedPoint")
            {
                SmallMatrixPtr_Type elementMatrixNBW = Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));
                this->assemblyOutflowNeumannBoundaryTermDev(elementMatrixNBW); //
                this->jacobian_->add((*elementMatrixNBW), (*this->jacobian_));
            }
        }
    }



    template <class SC, class LO, class GO, class NO>
    void AssembleFEGeneralizedNewtonian<SC,LO,GO,NO>::assembleFixedPoint() {

        SmallMatrixPtr_Type elementMatrixN = Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));
        SmallMatrixPtr_Type elementMatrixNC = Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));

	    if(this->newtonStep_ ==0){
            SmallMatrixPtr_Type elementMatrixB = Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));

            this->constantMatrix_.reset(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));
            // Construct the matrix B from FE formulation - as it is equivalent to Newtonian case we call the same function
            this->assemblyDivAndDivT(elementMatrixB); // For Matrix B
            elementMatrixB->scale(-1.);
            this->constantMatrix_->add((*elementMatrixB), (*this->constantMatrix_));
        }

        this->ANB_.reset(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_)); // A + B + N
        this->ANB_->add(((*this->constantMatrix_)), ((*this->ANB_)));

        // Nonlinear advection term \rho (u \cdot \nabla) u
        // As this class is derived from NavierStokes class we can call already implemented function
        //*************** ADVECTION************************
        this->assemblyAdvection(elementMatrixNC);
        elementMatrixNC->scale(this->density_);
        this->ANB_->add((*elementMatrixNC), ((*this->ANB_)));

        // For a generalized-newtonian fluid we add additional element matrix and fill it with specific contribution
        // Remember that this term is based on the stress-divergence formulation of the momentum equation
        // \nabla \dot \tau with \tau=\eta(\gammaDot)(\nabla u + (\nabla u)^T)
        //*************** STRESS TENSOR************************
        this->assemblyStress(elementMatrixN);
        this->ANB_->add((*elementMatrixN), ((*this->ANB_)));

        if (this->FEObject_->getNeumannBCElement() == true) // Our corresponding FE Elements corresponds to element on the outer boundary where we want to assign a special outflow boundary condition
        {   // @ToDo: If we have different flags, so i.e. we want to assigne different boundary conditions on different flags we should check here for the case that we are an element in the corner with two corrsponding boundaries
            SmallMatrixPtr_Type elementMatrixNB = Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));
            this->assemblyOutflowNeumannBoundaryTerm(elementMatrixNB);
            this->ANB_->add((*elementMatrixNB), ((*this->ANB_)));

        }

}

    // Extra stress term resulting from chosen non-newtonian constitutive model  - Compute element matrix entries
    template <class SC, class LO, class GO, class NO>
    void AssembleFEGeneralizedNewtonian<SC, LO, GO, NO>::assemblyStress(SmallMatrixPtr_Type &elementMatrix)
    {

        int dim = this->getDim();
        int numNodes = this->numNodesVelocity_;
        UN Grad = 1; // Needs to be fixed before 2
        string FEType = this->FETypeVelocity_;
        int dofs = this->dofsVelocity_; // For pressure it would be 1

        vec3D_dbl_ptr_Type dPhi;
        vec_dbl_ptr_Type weights = Teuchos::rcp(new vec_dbl_Type(0));

        UN deg = Helper::determineDegree(dim, FEType, Grad); //  e.g. for P1 3
        Helper::getDPhi(dPhi, weights, dim, FEType, deg);    //  e.g. for deg 5 we get weight vector with 7 entries
        // Example Values: dPhi->size() = 7 so number of quadrature points, dPhi->at(0).size() = 3 number of local element points, dPhi->at(0).at(0).size() = 2 as we have dim 2 therefore we have 2 derivatives (xi/eta in natural coordinates)
        // Phi is defined on reference element

        SC detB;
        SC absDetB;
        SmallMatrix<SC> B(dim);
        SmallMatrix<SC> Binv(dim);

        this->buildTransformation(B);
        detB = B.computeInverse(Binv); // The function computeInverse returns a double value corrsponding to determinant of B   
        absDetB = std::fabs(detB);     // Absolute value of B

        // dPhiTrans are the transorfmed basisfunctions, so B^(-T) * \grad_phi bzw. \grad_phi^T * B^(-1)
        // Corresponds to \hat{grad_phi}.
        vec3D_dbl_Type dPhiTrans(dPhi->size(), vec2D_dbl_Type(dPhi->at(0).size(), vec_dbl_Type(dim, 0.)));
        Helper::applyBTinv(dPhi, dPhiTrans, Binv); // dPhiTrans corresponds now to our basisfunction in natural coordinates

        TEUCHOS_TEST_FOR_EXCEPTION(dim == 1, std::logic_error, "AssemblyStress Not implemented for dim=1");
        //***************************************************************************
        if (dim == 2)
        {
            //************************************
            // Compute shear rate gammaDot, which is a vector because it is evaluated at each gaussian quadrature point
            // for that first compute velocity gradient
            vec_dbl_ptr_Type gammaDot(new vec_dbl_Type(weights->size(), 0.0)); // gammaDot->at(j) j=0...weights
            computeShearRate(dPhiTrans, gammaDot, dim);                        // updates gammaDot using velocity solution
            //************************************
            // Compute entries
            // Initialize some helper vectors/matrices
            double v11, v12, v21, v22, value1_j, value2_j, value1_i, value2_i, viscosity_atw;
            viscosity_atw = 0.;

            // Construct element matrices
            for (UN i = 0; i < numNodes; i++)
            {
                // Teuchos::Array<SC> value(dPhiTrans[0].size(), 0. ); // dPhiTrans[0].size() is 3
                for (UN j = 0; j < numNodes; j++)
                {
                    // Reset values
                    v11 = 0.0;
                    v12 = 0.0;
                    v21 = 0.0;
                    v22 = 0.0;

                    // So in general compute the components of eta*[ dPhiTrans_i : ( dPhiTrans_j + (dPhiTrans_j)^T )]
                    for (UN w = 0; w < dPhiTrans.size(); w++)
                    {

                        value1_j = dPhiTrans[w][j][0]; // so this corresponds to d\phi_j/dx
                        value2_j = dPhiTrans[w][j][1]; // so this corresponds to d\phi_j/dy

                        value1_i = dPhiTrans[w][i][0]; // so this corresponds to d\phi_i/dx
                        value2_i = dPhiTrans[w][i][1]; // so this corresponds to d\phi_i/dy

                        // viscosity function evaluated where we consider the dynamic viscosity!!
                        this->viscosityModel->evaluateMapping(this->params_, gammaDot->at(w), viscosity_atw);

                        v11 = v11 + viscosity_atw * weights->at(w) * (2.0 * value1_i * value1_j + value2_i * value2_j);
                        v12 = v12 + viscosity_atw * weights->at(w) * (value2_i * value1_j);
                        v21 = v21 + viscosity_atw * weights->at(w) * (value1_i * value2_j);
                        v22 = v22 + viscosity_atw * weights->at(w) * (2.0 * value2_i * value2_j + value1_i * value1_j);

                    } // loop end quadrature points

                    // multiply determinant from transformation
                    v11 *= absDetB;
                    v12 *= absDetB;
                    v21 *= absDetB;
                    v22 *= absDetB;

                    // Put values on the right position in element matrix - d=2 because we are in two dimensional case
                    // [v11  v12  ]
                    // [v21  v22  ]
                    (*elementMatrix)[i * dofs][j * dofs] = v11; // d=0, first dimension
                    (*elementMatrix)[i * dofs][j * dofs + 1] = v12;
                    (*elementMatrix)[i * dofs + 1][j * dofs] = v21;
                    (*elementMatrix)[i * dofs + 1][j * dofs + 1] = v22; // d=1, second dimension

                } // loop end over j node

            } // loop end over i node

        } // end if dim 2
        //***************************************************************************
        else if (dim == 3)
        {
            //************************************#

            // Compute shear rate gammaDot, which is a vector because it is evaluated at a gaussian quadrature point
            // for that compute velocity gradient
            vec_dbl_ptr_Type gammaDot(new vec_dbl_Type(weights->size(), 0.0)); // gammaDot->at(j) j=0...weights
            computeShearRate(dPhiTrans, gammaDot, dim);                        // updates gammaDot using velcoity solution

            // Initialize some helper vectors/matrices
            double v11, v12, v13, v21, v22, v23, v31, v32, v33, value1_j, value2_j, value3_j, value1_i, value2_i, value3_i, viscosity_atw;

            viscosity_atw = 0.;

            // Construct element matrices
            for (UN i = 0; i < numNodes; i++)
            {
                // Teuchos::Array<SC> value(dPhiTrans[0].size(), 0. ); // dPhiTrans[0].size() is 3

                for (UN j = 0; j < numNodes; j++)
                {
                    // Reset values
                    v11 = 0.0;
                    v12 = 0.0;
                    v13 = 0.0;
                    v21 = 0.0;
                    v22 = 0.0;
                    v23 = 0.0;
                    v31 = 0.0;
                    v32 = 0.0;
                    v33 = 0.0;

                    // So in general compute the components of eta*[ dPhiTrans_i : ( dPhiTrans_j + (dPhiTrans_j)^T )]
                    for (UN w = 0; w < dPhiTrans.size(); w++)
                    {

                        value1_j = dPhiTrans.at(w).at(j).at(0); // so this corresponds to d\phi_j/dx
                        value2_j = dPhiTrans.at(w).at(j).at(1); // so this corresponds to d\phi_j/dy
                        value3_j = dPhiTrans.at(w).at(j).at(2); // so this corresponds to d\phi_j/dz

                        value1_i = dPhiTrans.at(w).at(i).at(0); // so this corresponds to d\phi_i/dx
                        value2_i = dPhiTrans.at(w).at(i).at(1); // so this corresponds to d\phi_i/dy
                        value3_i = dPhiTrans.at(w).at(i).at(2); // so this corresponds to d\phi_i/dz

                        this->viscosityModel->evaluateMapping(this->params_, gammaDot->at(w), viscosity_atw);

                        // Construct entries - we go over all quadrature points and if j is updated we set v11 etc. again to zero
                        v11 = v11 + viscosity_atw * weights->at(w) * (2.0 * value1_j * value1_i + value2_j * value2_i + value3_j * value3_i);
                        v12 = v12 + viscosity_atw * weights->at(w) * (value2_i * value1_j);
                        v13 = v13 + viscosity_atw * weights->at(w) * (value3_i * value1_j);

                        v21 = v21 + viscosity_atw * weights->at(w) * (value1_i * value2_j);
                        v22 = v22 + viscosity_atw * weights->at(w) * (value1_i * value1_j + 2.0 * value2_j * value2_i + value3_j * value3_i);
                        v23 = v23 + viscosity_atw * weights->at(w) * (value3_i * value2_j);

                        v31 = v31 + viscosity_atw * weights->at(w) * (value1_i * value3_j);
                        v32 = v32 + viscosity_atw * weights->at(w) * (value2_i * value3_j);
                        v33 = v33 + viscosity_atw * weights->at(w) * (value1_i * value1_j + value2_i * value2_j + 2.0 * value3_i * value3_j);

                    } // loop end quadrature points

                    // multiply determinant from transformation
                    v11 *= absDetB;
                    v12 *= absDetB;
                    v13 *= absDetB;
                    v21 *= absDetB;
                    v22 *= absDetB;
                    v23 *= absDetB;
                    v31 *= absDetB;
                    v32 *= absDetB;
                    v33 *= absDetB;

                    // Put values on the right position in element matrix
                    // [v11  v12  v13]
                    // [v21  v22  v23]
                    // [v31  v32  v33]
                    (*elementMatrix)[i * dofs][j * dofs] = v11; // d=0, first dimension
                    (*elementMatrix)[i * dofs][j * dofs + 1] = v12;
                    (*elementMatrix)[i * dofs][j * dofs + 2] = v13;
                    (*elementMatrix)[i * dofs + 1][j * dofs] = v21;
                    (*elementMatrix)[i * dofs + 1][j * dofs + 1] = v22; // d=1, second dimension
                    (*elementMatrix)[i * dofs + 1][j * dofs + 2] = v23; // d=1, second dimension
                    (*elementMatrix)[i * dofs + 2][j * dofs] = v31;
                    (*elementMatrix)[i * dofs + 2][j * dofs + 1] = v32; // d=2, third dimension
                    (*elementMatrix)[i * dofs + 2][j * dofs + 2] = v33; // d=2, third dimension

                } // loop end over j node
            }     // loop end over i node
        }         // end if dim==3
    }

    // Directional Derivative of shear stress term resulting from chosen nonlinear non-newtonian model  -----
    // Same structure and functions as in assemblyStress
    //  ( -2.0*deta/dgammaDot * dgammaDot/dTau * (0.5(dv^k + (dvh^k)^T): 0.5( dPhiTrans_j + (dPhiTrans_j)^T))0.5(dv^k + (dvh^k)^T): 0.5( dPhiTrans_i + (dPhiTrans_i)^T)    )
    template <class SC, class LO, class GO, class NO>
    void AssembleFEGeneralizedNewtonian<SC, LO, GO, NO>::assemblyStressDev(SmallMatrixPtr_Type &elementMatrix)
    {

        int dim = this->getDim();
        int numNodes = this->numNodesVelocity_;
        UN Grad = 2; // Needs to be fixed
        string FEType = this->FETypeVelocity_;
        int dofs = this->dofsVelocity_; // for pressure it would be 1

        vec3D_dbl_ptr_Type dPhi;
        vec_dbl_ptr_Type weights = Teuchos::rcp(new vec_dbl_Type(0));

        UN deg = Helper::determineDegree(dim, FEType, Grad);
        Helper::getDPhi(dPhi, weights, dim, FEType, deg);

        SC detB;
        SC absDetB;
        SmallMatrix<SC> B(dim);
        SmallMatrix<SC> Binv(dim);

        this->buildTransformation(B);
        detB = B.computeInverse(Binv);
        absDetB = std::fabs(detB);

        vec3D_dbl_Type dPhiTrans(dPhi->size(), vec2D_dbl_Type(dPhi->at(0).size(), vec_dbl_Type(dim, 0.)));
        Helper::applyBTinv(dPhi, dPhiTrans, Binv);

        TEUCHOS_TEST_FOR_EXCEPTION(dim == 1, std::logic_error, "AssemblyStress Not implemented for dim=1");

        if (dim == 2)
        {
            //************************************
            //************************************
            // Due to the extra term related to the Gaetaeux-derivative there arise prefactors which depend on the velocity gradients solutions
            // which therefore also have to be computed here therefore we compute it directly here
            vec_dbl_Type u11(dPhiTrans.size(), -1.);                           // should correspond to du/dx at each quadrature point
            vec_dbl_Type u12(dPhiTrans.size(), -1.);                           // should correspond to du/dy at each quadrature point
            vec_dbl_Type u21(dPhiTrans.size(), -1.);                           // should correspond to dv/dx at each quadrature point
            vec_dbl_Type u22(dPhiTrans.size(), -1.);                           // should correspond to dv/dy at each quadrature point
            vec_dbl_ptr_Type gammaDot(new vec_dbl_Type(weights->size(), 0.0)); // gammaDot->at(j) j=0...weights

            vec_dbl_ptr_Type mixed_term_xy(new vec_dbl_Type(weights->size(), 0.0));
            for (UN w = 0; w < dPhiTrans.size(); w++)
            { // quads points
                // set again to zero
                u11[w] = 0.0; // du_dx
                u12[w] = 0.0; // du_dy
                u21[w] = 0.0; // dv_dx
                u22[w] = 0.0; // dv_dy
                for (UN i = 0; i < dPhiTrans[0].size(); i++)
                {                            // loop unrolling
                    LO index1 = dim * i + 0; // x
                    LO index2 = dim * i + 1; // y
                    // uLoc[d][w] += (*this->solution_)[index] * phi->at(w).at(i);
                    u11[w] += (*this->solution_)[index1] * dPhiTrans[w][i][0]; // u*dphi_dx
                    u12[w] += (*this->solution_)[index1] * dPhiTrans[w][i][1]; // because we are in 2D , 0 and 1
                    u21[w] += (*this->solution_)[index2] * dPhiTrans[w][i][0];
                    u22[w] += (*this->solution_)[index2] * dPhiTrans[w][i][1];
                }
                gammaDot->at(w) = sqrt(2.0 * u11[w] * u11[w] + 2.0 * u22[w] * u22[w] + (u12[w] + u21[w]) * (u12[w] + u21[w]));
                mixed_term_xy->at(w) = 0.5 * (u12[w] + u21[w]);
            }
            //*******************************

            // Initialize some helper vectors/matrices
            double v11, v12, v21, v22, value1_j, value2_j, value1_i, value2_i, deta_dgamma_dgamma_dtau;

            deta_dgamma_dgamma_dtau = 0.;

            // Construct element matrices
            for (UN i = 0; i < numNodes; i++)
            {
                // Teuchos::Array<SC> value(dPhiTrans[0].size(), 0. ); // dPhiTrans[0].size() is 3

                for (UN j = 0; j < numNodes; j++)
                {
                    // Reset values
                    v11 = 0.0;
                    v12 = 0.0;
                    v21 = 0.0;
                    v22 = 0.0;

                    // Only the part  deta/dgammaDot is different for all shear thinning models (because we make the assumption of incompressibility)? NO ALSO DIFFERENT FOR EXAMPLE FOR CASSON SO CONSIDERING YIELD STRESS
                    // but we put the two terms together because then we can multiply them together and get e.g. for carreau yasuda  : gammaDot^{a-2.0} which is for a=2.0 equals 0 and we do not have to worry about the problem what if gammaDot = 0.0
                    for (UN w = 0; w < dPhiTrans.size(); w++)
                    {

                        value1_j = dPhiTrans[w][j][0]; // so this corresponds to d\phi_j/dx
                        value2_j = dPhiTrans[w][j][1]; // so this corresponds to d\phi_j/dy

                        value1_i = dPhiTrans[w][i][0]; // so this corresponds to d\phi_i/dx
                        value2_i = dPhiTrans[w][i][1]; // so this corresponds to d\phi_i/dy

                        this->viscosityModel->evaluateDerivative(this->params_, gammaDot->at(w), deta_dgamma_dgamma_dtau);
                        /* EInfacher in unausmultiplizierter Form
                        v11 = v11 + (-2.0)*deta_dgamma_dgamma_dtau  * weights->at(w) *(u11[w]*u11[w]*value1_i*value1_j+u11[w]*mixed_terms->at(w)*(value1_i*value2_j+value2_i*value1_j)+ mixed_terms->at(w)*mixed_terms->at(w)*(value2_i*value2_j)); // xx contribution: (dv_x/dx)^2*dphi_i/dx*dphi_j/dx+dv_x/dx*f*(dphi_i/dx*dphi_j/dy+dphi_i/dy*dphi_j/dx)+f^2*dphi_i/dy*dphi_j/dy
                        v12 = v12 + (-2.0)*deta_dgamma_dgamma_dtau  * weights->at(w) *(u11[w]*mixed_terms->at(w)*value1_i*value1_j+u11[w]*u22[w]*(value1_i*value2_j)        +mixed_terms->at(w)*mixed_terms->at(w)*value2_i*value1_j+mixed_terms->at(w)*u22[w]*value2_i*value2_j ); // xy contribution:  dv_x/dx*f*dphi_i/dx*dphi_j/dx+dv_x/dx*dv_y/dy*dphi_i/dx*dphi_j/dy+f^2*dphi_i_dy*dphi_j/dx+f*dv_y/dy*dphi_i/dy*dphi_j/dy
                        v21 = v21 + (-2.0)*deta_dgamma_dgamma_dtau  * weights->at(w) *(u11[w]*mixed_terms->at(w)*value1_i*value1_j+mixed_terms->at(w)*mixed_terms->at(w)*value1_i*value2_j+u11[w]*u22[w]*value2_i*value1_j          +mixed_terms->at(w)*u22[w]*value2_i*value2_j ); // yx contribution:  dv_x/dx*f*dphi_i/dx*dphi_j/dx+dv_x/dx*dv_y/dy*dphi_i/dy*dphi_j/dx+f^2*dphi_i_dx*dphi_j/dy+f*dv_y/dy*dphi_i/dy*dphi_j/dy
                        v22 = v22 + (-2.0)*deta_dgamma_dgamma_dtau  * weights->at(w) *(u22[w]*u22[w]*value2_i*value2_j+u22[w]*mixed_terms->at(w)*(value1_i*value2_j+value2_i*value1_j)+ mixed_terms->at(w)*mixed_terms->at(w)*(value1_i*value1_j) ); // yy contribution: (dv_y/dy)^2*dphi_i/dy*dphi_j/dy+dv_y/dy*f*(dphi_i/dx*dphi_j/dy+dphi_i/dy*dphi_j/dx)+f^2*dphi_i/dx*dphi_j/dx
                        */
                        v11 = v11 + (-2.0) * deta_dgamma_dgamma_dtau * weights->at(w) * ((value1_j * u11[w] + value2_j * mixed_term_xy->at(w)) * (value1_i * u11[w] + value2_i * mixed_term_xy->at(w)));
                        v12 = v12 + (-2.0) * deta_dgamma_dgamma_dtau * weights->at(w) * ((value1_j * mixed_term_xy->at(w) + u22[w] * value2_j) * (value1_i * u11[w] + value2_i * mixed_term_xy->at(w)));

                        v21 = v21 + (-2.0) * deta_dgamma_dgamma_dtau * weights->at(w) * ((value1_j * u11[w] + value2_j * mixed_term_xy->at(w)) * (value1_i * mixed_term_xy->at(w) + value2_i * u22[w]));
                        v22 = v22 + (-2.0) * deta_dgamma_dgamma_dtau * weights->at(w) * ((value1_j * mixed_term_xy->at(w) + u22[w] * value2_j) * (value1_i * mixed_term_xy->at(w) + value2_i * u22[w]));

                    } // loop end quadrature points

                    // multiply determinant from transformation
                    v11 *= absDetB;
                    v12 *= absDetB;
                    v21 *= absDetB;
                    v22 *= absDetB;

                    // Put values on the right position in element matrix - d=2 because we are in two dimensional case
                    // [v11  v12  ]
                    // [v21  v22  ]
                    (*elementMatrix)[i * dofs][j * dofs] = v11; // d=0, first dimension
                    (*elementMatrix)[i * dofs][j * dofs + 1] = v12;
                    (*elementMatrix)[i * dofs + 1][j * dofs] = v21;
                    (*elementMatrix)[i * dofs + 1][j * dofs + 1] = v22; // d=1, second dimension

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
            vec_dbl_Type u11(dPhiTrans.size(), -1.); // should correspond to du/dx at each quadrature point
            vec_dbl_Type u12(dPhiTrans.size(), -1.); // should correspond to du/dy at each quadrature point
            vec_dbl_Type u13(dPhiTrans.size(), -1.); // should correspond to du/dz at each quadrature point

            vec_dbl_Type u21(dPhiTrans.size(), -1.); // should correspond to dv/dx at each quadrature point
            vec_dbl_Type u22(dPhiTrans.size(), -1.); // should correspond to dv/dy at each quadrature point
            vec_dbl_Type u23(dPhiTrans.size(), -1.); // should correspond to dv/dz at each quadrature point

            vec_dbl_Type u31(dPhiTrans.size(), -1.); // should correspond to dw/dx at each quadrature point
            vec_dbl_Type u32(dPhiTrans.size(), -1.); // should correspond to dw/dy at each quadrature point
            vec_dbl_Type u33(dPhiTrans.size(), -1.); // should correspond to dw/dz at each quadrature point

            vec_dbl_ptr_Type gammaDot(new vec_dbl_Type(weights->size(), 0.0)); // gammaDot->at(j) j=0...weights

            vec_dbl_ptr_Type mixed_term_xy(new vec_dbl_Type(weights->size(), 0.0));
            vec_dbl_ptr_Type mixed_term_xz(new vec_dbl_Type(weights->size(), 0.0));
            vec_dbl_ptr_Type mixed_term_yz(new vec_dbl_Type(weights->size(), 0.0));

            for (UN w = 0; w < dPhiTrans.size(); w++)
            { // quads points

                u11[w] = 0.0;
                u12[w] = 0.0;
                u13[w] = 0.0;
                u21[w] = 0.0;
                u22[w] = 0.0;
                u23[w] = 0.0;
                u31[w] = 0.0;
                u32[w] = 0.0;
                u33[w] = 0.0;

                for (UN i = 0; i < dPhiTrans[0].size(); i++)
                {
                    LO index1 = dim * i + 0; // x
                    LO index2 = dim * i + 1; // y
                    LO index3 = dim * i + 2; // z
                    // uLoc[d][w] += (*this->solution_)[index] * phi->at(w).at(i);
                    u11[w] += (*this->solution_)[index1] * dPhiTrans[w][i][0]; // u*dphi_dx
                    u12[w] += (*this->solution_)[index1] * dPhiTrans[w][i][1]; // because we are in 3D , 0 and 1, 2
                    u13[w] += (*this->solution_)[index1] * dPhiTrans[w][i][2];
                    u21[w] += (*this->solution_)[index2] * dPhiTrans[w][i][0]; // v*dphi_dx
                    u22[w] += (*this->solution_)[index2] * dPhiTrans[w][i][1];
                    u23[w] += (*this->solution_)[index2] * dPhiTrans[w][i][2];
                    u31[w] += (*this->solution_)[index3] * dPhiTrans[w][i][0]; // w*dphi_dx
                    u32[w] += (*this->solution_)[index3] * dPhiTrans[w][i][1];
                    u33[w] += (*this->solution_)[index3] * dPhiTrans[w][i][2];
                }
                gammaDot->at(w) = sqrt(2.0 * u11[w] * u11[w] + 2.0 * u22[w] * u22[w] + 2.0 * u33[w] * u33[w] + (u12[w] + u21[w]) * (u12[w] + u21[w]) + (u13[w] + u31[w]) * (u13[w] + u31[w]) + (u23[w] + u32[w]) * (u23[w] + u32[w]));

                mixed_term_xy->at(w) = 0.5 * (u12[w] + u21[w]);
                mixed_term_xz->at(w) = 0.5 * (u31[w] + u13[w]);
                mixed_term_yz->at(w) = 0.5 * (u32[w] + u23[w]);
            }

            // Initialize some helper vectors/matrices
            double v11, v12, v13, v21, v22, v23, v31, v32, v33, value1_j, value2_j, value3_j, value1_i, value2_i, value3_i, deta_dgamma_dgamma_dtau;

            deta_dgamma_dgamma_dtau = 0.;

            // Construct element matrices
            for (UN i = 0; i < numNodes; i++)
            {
                // Teuchos::Array<SC> value(dPhiTrans[0].size(), 0. ); // dPhiTrans[0].size() is 3

                for (UN j = 0; j < numNodes; j++)
                {
                    // Reset values
                    v11 = 0.0;
                    v12 = 0.0;
                    v13 = 0.0;
                    v21 = 0.0;
                    v22 = 0.0;
                    v23 = 0.0;
                    v31 = 0.0;
                    v32 = 0.0;
                    v33 = 0.0;

                    // So in general compute the components of eta*[ dPhiTrans_i : ( dPhiTrans_j + (dPhiTrans_j)^T )]
                    for (UN w = 0; w < dPhiTrans.size(); w++)
                    {

                        value1_j = dPhiTrans.at(w).at(j).at(0); // so this corresponds to d\phi_j/dx
                        value2_j = dPhiTrans.at(w).at(j).at(1); // so this corresponds to d\phi_j/dy
                        value3_j = dPhiTrans.at(w).at(j).at(2); // so this corresponds to d\phi_j/dz

                        value1_i = dPhiTrans.at(w).at(i).at(0); // so this corresponds to d\phi_i/dx
                        value2_i = dPhiTrans.at(w).at(i).at(1); // so this corresponds to d\phi_i/dy
                        value3_i = dPhiTrans.at(w).at(i).at(2); // so this corresponds to d\phi_i/dz

                        this->viscosityModel->evaluateDerivative(this->params_, gammaDot->at(w), deta_dgamma_dgamma_dtau);

                        // Construct entries - we go over all quadrature points and if j is updated we set v11 etc. again to zero
                        v11 = v11 + (-2.0) * deta_dgamma_dgamma_dtau * weights->at(w) * ((value1_j * u11[w] + value2_j * mixed_term_xy->at(w) + value3_j * mixed_term_xz->at(w)) * (value1_i * u11[w] + value2_i * mixed_term_xy->at(w) + value3_i * mixed_term_xz->at(w)));
                        v12 = v12 + (-2.0) * deta_dgamma_dgamma_dtau * weights->at(w) * ((value1_j * mixed_term_xy->at(w) + u22[w] * value2_j + mixed_term_yz->at(w) * value3_j) * (value1_i * u11[w] + value2_i * mixed_term_xy->at(w) + value3_i * mixed_term_xz->at(w)));
                        v13 = v13 + (-2.0) * deta_dgamma_dgamma_dtau * weights->at(w) * ((value3_j * mixed_term_xz->at(w) + value2_j * mixed_term_yz->at(w) + value3_j * u33[w]) * (value1_i * u11[w] + value2_i * mixed_term_xy->at(w) + value3_i * mixed_term_xz->at(w)));

                        v21 = v21 + (-2.0) * deta_dgamma_dgamma_dtau * weights->at(w) * ((value1_j * u11[w] + value2_j * mixed_term_xy->at(w) + value3_j * mixed_term_xz->at(w)) * (value1_i * mixed_term_xy->at(w) + value2_i * u22[w] + value3_i * mixed_term_yz->at(w)));
                        v22 = v22 + (-2.0) * deta_dgamma_dgamma_dtau * weights->at(w) * ((value1_j * mixed_term_xy->at(w) + u22[w] * value2_j + mixed_term_yz->at(w) * value3_j) * (value1_i * mixed_term_xy->at(w) + value2_i * u22[w] + value3_i * mixed_term_yz->at(w)));
                        v23 = v23 + (-2.0) * deta_dgamma_dgamma_dtau * weights->at(w) * ((value3_j * mixed_term_xz->at(w) + value2_j * mixed_term_yz->at(w) + value3_j * u33[w]) * (value1_i * mixed_term_xy->at(w) + value2_i * u22[w] + value3_i * mixed_term_yz->at(w)));

                        v31 = v31 + (-2.0) * deta_dgamma_dgamma_dtau * weights->at(w) * ((value1_j * u11[w] + value2_j * mixed_term_xy->at(w) + value3_j * mixed_term_xz->at(w)) * (value1_i * mixed_term_xz->at(w) + mixed_term_yz->at(w) * value2_i + u33[w] * value3_i));
                        v32 = v32 + (-2.0) * deta_dgamma_dgamma_dtau * weights->at(w) * ((value1_j * mixed_term_xy->at(w) + u22[w] * value2_j + mixed_term_yz->at(w) * value3_j) * (value1_i * mixed_term_xz->at(w) + mixed_term_yz->at(w) * value2_i + u33[w] * value3_i));
                        v33 = v33 + (-2.0) * deta_dgamma_dgamma_dtau * weights->at(w) * ((value3_j * mixed_term_xz->at(w) + value2_j * mixed_term_yz->at(w) + value3_j * u33[w]) * (value1_i * mixed_term_xz->at(w) + mixed_term_yz->at(w) * value2_i + u33[w] * value3_i));

                    } // loop end quadrature points

                    // multiply determinant from transformation
                    v11 *= absDetB;
                    v12 *= absDetB;
                    v13 *= absDetB;
                    v21 *= absDetB;
                    v22 *= absDetB;
                    v23 *= absDetB;
                    v31 *= absDetB;
                    v32 *= absDetB;
                    v33 *= absDetB;

                    // Put values on the right position in element matrix
                    // [v11  v12  v13]
                    // [v21  v22  v23]
                    // [v31  v32  v33]
                    (*elementMatrix)[i * dofs][j * dofs] = v11; // d=0, first dimension
                    (*elementMatrix)[i * dofs][j * dofs + 1] = v12;
                    (*elementMatrix)[i * dofs][j * dofs + 2] = v13;
                    (*elementMatrix)[i * dofs + 1][j * dofs] = v21;
                    (*elementMatrix)[i * dofs + 1][j * dofs + 1] = v22; // d=1, second dimension
                    (*elementMatrix)[i * dofs + 1][j * dofs + 2] = v23; // d=1, second dimension
                    (*elementMatrix)[i * dofs + 2][j * dofs] = v31;
                    (*elementMatrix)[i * dofs + 2][j * dofs + 1] = v32; // d=2, third dimension
                    (*elementMatrix)[i * dofs + 2][j * dofs + 2] = v33; // d=2, third dimension

                } // loop end over j node

            } // loop end over i node

        } // end if dim = 3
    }


    // Boundary integral over Neumann boundary resulting from the fact that we want our outflow boundary conditions
    // as in the case for reduced stress tensor - therefore we have to subtract the boundary integral
    // - int_NeumannBoundary ( \nabla u)^T n \cdot w dNeumannBoundary
    template <class SC, class LO, class GO, class NO>
    void AssembleFEGeneralizedNewtonian<SC, LO, GO, NO>::assemblyOutflowNeumannBoundaryTerm(SmallMatrixPtr_Type &elementMatrix) //, dblVecPtr normalVector)
    {

        int dim = this->getDim();
        int numNodes = this->numNodesVelocity_;
        string FEType = this->FETypeVelocity_;
        int dofs = this->dofsVelocity_; //

        vec3D_dbl_ptr_Type dPhi; // derivative of basisfunction
        vec2D_dbl_ptr_Type phi;  // basisfunction

        // Compute phi and derivative of phi at quadrature points
        SC detB;
        SC absDetB;
        SmallMatrix<SC> B(dim);
        SmallMatrix<SC> Binv(dim);

        double elscaling=0.0;
        vec_dbl_Type QuadWeightsReference;         // These are the Quadrature weights in terms of quadrature points defined in reference coordinate system so here [0 1]
        vec2D_dbl_Type QuadPointsGlobal;
        vec_dbl_Type normalVector;
        
        this->buildTransformation(B);   // In order to map from global coordinates to reference surface
        detB = B.computeInverse(Binv);  // The function computeInverse returns a double value corrsponding to determinant of B
        absDetB = std::fabs(detB);      // absolute value of B

        ElementsPtr_Type subEl = this->FEObject_->getSubElements(); // shouldn't be here null because we already checked that we have an underlying boundary element
        for (int surface=0; surface< this->FEObject_->numSubElements(); surface++) 
        {
            FiniteElement feSub = subEl->getElement( surface  );
            if(feSub.getNeumannBCElement() == true ) //So here we only consider the subelements which we set a flag that we want to add a Neumann boundary condition
            {

                        normalVector= feSub.getSurfaceNormal();
                        QuadWeightsReference=  feSub.getQuadratureWeightsReference();
                        QuadPointsGlobal    =  feSub.getQuadraturePointsGlobal();
                        elscaling  =  feSub.getElementScaling();


            }                   
        }    
    
        vec2D_dbl_Type QuadPointsMappedReference(QuadWeightsReference.size(), vec_dbl_Type(dim)); // now initialize the container with mapped quadrature coordinates
        // Now we have to compute the weights and the quadrature points for our line integral (2D), surface integral (3D)
        // where we want to evaluate our dPhi, phi
        // So we are now mapping back the quadrature points from the physical edge element to the reference element
        // via the inverse mapping xi = Binv(x-tau) where tau is p0
        // tau = [this->nodesRefConfig_.at(0).at(0) ;  this->nodesRefConfig_.at(0).at(1)]

        for (int l = 0; l < QuadPointsGlobal.size(); l++)
        {
            for (int p = 0; p < dim; p++)
            {
                for (int q = 0; q < dim; q++)
                {
                    QuadPointsMappedReference[l][p]  += Binv[p][q] * ( QuadPointsGlobal[l][q] - this->nodesRefConfig_.at(0).at(q));
                }
            }
        }

        // **** @TODO: With or without scaling? In the 2D case no points are projected 
        // Now we have successfully mapped the global quadrature points onto one of the reference faces (2D line / 3D surface)
        // But we still have to consider the case that if the global surface was mapped onto the diagonal line/ diagonal surface we
        // have to scale the weights (not the quadrature points because they were mapped on the right relative locations through the transformation) by the factor area change so in 2D it is just the length so sqrt(2) [0 sqrt(2)] and in 3D it should be the area which is 0.866025403784439
        // Check if a quadrature point lies on the diagonal line
        // We consider the point to be on the diagonal if both its x and y components are nonzero
        // https://www.math.ntnu.no/emner/TMA4130/2021h/lectures/CompositeQuadrature.pdf

        double lengthReferenceElementDiagonalLine = std::sqrt(2.0) ;
        double eps = 1e-12; // std::numeric_limits<double>::epsilon() is too small
        double areaReferenceElementDiagonalFace =   0.866025403784439; // 0.866025403784439;
        if (dim==2)
        {
            if (  (std::fabs( QuadPointsMappedReference[0][0] - 0.0) >  eps  )   &&  ( (QuadPointsMappedReference[0][1]  - 0.0) >  eps   ) ) // if the x and y component of quadrature point are non-zero we are on the diagonal but also only if the quadrature point was not defined in corners of element
            {
                for (int l = 0; l < QuadPointsGlobal.size(); l++)
                {
                QuadWeightsReference[l] =  lengthReferenceElementDiagonalLine * QuadWeightsReference[l]; // We have to only scale the weights as quadrature points are already mapped onto correct relative position on the diagonal
                }
                printf("We are on the diagonal\n");
            }
        } // Normal checken
        else if (dim==3) // Quadrature Points are already mapped onto the correct positions inside the 2D surface
        {
            if (  (std::fabs( QuadPointsMappedReference[0][0] - 0.0) > eps  )   &&  ( (QuadPointsMappedReference[0][1]  - 0.0) >  eps   ) &&  ( (QuadPointsMappedReference[0][2]  - 0.0) >  eps   ) ) // if the x and y component of quadrature point are non-zero we are on the diagonal but also only if the quadrature point was not defined in corners of element
            {
                for (int l = 0; l < QuadPointsGlobal.size(); l++)
                {
                QuadWeightsReference[l] = areaReferenceElementDiagonalFace  * QuadWeightsReference[l];
                }
                printf("We are on the diagonal\n");

                

            }
        }
        

        Helper::getPhi(phi, QuadWeightsReference, QuadPointsMappedReference, dim, FEType); // This should be zero for the basisfunction not laying on the line/ surface
        Helper::getDPhi(dPhi, QuadWeightsReference, QuadPointsMappedReference, dim, FEType);


        // dPhiTrans are the transorfmed basisfunctions, so B^(-T) * \grad_phi bzw. \grad_phi^T * B^(-1) Corresponds to \hat{grad_phi}.
        vec3D_dbl_Type dPhiTrans(dPhi->size(), vec2D_dbl_Type(dPhi->at(0).size(), vec_dbl_Type(dim, 0.)));
        Helper::applyBTinv(dPhi, dPhiTrans, Binv); // so dPhiTrans corresponds now to our basisfunction in natural coordinates

        // Compute shear rate gammaDot, which is a vector because it is evaluated at a gaussian quadrature point for that compute velocity gradient
        vec_dbl_ptr_Type gammaDot(new vec_dbl_Type(QuadWeightsReference.size(), 0.0)); // gammaDot->at(j) j=0...weights
        computeShearRate(dPhiTrans, gammaDot, dim);                      // updates gammaDot using velocity solution
        double viscosity_atw = 0.;

        TEUCHOS_TEST_FOR_EXCEPTION(dim == 1, std::logic_error, "AssemblyNeumannBoundaryTerm Not implemented for dim=1");
        // 2D
        if (dim == 2)
        {
            double v11, v12, v21, v22; // helper values for entries

            // loop over basis functions
            for (UN i = 0; i < phi->at(0).size(); i++)
            {
                for (UN j = 0; j < numNodes; j++)
                {
                    // Reset values
                    v11 = 0.0;
                    v12 = 0.0;
                    v21 = 0.0;
                    v22 = 0.0;

                    // loop over basis functions quadrature points
                    for (UN w = 0; w < phi->size(); w++)
                    {
                        this->viscosityModel->evaluateMapping(this->params_, gammaDot->at(w), viscosity_atw);

                        v11 = v11 + -1.0 * (viscosity_atw * QuadWeightsReference[w] * dPhiTrans[w][j][0] * normalVector[0]  * (*phi)[w][i]); // xx contribution:
                        v12 = v12 + -1.0 * (viscosity_atw * QuadWeightsReference[w] * dPhiTrans[w][j][0] * normalVector[1]  * (*phi)[w][i]); // xy contribution:
                        v21 = v21 + -1.0 * (viscosity_atw * QuadWeightsReference[w] * dPhiTrans[w][j][1] * normalVector[0]  * (*phi)[w][i]); // yx contribution:
                        v22 = v22 + -1.0 * (viscosity_atw * QuadWeightsReference[w] * dPhiTrans[w][j][1]  * normalVector[1]  * (*phi)[w][i]); // yy contribution:

                    } // End loop over quadrature points

                    // multiply determinant from transformation
                    v11 *=elscaling;
                    v12 *=elscaling;
                    v21 *=elscaling;
                    v22 *=elscaling;

                    // Put values on the right position in element matrix - d=2 because we are in two dimensional case
                    // [v11  v12 ]
                    // [v21  v22 ]
                    (*elementMatrix)[i * dofs][j * dofs] = v11;     // d=0, first dimension
                    (*elementMatrix)[i * dofs][j * dofs + 1] = v12; //
                    (*elementMatrix)[i * dofs + 1][j * dofs] = v21;
                    (*elementMatrix)[i * dofs + 1][j * dofs + 1] = v22; // d=1, second dimension

                } // End loop over j nodes

            } // End loop over i nodes
        }     // End dim==2
        else if (dim == 3)
        {
            double v11, v12, v13, v21, v22, v23, v31, v32, v33; // helper values for entries
                                                                // loop over basis functions
            for (UN i = 0; i < phi->at(0).size(); i++)
            {
                for (UN j = 0; j < numNodes; j++)
                {
                    // Reset values
                    v11 = 0.0;
                    v12 = 0.0;
                    v13 = 0.0;
                    v21 = 0.0;
                    v22 = 0.0;
                    v23 = 0.0;
                    v31 = 0.0;
                    v32 = 0.0;
                    v33 = 0.0;

                    // loop over basis functions quadrature points
                    for (UN w = 0; w < phi->size(); w++)
                    {
                        this->viscosityModel->evaluateMapping(this->params_, gammaDot->at(w), viscosity_atw);

                        v11 = v11 + -1.0 * (viscosity_atw * QuadWeightsReference[w] * dPhiTrans[w][j][0] *normalVector[0] * (*phi)[w][i]); // xx contribution:
                        v12 = v12 + -1.0 * (viscosity_atw * QuadWeightsReference[w] * dPhiTrans[w][j][0] *normalVector[1] * (*phi)[w][i]); // xy contribution:
                        v13 = v13 + -1.0 * (viscosity_atw * QuadWeightsReference[w] * dPhiTrans[w][j][0] *normalVector[2] * (*phi)[w][i]); // xz contribution:

                        v21 = v21 + -1.0 * (viscosity_atw * QuadWeightsReference[w] * dPhiTrans[w][j][1] * normalVector[0] * (*phi)[w][i]); // yx contribution:
                        v22 = v22 + -1.0 * (viscosity_atw * QuadWeightsReference[w]* dPhiTrans[w][j][1] * normalVector[1] * (*phi)[w][i]); // yy contribution:
                        v23 = v23 + -1.0 * (viscosity_atw * QuadWeightsReference[w] * dPhiTrans[w][j][1] *normalVector[2] * (*phi)[w][i]); // yz contribution:

                        v31 = v31 + -1.0 * (viscosity_atw * QuadWeightsReference[w] * dPhiTrans[w][j][2] * normalVector[0]* (*phi)[w][i]); // zx contribution:
                        v32 = v32 + -1.0 * (viscosity_atw * QuadWeightsReference[w] * dPhiTrans[w][j][2] * normalVector[1] * (*phi)[w][i]); // zy contribution:
                        v33 = v33 + -1.0 * (viscosity_atw * QuadWeightsReference[w] * dPhiTrans[w][j][2] * normalVector[2] * (*phi)[w][i]); // zz contribution:

                    } // End loop over quadrature points

                    // multiply determinant from transformation
                    v11 *=elscaling;
                    v12 *=elscaling;
                    v13 *=elscaling;
                    v21 *=elscaling;
                    v22 *=elscaling;
                    v23 *=elscaling;
                    v31 *=elscaling;
                    v32 *=elscaling;
                    v33 *=elscaling;

                    // Put values on the right position in element matrix - d=2 because we are in two dimensional case
                    // [v11  v12  v13]
                    // [v21  v22  v23]
                    // [v31  v32  v33]
                    (*elementMatrix)[i * dofs][j * dofs] = v11; // d=0, first dimension
                    (*elementMatrix)[i * dofs][j * dofs + 1] = v12;
                    (*elementMatrix)[i * dofs][j * dofs + 2] = v13;
                    (*elementMatrix)[i * dofs + 1][j * dofs] = v21;
                    (*elementMatrix)[i * dofs + 1][j * dofs + 1] = v22; // d=1, second dimension
                    (*elementMatrix)[i * dofs + 1][j * dofs + 2] = v23; // d=1, second dimension
                    (*elementMatrix)[i * dofs + 2][j * dofs] = v31;
                    (*elementMatrix)[i * dofs + 2][j * dofs + 1] = v32; // d=2, third dimension
                    (*elementMatrix)[i * dofs + 2][j * dofs + 2] = v33; // d=2, third dimension

                } // End loop over j nodes

            } // End loop over i nodes
        }     // end if 3d

    } // Function End loop




    // Boundary integral over Neumann boundary resulting from the fact that we want our outflow boundary conditions
    // as in the case for reduced stress tensor - therefore we have to subtract the boundary integral
    // int_NeumannBoundary ( \nabla u)^T n \cdot w dNeumannBoundary

    template <class SC, class LO, class GO, class NO>
    void AssembleFEGeneralizedNewtonian<SC, LO, GO, NO>::assemblyOutflowNeumannBoundaryTermDev(SmallMatrixPtr_Type &elementMatrix) //, dblVecPtr normalVector)
    {

       int dim = this->getDim();
        int numNodes = this->numNodesVelocity_;
        string FEType = this->FETypeVelocity_;
        int dofs = this->dofsVelocity_; //

        vec3D_dbl_ptr_Type dPhi; // derivative of basisfunction
        vec2D_dbl_ptr_Type phi;  // basisfunction

        // Compute phi and derivative of phi at quadrature points
        SC detB;
        SC absDetB;
        SmallMatrix<SC> B(dim);
        SmallMatrix<SC> Binv(dim);

        double elscaling=0.0;
        vec_dbl_Type QuadWeightsReference;         // These are the Quadrature weights in terms of quadrature points defined in reference coordinate system so here [0 1]
        vec2D_dbl_Type QuadPointsGlobal;
        vec_dbl_Type normalVector;
        
        this->buildTransformation(B);   // In order to map from global coordinates to reference surface
        detB = B.computeInverse(Binv);  // The function computeInverse returns a double value corrsponding to determinant of B
        absDetB = std::fabs(detB);      // absolute value of B

        ElementsPtr_Type subEl = this->FEObject_->getSubElements(); // shouldn't be here null because we already checked that we have an underlying boundary element
        for (int surface=0; surface< this->FEObject_->numSubElements(); surface++) 
        {
            FiniteElement feSub = subEl->getElement( surface  );
            if(feSub.getNeumannBCElement() == true ) //So here we only consider the subelements which we set a flag that we want to add a Neumann boundary condition
            {

                        normalVector= feSub.getSurfaceNormal();
                        QuadWeightsReference=  feSub.getQuadratureWeightsReference();
                        QuadPointsGlobal    =  feSub.getQuadraturePointsGlobal();
                        elscaling  =  feSub.getElementScaling();


            }                   
        }    
    
        vec2D_dbl_Type QuadPointsMappedReference(QuadWeightsReference.size(), vec_dbl_Type(dim)); // now initialize the container with mapped quadrature coordinates
        // Now we have to compute the weights and the quadrature points for our line integral (2D), surface integral (3D)
        // where we want to evaluate our dPhi, phi
        // So we are now mapping back the quadrature points from the physical edge element to the reference element
        // via the inverse mapping xi = Binv(x-tau) where tau is p0
        // tau = [this->nodesRefConfig_.at(0).at(0) ;  this->nodesRefConfig_.at(0).at(1)]

        for (int l = 0; l < QuadPointsGlobal.size(); l++)
        {
            for (int p = 0; p < dim; p++)
            {
                for (int q = 0; q < dim; q++)
                {
                    QuadPointsMappedReference[l][p]  += Binv[p][q] * ( QuadPointsGlobal[l][q] - this->nodesRefConfig_.at(0).at(q));
                }
            }
        }

        // Now we have successfully mapped the global quadrature points onto one of the reference faces (2D line / 3D surface)
        // But we still have to consider the case that if the global surface was mapped onto the diagonal line/ diagonal surface we
        // have to scale the weights (not the quadrature points because they were mapped on the right relative locations through the transformation) by the factor area change so in 2D it is just the length so sqrt(2) [0 sqrt(2)] and in 3D it should be the area which is 0.866025403784439
        // Check if a quadrature point lies on the diagonal line
        // We consider the point to be on the diagonal if both its x and y components are nonzero
        // https://www.math.ntnu.no/emner/TMA4130/2021h/lectures/CompositeQuadrature.pdf
        // !?! If I do not scale the weights I get better results ...

        double lengthReferenceElementDiagonalLine = std::sqrt(2.0) ;
        double eps = 1e-12; // std::numeric_limits<double>::epsilon() is too small
        double areaReferenceElementDiagonalFace =   0.866025403784439; // 0.866025403784439;
        if (dim==2)
        {
            if (  (std::fabs( QuadPointsMappedReference[0][0] - 0.0) >  eps  )   &&  ( (QuadPointsMappedReference[0][1]  - 0.0) >  eps   ) ) // if the x and y component of quadrature point are non-zero we are on the diagonal but also only if the quadrature point was not defined in corners of element
            {
                for (int l = 0; l < QuadPointsGlobal.size(); l++)
                {
                QuadWeightsReference[l] =  lengthReferenceElementDiagonalLine * QuadWeightsReference[l]; // We have to only scale the weights as quadrature points are already mapped onto correct relative position on the diagonal
                }
                printf("We are on the diagonal\n");
            }
        } // Normal checken
        else if (dim==3) // Quadrature Points are already mapped onto the correct positions inside the 2D surface
        {
            if (  (std::fabs( QuadPointsMappedReference[0][0] - 0.0) > eps  )   &&  ( (QuadPointsMappedReference[0][1]  - 0.0) >  eps   ) &&  ( (QuadPointsMappedReference[0][2]  - 0.0) >  eps   ) ) // if the x and y component of quadrature point are non-zero we are on the diagonal but also only if the quadrature point was not defined in corners of element
            {
                for (int l = 0; l < QuadPointsGlobal.size(); l++)
                {
                QuadWeightsReference[l] = areaReferenceElementDiagonalFace  * QuadWeightsReference[l];
                }
                printf("We are on the diagonal\n");

                

            }
        }
        

        Helper::getPhi(phi, QuadWeightsReference, QuadPointsMappedReference, dim, FEType); // This should be zero for the basisfunction not laying on the line/ surface
        Helper::getDPhi(dPhi, QuadWeightsReference, QuadPointsMappedReference, dim, FEType);


        // dPhiTrans are the transorfmed basisfunctions, so B^(-T) * \grad_phi bzw. \grad_phi^T * B^(-1) Corresponds to \hat{grad_phi}.
        vec3D_dbl_Type dPhiTrans(dPhi->size(), vec2D_dbl_Type(dPhi->at(0).size(), vec_dbl_Type(dim, 0.)));
        Helper::applyBTinv(dPhi, dPhiTrans, Binv); // so dPhiTrans corresponds now to our basisfunction in natural coordinates

        // Compute shear rate gammaDot, which is a vector because it is evaluated at a gaussian quadrature point for that compute velocity gradient
        vec_dbl_ptr_Type gammaDot(new vec_dbl_Type(QuadWeightsReference.size(), 0.0)); // gammaDot->at(j) j=0...weights
        computeShearRate(dPhiTrans, gammaDot, dim);                      // updates gammaDot using velocity solution
        double viscosity_atw = 0.;

        TEUCHOS_TEST_FOR_EXCEPTION(dim == 1, std::logic_error, "AssemblyNeumannBoundaryTerm Not implemented for dim=1");

        if (dim == 2)
        {
            double v11, v12, v21, v22, deta_dgamma_dgamma_dtau;
            deta_dgamma_dgamma_dtau = 0.;
            //************************************
            //************************************
            // Due to the extra term related to the Gaetaeux-derivative there arise prefactors which depend on the velocity gradients
            // which therefore also have to be computed here
            vec_dbl_Type u11(dPhiTrans.size(), -1.);                         // should correspond to du/dx at each quadrature point
            vec_dbl_Type u12(dPhiTrans.size(), -1.);                         // should correspond to du/dy at each quadrature point
            vec_dbl_Type u21(dPhiTrans.size(), -1.);                         // should correspond to dv/dx at each quadrature point
            vec_dbl_Type u22(dPhiTrans.size(), -1.);                         // should correspond to dv/dy at each quadrature point
            vec_dbl_ptr_Type gammaDot(new vec_dbl_Type(QuadWeightsReference.size(), 0.0)); // gammaDot->at(j) j=0...weights

            vec_dbl_ptr_Type a1(new vec_dbl_Type(QuadWeightsReference.size(), 0.0)); // prefactor a= 2*(du/dx)^2 + (du/dy+dv/dx)*dv/dx
            vec_dbl_ptr_Type b1(new vec_dbl_Type(QuadWeightsReference.size(), 0.0)); // prefactor b= (du/dy+dv/dx)*du/dx + 2*(dv/dx)*dv/dy
            vec_dbl_ptr_Type c1(new vec_dbl_Type(QuadWeightsReference.size(), 0.0)); //   prefactor c= (du/dy+dv/dx)*dv/dy + 2*(du/dx)*du/dy
            vec_dbl_ptr_Type d1(new vec_dbl_Type(QuadWeightsReference.size(), 0.0)); //   prefactor c= 2*(dv/dy)^2 + (du/dy+dv/dx)*du/dy
            for (UN w = 0; w < dPhiTrans.size(); w++)
            { // quads points
                // set again to zero
                u11[w] = 0.0;
                u12[w] = 0.0;
                u21[w] = 0.0;
                u22[w] = 0.0;
                for (UN i = 0; i < dPhiTrans[0].size(); i++)
                {                                                              // loop unrolling
                    LO index1 = dim * i + 0;                                   // x
                    LO index2 = dim * i + 1;                                   // y
                    u11[w] += (*this->solution_)[index1] * dPhiTrans[w][i][0]; // u*dphi_dx
                    u12[w] += (*this->solution_)[index1] * dPhiTrans[w][i][1]; // because we are in 2D , 0 and 1
                    u21[w] += (*this->solution_)[index2] * dPhiTrans[w][i][0];
                    u22[w] += (*this->solution_)[index2] * dPhiTrans[w][i][1];
                }
                gammaDot->at(w) = sqrt(2.0 * u11[w] * u11[w] + 2.0 * u22[w] * u22[w] + (u12[w] + u21[w]) * (u12[w] + u21[w]));

                a1->at(w) = 2.0 * (u11[w] * u11[w]) + (u12[w] + u21[w]) * (u21[w]);
                b1->at(w) = 2.0 * (u21[w] * u22[w]) + (u12[w] + u21[w]) * (u11[w]);
                c1->at(w) = 2.0 * (u11[w] * u12[w]) + (u12[w] + u21[w]) * (u22[w]);
                d1->at(w) = 2.0 * (u22[w] * u22[w]) + (u12[w] + u21[w]) * (u12[w]);
            }
            // loop over basis functions
            for (UN i = 0; i < phi->at(0).size(); i++)
            {
                for (UN j = 0; j < numNodes; j++)
                {
                    // Reset values
                    v11 = 0.0;
                    v12 = 0.0;
                    v21 = 0.0;
                    v22 = 0.0;

                    // loop over basis functions quadrature points
                    for (UN w = 0; w < phi->size(); w++)
                    {
                        this->viscosityModel->evaluateDerivative(this->params_, gammaDot->at(w), deta_dgamma_dgamma_dtau);
                        v11 = v11 + (0.25 * deta_dgamma_dgamma_dtau) * QuadWeightsReference[w]  * ((2.0 * dPhiTrans[w][j][0] * a1->at(w) + dPhiTrans[w][j][1] * b1->at(w)) * normalVector[0] + (dPhiTrans[w][j][1] * a1->at(w)) * normalVector[1]) * ((*phi)[w][i]); // xx contribution:
                        v12 = v12 + (0.25 * deta_dgamma_dgamma_dtau) * QuadWeightsReference[w]  * ((dPhiTrans[w][j][0] * b1->at(w)) * normalVector[0] + (dPhiTrans[w][j][0] * a1->at(w) + 2.0 * dPhiTrans[w][j][1] * b1->at(w)) * normalVector[1]) * ((*phi)[w][i]); // xy contribution:
                        v21 = v21 + (0.25 * deta_dgamma_dgamma_dtau) * QuadWeightsReference[w]  * ((2.0 * dPhiTrans[w][j][0] * c1->at(w) + dPhiTrans[w][j][1] * d1->at(w)) * normalVector[0] + (dPhiTrans[w][j][1] * c1->at(w)) * normalVector[1]) * ((*phi)[w][i]); // yx contribution:
                        v22 = v22 + (0.25 * deta_dgamma_dgamma_dtau) * QuadWeightsReference[w]  * ((dPhiTrans[w][j][0] * d1->at(w)) * normalVector[0] + (dPhiTrans[w][j][0] * c1->at(w) + 2.0 * dPhiTrans[w][j][1] * d1->at(w)) * normalVector[1]) * ((*phi)[w][i]); // yy contribution:

                    } // End loop over quadrature points

                    // multiply determinant from transformation
                    v11 *= elscaling;
                    v12 *= elscaling;
                    v21 *= elscaling;
                    v22 *= elscaling;

                    // Put values on the right position in element matrix - d=2 because we are in two dimensional case
                    // [v11  v12 ]
                    // [v21  v22 ]
                    (*elementMatrix)[i * dofs][j * dofs] = v11;     // d=0, first dimension
                    (*elementMatrix)[i * dofs][j * dofs + 1] = v12; //
                    (*elementMatrix)[i * dofs + 1][j * dofs] = v21;
                    (*elementMatrix)[i * dofs + 1][j * dofs + 1] = v22; // d=1, second dimension

                } // End loop over j nodes

            } // End loop over i nodes
        }     // End dim==2
        else if (dim == 3)
        {
            vec_dbl_Type u11(dPhiTrans.size(), -1.); // should correspond to du/dx at each quadrature point
            vec_dbl_Type u12(dPhiTrans.size(), -1.); // should correspond to du/dy at each quadrature point
            vec_dbl_Type u13(dPhiTrans.size(), -1.); // should correspond to du/dz at each quadrature point

            vec_dbl_Type u21(dPhiTrans.size(), -1.); // should correspond to dv/dx at each quadrature point
            vec_dbl_Type u22(dPhiTrans.size(), -1.); // should correspond to dv/dy at each quadrature point
            vec_dbl_Type u23(dPhiTrans.size(), -1.); // should correspond to dv/dz at each quadrature point

            vec_dbl_Type u31(dPhiTrans.size(), -1.); // should correspond to dw/dx at each quadrature point
            vec_dbl_Type u32(dPhiTrans.size(), -1.); // should correspond to dw/dy at each quadrature point
            vec_dbl_Type u33(dPhiTrans.size(), -1.); // should correspond to dw/dz at each quadrature point

            vec_dbl_ptr_Type gammaDot(new vec_dbl_Type(QuadWeightsReference.size(), 0.0)); // gammaDot->at(j) j=0...weights

            vec_dbl_ptr_Type a1(new vec_dbl_Type(QuadWeightsReference.size(), 0.0)); // prefactor a= 2*(du/dx)^2 + (du/dy+dv/dx)*xv/dy +  (dw/dx+du/dz)*dw/dx
            vec_dbl_ptr_Type b1(new vec_dbl_Type(QuadWeightsReference.size(), 0.0)); // prefactor b=
            vec_dbl_ptr_Type c1(new vec_dbl_Type(QuadWeightsReference.size(), 0.0)); // prefactor c=
            vec_dbl_ptr_Type d1(new vec_dbl_Type(QuadWeightsReference.size(), 0.0)); // prefactor d=
            vec_dbl_ptr_Type e1(new vec_dbl_Type(QuadWeightsReference.size(), 0.0)); // prefactor e=
            vec_dbl_ptr_Type f1(new vec_dbl_Type(QuadWeightsReference.size(), 0.0)); // prefactor e=
            vec_dbl_ptr_Type g1(new vec_dbl_Type(QuadWeightsReference.size(), 0.0)); // prefactor g=
            vec_dbl_ptr_Type h1(new vec_dbl_Type(QuadWeightsReference.size(), 0.0)); // prefactor g=
            vec_dbl_ptr_Type i1(new vec_dbl_Type(QuadWeightsReference.size(), 0.0)); // prefactor g=

            for (UN w = 0; w < dPhiTrans.size(); w++)
            { // quads points

                u11[w] = 0.0;
                u12[w] = 0.0;
                u13[w] = 0.0;
                u21[w] = 0.0;
                u22[w] = 0.0;
                u23[w] = 0.0;
                u31[w] = 0.0;
                u32[w] = 0.0;
                u33[w] = 0.0;

                for (UN i = 0; i < dPhiTrans[0].size(); i++)
                {
                    LO index1 = dim * i + 0; // x
                    LO index2 = dim * i + 1; // y
                    LO index3 = dim * i + 2; // z
                    // uLoc[d][w] += (*this->solution_)[index] * phi->at(w).at(i);
                    u11[w] += (*this->solution_)[index1] * dPhiTrans[w][i][0]; // u*dphi_dx
                    u12[w] += (*this->solution_)[index1] * dPhiTrans[w][i][1]; // because we are in 3D , 0 and 1, 2
                    u13[w] += (*this->solution_)[index1] * dPhiTrans[w][i][2];
                    u21[w] += (*this->solution_)[index2] * dPhiTrans[w][i][0]; // v*dphi_dx
                    u22[w] += (*this->solution_)[index2] * dPhiTrans[w][i][1];
                    u23[w] += (*this->solution_)[index2] * dPhiTrans[w][i][2];
                    u31[w] += (*this->solution_)[index3] * dPhiTrans[w][i][0]; // w*dphi_dx
                    u32[w] += (*this->solution_)[index3] * dPhiTrans[w][i][1];
                    u33[w] += (*this->solution_)[index3] * dPhiTrans[w][i][2];
                }
                gammaDot->at(w) = sqrt(2.0 * u11[w] * u11[w] + 2.0 * u22[w] * u22[w] + 2.0 * u33[w] * u33[w] + (u12[w] + u21[w]) * (u12[w] + u21[w]) + (u13[w] + u31[w]) * (u13[w] + u31[w]) + (u23[w] + u32[w]) * (u23[w] + u32[w]));

                a1->at(w) = 2.0 * (u11[w] * u11[w]) + (u12[w] + u21[w]) * (u21[w]) + (u31[w] + u13[w]) * (u31[w]);
                b1->at(w) = 2.0 * (u21[w] * u22[w]) + (u12[w] + u21[w]) * (u11[w]) + (u32[w] + u23[w]) * (u31[w]);
                c1->at(w) = 2.0 * (u33[w] * u31[w]) + (u13[w] + u31[w]) * (u11[w]) + (u32[w] + u23[w]) * (u21[w]);
                d1->at(w) = 2.0 * (u11[w] * u12[w]) + (u12[w] + u21[w]) * (u22[w]) + (u31[w] + u13[w]) * (u32[w]);
                e1->at(w) = 2.0 * (u22[w] * u22[w]) + (u12[w] + u21[w]) * (u12[w]) + (u32[w] + u23[w]) * (u32[w]);
                f1->at(w) = 2.0 * (u33[w] * u32[w]) + (u13[w] + u31[w]) * (u12[w]) + (u32[w] + u23[w]) * (u22[w]);
                g1->at(w) = 2.0 * (u11[w] * u13[w]) + (u12[w] + u21[w]) * (u23[w]) + (u31[w] + u13[w]) * (u33[w]);
                h1->at(w) = 2.0 * (u22[w] * u23[w]) + (u12[w] + u21[w]) * (u13[w]) + (u32[w] + u23[w]) * (u33[w]);
                i1->at(w) = 2.0 * (u33[w] * u33[w]) + (u13[w] + u31[w]) * (u13[w]) + (u32[w] + u23[w]) * (u23[w]);
            }
            double v11, v12, v13, v21, v22, v23, v31, v32, v33, deta_dgamma_dgamma_dtau; // helper values for entries
            deta_dgamma_dgamma_dtau = 0.;

            // loop over basis functions
            for (UN i = 0; i < phi->at(0).size(); i++)
            {
                for (UN j = 0; j < numNodes; j++)
                {
                    // Reset values
                    v11 = 0.0;
                    v12 = 0.0;
                    v13 = 0.0;
                    v21 = 0.0;
                    v22 = 0.0;
                    v23 = 0.0;
                    v31 = 0.0;
                    v32 = 0.0;
                    v33 = 0.0;

                    // loop over basis functions quadrature points
                    for (UN w = 0; w < phi->size(); w++)
                    {
                        this->viscosityModel->evaluateMapping(this->params_, gammaDot->at(w), viscosity_atw);

                        v11 = v11 + (0.25 * deta_dgamma_dgamma_dtau) * QuadWeightsReference[w]  * ((2.0 * dPhiTrans[w][j][0] * a1->at(w) + dPhiTrans[w][j][1] * b1->at(w) + dPhiTrans[w][j][2] * c1->at(w)) * normalVector[0] + (dPhiTrans[w][j][1] * a1->at(w)) * normalVector[1] + (dPhiTrans[w][j][2] * a1->at(w)) * normalVector[2]) * ((*phi)[w][i]); // xx contribution:
                        v12 = v12 + (0.25 * deta_dgamma_dgamma_dtau) * QuadWeightsReference[w]  * ((dPhiTrans[w][j][0] * b1->at(w)) * normalVector[0] + (dPhiTrans[w][j][0] * a1->at(w) + 2.0 * dPhiTrans[w][j][1] * b1->at(w) + dPhiTrans[w][j][2] * c1->at(w)) * normalVector[1] + (dPhiTrans[w][j][2] * b1->at(w)) * normalVector[2]) * ((*phi)[w][i]); // xy contribution:
                        v13 = v13 + (0.25 * deta_dgamma_dgamma_dtau) * QuadWeightsReference[w]  * ((dPhiTrans[w][j][0] * c1->at(w)) * normalVector[0] + (dPhiTrans[w][j][1] * c1->at(w)) * normalVector[1] + (dPhiTrans[w][j][0] * a1->at(w) + dPhiTrans[w][j][1] * b1->at(w) + 2.0 * dPhiTrans[w][j][2] * c1->at(w)) * normalVector[2]) * ((*phi)[w][i]); // xz contribution:

                        v21 = v21 + (0.25 * deta_dgamma_dgamma_dtau) * QuadWeightsReference[w]  * ((2.0 * dPhiTrans[w][j][0] * d1->at(w) + dPhiTrans[w][j][1] * e1->at(w) + dPhiTrans[w][j][2] * f1->at(w)) * normalVector[0] + (dPhiTrans[w][j][1] * d1->at(w)) * normalVector[1] + (dPhiTrans[w][j][2] * d1->at(w)) * normalVector[2]) * ((*phi)[w][i]); // xx contribution:
                        v22 = v22 + (0.25 * deta_dgamma_dgamma_dtau) * QuadWeightsReference[w]  * ((dPhiTrans[w][j][0] * e1->at(w)) * normalVector[0] + (dPhiTrans[w][j][0] * d1->at(w) + 2.0 * dPhiTrans[w][j][1] * e1->at(w) + dPhiTrans[w][j][2] * f1->at(w)) * normalVector[1] + (dPhiTrans[w][j][2] * e1->at(w)) * normalVector[2]) * ((*phi)[w][i]); // xy contribution:
                        v23 = v23 + (0.25 * deta_dgamma_dgamma_dtau) * QuadWeightsReference[w]  * ((dPhiTrans[w][j][0] * f1->at(w)) * normalVector[0] + (dPhiTrans[w][j][1] * f1->at(w)) * normalVector[1] + (dPhiTrans[w][j][0] * f1->at(w) + dPhiTrans[w][j][1] * e1->at(w) + 2.0 * dPhiTrans[w][j][2] * f1->at(w)) * normalVector[2]) * ((*phi)[w][i]); // xz contribution:

                        v31 = v31 + (0.25 * deta_dgamma_dgamma_dtau) * QuadWeightsReference[w]  * ((2.0 * dPhiTrans[w][j][0] * g1->at(w) + dPhiTrans[w][j][1] * h1->at(w) + dPhiTrans[w][j][2] * i1->at(w)) * normalVector[0] + (dPhiTrans[w][j][1] * g1->at(w)) * normalVector[1] + (dPhiTrans[w][j][2] * g1->at(w)) * normalVector[2]) * ((*phi)[w][i]); // xx contribution:
                        v32 = v32 + (0.25 * deta_dgamma_dgamma_dtau) * QuadWeightsReference[w]  * ((dPhiTrans[w][j][0] * h1->at(w)) * normalVector[0] + (dPhiTrans[w][j][0] * g1->at(w) + 2.0 * dPhiTrans[w][j][1] * h1->at(w) + dPhiTrans[w][j][2] * i1->at(w)) * normalVector[1] + (dPhiTrans[w][j][2] * h1->at(w)) * normalVector[2]) * ((*phi)[w][i]); // xy contribution:
                        v33 = v33 + (0.25 * deta_dgamma_dgamma_dtau) * QuadWeightsReference[w]  * ((dPhiTrans[w][j][0] * i1->at(w)) * normalVector[0] + (dPhiTrans[w][j][1] * i1->at(w)) * normalVector[1] + (dPhiTrans[w][j][0] * g1->at(w) + dPhiTrans[w][j][1] * h1->at(w) + 2.0 * dPhiTrans[w][j][2] * i1->at(w)) * normalVector[2]) * ((*phi)[w][i]); // xz contribution:

                    } // End loop over quadrature points

                    // multiply determinant from transformation
                    v11 *= elscaling;
                    v12 *= elscaling;
                    v13 *= elscaling;
                    v21 *= elscaling;
                    v22 *= elscaling;
                    v23 *= elscaling;
                    v31 *= elscaling;
                    v32 *= elscaling;
                    v33 *= elscaling;

                    // Put values on the right position in element matrix - d=2 because we are in two dimensional case
                    // [v11  v12  v13]
                    // [v21  v22  v23]
                    // [v31  v32  v33]
                    (*elementMatrix)[i * dofs][j * dofs] = v11; // d=0, first dimension
                    (*elementMatrix)[i * dofs][j * dofs + 1] = v12;
                    (*elementMatrix)[i * dofs][j * dofs + 2] = v13;
                    (*elementMatrix)[i * dofs + 1][j * dofs] = v21;
                    (*elementMatrix)[i * dofs + 1][j * dofs + 1] = v22; // d=1, second dimension
                    (*elementMatrix)[i * dofs + 1][j * dofs + 2] = v23; // d=1, second dimension
                    (*elementMatrix)[i * dofs + 2][j * dofs] = v31;
                    (*elementMatrix)[i * dofs + 2][j * dofs + 1] = v32; // d=2, third dimension
                    (*elementMatrix)[i * dofs + 2][j * dofs + 2] = v33; // d=2, third dimension

                } // End loop over j nodes

            } // End loop over i nodes
        }     // end if 3d
        // TEUCHOS_TEST_FOR_EXCEPTION(dim == 3,std::logic_error, "AssemblyNeumannBoundaryTerm Not implemented for dim=3");

    } // Function End loop



    

    // "Fixpunkt"- Matrix without jacobian for calculating Ax
    // Here update please to unlinearized System Matrix accordingly.
    template <class SC, class LO, class GO, class NO>
    void AssembleFEGeneralizedNewtonian<SC, LO, GO, NO>::assembleRHS()
    {

        SmallMatrixPtr_Type elementMatrixN = Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));
        SmallMatrixPtr_Type elementMatrixNC = Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));

        this->ANB_.reset(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_)); // A + B + N
        this->ANB_->add((*this->constantMatrix_), (*this->ANB_));

        // Nonlinear shear stress tensor *******************************
        this->assemblyStress(elementMatrixN);
        this->ANB_->add((*elementMatrixN), (*this->ANB_));

        // Nonlinear convection term *******************************
        this->assemblyAdvection(elementMatrixNC);
        elementMatrixNC->scale(this->density_);
        this->ANB_->add((*elementMatrixNC), (*this->ANB_));

        // If boundary element - nonlinear boundar term *******************************
        if (this->FEObject_->getNeumannBCElement() == true) // Our corresponding FE Elements corresponds to element on the outer boundary where we want to assign a special outflow boundary condition
        {
            SmallMatrixPtr_Type elementMatrixNB = Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));
            this->assemblyOutflowNeumannBoundaryTerm(elementMatrixNB);
            this->ANB_->add((*elementMatrixNB), ((*this->ANB_)));
        }

        this->rhsVec_.reset(new vec_dbl_Type(this->dofsElement_, 0.));
        // Multiplying ANB_ * solution // ANB Matrix without nonlinear part.
        int s = 0, t = 0;
        for (int i = 0; i < this->ANB_->size(); i++)
        {
            if (i >= this->dofsElementVelocity_)
                s = 1;
            for (int j = 0; j < this->ANB_->size(); j++)
            {
                if (j >= this->dofsElementVelocity_)
                    t = 1;
                (*this->rhsVec_)[i] += (*this->ANB_)[i][j] * (*this->solution_)[j] * this->coeff_[s][t];
            }
            t = 0;
        }
    }

    /*!
    Building Transformation
    */
    /* In 2D B=[ nodesRefConfig(1).at(0)-nodesRefConfig(0).at(0)    nodesRefConfig(2).at(0)-nodesRefConfig(0).at(0)    ]
               [ nodesRefConfig(1).at(1)-nodesRefConfig(0).at(1)    nodesRefConfig(2).at(1)-nodesRefConfig(0).at(1)   ]
    /*!
        - Triangle numbering

                        2
                        * *
                        *   *
                        5     4
                        *       *
                        *         *
                        0 * * 3 * * 1

        In 2D B=[ 1(x)-0(x)    2(x)-0(x)  ]
                [ 1(y)-0(y)    2(y)-0(y)  ]
    */


    // Compute Shear Rate on quadrature points depending on gradient of velocity solution at nodes
    template <class SC, class LO, class GO, class NO>
    void AssembleFEGeneralizedNewtonian<SC, LO, GO, NO>::computeShearRate(vec3D_dbl_Type dPhiTrans,
                                                                          vec_dbl_ptr_Type &gammaDot, int dim)
    {

        //****************** TWO DIMENSIONAL *********************************
        if (dim == 2)
        {

            vec_dbl_Type u11(dPhiTrans.size(), -1.); // should correspond to du/dx at each quadrature point
            vec_dbl_Type u12(dPhiTrans.size(), -1.); // should correspond to du/dy at each quadrature point
            vec_dbl_Type u21(dPhiTrans.size(), -1.); // should correspond to dv/dx at each quadrature point
            vec_dbl_Type u22(dPhiTrans.size(), -1.); // should correspond to dv/dy at each quadrature point

            for (UN w = 0; w < dPhiTrans.size(); w++)
            { // quads points
                // set again to zero
                u11[w] = 0.0;
                u12[w] = 0.0;
                u21[w] = 0.0;
                u22[w] = 0.0;
                for (UN i = 0; i < dPhiTrans[0].size(); i++)
                {                            // loop unrolling
                    LO index1 = dim * i + 0; // x
                    LO index2 = dim * i + 1; // y
                    // uLoc[d][w] += (*this->solution_)[index] * phi->at(w).at(i);
                    u11[w] += (*this->solution_)[index1] * dPhiTrans[w][i][0]; // u*dphi_dx
                    u12[w] += (*this->solution_)[index1] * dPhiTrans[w][i][1]; // because we are in 2D , 0 and 1
                    u21[w] += (*this->solution_)[index2] * dPhiTrans[w][i][0];
                    u22[w] += (*this->solution_)[index2] * dPhiTrans[w][i][1];
                }
                gammaDot->at(w) = sqrt(2.0 * u11[w] * u11[w] + 2.0 * u22[w] * u22[w] + (u12[w] + u21[w]) * (u12[w] + u21[w])); 
            }
        } // end if dim == 2
        //****************** THREE DIMENSIONAL *********************************
        else if (dim == 3)
        {

            vec_dbl_Type u11(dPhiTrans.size(), -1.); // should correspond to du/dx at each quadrature point
            vec_dbl_Type u12(dPhiTrans.size(), -1.); // should correspond to du/dy at each quadrature point
            vec_dbl_Type u13(dPhiTrans.size(), -1.); // should correspond to du/dz at each quadrature point

            vec_dbl_Type u21(dPhiTrans.size(), -1.); // should correspond to dv/dx at each quadrature point
            vec_dbl_Type u22(dPhiTrans.size(), -1.); // should correspond to dv/dy at each quadrature point
            vec_dbl_Type u23(dPhiTrans.size(), -1.); // should correspond to dv/dz at each quadrature point

            vec_dbl_Type u31(dPhiTrans.size(), -1.); // should correspond to dw/dx at each quadrature point
            vec_dbl_Type u32(dPhiTrans.size(), -1.); // should correspond to dw/dy at each quadrature point
            vec_dbl_Type u33(dPhiTrans.size(), -1.); // should correspond to dw/dz at each quadrature point

            for (UN w = 0; w < dPhiTrans.size(); w++)
            { // quads points
                // set again to zero
                u11[w] = 0.0;
                u12[w] = 0.0;
                u13[w] = 0.0;
                u21[w] = 0.0;
                u22[w] = 0.0;
                u23[w] = 0.0;
                u31[w] = 0.0;
                u32[w] = 0.0;
                u33[w] = 0.0;

                for (UN i = 0; i < dPhiTrans[0].size(); i++)
                {
                    LO index1 = dim * i + 0; // x
                    LO index2 = dim * i + 1; // y
                    LO index3 = dim * i + 2; // z
                    // uLoc[d][w] += (*this->solution_)[index] * phi->at(w).at(i);
                    u11[w] += (*this->solution_)[index1] * dPhiTrans[w][i][0]; // u*dphi_dx
                    u12[w] += (*this->solution_)[index1] * dPhiTrans[w][i][1]; // because we are in 3D , 0 and 1, 2
                    u13[w] += (*this->solution_)[index1] * dPhiTrans[w][i][2];
                    u21[w] += (*this->solution_)[index2] * dPhiTrans[w][i][0]; // v*dphi_dx
                    u22[w] += (*this->solution_)[index2] * dPhiTrans[w][i][1];
                    u23[w] += (*this->solution_)[index2] * dPhiTrans[w][i][2];
                    u31[w] += (*this->solution_)[index3] * dPhiTrans[w][i][0]; // w*dphi_dx
                    u32[w] += (*this->solution_)[index3] * dPhiTrans[w][i][1];
                    u33[w] += (*this->solution_)[index3] * dPhiTrans[w][i][2];
                }
                gammaDot->at(w) = sqrt(2.0 * u11[w] * u11[w] + 2.0 * u22[w] * u22[w] + 2.0 * u33[w] * u33[w] + (u12[w] + u21[w]) * (u12[w] + u21[w]) + (u13[w] + u31[w]) * (u13[w] + u31[w]) + (u23[w] + u32[w]) * (u23[w] + u32[w]));
            }
        } // end if dim == 3
    }

    /* Based on the current solution (velocity, pressure etc.) we want to be able to compute postprocessing fields
    like here the viscosity inside an element.
    */
    template <class SC, class LO, class GO, class NO>
    void AssembleFEGeneralizedNewtonian<SC, LO, GO, NO>::computeLocalconstOutputField()
    {
        int dim = this->getDim();
        string FEType = this->FETypeVelocity_;

        SC detB;
        SmallMatrix<SC> B(dim);
        SmallMatrix<SC> Binv(dim);

        this->buildTransformation(B);
        detB = B.computeInverse(Binv);

        vec3D_dbl_ptr_Type dPhiAtCM;

        // Compute viscosity at center of mass using nodal values and shape function **********************************************************************************
        TEUCHOS_TEST_FOR_EXCEPTION(dim == 1, std::logic_error, "computeLocalconstOutputField Not implemented for dim=1");

        Helper::getDPhiAtCM(dPhiAtCM, dim, FEType); // These are the original coordinates of the reference element
        vec3D_dbl_Type dPhiTransAtCM(dPhiAtCM->size(), vec2D_dbl_Type(dPhiAtCM->at(0).size(), vec_dbl_Type(dim, 0.)));
        Helper::applyBTinv(dPhiAtCM, dPhiTransAtCM, Binv); // We need transformation because of velocity gradient in shear rate equation

        vec_dbl_ptr_Type gammaDoti(new vec_dbl_Type(dPhiAtCM->size(), 0.0)); // Only one value because size is one
        computeShearRate(dPhiTransAtCM, gammaDoti, dim);                     // updates gammaDot using velocity solution
        this->viscosityModel->evaluateMapping(this->params_, gammaDoti->at(0), this->constOutputField_.at(0));
    }

}
#endif
