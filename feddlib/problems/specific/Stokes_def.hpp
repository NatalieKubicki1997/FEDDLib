#ifndef STOKES_def_hpp
#define STOKES_def_hpp
#include "Stokes_decl.hpp"
/*!
 Definition of Stokes
 
 @brief Stokes
 @author Christian Hochmuth
 @version 1.0
 @copyright CH
 */

namespace FEDD {
void ZeroDirichlet(double* x, double* res, double t, double* parameters){
    
    res[0] = 0.;
    
    return;
}
    
double OneFunction(double* x, int* parameter)
{
    return 1.0;
}
    
template<class SC,class LO,class GO,class NO>
Stokes<SC,LO,GO,NO>::Stokes(const DomainConstPtr_Type &domainVelocity, std::string FETypeVelocity, const DomainConstPtr_Type &domainPressure, std::string FETypePressure, ParameterListPtr_Type parameterList):
Problem<SC,LO,GO,NO>(parameterList, domainVelocity->getComm())
{

    this->addVariable( domainVelocity , FETypeVelocity , "u" , domainVelocity->getDimension());
    this->addVariable( domainPressure , FETypePressure , "p" , 1);
    
    //This will probably not be necessary but we keep it for now
   /*if(this->parameterList_->sublist("Parameter").get("Use Pressure Correction",true) && !this->getFEType(0).compare("P2") && !this->parameterList_->sublist("General").get("Preconditioner Method","Monolithic").compare("Monolithic")){ // We only correct pressure in P2 Case
        
        Teuchos::RCP<Domain<SC,LO,GO,NO> > domainLambda( new Domain<SC,LO,GO,NO>( this->getDomain(0)->getComm(), this->dim_ ) );
        
        vec_GO_Type globalInterfaceIDs(0);
		if(this->getDomain(0)->getComm()->getRank() ==0 )
			globalInterfaceIDs.push_back(0);
		Teuchos::ArrayView<GO> globalEdgesInterfaceArray = Teuchos::arrayViewFromVector( globalInterfaceIDs);

		MapPtr_Type mapNode = Teuchos::rcp( new Map_Type( this->getDomain(1)->getMapUnique()->getUnderlyingLib(), Teuchos::OrdinalTraits<GO>::invalid(), globalEdgesInterfaceArray, 0, this->getDomain(0)->getComm()) );

        domainLambda->initDummyMesh(mapNode);

        this->addVariable( domainLambda , FETypePressure , "lambda" , 1);

    }*/
    this->dim_ = this->getDomain(0)->getDimension();
}

template<class SC,class LO,class GO,class NO>
Stokes<SC,LO,GO,NO>::~Stokes(){

}

template<class SC,class LO,class GO,class NO>
void Stokes<SC,LO,GO,NO>::info(){
    this->infoProblem();
}
    
template<class SC,class LO,class GO,class NO>
void Stokes<SC,LO,GO,NO>::assemble( std::string type ) const{
    
    if (this->verbose_)
        std::cout << "-- Assembly ... " << std::flush;

    double viscosity = this->parameterList_->sublist("Parameter").get("Viscosity",1.);

    MatrixPtr_Type A(new Matrix_Type( this->getDomain(0)->getMapVecFieldUnique(), this->getDomain(0)->getApproxEntriesPerRow() ) );
    MatrixPtr_Type BT(new Matrix_Type( this->getDomain(0)->getMapVecFieldUnique(), this->getDomain(1)->getDimension() * this->getDomain(1)->getApproxEntriesPerRow() ) );
    
    MapConstPtr_Type pressureMap;
    if ( this->getDomain(1)->getFEType() == "P0" )
        pressureMap = this->getDomain(1)->getElementMap();
    else
        pressureMap = this->getDomain(1)->getMapUnique();

    MatrixPtr_Type B(new Matrix_Type( pressureMap, this->getDomain(0)->getDimension() * this->getDomain(0)->getApproxEntriesPerRow() ) );
    
    MatrixPtr_Type C;
    if (this->verbose_)
        std::cout << " A ... " << std::flush;
    int* dummy;
    if ( this->parameterList_->sublist("Parameter").get("Symmetric gradient",false) )
        this->feFactory_->assemblyStress(this->dim_, this->domain_FEType_vec_.at(0), A, OneFunction, dummy, true);
    else
        this->feFactory_->assemblyLaplaceVecField(this->dim_, this->domain_FEType_vec_.at(0), 2, A, true);

    if (this->verbose_)
        std::cout << "B and B^T ... " << std::flush;

    this->feFactory_->assemblyDivAndDivT(this->dim_, this->getFEType(0), this->getFEType(1), 2, B, BT, this->getDomain(0)->getMapVecFieldUnique(), pressureMap, true );
    
    A->resumeFill();
    B->resumeFill();
    BT->resumeFill();
    
    A->scale(viscosity);
    B->scale(-1.);
    BT->scale(-1.);

    A->fillComplete( this->getDomain(0)->getMapVecFieldUnique(), this->getDomain(0)->getMapVecFieldUnique());
    B->fillComplete( this->getDomain(0)->getMapVecFieldUnique(), pressureMap );
    BT->fillComplete( pressureMap, this->getDomain(0)->getMapVecFieldUnique() );
    
    this->system_.reset(new BlockMatrix_Type(2));
    
    // In case of a monolithic preconditioner and a P2-P1 discretization we have the option to correct the pressure to have mean value = 0. This way, generally, we can improve scalabilty and results. 
    // The real correction is then done via projection in the Overlapping Operator of FROSch,here we only assemble a as \int p dx . a is assembled as a column vector but in the Dissertation of C. Hochmuth defined as row.
    if(this->parameterList_->sublist("Parameter").get("Use Pressure Correction",false) && !this->getFEType(0).compare("P2") && !this->parameterList_->sublist("General").get("Preconditioner Method","Monolithic").compare("Monolithic")){ 
        // Projection vector a: \int p dx, for pressure component and 0 for velocity.
        BlockMultiVectorPtr_Type projection(new BlockMultiVector_Type (2));

        MultiVectorPtr_Type P(new MultiVector_Type( this->getDomain(1)->getMapUnique(), 1 ) );

        this->feFactory_->assemblyPressureMeanValue( this->dim_,"P1",P) ;

        MultiVectorPtr_Type vel0(new MultiVector_Type( this->getDomain(0)->getMapVecFieldUnique(), 1 ) );
        vel0->putScalar(0.);

        // Adding components to projection vector 
        projection->addBlock(vel0,0);
        projection->addBlock(P,1);

        // Setting projection vector in preconditioner to later pass to paramterlist in FROSch
        this->getPreconditionerConst()->setPressureProjection( projection );    

        if (this->verbose_)
            std::cout << "\n 'Use pressure correction' was set to 'true'. This requieres a version of Trilinos of that includes pressure correction in the FROSch_OverlappingOperator!!" << std::endl;  


    }
    this->system_->addBlock( A, 0, 0 );
    this->system_->addBlock( BT, 0, 1 );
    this->system_->addBlock( B, 1, 0 );
    
//    this->initializeVectors();
    
    if ( !this->getFEType(0).compare("P1") ) {
        C.reset(new Matrix_Type( this->getDomain(1)->getMapUnique(), this->getDomain(1)->getApproxEntriesPerRow() ) );
        this->feFactory_->assemblyBDStabilization( this->dim_, "P1", C, true);
        C->resumeFill();
        C->scale( -1./viscosity );
        C->fillComplete( pressureMap, pressureMap );
        this->system_->addBlock( C, 1, 1 );
    }
#ifdef FEDD_HAVE_TEKO
    if ( !this->parameterList_->sublist("General").get("Preconditioner Method","Monolithic").compare("Teko") ) {
        if (this->parameterList_->sublist("General").get("Assemble Velocity Mass",false)) {
            MatrixPtr_Type Mvelocity(new Matrix_Type( this->getDomain(0)->getMapVecFieldUnique(), this->getDomain(0)->getApproxEntriesPerRow() ) );
            //
            this->feFactory_->assemblyMass( this->dim_, this->domain_FEType_vec_.at(0), "Vector", Mvelocity, true );
            //
            this->getPreconditionerConst()->setVelocityMassMatrix( Mvelocity );
           if (this->verbose_)
                std::cout << "\nVelocity mass matrix for LSC block preconditioner is assembled and used for the preconditioner." << std::endl;
        } else {
            if (this->verbose_)
                std::cout << "\nVelocity mass matrix for LSC block preconditioner not assembled and thus approximated/replaced with matrix F (fluid)." << std::endl;
        }
        if(!this->parameterList_->sublist("Teko Parameters").sublist("Preconditioner Types").sublist("Teko").get("Inverse Type","SIMPLE").compare("PCD")){
             // Pressure mass matrix
            MatrixPtr_Type Mpressure(new Matrix_Type( this->getDomain(1)->getMapUnique(), this->getDomain(1)->getApproxEntriesPerRow() ) );
            this->feFactory_->assemblyMass( this->dim_, this->domain_FEType_vec_.at(1), "Scalar", Mpressure, true ); //assemblyIdentity(Mpressure);//
            this->getPreconditionerConst()->setPressureMass( Mpressure );

            // Pressure Laplace matrix
            MatrixPtr_Type Lpressure(new Matrix_Type( this->getDomain(1)->getMapUnique(), this->getDomain(1)->getApproxEntriesPerRow() ) );
            this->feFactory_->assemblyLaplace( this->dim_, this->domain_FEType_vec_.at(1), 2, Lpressure, true );//assemblyIdentity(Lpressure); //
            BlockMatrixPtr_Type dummy(new BlockMatrix_Type (1));
            dummy->addBlock(Lpressure,0,0);
            this->bcFactoryPressureLaplace_->setSystem(dummy); 
            this->getPreconditionerConst()->setPressureLaplaceMatrix( Lpressure );
            
            // PCD Operator  
            MultiVectorConstPtr_Type u = this->solution_->getBlock(0);
            MultiVectorPtr_Type u_rep_(new MultiVector_Type( this->getDomain(0)->getMapVecFieldRepeated(), 1 ) );
            u_rep_->importFromVector(u, true); // making repeated solution

            MatrixPtr_Type AdvPressure(new Matrix_Type( this->getDomain(1)->getMapUnique(), this->getDomain(1)->getApproxEntriesPerRow() ) );
            this->feFactory_->assemblyAdvectionVecFieldScalar( this->dim_, this->domain_FEType_vec_.at(1),this->domain_FEType_vec_.at(0), AdvPressure, u_rep_, false ); 

            // \nu * \Delta
            MatrixPtr_Type Lpressure2(new Matrix_Type( this->getDomain(1)->getMapUnique(), this->getDomain(1)->getApproxEntriesPerRow() ) );
            this->feFactory_->assemblyLaplace( this->dim_, this->domain_FEType_vec_.at(1), 2, Lpressure2, true );//assemblyIdentity(Lpressure); //
            SC kinVisco = this->parameterList_->sublist("Parameter").get("Viscosity",1.);
            Lpressure2->resumeFill();
            Lpressure2->scale(kinVisco);
            Lpressure2->fillComplete();

            // Adding laplace an convection together
            Lpressure2->addMatrix(1.,AdvPressure,1.); // adding advection to diffusion
            AdvPressure->fillComplete();
            BlockMatrixPtr_Type dummy2(new BlockMatrix_Type (1));
            dummy2->addBlock(AdvPressure,0,0);
            this->bcFactoryPressureFp_->setSystem(dummy2); 
            this->getPreconditionerConst()->setPCDOperator( AdvPressure );

        }
    }
#endif
    string precType = this->parameterList_->sublist("General").get("Preconditioner Method","Monolithic");
    if ( precType == "Diagonal" || precType == "Triangular" ) {
        MatrixPtr_Type Mpressure(new Matrix_Type( this->getDomain(1)->getMapUnique(), this->getDomain(1)->getApproxEntriesPerRow() ) );
        
        this->feFactory_->assemblyMass( this->dim_, this->domain_FEType_vec_.at(1), "Scalar", Mpressure, true );
        
        Mpressure->resumeFill();
        Mpressure->scale(-1./viscosity);
        Mpressure->fillComplete( pressureMap, pressureMap );
        this->getPreconditionerConst()->setPressureMassMatrix( Mpressure );
    }

    this->assembleSourceTerm( 0. );
    this->addToRhs( this->sourceTerm_ );
    //this->rhs_->print();
    
    if (this->verbose_)
        std::cout << "done -- " << std::endl;
    
}

}
#endif
