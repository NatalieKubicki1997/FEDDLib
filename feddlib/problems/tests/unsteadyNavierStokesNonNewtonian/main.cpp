#ifndef MAIN_TIMER_START
#define MAIN_TIMER_START(A,S) Teuchos::RCP<Teuchos::TimeMonitor> A = Teuchos::rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer(std::string("Main") + std::string(S))));
#endif

#ifndef MAIN_TIMER_STOP
#define MAIN_TIMER_STOP(A) A.reset();
#endif

#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/FE/Domain.hpp"
#include "feddlib/core/Mesh/MeshPartitioner.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/General/ExporterParaView.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector.hpp"
#include "feddlib/problems/Solver/DAESolverInTime.hpp"
#include "feddlib/problems/Solver/NonLinearSolver.hpp"
#include "feddlib/problems/specific/NavierStokes.hpp"
#include "feddlib/problems/specific/NavierStokesAssFE.hpp"
#include <Xpetra_DefaultPlatform.hpp>

/*!
 main of time-dependent Navier-Stokes problem
 
 @brief time-dependent Navier-Stokes main
 @author Christian Hochmuth
 @version 1.0
 @copyright CH
 */

using namespace std;

void zeroDirichlet(double* x, double* res, double t, const double* parameters){

    res[0] = 0.;

    return;
}

void zeroDirichlet2D(double* x, double* res, double t, const double* parameters){

    res[0] = 0.;
    res[1] = 0.;

    return;
}

void zeroDirichlet3D(double* x, double* res, double t, const double* parameters){

    res[0] = 0.;
    res[1] = 0.;
    res[2] = 0.;

    return;
}

void inflowPartialCFD(double* x, double* res, double t, const double* parameters){

    double H = parameters[1];

    if(t < 0.5)
    {
        res[0] = (4.0*1.5*parameters[0]*x[1]*(H-x[1])/(H*H))*((1 - cos(2.0*M_PI*t))/2.0);
        res[1] = 0.;
    }
    else
    {
        res[0] = 4.0*1.5*parameters[0]*x[1]*(H-x[1])/(H*H);
        res[1] = 0.;
    }

    return;
}

// Stepwise Inflow profile 
void TimeDependentinflowParabolic2D(double* x, double* res, double t, const double* parameters){


    double H = 0.1;
    double n = 0.6;
    double rho_ref = 1000.0;
    double nu_0 = 0.035;
    double dp = 10000.0;


    if(t <= 0.25)
    {
    res[0] = 0.25*(n/(n+1.0))*pow( dp/(rho_ref*nu_0), 1.0/n)*( pow( H/(2.0), (n+1.0)/n) - pow( abs( (H/2.0) -x[1] ) , (n+1.0)/n  ) );
    res[1] = 0.;
    }
    else if((t <= 0.5) && (t > 0.25 ))
    {
    res[0] = 0.5*(n/(n+1.0))*pow( dp/(rho_ref*nu_0), 1.0/n)*( pow( H/(2.0), (n+1.0)/n) - pow( abs( (H/2.0) -x[1] ) , (n+1.0)/n  ) );
    res[1] = 0.;
    }
    else if((t <= 0.75) && (t > 0.5 ))
    {
    res[0] = 0.75*(n/(n+1.0))*pow( dp/(rho_ref*nu_0), 1.0/n)*( pow( H/(2.0), (n+1.0)/n) - pow( abs( (H/2.0) -x[1] ) , (n+1.0)/n  ) );
    res[1] = 0.;
    }
    else 
    {
    res[0] = (n/(n+1.0))*pow( dp/(rho_ref*nu_0), 1.0/n)*( pow( H/(2.0), (n+1.0)/n) - pow( abs( (H/2.0) -x[1] ) , (n+1.0)/n  ) );
    res[1] = 0.;
    }


    
    return;
}

// Slowly increasing velcoity profile depending on time t
void TimeDependentSmoothinflowParabolic2D(double* x, double* res, double t, const double* parameters){


    double H = 0.1;
    double n = 0.6;
    double rho_ref = 1000.0;
    double nu_0 = 0.035;
    double dp = 10000.0;

    if(t < 1.)
    {
    res[0] = t*(n/(n+1.0))*pow( dp/(rho_ref*nu_0), 1.0/n)*( pow( H/(2.0), (n+1.0)/n) - pow( abs( (H/2.0) -x[1] ) , (n+1.0)/n  ) );
    res[1] = 0.;
    }
    else
    {
    res[0] = (n/(n+1.0))*pow( dp/(rho_ref*nu_0), 1.0/n)*( pow( H/(2.0), (n+1.0)/n) - pow( abs( (H/2.0) -x[1] ) , (n+1.0)/n  ) );
    res[1] = 0.;   
    }
    
    return;
}


void inflowParabolic2D(double* x, double* res, double t, const double* parameters){


    double H = 0.1;
    double n = 0.6;
    double rho_ref = 1000.0;
    double nu_0 = 0.035;
    double dp = 10000.0;

    res[0] = (n/(n+1.0))*pow( dp/(rho_ref*nu_0), 1.0/n)*( pow( H/(2.0), (n+1.0)/n) - pow( abs( (H/2.0) -x[1] ) , (n+1.0)/n  ) );
    res[1] = 0.;
    
    return;
}

// Inflow profile analytical solution of powr law problem 
void inflowPowerLaw2D(double *x, double *res, double t, const double *parameters)
{

double H = 0.1;
double dp = 10.0;

double K =  parameters[0]; // 0.035;
double n =  parameters[1]; // 1.0; // For n=1.0 we have parabolic inflow profile (Newtonian case)

res[0] = (n / (n + 1.0)) * pow(dp / (K), 1.0 / n) * (pow(H / (2.0), (n + 1.0) / n) - pow(abs((H / 2.0) - x[1]), (n + 1.0) / n));
res[1] = 0.;

    return;
}



void inflowParabolic2DSin(double* x, double* res, double t, const double* parameters){

    double H = parameters[1];
    res[0] = sin(M_PI*t*0.125)*( 6*x[1]*(H-x[1]) ) / (H*H);
    res[1] = 0.;

    return;
}

void inflowParabolic3D(double* x, double* res, double t, const double* parameters){

    double H = parameters[1];
    res[0] = 16*parameters[0]*x[1]*(H-x[1])*x[2]*(H-x[2])/(H*H*H*H);
    res[1] = 0.;
    res[2] = 0.;

    return;
}

void inflow3DRichter(double* x, double* res, double t, const double* parameters)
{
    double H = parameters[1];
    
    if(t < 1.)
    {
        res[0] = 9./8 * parameters[0] *x[1]*(H-x[1])*(H*H-x[2]*x[2])/( H*H*(H/2.)*(H/2.) ) * ((1 - cos(2.0*M_PI*t))/2.0);
        res[1] = 0.;
        res[2] = 0.;
    }
    else
    {
        res[0] = 9./8 * parameters[0] *x[1]*(H-x[1])*(H*H-x[2]*x[2])/( H*H*(H/2.)*(H/2.) );
        res[1] = 0.;
        res[2] = 0.;
    }
    
    return;
}

void dummyFunc(double* x, double* res, double t, const double* parameters){

    return;
}

typedef unsigned UN;
typedef default_sc SC;
typedef default_lo LO;
typedef default_go GO;
typedef default_no NO;

using namespace Teuchos;
using namespace FEDD;
int main(int argc, char *argv[]) {
    typedef MeshPartitioner<SC,LO,GO,NO> MeshPartitioner_Type;
    typedef Teuchos::RCP<Domain<SC,LO,GO,NO> > DomainPtr_Type;

    typedef Matrix<SC,LO,GO,NO> Matrix_Type;
    typedef Teuchos::RCP<Matrix_Type> MatrixPtr_Type;

    Teuchos::oblackholestream blackhole;
    Teuchos::GlobalMPISession mpiSession(&argc,&argv,&blackhole);

    Teuchos::RCP<const Teuchos::Comm<int> > comm = Xpetra::DefaultPlatform::getDefaultPlatform().getComm();
    bool verbose (comm->getRank() == 0);
    if (verbose) {
        cout << "###############################################################" <<endl;
        cout << "################### Unsteady Navier-Stokes ####################" <<endl;
        cout << "###############################################################" <<endl;
    }

    // Command Line Parameters
    Teuchos::CommandLineProcessor myCLP;

    string xmlProblemFile = "parametersProblem.xml";
    myCLP.setOption("problemfile",&xmlProblemFile,".xml file with Inputparameters.");
    string xmlPrecFile = "parametersPrec.xml";
    myCLP.setOption("precfile",&xmlPrecFile,".xml file with Inputparameters.");
    string xmlSolverFile = "parametersSolver.xml";
    myCLP.setOption("solverfile",&xmlSolverFile,".xml file with Inputparameters.");

    string xmlTekoPrecFile = "parametersTeko.xml";
    myCLP.setOption("tekoprecfile",&xmlTekoPrecFile,".xml file with Inputparameters.");

    double length = 4.;
    myCLP.setOption("length",&length,"length of domain.");

    myCLP.recogniseAllOptions(true);
    myCLP.throwExceptions(false);
    Teuchos::CommandLineProcessor::EParseCommandLineReturn parseReturn = myCLP.parse(argc,argv);
    if(parseReturn == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED) {
        MPI_Finalize();
        return 0;
    }

    {
        ParameterListPtr_Type parameterListProblem = Teuchos::getParametersFromXmlFile(xmlProblemFile);

        ParameterListPtr_Type parameterListPrec = Teuchos::getParametersFromXmlFile(xmlPrecFile);

        ParameterListPtr_Type parameterListSolver = Teuchos::getParametersFromXmlFile(xmlSolverFile);

        ParameterListPtr_Type parameterListPrecTeko = Teuchos::getParametersFromXmlFile(xmlTekoPrecFile);
        int 		dim				= parameterListProblem->sublist("Parameter").get("Dimension",3);
        std::string discVelocity = parameterListProblem->sublist("Parameter").get("Discretization Velocity","P2");
        std::string discPressure = parameterListProblem->sublist("Parameter").get("Discretization Pressure","P1");
        string		meshType = parameterListProblem->sublist("Parameter").get("Mesh Type","structured");
        string		meshName = parameterListProblem->sublist("Parameter").get("Mesh Name","structured");
        string		meshDelimiter = parameterListProblem->sublist("Parameter").get("Mesh Delimiter"," ");
        int 		m = parameterListProblem->sublist("Parameter").get("H/h",5);
        string		linearization = parameterListProblem->sublist("General").get("Linearization","FixedPoint");
        string		precMethod = parameterListProblem->sublist("General").get("Preconditioner Method","Monolithic");
        bool computeInflow = parameterListProblem->sublist("Parameter").get("Compute Inflow",false);
        int         n;


        ParameterListPtr_Type parameterListAll(new Teuchos::ParameterList(*parameterListProblem)) ;
        if (!precMethod.compare("Monolithic"))
            parameterListAll->setParameters(*parameterListPrec);
        else
            parameterListAll->setParameters(*parameterListPrecTeko);

        parameterListAll->setParameters(*parameterListSolver);

        std::string bcType = parameterListProblem->sublist("Parameter").get("BC Type","parabolic");

        int minNumberSubdomains;
        if (!meshType.compare("structured")) {
            minNumberSubdomains = 1;
        }
        else if(!meshType.compare("structured_rec")){
            minNumberSubdomains = length;
        }
        else if(!meshType.compare("structured_bfs")){
            minNumberSubdomains = (int) 2*length+1;
        }

        int numProcsCoarseSolve = parameterListProblem->sublist("General").get("Mpi Ranks Coarse",0);
        int size = comm->getSize() - numProcsCoarseSolve;

        double viscosity = parameterListProblem->sublist("Parameter").get("Viscosity",1.e-3);

        Teuchos::RCP<Teuchos::Time> totalTime(Teuchos::TimeMonitor::getNewCounter("main: Total Time"));
        Teuchos::RCP<Teuchos::Time> buildMesh(Teuchos::TimeMonitor::getNewCounter("main: Build Mesh"));
        Teuchos::RCP<Teuchos::Time> solveTime(Teuchos::TimeMonitor::getNewCounter("main: Solve problem time"));
        
        {
           
            DomainPtr_Type domainPressure;
            DomainPtr_Type domainVelocity;
                              
            domainPressure.reset( new Domain<SC,LO,GO,NO>( comm, dim ) );
            domainVelocity.reset( new Domain<SC,LO,GO,NO>( comm, dim ) );
            
            MeshPartitioner_Type::DomainPtrArray_Type domainP1Array(1);
            domainP1Array[0] = domainPressure;
            
            ParameterListPtr_Type pListPartitioner = sublist( parameterListProblem, "Mesh Partitioner" );
            MeshPartitioner<SC,LO,GO,NO> partitionerP1 ( domainP1Array, pListPartitioner, "P1", dim );
            
            partitionerP1.readAndPartition();

            if (discVelocity=="P2")
                domainVelocity->buildP2ofP1Domain( domainPressure );
            else
                domainVelocity = domainPressure;
      
           // std::vector<double> parameter_vec(1, parameterListProblem->sublist("Parameter").get("MaxVelocity",1.));
           // ####################
           // Teuchos::RCP<BCBuilder<SC,LO,GO,NO> > bcFactory( new BCBuilder<SC,LO,GO,NO>( ) );


                      /***** For boundary condition read parameter values */
            std::vector<double> parameter_vec(1, parameterListProblem->sublist("Material").get("PowerLawParameter K",0.));
            parameter_vec.push_back(parameterListProblem->sublist("Material").get("PowerLaw index n",1.)); 

            //          **********************  BOUNDARY CONDITIONS ***********************************
            //          **********************  BOUNDARY CONDITIONS ***********************************     
            //          **********************  BOUNDARY CONDITIONS ***********************************
        
            Teuchos::RCP<BCBuilder<SC, LO, GO, NO>> bcFactory(new BCBuilder<SC, LO, GO, NO>());
            parameter_vec.push_back(1); // 0.41);

            //** COUETTE FLOW
            /*    if (dim==2){
                // So for Couette Flow we have a moving upper plate and rigid lower plate
                    bcFactory->addBC(zeroDirichlet2D, 1, 0, domainVelocity, "Dirichlet", dim); // wall
                    bcFactory->addBC(couette2D, 2, 0, domainVelocity, "Dirichlet", dim, parameter_vec); // wall
                // The flow will be induced by the moving plate so there should be no pressure gradient in this testcase
                    bcFactory->addBC(zeroDirichlet2D, 3, 1, domainPressure, "Dirichlet", 1); // outflow
                    bcFactory->addBC(zeroDirichlet2D, 4, 1, domainPressure, "Dirichlet", 1); // inflow
                }
             */
            
            //** POISEUILLE FLOW - Rectangle Grid //**//**//**//**//**//**//**//**//**//**//**
            bcFactory->addBC(zeroDirichlet2D, 1, 0, domainVelocity, "Dirichlet", dim);                // wall
            bcFactory->addBC(zeroDirichlet2D, 2, 0, domainVelocity, "Dirichlet", dim, parameter_vec); // wall
            bcFactory->addBC(inflowPowerLaw2D, 4, 0, domainVelocity, "Dirichlet", dim, parameter_vec); //original bc Inlet
            // After we added the proper code line in NavierStokesAssFE we can set this for P2-P1 element
            bcFactory->addBC(zeroDirichlet, 3, 1, domainPressure, "Dirichlet", 1); //Outflow
            // For test 1x1, 2x2, 4x4 because else inflow is unsymmetric
            //bcFactory->addBC(onex, 4, 0, domainVelocity, "Dirichlet", dim, parameter_vec);
    
       
            //** Flow over step//**//**//**//**//**//**//**//**//**//**//**//**//**//**//**
            /*
            bcFactory->addBC(zeroDirichlet2D, 1, 0, domainVelocity, "Dirichlet", dim);                // wall
            bcFactory->addBC(zeroDirichlet2D, 2, 0, domainVelocity, "Dirichlet", dim, parameter_vec); // wall
            bcFactory->addBC(zeroDirichlet2D, 3, 0, domainVelocity, "Dirichlet", dim, parameter_vec); // wall
            bcFactory->addBC(zeroDirichlet2D, 4, 0, domainVelocity, "Dirichlet", dim, parameter_vec); // wall
            bcFactory->addBC(inflowParabolicAverageVelocity2D, 5, 0, domainVelocity, "Dirichlet", dim, parameter_vec); //original bc Inlet
            */
            // After we added the proper code line in NavierStokesAssFE we can set this for P2-P1 element
            //bcFactory->addBC(zeroDirichlet, , 1, domainPressure, "Dirichlet", 1); //Outflow
            // Ich muss irgendwo ein Druck Punkt festlegen bestimmt!!
   
                     
            int timeDisc = parameterListProblem->sublist("Timestepping Parameter").get("Butcher table",0);

			DAESolverInTime<SC,LO,GO,NO> daeTimeSolver(parameterListAll, comm);
            SmallMatrix<int> defTS(2);
            defTS[0][0] = 1;
            defTS[0][1] = 1;
            defTS[1][0] = 0;
            defTS[1][1] = 0;

            daeTimeSolver.defineTimeStepping(defTS);

			DAESolverInTime<SC,LO,GO,NO> daeTimeSolverAssFE(parameterListAll, comm);


            daeTimeSolverAssFE.defineTimeStepping(defTS);

			// ###########################################################################################################
			// New Assembly
			MAIN_TIMER_START(FE_test," New: Solve equation");

  			NavierStokesAssFE<SC,LO,GO,NO> navierStokesFE( domainVelocity, discVelocity, domainPressure, discPressure, parameterListAll );
			navierStokesFE.addBoundaries(bcFactory);
            
            navierStokesFE.initializeProblem();
            
            navierStokesFE.assemble();

            navierStokesFE.setBoundariesRHS();

            daeTimeSolverAssFE.setProblem(navierStokesFE);

            daeTimeSolverAssFE.setupTimeStepping();

            daeTimeSolverAssFE.advanceInTime();
			MAIN_TIMER_STOP(FE_test);	
			Teuchos::TimeMonitor::report(cout,"Main");
			// ###########################################################################################################


      //****************************************************************************************
            //****************************************************************************************
            //          **********************  POST-PROCESSING - VISCOSITY COMPUTATION FOR NON-NEWTONIAN FLUID ***********************************
            // We only write out viscosity field if we consider non-Newtonian fluid because otherwise it is constant
          if((parameterListProblem->sublist("Material").get("Newtonian",true) == false) && (parameterListProblem->sublist("Material").get("WriteOutViscosity",false)) == true ) 
           {

            Teuchos::RCP<ExporterParaView<SC, LO, GO, NO>> exParaViscsoity(new ExporterParaView<SC, LO, GO, NO>());
            DomainPtr_Type domV = domainVelocity;
            //int nmbElementsGlob = domV->getMesh()->getNumElementsGlobal();

            /*Viskosität berechnen auf Basis der berechnet Geschwindigkeitslösung*/
            navierStokesFE.computeViscosity_Solution();
            navierStokesFE.getViscosity_Solution();
           //Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> exportSolutionViscosityAssFE = navierStokesAssFE.viscosity_element_; 

            //**************** Write out viscosity ****************** so we need something from type multivector so this is not working because we can not access navierStokesAssFE.feFactory_->visco_output_->getBlock(0)
            Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> exportSolutionViscosityAssFE = navierStokesFE.viscosity_element_; 
            exParaViscsoity->setup("viscosity", domV->getMesh(), "P0");
            exParaViscsoity->addVariable(exportSolutionViscosityAssFE, "viscosityAssFE", "Scalar", 1,  domV->getElementMap() );
            exParaViscsoity->save(0.0);        
            }

        }

        
    }

    Teuchos::TimeMonitor::report(cout);

    return(EXIT_SUCCESS);
}
