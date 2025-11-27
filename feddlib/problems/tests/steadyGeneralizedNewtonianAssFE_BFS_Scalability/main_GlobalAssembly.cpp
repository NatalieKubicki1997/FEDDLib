#ifndef MAIN_TIMER_START
#define MAIN_TIMER_START(A, S) Teuchos::RCP<Teuchos::TimeMonitor> A = Teuchos::rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer(std::string("Main") + std::string(S))));
#endif

#ifndef MAIN_TIMER_STOP
#define MAIN_TIMER_STOP(A) A.reset();
#endif

#include <Tpetra_Core.hpp>

#include "feddlib/core/FEDDCore.hpp"

#include "feddlib/core/Mesh/MeshPartitioner.hpp"
#include "feddlib/core/FE/Domain.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/General/ExporterParaView.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector.hpp"

#include "feddlib/problems/Solver/NonLinearSolver.hpp"
#include "feddlib/problems/specific/NavierStokes.hpp"


#include <boost/function.hpp>


/*!
Investigate the weak scalability performance for a three-dimensional backward facing Step problem 

 @brief Stationary Flow of Generalized Newtonian Fluid through Backward-Facing Step Flow
 @author Natalie Kubicki
 @version 1.0
 @copyright NK
 */

using namespace std;
using namespace Teuchos;
using namespace FEDD;


/*
------------------------------------------------------------------------------------------------------------------------------
--------------------------------- BOUNDARY CONDITIONS ------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------
*/


void constx3D(double *x, double *res, double t, const double *parameters)
{

    res[0] = 0.1;
    res[1] = 0.;
    res[2] = 0.;

    return;
}

void zeroDirichlet3D(double *x, double *res, double t, const double *parameters)
{
    res[0] = 0.;
    res[1] = 0.;
    res[2] = 0.;
    return;
}


void inflowParabolic3D(double *x, double *res, double t, const double *parameters)
{

    double H = parameters[2];
    double mu = 0.035;
    double dp = parameters[3];
    ;

    res[0] = (1 / (2 * mu)) * (-1.0) * dp * (x[1] * x[1] - x[1] * H);
    res[1] = 0.;
    res[2] = 0.;

    return;
}

void inflowParabolic3D_structured(double *x, double *res, double t, const double *parameters)
{
    // Dependent of height and max velocity!
    double H = parameters[2];
    res[0] = 16 * parameters[4] * x[1] * (H - x[1]) * x[2] * (H - x[2]) / (H * H * H * H);
    res[1] = 0.;
    res[2] = 0.;

    return;
}
void inflow3DRichter(double *x, double *res, double t, const double *parameters)
{

    double H = parameters[1];

    res[0] = 9. / 8 * parameters[0] * x[1] * (H - x[1]) * (H * H - x[2] * x[2]) / (H * H * (H / 2.) * (H / 2.));
    res[1] = 0.;
    res[2] = 0.;

    return;
}
void dummyFunc(double *x, double *res, double t, const double *parameters)
{

    return;
}


/*
------------------------------------------------------------------------------------------------------------------------------
--------------------------------- SWITCHING STRATEGIES FOR NONLINEAR SOLVER --------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------
*/

// Add here predefined switching strategies for FixedPoint-Newtons Method

// Simple switch to the other linearization method after a defined number of nonlinear iterations
bool switchingAfterMaxIter(std::string& linearization, int nlIts, double criterionValue, ParameterListPtr_Type parameterList )
{
    if (nlIts == parameterList->get("MaximumIteration_Switch", 1) )
    {
        if( (parameterList->get("Initial_Linearization", "FixedPoint") == "FixedPoint") && (linearization == "FixedPoint") )   
        {
            linearization = "Newton";
            return true;
        }
        else if(parameterList->get("Initial_Linearization", "FixedPoint")  == "Newton" && (linearization == "Newton") )
        {
            linearization = "FixedPoint";
            return true;
        }
        else
        {
            // Linearization not changed
            return false;
        }
    }
    else // Stay with the current linearization if number of iterations to switch not yet reached
    {
        return false;
    }
}

// Switch to the other linearization method if the nonlinear criterion value is below a certain tolerance
bool switchingAfterNonlinearTolerance(std::string& linearization, int nlIts, double criterionValue, ParameterListPtr_Type parameterList)
{
    // Switch to Newtons Method if relative residual is below a certain value
    if (criterionValue < parameterList->get("NonlinearTolerance_Switch", 0.1))
    {
        if( (parameterList->get("Initial_Linearization", "FixedPoint") == "FixedPoint") && (linearization == "FixedPoint") )   
        {
            linearization = "Newton";
            return true;
        }
        else if(parameterList->get("Initial_Linearization", "FixedPoint")  == "Newton" && (linearization == "Newton") )
        {
            linearization = "FixedPoint";
            return true;
        }
        else
        {
            // Linearization not changed
            return false;
        }
    }
    else // Stay with the current linearization if tolerance not yet reached
    {
        return false;
    }
}


// Switch linearization method after every iteration
bool switchingAlternating(std::string& linearization, int nlIts, double criterionValue, ParameterListPtr_Type parameterList)
{
    if (nlIts > 0) // Switch only after first iteration, because one solution step should be done with the initial linearization strategy
    {
        if (linearization == "FixedPoint")
        {
            linearization = "Newton";
            return true;
        }
        else if (linearization == "Newton")
        {
            linearization = "FixedPoint";
            return true;
        }
        else
        {
            std::cerr << "Error: Current Linearization not FixedPoint or Newton - Staying with current one!" << std::endl;
            return false;
        }
    }
    else
    {
        return false; // Do not change in first iteration, because we want to start one computation with the initial linearization strategy
    }
}




typedef unsigned UN;
typedef default_sc SC;
typedef default_lo LO;
typedef default_go GO;
typedef default_no NO;

using namespace FEDD;

// Now the main starts
int main(int argc, char *argv[])
{

    typedef MeshPartitioner<SC, LO, GO, NO> MeshPartitioner_Type;
    typedef Teuchos::RCP<Domain<SC, LO, GO, NO>> DomainPtr_Type;

    typedef Matrix<SC, LO, GO, NO> Matrix_Type;
    typedef Teuchos::RCP<Matrix_Type> MatrixPtr_Type;

    // boost function type for switching strategy: 
    typedef boost::function<bool( std::string& , int, double, ParameterListPtr_Type)> SwitchingStrategyFunc;

    

    // MPI boilerplate
    Tpetra::ScopeGuard tpetraScope (&argc, &argv); // initializes MPI
    {
    Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();

    bool verbose(comm->getRank() == 0);

    //    Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();

    if (verbose)
    {
        cout << "###############################################################" << endl;
        cout << "##################### Stationary Flow of Generalized Newtonian Fluid ####################" << endl;
        cout << "###############################################################" << endl;
    }

    // Command Line Parameters
    Teuchos::CommandLineProcessor myCLP;

    string xmlProblemFile = "parametersProblem.xml";
    myCLP.setOption("problemfile", &xmlProblemFile, ".xml file with Inputparameters.");

    string xmlPrecFile = "parametersPrec.xml";
    myCLP.setOption("precfile", &xmlPrecFile, ".xml file with Inputparameters.");
    string xmlSolverFile = "parametersSolver.xml";
    myCLP.setOption("solverfile", &xmlSolverFile, ".xml file with Inputparameters.");

    string xmlTekoPrecFile = "parametersTeko.xml";
    myCLP.setOption("tekoprecfile", &xmlTekoPrecFile, ".xml file with Inputparameters.");

    double length = 4.; // This a constant which has to be set for strcutured grids
    myCLP.setOption("length", &length, "length of domain.");

    myCLP.recogniseAllOptions(true);
    myCLP.throwExceptions(false);
    Teuchos::CommandLineProcessor::EParseCommandLineReturn parseReturn = myCLP.parse(argc, argv);
    if (parseReturn == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED)
    {
        MPI_Finalize();
        return 0;
    }
    Teuchos::RCP<Teuchos::Time> totalTime(Teuchos::TimeMonitor::getNewCounter("Main: Total Time = Build Mesh + Solve + Visualization (if on)"));

    // Einlesen von Parameterwerten
    {
        Teuchos::TimeMonitor totalTimeMonitor(*totalTime);



        ParameterListPtr_Type parameterListProblem = Teuchos::getParametersFromXmlFile(xmlProblemFile);

        ParameterListPtr_Type parameterListPrec = Teuchos::getParametersFromXmlFile(xmlPrecFile);

        ParameterListPtr_Type parameterListSolver = Teuchos::getParametersFromXmlFile(xmlSolverFile);

        ParameterListPtr_Type parameterListPrecTeko = Teuchos::getParametersFromXmlFile(xmlTekoPrecFile);

        int dim = parameterListProblem->sublist("Parameter").get("Dimension", 3);

        std::string discVelocity = parameterListProblem->sublist("Parameter").get("Discretization Velocity", "P2");
        std::string discPressure = parameterListProblem->sublist("Parameter").get("Discretization Pressure", "P1");

        string meshType = parameterListProblem->sublist("Parameter").get("Mesh Type", "structured");
        string meshName = parameterListProblem->sublist("Parameter").get("Mesh Name", "circle2D_1800.mesh");
        string meshDelimiter = parameterListProblem->sublist("Parameter").get("Mesh Delimiter", " ");
        int m = parameterListProblem->sublist("Parameter").get("H/h", 5);
        string linearization = parameterListProblem->sublist("General").get("Linearization", "FixedPoint");

        bool linearization_SwitchToNewton_ = parameterListProblem->sublist("General").get("Linearization_SwitchToNewton",false);
        string precMethod = parameterListProblem->sublist("General").get("Preconditioner Method", "Monolithic");
        int mixedFPIts = parameterListProblem->sublist("General").get("MixedFPIts", 1);
        int n;

        ParameterListPtr_Type parameterListAll(new Teuchos::ParameterList(*parameterListProblem));
        if (!precMethod.compare("Monolithic"))
            parameterListAll->setParameters(*parameterListPrec);
        else
            parameterListAll->setParameters(*parameterListPrecTeko);
        parameterListAll->setParameters(*parameterListSolver);


        int minNumberSubdomains;
        if (!meshType.compare("structured"))
        {
            minNumberSubdomains = 1;
        }
        else if (!meshType.compare("structured_bfs"))
        {
            minNumberSubdomains = (int)2 * length + 1;
        }

        int numProcsCoarseSolve = parameterListProblem->sublist("General").get("Mpi Ranks Coarse",0);
        int size = comm->getSize() - numProcsCoarseSolve;
        Teuchos::RCP<Teuchos::Time> buildMesh(Teuchos::TimeMonitor::getNewCounter("main: Build Mesh = Construction of structured Mesh"));
        Teuchos::RCP<Teuchos::Time> solveTime(Teuchos::TimeMonitor::getNewCounter("main: Solve problem time = assembly + Solve"));
        
        {


            DomainPtr_Type domainPressure;
            DomainPtr_Type domainVelocity;
                              
           if (!meshType.compare("structured_bfs")) {
                Teuchos::TimeMonitor buildMeshMonitor(*buildMesh);

                TEUCHOS_TEST_FOR_EXCEPTION( size%minNumberSubdomains != 0 , std::logic_error, "Wrong number of processors for structured BFS mesh.");
                if (dim == 2) {
                    n = (int) (std::pow( size/minNumberSubdomains ,1/2.) + 100*Teuchos::ScalarTraits<double>::eps()); // 1/H
                    std::vector<double> x(2);
                    x[0]=-1.0;    x[1]=-1.0;
                    domainPressure.reset(new Domain<SC,LO,GO,NO>( x, length+1., 2., comm ) );
                    domainVelocity.reset(new Domain<SC,LO,GO,NO>( x, length+1., 2., comm ) );
                }
                else if (dim == 3){
                    n = (int) (std::pow( size/minNumberSubdomains ,1/3.) + 100*Teuchos::ScalarTraits<double>::eps()); // 1/H
                    std::vector<double> x(3);
                    x[0]=-1.0;    x[1]=0.0;    x[2]=-1.0;
                    domainPressure.reset(new Domain<SC,LO,GO,NO>( x, length+1., 1., 2., comm));
                    domainVelocity.reset(new Domain<SC,LO,GO,NO>( x, length+1., 1., 2., comm));
                }
                domainPressure->buildMesh( 2,"BFS", dim, discPressure, n, m, numProcsCoarseSolve); // m is H/h
                domainVelocity->buildMesh( 2,"BFS", dim, discVelocity, n, m, numProcsCoarseSolve);
            }
	        else if (!meshType.compare("unstructured")) {
	        
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
	        }
            // **********************  BOUNDARY CONDITIONS ************************************
            // We can here differentiate between different problem cases and boundary conditions

            /***** For boundary condition read parameter values */
            /*******************************************************************************/
            std::vector<double> parameter_vec(1, parameterListProblem->sublist("Material").get("PowerLawParameter K", 0.));
            parameter_vec.push_back(parameterListProblem->sublist("Material").get("PowerLaw index n", 1.));
            // So parameter[0] is K, [1] is n
            parameter_vec.push_back(parameterListProblem->sublist("Parameter").get("Height Inflow", 1.));
            parameter_vec.push_back(parameterListProblem->sublist("Parameter").get("Constant Pressure Gradient", 1.));
            parameter_vec.push_back(parameterListProblem->sublist("Parameter").get("MaxVelocity", 1.));

            Teuchos::RCP<BCBuilder<SC, LO, GO, NO>> bcFactory(new BCBuilder<SC, LO, GO, NO>());
            std::string bcType = parameterListProblem->sublist("Parameter").get("BC Type","parabolic");


            /*** Nonlinear Solver Switching Strategy Parameters ******************************************************************************************************************************/
            // Define switching strategy function based on user input -  <Parameter name="SwitchingStrategy" type="string" value="AfterMaxIter"/>  Predefined switching strategies are: AfterMaxIter, AfterResidual -->
            std::string switchingStrategyType = parameterListProblem->sublist("General").get("SwitchingStrategy", "AfterMaxIter"); // Default is AfterMaxIter
            SwitchingStrategyFunc switchingStrategy = []( std::string& currentLinearization , int nlIts, double criterionValue, ParameterListPtr_Type parameterList ) {return false;};
            if (switchingStrategyType == "AfterMaxIter")
            {
                if (verbose) std::cout << "Selected Switching Strategy for FixedPointNewton: " << switchingStrategyType << std::endl;
                switchingStrategy = switchingAfterMaxIter;
            }
            else if (switchingStrategyType == "Alternating")
            {
                if (verbose) std::cout << "Selected Switching Strategy for FixedPointNewton: " << switchingStrategyType << std::endl;
                switchingStrategy = switchingAlternating;
            }
            else if (switchingStrategyType == "AfterResidual")
            {
                if (verbose) std::cout << "Selected Switching Strategy for FixedPointNewton: " << switchingStrategyType << std::endl;
                switchingStrategy = switchingAfterNonlinearTolerance;
            }
            else
            {
                if (verbose) std::cerr << "Error: Selected Switching Strategy for FixedPointNewton not recognized - No switching will be performed!" << std::endl;
            }


             if ( !bcType.compare("parabolic") && dim==3 ) {//flag of obstacle
                    bcFactory->addBC(zeroDirichlet3D, 1, 0, domainVelocity, "Dirichlet", dim);
                    bcFactory->addBC(inflowParabolic3D_structured, 2, 0, domainVelocity, "Dirichlet", dim, parameter_vec);
                    bcFactory->addBC(zeroDirichlet3D, 4, 0, domainVelocity, "Dirichlet", dim);    
                    // Flag 3 is outflow
                }
                else
                {
                  if (dim == 2)
                  {
                    // We consider backward facing step problem 

                  }
                 else if (dim == 3)
                 {
                  bcFactory->addBC(zeroDirichlet3D, 1, 0, domainVelocity, "Dirichlet", dim);                  // upper
                  bcFactory->addBC(zeroDirichlet3D, 2, 0, domainVelocity, "Dirichlet", dim);                  // lower
                  bcFactory->addBC(zeroDirichlet3D, 3, 0, domainVelocity, "Dirichlet", dim);                  // sides
                  bcFactory->addBC(zeroDirichlet3D, 4, 0, domainVelocity, "Dirichlet", dim);                  // sides
                  bcFactory->addBC(inflowParabolic3D, 6, 0, domainVelocity, "Dirichlet", dim, parameter_vec); // inlet

                 }
               }


            //          **********************  CALL SOLVER ***********************************
            NavierStokes<SC, LO, GO, NO> navierStokes(domainVelocity, discVelocity, domainPressure, discPressure, parameterListAll);

            {
                Teuchos::TimeMonitor solveTimeMonitor(*solveTime);
                MAIN_TIMER_START(NavierStokes, " AssFE:   Assemble System and solve");
                navierStokes.addBoundaries(bcFactory);
                navierStokes.initializeProblem();
                navierStokes.assemble();

                navierStokes.setBoundariesRHS();

                std::string nlSolverType = parameterListProblem->sublist("General").get("Linearization", "FixedPoint");
                NonLinearSolver<SC, LO, GO, NO> nlSolverAssFE(nlSolverType);

                nlSolverAssFE.addSwitchingStrategyFunction( switchingStrategy); // Add switching strategy function of NonLinearSolver to User defined one


                nlSolverAssFE.solve(navierStokes); // jumps into NonLinearSolver_def.hpp

                MAIN_TIMER_STOP(NavierStokes);
                comm->barrier();
            }

            Teuchos::TimeMonitor::report(cout, "Main");
            
            //**********************  POST-PROCESSING ***********************************
            //****************************************************************************************
            ///*******************************************************************************///

             // Plotte subdomain   
             if (parameterListAll->sublist("General").get("ParaView export subdomains",false) )
             {
                
                if (verbose)
                    std::cout << "\t### Exporting fluid subdomains ###\n";

                typedef MultiVector<SC,LO,GO,NO> MultiVector_Type;
                typedef RCP<MultiVector_Type> MultiVectorPtr_Type;
                typedef RCP<const MultiVector_Type> MultiVectorConstPtr_Type;
                typedef BlockMultiVector<SC,LO,GO,NO> BlockMultiVector_Type;
                typedef RCP<BlockMultiVector_Type> BlockMultiVectorPtr_Type;

                {
                    MultiVectorPtr_Type vecDecomposition = rcp(new MultiVector_Type( domainVelocity->getElementMap() ) );
                    MultiVectorConstPtr_Type vecDecompositionConst = vecDecomposition;
                    vecDecomposition->putScalar(comm->getRank()+1.);
                    
                    Teuchos::RCP<ExporterParaView<SC,LO,GO,NO> > exPara(new ExporterParaView<SC,LO,GO,NO>());
                    
                    exPara->setup( "subdomains_fluid", domainVelocity->getMesh(), "P0" );
                    
                    exPara->addVariable( vecDecompositionConst, "subdomains", "Scalar", 1, domainVelocity->getElementMap());
                    exPara->save(0.0);
                    exPara->closeExporter();
                }
                

            }
            

        
            //****************************************************************************************
            //          **********************  POST-PROCESSING - WRITE OUT VELOCITY AND PRESSURE ***********************************
            if ( parameterListAll->sublist("General").get("ParaViewExport",false) ) {
              
            Teuchos::RCP<ExporterParaView<SC, LO, GO, NO>> exParaVelocity(new ExporterParaView<SC, LO, GO, NO>());
            Teuchos::RCP<ExporterParaView<SC, LO, GO, NO>> exParaPressure(new ExporterParaView<SC, LO, GO, NO>());

            Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> exportSolutionVAssFE = navierStokes.getSolution()->getBlock(0);
            Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> exportSolutionPAssFE = navierStokes.getSolution()->getBlock(1);

            DomainPtr_Type dom = domainVelocity;
            exParaVelocity->setup("velocity", dom->getMesh(), dom->getFEType());
            UN dofsPerNode = dim;
            exParaVelocity->addVariable(exportSolutionVAssFE, "uAssFE", "Vector", dofsPerNode, dom->getMapUnique());

            dom = domainPressure;
            exParaPressure->setup("pressure", dom->getMesh(), dom->getFEType());
            exParaPressure->addVariable(exportSolutionPAssFE, "pAssFE", "Scalar", 1, dom->getMapUnique());

            exParaVelocity->save(0.0);
            exParaPressure->save(0.0);
            }
         
        
    }
    }
    Teuchos::TimeMonitor::report(cout);
    } // end Tpetra::ScopeGuard
    return (EXIT_SUCCESS);
 
}
