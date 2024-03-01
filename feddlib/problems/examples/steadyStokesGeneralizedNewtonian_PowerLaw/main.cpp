#ifndef MAIN_TIMER_START
#define MAIN_TIMER_START(A, S) Teuchos::RCP<Teuchos::TimeMonitor> A = Teuchos::rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer(std::string("Main") + std::string(S))));
#endif

#ifndef MAIN_TIMER_STOP
#define MAIN_TIMER_STOP(A) A.reset();
#endif

#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/Mesh/MeshPartitioner.hpp"
#include "feddlib/core/FE/Domain.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/General/ExporterParaView.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector.hpp"

#include "feddlib/problems/Solver/NonLinearSolver.hpp"
#include "feddlib/problems/specific/NavierStokesAssFE.hpp"

#include <Teuchos_GlobalMPISession.hpp>
#include <Xpetra_DefaultPlatform.hpp>

/*!
 Main of steady-state Generalized Newtonian fluid flow problem with Non-Newtonian stress tensor assumption
 where we use for this example the Power law model of the form eta = K*gamma^(n-1)
 In this main we also neglectthe convective part of the momentum euqation which is why we call it
 Stokes NonNewtonian

 ***********************************************************************************************
 Here we have to check that IF we want to compute a comparison between Navier-Stokes and 
 Power-Law Model that K has to correspond to kinematicViscosity*density 
 If not we are not comparing the same flow

 @brief steady-state Non-Newtonian Flow main 
 @author Natalie Kubicki
 @version 1.0
 @copyright NK
 */


using namespace std;
using namespace Teuchos;
using namespace FEDD;

// These are the boundary conditions already defined
void zeroDirichlet(double *x, double *res, double t, const double *parameters)
{

    res[0] = 0.0;
    return;
}
void zeroDirichlet2D(double *x, double *res, double t, const double *parameters)
{

    res[0] = 0.;
    res[1] = 0.;

    return;
}

void zeroDirichlet2D_onlyY(double *x, double *res, double t, const double *parameters)
{

    // res[0] = 0.;
    res[1] = 0.;

    return;
}

void couette2D(double *x, double *res, double t, const double *parameters)
{

    res[0] = parameters[4];
    res[1] = 0.;

    return;
}
void one(double *x, double *res, double t, const double *parameters)
{

    res[0] = 1.;
    res[1] = 1.;

    return;
}

void onex(double *x, double *res, double t, const double *parameters)
{

    res[0] = 1.;
    res[1] = 0.;

    return;
}

void constx3D(double *x, double *res, double t, const double *parameters)
{

    res[0] = 0.1;
    res[1] = 0.;
    res[2] = 0.;

    return;
}

void oney(double *x, double *res, double t, const double *parameters)
{

    res[1] = 1.;
    res[0] = 0.;

    return;
}

void two(double *x, double *res, double t, const double *parameters)
{

    res[0] = 2.;
    res[1] = 2.;

    return;
}
void three(double *x, double *res, double t, const double *parameters)
{

    res[0] = 3.;
    res[1] = 3.;
    return;
}
void four(double *x, double *res, double t, const double *parameters)
{
    res[0] = 4.;
    res[1] = 4.;
    return;
}

void zeroDirichlet3D(double *x, double *res, double t, const double *parameters)
{
    res[0] = 0.;
    res[1] = 0.;
    res[2] = 0.;
    return;
}

/*For a 2D-Poiseulle-Flow of a Power-Law fluid with eta=K*gamma^(n-1) an analytical solution for the velocity field
  can be derived depending on K, n, H and the pressure gradient dp/dx
  So a simple test case for the generalized-Newtonian fluid solver is a 2D-Poiseuille Flow, prescribing the analytical velocity profile u(y) and 
  checking if correct pressure gradient is recovered
*/ 
void inflowPowerLaw2D(double *x, double *res, double t, const double *parameters)
{

    double K = parameters[0]; // 0.0035;
    double n = parameters[1]; // 1.0; // For n=1.0 we have parabolic inflow profile (Newtonian case)
    double H = parameters[2]; // Height of Channel
    double dp = parameters[3]; // dp/dx constant pressure gradient along channel

    // This corresponds to the analytical solution of a Poiseuille like flow for a Power-Law fluid
    res[0] = (n / (n + 1.0)) * pow(dp / (K), 1.0 / n) * (pow(H / (2.0), (n + 1.0) / n) - pow(abs((H / 2.0) - x[1]), (n + 1.0) / n));
    res[1] = 0.;

    return;
}

// Same as above but for inflow in y-direction
void inflowPowerLaw2D_y(double *x, double *res, double t, const double *parameters)
{

    double K = parameters[0]; // 
    double n = parameters[1]; // 1.0; // For n=1.0 we have parabolic inflow profile (Newtonian case)
    double H = parameters[2];
    double dp = parameters[3];

    res[1] = (n / (n + 1.0)) * pow(dp / (K), 1.0 / n) * (pow(H / (2.0), (n + 1.0) / n) - pow(abs((H / 2.0) - x[0]), (n + 1.0) / n));
    res[0] = 0.;

    return;
}

void inflowParabolicAverageVelocity2D(double *x, double *res, double t, const double *parameters)
{

    double H = parameters[2];
    double mu = 0.00345;
    double rho = 1050.0;
    double Re=1.0;
    double averageVelocity = (mu / (rho * H)) * Re;
    res[0] = 4 * (2 * averageVelocity) * ((x[1] - H) * H - pow(x[1] - H, 2)) / (H * H);
    res[1] = 0.;

    return;
}

void inflowParabolic2D(double *x, double *res, double t, const double *parameters)
{
    /*
        double H =  parameters[2];
        double mu = 0.035;
        double dp = parameters[3];

        res[0] = (1/(2*mu))*(-1.0)*dp*(x[1]*x[1] - x[1]*H);
        res[1] = 0.;
    */
    double H = parameters[2];
    double v_max = parameters[4];
    res[0] = 4 * v_max * x[1] * (H - x[1]) / (H * H);
    res[1] = 0.;

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

void inflowParabolic3D_Old(double *x, double *res, double t, const double *parameters)
{
    double H = parameters[2];
    res[0] = 16 * parameters[0] * x[1] * (H - x[1]) * x[2] * (H - x[2]) / (H * H * H * H);
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

    Teuchos::oblackholestream blackhole;
    Teuchos::GlobalMPISession mpiSession(&argc, &argv, &blackhole);

    Teuchos::RCP<const Teuchos::Comm<int>> comm = Xpetra::DefaultPlatform::getDefaultPlatform().getComm();
    bool verbose(comm->getRank() == 0);

    //    Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();

    if (verbose)
    {
        cout << "###############################################################" << endl;
        cout << "##################### Steady Non-Newtonian Flow of Generalized Newtonian Fluid ####################" << endl;
        cout << "###############################################################" << endl;
    }

    // Command Line Parameters
    Teuchos::CommandLineProcessor myCLP;

    string xmlProblemFile = "parametersProblem.xml";
    myCLP.setOption("problemfile", &xmlProblemFile, ".xml file with Inputparameters.");

    // So this we are reading in only if we want to compare with Newtonian solver
    string xmlProblemFile2 = "parametersProblem_Newtonian.xml";
    myCLP.setOption("problemfile2", &xmlProblemFile2, ".xml file with Inputparameters.");

    string xmlPrecFile = "parametersPrec.xml";
    myCLP.setOption("precfile", &xmlPrecFile, ".xml file with Inputparameters.");
    string xmlSolverFile = "parametersSolver.xml";
    myCLP.setOption("solverfile", &xmlSolverFile, ".xml file with Inputparameters.");

    string xmlTekoPrecFile = "parametersTeko.xml";
    myCLP.setOption("tekoprecfile", &xmlTekoPrecFile, ".xml file with Inputparameters.");

    double length = 4.;
    myCLP.setOption("length", &length, "length of domain.");

    myCLP.recogniseAllOptions(true);
    myCLP.throwExceptions(false);
    Teuchos::CommandLineProcessor::EParseCommandLineReturn parseReturn = myCLP.parse(argc, argv);
    if (parseReturn == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED)
    {
        MPI_Finalize();
        return 0;
    }
    // Einlesen von Parameterwerten
    {
        ParameterListPtr_Type parameterListProblem = Teuchos::getParametersFromXmlFile(xmlProblemFile);

        ParameterListPtr_Type parameterListProblem2 = Teuchos::getParametersFromXmlFile(xmlProblemFile2);

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

        // For comparison with Navier-Stokes solver
        ParameterListPtr_Type parameterListAll2(new Teuchos::ParameterList(*parameterListProblem2));
        if (!precMethod.compare("Monolithic"))
            parameterListAll2->setParameters(*parameterListPrec);
        else
            parameterListAll2->setParameters(*parameterListPrecTeko);
        parameterListAll2->setParameters(*parameterListSolver);

        int minNumberSubdomains;
        if (!meshType.compare("structured"))
        {
            minNumberSubdomains = 1;
        }
        else if (!meshType.compare("structured_bfs"))
        {
            minNumberSubdomains = (int)2 * length + 1;
        }

        int numProcsCoarseSolve = parameterListProblem->sublist("General").get("Mpi Ranks Coarse", 0);
        int size = comm->getSize() - numProcsCoarseSolve;

        // Inside we construct the mesh so define connectivities etc.
        {
            DomainPtr_Type domainPressure;
            DomainPtr_Type domainVelocity;

            domainPressure.reset(new Domain<SC, LO, GO, NO>(comm, dim));
            domainVelocity.reset(new Domain<SC, LO, GO, NO>(comm, dim));

            MeshPartitioner_Type::DomainPtrArray_Type domainP1Array(1);
            domainP1Array[0] = domainPressure;

            ParameterListPtr_Type pListPartitioner = sublist(parameterListProblem, "Mesh Partitioner");
            MeshPartitioner<SC, LO, GO, NO> partitionerP1(domainP1Array, pListPartitioner, "P1", dim);

            partitionerP1.readAndPartition();

            if (discVelocity == "P2")
                domainVelocity->buildP2ofP1Domain(domainPressure); // jumps
            else
                domainVelocity = domainPressure;

            //          **********************  BOUNDARY CONDITIONS ***********************************
            std::string bcType = parameterListProblem->sublist("Parameter").get("BC Type", "parabolic");
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

            if (dim == 2)
            {
                // Flags for rectangular channel inside this folder rectangular_200.mesh
                //            1
                //   ------------------
                //   |                |
                // 4 |                | 3
                //   |                |   
                //   ------------------
                //            2
                //************* POISEUILLE FLOW  *************
                bcFactory->addBC(zeroDirichlet2D, 1, 0, domainVelocity, "Dirichlet", dim);                 // wall
                bcFactory->addBC(zeroDirichlet2D, 2, 0, domainVelocity, "Dirichlet", dim, parameter_vec);  // wall
                
                bcFactory->addBC(inflowPowerLaw2D, 4, 0, domainVelocity, "Dirichlet", dim, parameter_vec); // Analytical solution Power-Law 2D Poiseuille flow based on parameters and pressure gradient
                //bcFactory->addBC(couette2D, 4, 0, domainVelocity, "Dirichlet", dim, parameter_vec); // Analytical solution Power-Law 2D Poiseuille flow based on parameters and pressure gradient
                // Couette 2D gives us constant inflow velocity specified by Max Velocity entry
                // bcFactory->addBC(zeroDirichlet, , 1, domainPressure, "Dirichlet", 1); //Outflow
                //  Ich muss irgendwo ein Druck Punkt festlegen bestimmt!!
            }
            else if (dim == 3)
            {
                bcFactory->addBC(zeroDirichlet3D, 1, 0, domainVelocity, "Dirichlet", dim);                  // upper
                bcFactory->addBC(zeroDirichlet3D, 2, 0, domainVelocity, "Dirichlet", dim);                  // lower
                bcFactory->addBC(zeroDirichlet3D, 3, 0, domainVelocity, "Dirichlet", dim);                  // sides
                bcFactory->addBC(zeroDirichlet3D, 4, 0, domainVelocity, "Dirichlet", dim);                  // sides
                bcFactory->addBC(inflowParabolic3D, 6, 0, domainVelocity, "Dirichlet", dim, parameter_vec); // inlet
                bcFactory->addBC(zeroDirichlet, 5, 1, domainPressure, "Dirichlet", 1);                      // Outflow - Try Neumann but then we have to set a pressure point anywhere else that why // After we added the proper code line in NavierStokesAssFE we can set this for P2-P1 element
                // bcFactory->addBC(zeroDirichlet3D, 3, 0, domainVelocity, "Dirichlet", dim); // lower

            }
            //** Stenosis 2D **********************************************************************/*


            //          **********************  CALL SOLVER ***********************************
            NavierStokesAssFE<SC, LO, GO, NO> navierStokesAssFE(domainVelocity, discVelocity, domainPressure, discPressure, parameterListAll);

            {
                MAIN_TIMER_START(NavierStokesAssFE, " AssFE:   Assemble System and solve");
                navierStokesAssFE.addBoundaries(bcFactory);
                navierStokesAssFE.initializeProblem();
                navierStokesAssFE.assemble();

                navierStokesAssFE.setBoundariesRHS();

                std::string nlSolverType = parameterListProblem->sublist("General").get("Linearization", "FixedPoint");
                NonLinearSolver<SC, LO, GO, NO> nlSolverAssFE(nlSolverType, linearization_SwitchToNewton_);
                nlSolverAssFE.solve(navierStokesAssFE); // jumps into NonLinearSolver_def.hpp

                MAIN_TIMER_STOP(NavierStokesAssFE);
                comm->barrier();
            }

            //          **********************  POST-PROCESSING ***********************************
            Teuchos::TimeMonitor::report(cout, "Main");

            //****************************************************************************************
            //****************************************************************************************
            //          **********************  POST-PROCESSING - VISCOSITY COMPUTATION FOR NON-NEWTONIAN FLUID ***********************************
            // We only write out viscosity field if we consider non-Newtonian fluid because otherwise it is constant
            if ((parameterListProblem->sublist("Material").get("Newtonian", true) == false) && (parameterListProblem->sublist("Material").get("WriteOutViscosity", false)) == true)
            {

                Teuchos::RCP<ExporterParaView<SC, LO, GO, NO>> exParaViscsoity(new ExporterParaView<SC, LO, GO, NO>());
                DomainPtr_Type domV = domainVelocity;

                /*Viskosität berechnen auf Basis der berechnet Geschwindigkeitslösung*/
                navierStokesAssFE.computeViscosity_Solution();
                navierStokesAssFE.getViscosity_Solution();
       
                //**************** Write out viscosity ****************** 
                //so we need something from type multivector so this is not working because we can not access
                // navierStokesAssFE.feFactory_->visco_output_->getBlock(0)
                Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> exportSolutionViscosityAssFE = navierStokesAssFE.viscosity_element_;
                exParaViscsoity->setup("viscosity", domV->getMesh(), "P0");
                exParaViscsoity->addVariable(exportSolutionViscosityAssFE, "viscosityAssFE", "Scalar", 1, domV->getElementMap());
                exParaViscsoity->save(0.0);
            }

            //****************************************************************************************
            //****************************************************************************************
            //          **********************  POST-PROCESSING - WRITE OUT VELOCITY AND PRESSURE ***********************************
            Teuchos::RCP<ExporterParaView<SC, LO, GO, NO>> exParaVelocity(new ExporterParaView<SC, LO, GO, NO>());
            Teuchos::RCP<ExporterParaView<SC, LO, GO, NO>> exParaPressure(new ExporterParaView<SC, LO, GO, NO>());

            Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> exportSolutionVAssFE = navierStokesAssFE.getSolution()->getBlock(0);
            Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> exportSolutionPAssFE = navierStokesAssFE.getSolution()->getBlock(1);

            DomainPtr_Type dom = domainVelocity;
            exParaVelocity->setup("velocity", dom->getMesh(), dom->getFEType());
            UN dofsPerNode = dim;
            exParaVelocity->addVariable(exportSolutionVAssFE, "uAssFE", "Vector", dofsPerNode, dom->getMapUnique());

            dom = domainPressure;
            exParaPressure->setup("pressure", dom->getMesh(), dom->getFEType());
            exParaPressure->addVariable(exportSolutionPAssFE, "pAssFE", "Scalar", 1, dom->getMapUnique());

            exParaVelocity->save(0.0);
            exParaPressure->save(0.0);


            //*************************** FLAGS *************************************
            // Leas Code to obtain Flags in Paraview
            Teuchos::RCP<ExporterParaView<SC, LO, GO, NO>> exParaF(new ExporterParaView<SC, LO, GO, NO>());
            Teuchos::RCP<MultiVector<SC, LO, GO, NO>> exportSolution(new MultiVector<SC, LO, GO, NO>(domainVelocity->getMapUnique()));
            vec_int_ptr_Type BCFlags = domainVelocity->getBCFlagUnique();
            Teuchos::ArrayRCP<SC> entries = exportSolution->getDataNonConst(0);
            for (int i = 0; i < entries.size(); i++)
            {
                entries[i] = BCFlags->at(i);
            }
            Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> exportSolutionConst = exportSolution;
            exParaF->setup("Flags", domainVelocity->getMesh(), domainVelocity->getFEType());
            exParaF->addVariable(exportSolutionConst, "Flags", "Scalar", 1, domainVelocity->getMapUnique(), domainVelocity->getMapUniqueP2());
            exParaF->save(0.0);

        
            //*********************************************************************************************************
            //*********************************************************************************************************
            //*********************************************************************************************************
            //*********************************************************************************************************
            //*********************  POST-PROCESSING - COMPARISON BETWEEN NON-NEWTONIAN POWER-LAW FLOW FOR n=1 with NAVIER STOKES SOLVER ***********************************
            // So basically they should give same results for n=1.0 as then Power-Law reduces to constant viscosity case
            // BUT as the Navier-Stokes equations are implemented in the conventional formulation with the laplacian operator the outlet boudary condition 
            // and therefore the solution at the outlet will differ -> if we want to have the exactly same formulation we have to consider the addiional boundary integral 
            if ((parameterListProblem->sublist("Material").get("compareNavierStokes", false)) == true && (parameterListProblem->sublist("Material").get("PowerLaw index n", 1.) == 1.) && (parameterListProblem->sublist("Material").get("ShearThinningModel", "") == "Power-Law"))
            {
                if (verbose)
                {
                    cout << "###############################################################" << endl;
                    cout << "##################### Start Navier-Stokes Newtonian Solver ####################" << endl;
                    cout << "###############################################################" << endl;
                }

                NavierStokesAssFE<SC, LO, GO, NO> navierStokesAssFEModel2(domainVelocity, discVelocity, domainPressure, discPressure, parameterListAll2);
                {
                    MAIN_TIMER_START(NavierStokesAssFE, " AssFE:   Assemble System and solve");
                    navierStokesAssFEModel2.addBoundaries(bcFactory);
                    navierStokesAssFEModel2.initializeProblem();
                    navierStokesAssFEModel2.assemble();
                    navierStokesAssFEModel2.setBoundariesRHS();
                    std::string nlSolverType = parameterListProblem->sublist("General").get("Linearization", "FixedPoint");
                    NonLinearSolver<SC, LO, GO, NO> nlSolverAssFE(nlSolverType);
                    nlSolverAssFE.solve(navierStokesAssFEModel2); // jumps into NonLinearSolver_def.hpp
                    MAIN_TIMER_STOP(NavierStokesAssFE);
                    comm->barrier();
                }

                // **********************  POST-PROCESSING NS Solver comparison ***********************************
                Teuchos::RCP<ExporterParaView<SC, LO, GO, NO>> exParaVelocityModel2(new ExporterParaView<SC, LO, GO, NO>());
                Teuchos::RCP<ExporterParaView<SC, LO, GO, NO>> exParaPressureModel2(new ExporterParaView<SC, LO, GO, NO>());
                Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> exportSolutionVAssFEModel2 = navierStokesAssFEModel2.getSolution()->getBlock(0);
                Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> exportSolutionPAssFEModel2 = navierStokesAssFEModel2.getSolution()->getBlock(1);
                dom = domainVelocity;
                exParaVelocityModel2->setup("velocity_Newtonian_and_error", dom->getMesh(), dom->getFEType());
                exParaVelocityModel2->addVariable(exportSolutionVAssFEModel2, "uAssFE_newtonian", "Vector", dofsPerNode, dom->getMapUnique());

                dom = domainPressure;
                exParaPressureModel2->setup("pressure_Newtonian_and_error", dom->getMesh(), dom->getFEType());
                exParaPressureModel2->addVariable(exportSolutionPAssFEModel2, "pAssFE_newtonian", "Scalar", 1, dom->getMapUnique());

                // Compute error between Navier-Stokes Solver and Power-Law Solver (only reasonable if n=1 in Power-Law!)
                // Calculating the error per node
                Teuchos::RCP<MultiVector<SC, LO, GO, NO>> errorValues = Teuchos::rcp(new MultiVector<SC, LO, GO, NO>(navierStokesAssFE.getSolution()->getBlock(0)->getMap()));
                // this = alpha*A + beta*B + gamma*this
                errorValues->update(1., exportSolutionVAssFE, -1., exportSolutionVAssFEModel2, 0.);
                // Taking abs norm
                Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> errorValuesAbsV = errorValues;
                errorValues->abs(errorValuesAbsV);
                dom = domainVelocity;
                exParaVelocityModel2->addVariable(errorValuesAbsV, "u_pl_u_n", "Vector", dofsPerNode, dom->getMapUnique());

                // Calculating the error per node
                Teuchos::RCP<MultiVector<SC, LO, GO, NO>> errorValuesP = Teuchos::rcp(new MultiVector<SC, LO, GO, NO>(navierStokesAssFE.getSolution()->getBlock(1)->getMap()));
                // this = alpha*A + beta*B + gamma*this
                errorValuesP->update(1., exportSolutionPAssFE, -1., exportSolutionPAssFEModel2, 0.);

                // Taking abs norm
                Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> errorValuesAbsP = errorValuesP;

                errorValuesP->abs(errorValuesAbsP);
                dom = domainPressure;
                exParaPressureModel2->addVariable(errorValuesAbsP, "p_pl_p_n", "Scalar", 1, dom->getMapUnique());

                exParaVelocityModel2->save(0.0);
                exParaPressureModel2->save(0.0);

                // Error comparison
                Teuchos::Array<SC> norm(1);
                errorValues->norm2(norm); // const Teuchos::ArrayView<typename Teuchos::ScalarTraits<SC>::magnitudeType> &norms);
                double res = norm[0];
                if (comm->getRank() == 0)
                    cout << " Inf Norm of Error of Solutions:" << res << endl;
                double twoNormError = res;

                navierStokesAssFE.getSolution()->norm2(norm);
                res = norm[0];
                if (comm->getRank() == 0)
                    cout << " 2 rel. Norm of solutions n=1 power law:" << twoNormError / res << endl;
                navierStokesAssFEModel2.getSolution()->norm2(norm);
                res = norm[0];
                if (comm->getRank() == 0)
                    cout << " 2 rel. Norm of solutions navier stokes assemFE:" << twoNormError / res << endl;

                // Get Block A
                MatrixPtr_Type Sum2 = Teuchos::rcp(new Matrix_Type(domainVelocity->getMapVecFieldUnique(), domainVelocity->getDimension() * domainVelocity->getApproxEntriesPerRow()));
                navierStokesAssFEModel2.getSystem()->getBlock(0, 0)->addMatrix(1, Sum2, 1);
                navierStokesAssFE.getSystem()->getBlock(0, 0)->addMatrix(-1, Sum2, 1);
                Teuchos::ArrayView<const GO> indices;
                Teuchos::ArrayView<const SC> values;
                res = 0.;
                for (UN i = 0; i < domainVelocity->getMapUnique()->getMaxLocalIndex() + 1; i++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        GO row = dim * domainVelocity->getMapUnique()->getGlobalElement(i) + d;
                        Sum2->getGlobalRowView(row, indices, values);

                        for (int j = 0; j < values.size(); j++)
                        {
                            if (fabs(values[j]) > res)
                                res = fabs(values[j]);
                        }
                    }
                }
                res = fabs(res);
                reduceAll<int, double>(*comm, REDUCE_MAX, res, outArg(res));
                if (comm->getRank() == 0)
                    cout << "Inf Norm of Difference between Block A: " << res << endl;

                // Get Block B
                MatrixPtr_Type Sum1 = Teuchos::rcp(new Matrix_Type(domainPressure->getMapUnique(), domainVelocity->getDimension() * domainVelocity->getApproxEntriesPerRow()));
                navierStokesAssFEModel2.getSystem()->getBlock(1, 0)->addMatrix(1, Sum1, 1);
                navierStokesAssFE.getSystem()->getBlock(1, 0)->addMatrix(-1, Sum1, 1);
                res = 0.;
                for (UN i = 0; i < domainPressure->getMapUnique()->getMaxLocalIndex() + 1; i++)
                {
                    GO row = domainPressure->getMapUnique()->getGlobalElement(i);
                    Sum1->getGlobalRowView(row, indices, values);
                    for (int j = 0; j < values.size(); j++)
                    {
                        res += fabs(values[j]);
                    }
                }
                res = fabs(res);
                reduceAll<int, double>(*comm, REDUCE_SUM, res, outArg(res));
                if (comm->getRank() == 0)
                    cout << " Norm of Difference between Block B: " << res << endl;
                /*
                if(comm->getRank() == 0)
                    cout << " Print A matrix ns solver: " << res << endl;
                navierStokesAssFEModel2.getSystem()->getBlock(0,0)->print(Teuchos::VERB_EXTREME);
                 if(comm->getRank() == 0)
                    cout << " Print A matrix pl solver: " << res << endl;
                navierStokesAssFE.getSystem()->getBlock(0,0)->print(Teuchos::VERB_EXTREME);
                */
            }
        }
    }
    Teuchos::TimeMonitor::report(cout);
    return (EXIT_SUCCESS);
}
