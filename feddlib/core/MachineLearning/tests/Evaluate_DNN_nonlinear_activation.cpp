#define MAIN_ASSERT(A,S) if(!(A)) { cerr<<"Assertion failed. "<<S<<endl; cout.flush(); throw out_of_range("Assertion.");};
#define VERBOSE

#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector.hpp"
#include <Teuchos_GlobalMPISession.hpp>
#include <Xpetra_DefaultPlatform.hpp>
#include <iostream>
#include <Xpetra_MultiVector.hpp>
#include <Xpetra_MapFactory.hpp>
#include <Teuchos_Array.hpp>
#include <iostream>
#include <unistd.h>
#include <string>

#include "feddlib/core/LinearAlgebra/Matrix.hpp"

#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_MatrixMatrix.hpp>
#include <Xpetra_IO.hpp>

#include "feddlib/core/General/DifferentiableFuncClass.hpp"
#include "feddlib/core/MachineLearning/ActivationFunctions/Tanh.hpp"


// If you make changes to this file always remember to compile

/*!
Write a simple test script for the evaluation of a trained neuronal network
with a specific number of layers

 @brief  MultiLayerPerceptron Test
 @author Natalie Kubicki
 @version 1.0
 @copyright CH
 */

using namespace std;
using namespace Teuchos;

typedef unsigned UN;
typedef double SC;
typedef int LO;
typedef default_go GO;
typedef Tpetra::KokkosClassic::DefaultNode::DefaultNodeType NO;
using namespace FEDD;
int main(int argc, char *argv[]) {

    // Everything needed  for parallelization
    /*
    In our code each processor will get local copies of the matrices -
    this is okey because the matrices will be in the start not too large
    We then want to use a serial communicator because for the single processor it is not important to know 
    what other processors do we the matrices
    
    */
    oblackholestream blackhole;
    GlobalMPISession mpiSession(&argc,&argv,&blackhole);

    RCP<const Comm<int> > commWorld = Xpetra::DefaultPlatform::getDefaultPlatform().getComm();
    int rank = commWorld->getRank();
    // Command Line Parameters
    Teuchos::CommandLineProcessor myCLP;
    string ulib_str = "Tpetra";    // Set here that I want to use TPetra and nott EPetra
    myCLP.setOption("ulib",&ulib_str,"Underlying lib");

    TEUCHOS_TEST_FOR_EXCEPTION(!(!ulib_str.compare("Tpetra") || !ulib_str.compare("Epetra") ) , std::runtime_error, "Unknown algebra type");

        Xpetra::UnderlyingLib ulib;
    if (ulib_str .compare("Epetra"))
        ulib = Xpetra::UseEpetra;
    else if (ulib_str .compare("Tpetra"))
        ulib = Xpetra::UseTpetra;

    myCLP.recogniseAllOptions(true);
    myCLP.throwExceptions(false);
    Teuchos::CommandLineProcessor::EParseCommandLineReturn parseReturn = myCLP.parse(argc,argv);
    if(parseReturn == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED) {
        mpiSession.~GlobalMPISession();
        return 0;
    }

    // Type definitions
    typedef Map<LO,GO,NO> Map_Type;
    typedef RCP<Map_Type> MapPtr_Type;
    typedef RCP<const Map_Type> MapConstPtr_Type;
    typedef MultiVector<SC,LO,GO,NO> MV_Type;
    typedef RCP<MV_Type> MVPtr_Type;

    typedef Map<LO,GO,NO> Map_Type;
    typedef RCP<Map_Type> MapPtr_Type;
    typedef RCP<const Map_Type> MapConstPtr_Type;

    typedef MultiVector<SC,LO,GO,NO> MV_Type;
    typedef RCP<MV_Type> MVPtr_Type;
    typedef RCP<const MV_Type> MVPtrConst_Type;

    typedef Xpetra::Map<LO,GO,NO> XpetraMap_Type;
    typedef RCP<XpetraMap_Type> XpetraMapPtr_Type;
    typedef RCP<const XpetraMap_Type> XpetraMapConstPtr_Type;

    typedef Xpetra::Matrix<SC,LO,GO,NO> XpetraMatrix_Type;
    typedef RCP<XpetraMatrix_Type> XpetraMatrixPtr_Type;

    typedef Matrix<SC,LO,GO,NO> Matrix_Type;
    typedef RCP<Matrix_Type> MatrixPtr_Type;

    typedef Map<LO,GO,NO> Map_Type;
    typedef RCP<Map_Type> MapPtr_Type;

    typedef Xpetra::Map<LO,GO,NO> XpetraMap_Type;
    typedef RCP<XpetraMap_Type> XpetraMapPtr_Type;
    typedef RCP<const XpetraMap_Type> XpetraMapConstPtr_Type;

    typedef Xpetra::Matrix<SC,LO,GO,NO> XpetraMatrix_Type;
    typedef RCP<XpetraMatrix_Type> XpetraMatrixPtr_Type;

    typedef Matrix<SC,LO,GO,NO> Matrix_Type;
    typedef RCP<Matrix_Type> MatrixPtr_Type;

    typedef Map<LO,GO,NO> Map_Type;
    typedef RCP<Map_Type> MapPtr_Type;

    typedef Xpetra::IO<SC,LO,GO,NO> XPetra_IO_Type;


    typedef InputToOutputMappingClass<SC,LO,GO,NO>  InputToOutputMappingClass_Type;    
	typedef Teuchos::RCP<InputToOutputMappingClass_Type> InputToOutputMappingClassPtr_Type;

    /*
    Simple test where each hidden layer has a fixed number of neurons
    so we need to know the length of 
    Input    Hidden Layer   Output
    */

    // We have to somehow initialize the size of the matrix/ vector we want to read - therefore user it has to be set 
    GO numGlobalElements_input = 5;
    GO numGlobalElements_output = 2;
    GO numGlobalElements_hidden = 6;

    myCLP.setOption("nge",&numGlobalElements_input,"numGlobalElements_input.");
    myCLP.setOption("nge",&numGlobalElements_output,"numGlobalElements_output.");
    myCLP.setOption("nge",&numGlobalElements_hidden,"numGlobalElements_hidden.");


    // Construct a simple map for each type in order 
    Array<GO> indices_input(numGlobalElements_input);
    for (UN i=0; i<indices_input.size(); i++) {
        indices_input[i] = i; 
    }
    Array<GO> indices_output(numGlobalElements_output);
    for (UN i=0; i<indices_output.size(); i++) {
        indices_output[i] = i; 
    }
    Array<GO> indices_hidden(numGlobalElements_hidden);
    for (UN i=0; i<indices_hidden.size(); i++) {
        indices_hidden[i] = i; 
    }

    // For this simple example construct repeated map just in such that way that each processor access every other processor
    MapConstPtr_Type mapRepeated_Input = rcp( new Map_Type(ulib_str, commWorld->getSize()*numGlobalElements_input, indices_input(), 0, commWorld) ); // The repeated map has size 9 if we have 3 processors and 3 global data
    MapConstPtr_Type mapUnique_Input = mapRepeated_Input->buildUniqueMap();

    MapConstPtr_Type mapRepeated_Output = rcp( new Map_Type(ulib_str, commWorld->getSize()*numGlobalElements_output, indices_output(), 0, commWorld) ); // The repeated map has size 9 if we have 3 processors and 3 global data
    MapConstPtr_Type mapUnique_Output = mapRepeated_Output->buildUniqueMap();

    MapConstPtr_Type mapRepeated_Hidden = rcp( new Map_Type(ulib_str, commWorld->getSize()*numGlobalElements_hidden, indices_hidden(), 0, commWorld) ); // The repeated map has size 9 if we have 3 processors and 3 global data
    MapConstPtr_Type mapUnique_Hidden = mapRepeated_Hidden->buildUniqueMap();

    
    MVPtr_Type mvInput_a = rcp( new MV_Type( mapUnique_Input ) ); 
    mvInput_a->putScalar(1.0);
    mvInput_a->print();
    MVPtrConst_Type mvInput = rcp( new MV_Type( mapUnique_Input ) ); 
    mvInput = mvInput_a ;// Es wird lokal ein neuer Multivektor erzeugt auf Basis der Unique Map
    mvInput ->print();
    MVPtr_Type mvOutput = rcp( new MV_Type( mapUnique_Output ) ); // Es wird lokal ein neuer Multivektor erzeugt auf Basis der Unique Map
    MVPtr_Type mvHidden = rcp( new MV_Type( mapUnique_Hidden ) ); // Es wird lokal ein neuer Multivektor erzeugt auf Basis der Unique Map

    // If they would vary alot we could also make again a vector list if the resulting vectors

    // Es macht total Sinn einmal vorab die ganzen Objekte einzulesen - unser Fokus sollte bei einer schnellen Auswertung liegen!

    // First read all matrices
     MatrixPtr_Type matrix_w1 = rcp( new Matrix_Type( mapUnique_Hidden, numGlobalElements_input) ); // Gebe hier immer die Map für den Output vektor an weil das netspricht dann auch unser row map
     MatrixPtr_Type matrix_w2 = rcp( new Matrix_Type( mapUnique_Hidden, numGlobalElements_hidden ) ); // Gebe hier immer die Map für den Output vektor an weil das netspricht dann auch unser row map
     MatrixPtr_Type matrix_w3 = rcp( new Matrix_Type( mapUnique_Hidden, numGlobalElements_hidden ) ); // Gebe hier immer die Map für den Output vektor an weil das netspricht dann auch unser row map
     MatrixPtr_Type matrix_w4 = rcp( new Matrix_Type( mapUnique_Output, numGlobalElements_hidden ) ); // Gebe hier immer die Map für den Output vektor an weil das netspricht dann auch unser row map

     
     MVPtrConst_Type mv_b1 = rcp( new MV_Type( mapUnique_Hidden ) ); // Es wird lokal ein neuer Multivektor erzeugt auf Basis der Unique Map
     MVPtrConst_Type mv_b2 = rcp( new MV_Type( mapUnique_Hidden ) ); // Es wird lokal ein neuer Multivektor erzeugt auf Basis der Unique Map
     MVPtrConst_Type mv_b3 = rcp( new MV_Type( mapUnique_Hidden ) ); // Es wird lokal ein neuer Multivektor erzeugt auf Basis der Unique Map
     MVPtrConst_Type mv_b4 = rcp( new MV_Type( mapUnique_Output ) ); // Es wird lokal ein neuer Multivektor erzeugt auf Basis der Unique Map

    

     std::string filename_matrix_w1 = "Test_Data_Linear_Activation_Function/weight_matrix_layer_1.mtx"; // Change the filename as per your requirement
     std::string filename_matrix_w2 = "Test_Data_Linear_Activation_Function/weight_matrix_layer_2.mtx"; // Change the filename as per your requirement
     std::string filename_matrix_w3 = "Test_Data_Linear_Activation_Function/weight_matrix_layer_3.mtx"; // Change the filename as per your requirement
     std::string filename_matrix_w4 = "Test_Data_Linear_Activation_Function/weight_matrix_layer_4.mtx"; // Change the filename as per your requirement

     std::string filename_bias_1 = "Test_Data_Linear_Activation_Function/biases_layer_1.mtx"; // Change the filename as per your requirement
     std::string filename_bias_2 = "Test_Data_Linear_Activation_Function/biases_layer_2.mtx"; // Change the filename as per your requirement
     std::string filename_bias_3 = "Test_Data_Linear_Activation_Function/biases_layer_3.mtx"; // Change the filename as per your requirement
     std::string filename_bias_4 = "Test_Data_Linear_Activation_Function/biases_layer_4.mtx"; // Change the filename as per your requirement


    /* This works lets try it out in a loop
     matrix_w1->readMM(filename_matrix_w1, commWorld); //= ExampleIO.Read(filename_matrix,   ulib, commWorld, false);
     matrix_w2->readMM(filename_matrix_w2, commWorld); //= ExampleIO.Read(filename_matrix,   ulib, commWorld, false);
     matrix_w3->readMM(filename_matrix_w3, commWorld); //= ExampleIO.Read(filename_matrix,   ulib, commWorld, false);
     matrix_w4->readMM(filename_matrix_w4, commWorld); //= ExampleIO.Read(filename_matrix,   ulib, commWorld, false);

     // In real code save filenames in a list and loop through entries

    //Map->getGlobalNumElements()	const The number of elements in this Map.
    //Map->getLocalNumElements()	const The number of elements belonging to the calling process.
    // \brief Filling of Matrix based on specific domainmap (column map) and rangeMap (rowmap).
    matrix_w1->fillComplete( mapUnique_Input , mapUnique_Hidden ); // 
    matrix_w2->fillComplete( mapUnique_Hidden , mapUnique_Hidden ); // 
    matrix_w3->fillComplete( mapUnique_Hidden , mapUnique_Hidden ); // 
    matrix_w4->fillComplete( mapUnique_Hidden , mapUnique_Output ); // 

       matrix_w1->print();
    */

     std::string filenames_w[] = {  filename_matrix_w1,  filename_matrix_w2 ,  filename_matrix_w3,  filename_matrix_w4};
     std::string filenames_b[] = {  filename_bias_1,  filename_bias_2 ,  filename_bias_3,  filename_bias_4};

     std::vector<MatrixPtr_Type>  weight_matrices = {matrix_w1 , matrix_w2, matrix_w3 , matrix_w4} ; 
     std::vector<MVPtrConst_Type> biases = {mv_b1, mv_b2, mv_b3 ,mv_b4} ; 
     int number_layers = weight_matrices.size();
     int number_hidden_layers = number_layers-1 ; // Just the number of hidden layer where we except that each hidden layer has same number of neurons

      InputToOutputMappingClassPtr_Type activationFunction;

      ParameterListPtr_Type params;
      Teuchos::RCP<Tanh<SC,LO,GO,NO>> activationFunctionSpecific(new Tanh<SC,LO,GO,NO>(params) );
      activationFunction=activationFunctionSpecific;
           
     for (int i=0; i<number_layers ; i++)
     {
        weight_matrices[i]->readMM(filenames_w[i], commWorld); 
        biases[i]->readMM(filenames_b[i]); // Und jeder Prozessor liest jetzt ein Teil des Vektors ein

        //Now we have the edge cases of the input and output file
     }
    weight_matrices[0]->fillComplete( mapUnique_Input , mapUnique_Hidden ); // 
    weight_matrices[1]->fillComplete( mapUnique_Hidden , mapUnique_Hidden ); // 
    weight_matrices[2]->fillComplete( mapUnique_Hidden , mapUnique_Hidden ); // 
    weight_matrices[3]->fillComplete( mapUnique_Hidden , mapUnique_Output );

    MVPtrConst_Type mvHidden_copy = rcp( new MV_Type( mapUnique_Hidden  ) );  // 
     
    // So now we can perform the Matrix-Vektor-Produkt:
    // Matrix Vector Operation. Applying MultiVector X to this. Y = alpha * (this)^mode * X + beta * Y. Mode being transposed or not. 
     weight_matrices[0]->apply(mvInput, *mvHidden,Teuchos::NO_TRANS,1.0,0.0); // First Layer is special
    //this = alpha*A + beta*this
    //void update( const SC& alpha, const MultiVector_Type& A, const SC& beta );
    mvHidden->update(1.0, biases[0], 1.0 );
    // we have to add additional bias
     mvHidden_copy = mvHidden;

    // Now we will have here an additional step where we want to apply an activation function to each entry:
    // something like mvHidden->applyNonlinearFunction( InputToOutputMappingClassPtr_Type activation_function )
    activationFunction->evaluateMapping(params, mvHidden_copy , mvHidden); // We give an input and obtain an output

 

     for(int i=1; i<number_hidden_layers ; i++) // Layer in between
     {
                                                                               // Now we have to copy the entries from the one vector into the other
     weight_matrices[i]->apply(mvHidden_copy , *mvHidden,Teuchos::NO_TRANS,1.0,0.0); // Inner Layers with constant number of neurons - problem with that apply(mvHidden, *mvHidden,Teuchos::NO_TRANS,1.0,0.0) is that mvHiddenLayer should be const and non-const at the same time
     mvHidden->update(1.0, biases[i], 1.0 );
     mvHidden_copy = mvHidden;


     }
     weight_matrices[number_layers-1]->apply(mvHidden_copy , *mvOutput,Teuchos::NO_TRANS,1.0,0.0); // Last Layer is special
     mvOutput->update(1.0, biases[number_layers-1], 1.0 );

     mvOutput->print();
     std::cout << "****************************" << std::endl;


   
    return(EXIT_SUCCESS);
}
