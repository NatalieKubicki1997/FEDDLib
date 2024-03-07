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
// If you make changes to this file always remember to compile

/*!
 Write a simple sample script for computing a Matrix Vector product
 where we read a vector from a file and we read a matrix from a file

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

    /*
    Simple test
    Matrix 3x4
    Vector 4x1
    Result Vector 3x1

    1 1 1 1        1          10
    2 2 2 2   x    2    =     20   should be the results
    3 3 3 3        3          30
                   4  

    */


    // We have to somehow initialize the size of the matrix/ vector we want to read - therefore user it has to be set 
    GO numGlobalElements_vec = 4;
    myCLP.setOption("nge",&numGlobalElements_vec,"numGlobalElements_vec.");


    Array<GO> indicesa(numGlobalElements_vec);
    for (UN i=0; i<indicesa.size(); i++) {
        indicesa[i] = i; // [0 1 2 3]
    }

    // This lines were added for a simple check what happens if we do not have fillComplete(...) in the Code and have really "random" domain map -> what we saw was that
    // the domain map of the matrix did not match because it was constructed arbitray BUT if we add fillComplete(mapUniqueIn, ...) it works!
    /*Teuchos::Array<GO> indicesa;
    if(rank==0){
       indicesa.push_back(3);
    }
    else{
        indicesa.push_back(0);
        indicesa.push_back(1);
        indicesa.push_back(2);
    }
      MapConstPtr_Type mapUnique_In = rcp( new Map_Type(ulib_str, numGlobalElements_vec, indicesa(), 0, commWorld) );    
    */

   

    MapConstPtr_Type mapRepeated_In = rcp( new Map_Type(ulib_str, commWorld->getSize()*numGlobalElements_vec, indicesa(), 0, commWorld) ); // The repeated map has size 9 if we have 3 processors and 3 global data
    MapConstPtr_Type mapUnique_In = mapRepeated_In->buildUniqueMap();

    mapRepeated_In->print(); // In der repeated Map hat jeder Prozessor alle Informationen
    mapUnique_In->print();   // In der unique Map hat ein Prozessor nur Informationen über sein globalen Stand im Vektor   


    MVPtrConst_Type mvUni = rcp( new MV_Type( mapUnique_In ) ); // Es wird lokal ein neuer Multivektor erzeugt auf Basis der Unique Map
    mvUni->readMM("test_vector_4x1.csv"); // Und jeder Prozessor liest jetzt ein Teil des Vektors ein
    mvUni->print();


    // Nun erzeugen wir den Vektor wo nachher die Ergebnisse rein sollen
    Teuchos::Array<GO> indices2(numGlobalElements_vec-1);
    for (int i=0; i<indices2.size(); i++) {
        indices2[i] = i;
    }
    MapConstPtr_Type map2rep = rcp( new Map_Type( ulib_str, commWorld->getSize()*(numGlobalElements_vec-1), indices2(), 0, commWorld ) );
    MapConstPtr_Type mapUnique_Out = map2rep->buildUniqueMap(); // Hier auch wieder auf Basis der UniqueMap


    MVPtr_Type mvRes= rcp( new MV_Type(mapUnique_Out  )); // Erstellung des Ergebniss-Vektor
    mvRes->print(); 

   
    TEUCHOS_TEST_FOR_EXCEPTION(!(!ulib_str.compare("Tpetra") || !ulib_str.compare("Epetra") ) , std::runtime_error,"Unknown algebra type");

    // We want to construct a Matrix
    // with numGlobalElements_vec columns
    // and  numGlobalElemnts -1 rows

    // Uses internally: 
    // Build (const RCP< const Map > &rowMap, size_t maxNumEntriesPerRow)
    // Constructor specifying the max number of non-zeros per row and providing column map.

    // Wir wollen eine 3x4 Matrix erstellen
    // Constructor specifying the number of non-zeros for all rows.

     MatrixPtr_Type matrix = rcp( new Matrix_Type( mapUnique_Out, numGlobalElements_vec ) );
     //XpetraMatrixPtr_Type matrix_ = matrix->getXpetraMatrix(); // This is constant
     std::string filename_matrix = "test_matrx_3x4_sparse.csv"; // Change the filename as per your requirement
     // Wir lesen allgemein eine sparse Matrix ein!


     XPetra_IO_Type ExampleIO;
     //ExampleIO.Write(filename, *(mapUnique_In->getXpetraMap()) );
     
    // matrix->matrix_ is XpetraMatrixPtr_Type matrix_
     std::string filename = "output_file.txt"; // Change the filename as per your requirement
     matrix->readMM(filename_matrix, commWorld); //= ExampleIO.Read(filename_matrix,   ulib, commWorld, false);
    
    //Map->getGlobalNumElements()	const The number of elements in this Map.
    //Map->getLocalNumElements()	const The number of elements belonging to the calling process.
    // \brief Filling of Matrix based on specific domainmap (column map) and rangeMap (rowmap).
    matrix->fillComplete( mapUnique_In , mapUnique_Out ); // 
    matrix->print(VERB_EXTREME);
                
    std::cout << "****************************" << std::endl;

    matrix->print();

    mvRes->print();
    mvUni->print();

    matrix->apply(mvUni, *mvRes,Teuchos::NO_TRANS,1.0,0.0);
    mvRes->print();

   
    return(EXIT_SUCCESS);
}
