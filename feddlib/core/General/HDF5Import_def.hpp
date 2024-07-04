#ifndef HDF5IMPORT_DEF_hpp
#define HDF5IMPORT_DEF_hpp

#include "HDF5Import_decl.hpp"

/*!
 Definition of ExporterParaView

 @brief  ExporterParaView
 @author Christian Hochmuth
 @version 1.0
 @copyright CH
 */

using namespace std;
namespace FEDD {

template<class SC,class LO,class GO,class NO>
HDF5Import<SC,LO,GO,NO>::HDF5Import(MapConstPtr_Type readMap, std::string inputFilename):
hdf5exporter_(),
comm_(),
commEpetra_()
{
    Teuchos::RCP<const Teuchos::MpiComm<int> > mpiComm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int> >( readMap->getComm() );
    commEpetra_.reset( new Epetra_MpiComm( *mpiComm->getRawMpiComm() ) );

    Teuchos::ArrayView< const GO > indices = readMap->getNodeElementList();
    int* intGlobIDs = new int[indices.size()];
    for (int i=0; i<indices.size(); i++) {
        intGlobIDs[i] = (int) indices[i];
    }

    int nmbPointsGlob = readMap->getGlobalNumElements();

    EpetraMapPtr_Type mapEpetra = Teuchos::rcp(new Epetra_Map((int)nmbPointsGlob,indices.size(),intGlobIDs,0,*commEpetra_));

    readMap_ = mapEpetra;

   
    hdf5exporter_.reset( new HDF5_Type(*commEpetra_) );
    //hdf5exporter_->Create(outputFilename_);
  
    inputFilename_ = inputFilename;
    hdf5exporter_->Open(inputFilename_+".h5");

    u_import_mv_.reset(new MultiVector_Type(readMap)); // ParaView always uses 3D Data. for 2D Data the last entries (for the 3rd Dim) are all zero.


}

template<class SC,class LO,class GO,class NO>
typename HDF5Import<SC, LO, GO, NO>::MultiVectorPtr_Type HDF5Import<SC,LO,GO,NO>::readVariablesHDF5(string varName){
    TEUCHOS_TEST_FOR_EXCEPTION( !hdf5exporter_->IsContained(varName), std::logic_error, "Requested varName: " << varName << " not contained in hdf file "<< inputFilename_ << ".h5.");
    hdf5exporter_->Read(varName,*readMap_,u_import_);
    for (int i=0; i<u_import_mv_->getLocalLength(); i++) {
        Teuchos::ArrayRCP<SC> tmpData = u_import_mv_->getDataNonConst(0);
        tmpData[i] = u_import_->Values()[i];
    }

    hdf5exporter_->Flush();

    return u_import_mv_;
    
}


}
#endif
