#ifndef HDF5EXPORT_DEF_hpp
#define HDF5EXPORT_DEF_hpp

#include "HDF5Export_decl.hpp"


using namespace std;
namespace FEDD {

template<class SC,class LO,class GO,class NO>
HDF5Export<SC,LO,GO,NO>::HDF5Export():
hdf5exporter_(),
comm_(),
commEpetra_()
{

}

template<class SC,class LO,class GO,class NO>
HDF5Export<SC,LO,GO,NO>::HDF5Export(MapConstPtr_Type writeMap, MultiVectorPtr_Type writeVector, int dim, std::string outputFilename):
hdf5exporter_(),
comm_(),
commEpetra_()
{

    Teuchos::RCP<const Teuchos::MpiComm<int> > mpiComm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int> >( writeMap->getComm() );
    commEpetra_.reset( new Epetra_MpiComm( *mpiComm->getRawMpiComm() ) );

    Teuchos::ArrayView< const GO > indices = writeMap->getNodeElementList();
    int* intGlobIDs = new int[indices.size()];
    for (int i=0; i<indices.size(); i++) {
        intGlobIDs[i] = (int) indices[i];
    }

    int nmbPointsGlob = writeMap->getGlobalNumElements();

    EpetraMapPtr_Type mapEpetra = Teuchos::rcp(new Epetra_Map((int)nmbPointsGlob,indices.size(),intGlobIDs,0,*commEpetra_));

    readMap_ = mapEpetra;

  
    hdf5exporter_.reset( new HDF5_Type(*commEpetra_) );

    hdf5exporter_->Create(outputFilename+".h5");

    //hdf5exporter_->Create(outputFilename_);
    EpetraMVPtr_Type u_export(new Epetra_MultiVector(*(mapEpetra),1)); // ParaView always uses 3D Data. for 2D Data the last entries (for the 3rd Dim) are all zero.

    for (int i=0; i<writeVector->getLocalLength(); i++) {
        Teuchos::ArrayRCP<const SC> tmpData = writeVector->getData(0);
        u_export->ReplaceMyValue( i, 0, tmpData[i] );
    }

    u_export_ = u_export;
  
}

template<class SC,class LO,class GO,class NO>
void HDF5Export<SC,LO,GO,NO>::writeVariablesHDF5(string varName){

    hdf5exporter_->Write(varName,*u_export_);

    hdf5exporter_->Flush();
    
}

template<class SC,class LO,class GO,class NO>
void HDF5Export<SC,LO,GO,NO>::closeExporter(){
    hdf5exporter_->Close();
}

}
#endif
