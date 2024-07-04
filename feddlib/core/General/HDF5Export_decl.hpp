#ifndef HDF5EXPORT_DECL_hpp
#define HDF5EXPORT_DECL_hpp

#include <fstream>
#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector.hpp"
#include "feddlib/core/Mesh/Mesh.hpp"
// Trilinos
#include <Teuchos_Array.hpp>

#include "Epetra_ConfigDefs.h"
#ifdef HAVE_MPI
#include "mpi.h"
#include "Epetra_MpiComm.h"
#include "Epetra_SerialComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <Epetra_Vector.h>
#include <Epetra_LongLongVector.h>
#include <Epetra_IntVector.h>

#include <EpetraExt_HDF5.h>
#include <hdf5.h>


namespace FEDD {
template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class HDF5Export{
public:
    typedef std::vector<double>										vec_dbl;
    typedef std::vector<std::vector<double> >						vec2D_dbl;
    typedef std::vector<std::vector<int> >							vec2D_int;
    typedef std::vector<std::vector<long long> >					vec2D_longlong;
    typedef Teuchos::RCP<std::vector<int> >							vec_int_ptr;
    typedef Teuchos::RCP<std::vector<long long> >					vec_longlong_ptr;
    typedef Teuchos::RCP<vec_dbl>									vec_dbl_ptr;
    typedef Teuchos::RCP<std::vector<std::vector<double> > >     	vec2D_dbl_ptr;
    typedef Teuchos::RCP<std::vector<std::vector<int> > >        	vec2D_int_ptr;
    typedef Teuchos::RCP<vec2D_longlong >				        	vec2D_longlong_ptr;
    typedef Teuchos::RCP<Epetra_Vector> 							EpetraVec_ptr;
    typedef Teuchos::RCP<Epetra_MpiComm>		 					EpetraComm_ptr;
    typedef Teuchos::RCP<Epetra_IntVector>	 						EpetraVecInt_ptr;
    typedef Teuchos::RCP<Epetra_LongLongVector>	 					EpetraVecLongLong_ptr;
    typedef Teuchos::RCP<Epetra_MultiVector>	 					EpetraMVPtr_Type;
    typedef Teuchos::RCP<Epetra_Map>                               	EpetraMapPtr_Type;

    
    typedef EpetraExt::HDF5 HDF5_Type;
    typedef Teuchos::RCP<HDF5_Type> HDF5Ptr_Type;
    
    typedef Teuchos::Comm<int> Comm_Type;
    typedef Teuchos::RCP<const Comm_Type> CommConstPtr_Type;
    typedef const Teuchos::RCP<const Comm_Type> CommConstPtrConst_Type;
    
    typedef Map<LO,GO,NO> Map_Type;
    typedef Teuchos::RCP<const Map_Type> MapConstPtr_Type;
    typedef const MapConstPtr_Type MapConstPtrConst_Type;
    
    typedef MultiVector<SC,LO,GO,NO> MultiVec_Type;
    typedef Teuchos::RCP<MultiVec_Type> MultiVectorPtr_Type;
    typedef Teuchos::RCP<const MultiVec_Type> MultiVecConstPtr_Type;
    typedef const MultiVecConstPtr_Type MultiVecConstPtrConst_Type;
    
    typedef Mesh<SC,LO,GO,NO> Mesh_Type;
    typedef Teuchos::RCP<Mesh_Type> MeshPtr_Type;
    
    typedef typename Mesh_Type::ElementsPtr_Type ElementsPtr_Type;
    
    HDF5Export();

    HDF5Export(MapConstPtr_Type readMap, MultiVectorPtr_Type writeVector, std::string outputFilename);

    void writeVariablesHDF5(string varName);

    void closeExporter();

protected:
    
    HDF5Ptr_Type hdf5exporter_;
    CommConstPtr_Type comm_;
    Teuchos::RCP<Epetra_MpiComm> commEpetra_;
    
    // ------------------------
    // READ 
    // ------------------------
    std::vector<std::string>   		varNamesRead_;
    EpetraMapPtr_Type               readMap_;
    EpetraMVPtr_Type u_export_;

};

}

#endif
