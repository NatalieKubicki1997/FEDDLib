#ifndef CARREAUYASUDA_DEF_hpp
#define CARREAUYASUDA_DEF_hpp

#include "Tanh_decl.hpp"

namespace FEDD {

template <class SC, class LO, class GO, class NO>
Tanh<SC,LO,GO,NO>::Tanh(ParameterListPtr_Type params):
DifferentiableFuncClass<SC,LO,GO,NO>(params)
{
    this->params_=params;
    //	TEUCHOS_TEST_FOR_EXCEPTION( true, std::logic_error, "No discretisation Information for Velocity in Navier Stokes Element." );

	
}

template <class SC, class LO, class GO, class NO>
void Tanh<SC,LO,GO,NO>::evaluateMapping(ParameterListPtr_Type params, MultiVectorConstPtr_Type input, MultiVectorPtr_Type &output) {
    // We should check that the mapping of our input and output vector are the same!!
	
    //TEUCHOS_TEST_FOR_EXCEPTION( (input->getMap()->getXpetraMap())->isSameAs( output->getMap()->getXpetraMap() ), std::logic_error,"Source maps of Importer and Multivector are not the same."); // Something like this
    // Write here the function that reads in on each processor all values that it has
    // Then apply to each value the tanh function
    // Next write it back

}



template <class SC, class LO, class GO, class NO>
void Tanh<SC,LO,GO,NO>::setParams(ParameterListPtr_Type params){
    this->params_=params;
 }




template <class SC, class LO, class GO, class NO>
void Tanh<SC,LO,GO,NO>::echoInformationMapping(){
            std::cout << "************************************************************ "  <<std::endl;
            std::cout << "-- Used Functional Mapping is Output = Tanh (Input) ..."  <<std::endl;
  }




}
#endif

