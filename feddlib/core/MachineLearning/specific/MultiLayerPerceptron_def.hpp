#ifndef MULTILAYERPERCEPTRON_DEF_hpp
#define MULTILAYERPERCEPTRON_DEF_hpp

#include "MultiLayerPerceptron_decl.hpp"

namespace FEDD {


template <class SC, class LO, class GO, class NO>
MultiLayerPerceptron<SC,LO,GO,NO>::MultiLayerPerceptron(ParameterListPtr_Type params):FeedForwardNeuronalNetworkClass<SC,LO,GO,NO>(params)
{

	this->params_=params;

}

/* This should be inherited from the base abstract class
template <class SC, class LO, class GO, class NO>
void DifferentiableFuncClass<SC,LO,GO,NO>::updateParams( ParameterListPtr_Type params){
	params_ = params;
};
*/

}
#endif
