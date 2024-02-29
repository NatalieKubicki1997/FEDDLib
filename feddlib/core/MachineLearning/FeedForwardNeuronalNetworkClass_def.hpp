#ifndef FEEDFORWARDNEURONALNETWORKCLASS_DEF_hpp
#define FEEDFORWARDNEURONALNETWORKCLASS_DEF_hpp

#include "FeedForwardNeuronalNetworkClass_decl.hpp"

namespace FEDD {


template <class SC, class LO, class GO, class NO>
FeedForwardNeuronalNetworkClass<SC,LO,GO,NO>::FeedForwardNeuronalNetworkClass(ParameterListPtr_Type params):TrainedMLModelClass<SC,LO,GO,NO>(params)
{

	params_=params;

}

/* This should be inherited from the base abstract class
template <class SC, class LO, class GO, class NO>
void DifferentiableFuncClass<SC,LO,GO,NO>::updateParams( ParameterListPtr_Type params){
	params_ = params;
};
*/

}
#endif
