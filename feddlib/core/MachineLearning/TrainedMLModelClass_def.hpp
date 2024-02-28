#ifndef TRAINEDMLMODELCLASS_DEF_hpp
#define TRAINEDMLMODELCLASS_DEF_hpp

#include "TrainedMLModelClass_decl.hpp"

namespace FEDD {


template <class SC, class LO, class GO, class NO>
TrainedMLModelClassClass<SC,LO,GO,NO>::TrainedMLModelClassClass(ParameterListPtr_Type params):InputToOutputMappingClass<SC,LO,GO,NO>(params)
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
