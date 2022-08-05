#ifndef DIFFERENTIABLEFUNCCLASS_DEF_hpp
#define DIFFERENTIABLEFUNCCLASS_DEF_hpp

#include "DifferentiableFuncClass_decl.hpp"

namespace FEDD {


template <class SC, class LO, class GO, class NO>
DifferentiableFuncClass<SC,LO,GO,NO>::DifferentiableFuncClass(ParameterListPtr_Type params)
{

	params_=params;

}

template <class SC, class LO, class GO, class NO>
void DifferentiableFuncClass<SC,LO,GO,NO>::updateParams( ParameterListPtr_Type params){
	params_ = params;
};

}
#endif
