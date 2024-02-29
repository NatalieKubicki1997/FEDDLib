#ifndef INPUTTOOUTPUTMAPPINGCLASS_DECL_hpp
#define INPUTTOOUTPUTMAPPINGCLASS_DECL_hpp


#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector.hpp"

//#include "feddlib/core/LinearAlgebra/Matrix.hpp" maybe we will need something like this for multidimensional?


namespace FEDD {


    /*!
    \class InputToOutputMappingClass 
    \brief This abstract class defining the general concepts of a mapping between an input and an output. To hold it general the input can be any high-dimensional array

    \tparam SC The scalar type. So far, this is always double, but having it as a template parameter would allow flexibily, e.g., for using complex instead
    \tparam LO The local ordinal type. The is the index type for local indices
    \tparam GO The global ordinal type. The is the index type for global indices
    @todo This should actually be removed since the class should operate only on element level)
    \tparam NO The Kokkos Node type. This would allow for performance portibility when using Kokkos. Currently, this is not used.

    The material parameters can be provided through a Teuchos::ParameterList object which will contain 
    all the parameters specified in the input file `ABC.xml`.
    The structure of the input file and, hence, of the resulting parameter list can be chosen freely. The FEDDLib will take care of reading the parameters 
    from the file and making them available.


    I made this general classification of an general Input to Output mapping in order to differentiate between different complexeties of Input to Output mapping 
    For example a really simple one is a general differentiable function - therefore we derive another abstract subclass called differentiableFuncClass.
    We could also derive another subclass called NonDifferentiableFuncClass and for evaluating the derivatives throw a error
    Another General Input to Output Mapping are our TrainedMlModels.



    */
    template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
    class InputToOutputMappingClass{
    public:
 
        typedef MultiVector<SC,LO,GO,NO> MultiVector_Type;
        typedef Teuchos::RCP<MultiVector_Type> MultiVectorPtr_Type;
        typedef Teuchos::RCP<const MultiVector_Type> MultiVectorConstPtr_Type;
        /*!
         \brief Implement a mapping description for evaluating output in dependence of given input and specified parameters
         @param[in] params Parameterlist as read from the xml file (maybe redundant)
         @param[in] x Independent variable
         @param[in,out] res Dependent variable
        */
        virtual void evaluateMapping(ParameterListPtr_Type params, MultiVectorConstPtr_Type input, MultiVectorPtr_Type &output) = 0;


        /*!
         \brief Computes value of derivative of defined function in evaluateMapping
         @param[in] params Parameterlist as read from the xml file (maybe redundant)
         @param[in] x Independent variable
         @param[in,out] res Dependent variable
        */
        virtual void evaluateDerivative(ParameterListPtr_Type params, MultiVectorConstPtr_Type x, MultiVectorPtr_Type &res) = 0;


        // Consider also specific case where we only have ONE INPUT and ONE Output
        /*
        \brief Implements a functional description for evaluating res in dependence of given dependent variables and specified parameters
                Here we overload the functional evaluation because we often can have the case that we have simple one dimensional function -
                but remember that MANY functional evaluations can be costly so maybe it is computationally more performant to give one array as
                an input and one array as an output
         @param[in] params Parameterlist as read from the xml file (maybe redundant)
         @param[in] x Independent variable
         @param[in,out] res Dependent variable
        */
        virtual void evaluateMapping(ParameterListPtr_Type params, double x, double &res) = 0;

        /*!
         \brief Computes value of derivative of defined function in evaluateMapping specific 
         @param[in] params Parameterlist as read from the xml file (maybe redundant)
         @param[in] x Independent variable
         @param[in,out] res Dependent variable
        */
        virtual void evaluateDerivative(ParameterListPtr_Type params, double x, double &res) = 0;

        /*!
         \brief Function could include different parameters which will be specified in *.xml
                This function should set the needed parameters for defined function to the specified values.
        @param[in] ParameterList as read from the xml file (maybe redundant)
        */
        virtual void setParams(ParameterListPtr_Type params) = 0;

        /*!
        \brief Print e.g. parameter values used in model at runtime or problem specific 
        */
        virtual void echoInformationMapping() = 0;

        /*!
         \brief Set or update the parameters read from the ParameterList.
         @param[in] ParameterList as read from the xml file
        */
        virtual void updateParams(ParameterListPtr_Type params);


    protected:

        /*!
         \brief Constructor
         @param[in] params Parameterlist for current problem
        */
        InputToOutputMappingClass(ParameterListPtr_Type parameters);


        ParameterListPtr_Type params_;

    };
}
#endif
