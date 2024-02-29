#ifndef MULTILAYERPERCEPTRON_DECL_hpp
#define MULTILAYERPERCEPTRON_DECL_hpp


#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector.hpp"
// #include "feddlib/core/General/InputToOutputMappingClass.hpp" Should be not necessary because already included in TrainedMLModelClass
#include "feddlib/core/MachineLearning/FeedForwardNeuronalNetworkClass.hpp"

//#include "feddlib/core/LinearAlgebra/Matrix.hpp" maybe we will need something like this for multidimensional?


namespace FEDD {


    /*!
    \class MultiLayerPerceptronClass
    \brief This abstract class is derived from the abstract class of general input to output mapping.
           It is defining the general concepts of a function and computing function evaluation and evaluation of its derivative.

    \tparam SC The scalar type. So far, this is always double, but having it as a template parameter would allow flexibily, e.g., for using complex instead
    \tparam LO The local ordinal type. The is the index type for local indices
    \tparam GO The global ordinal type. The is the index type for global indices
    @todo This should actually be removed since the class should operate only on element level)
    \tparam NO The Kokkos Node type. This would allow for performance portibility when using Kokkos. Currently, this is not used.

Multilayer perceptron network is one of the most popular NN architectures, which consists of a series
of fully connected layers, called input, hidden, and output layers. The layers are connected by a
directed graph
    */
    template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
    class MultiLayerPerceptron : public FeedForwardNeuronalNetworkClass<SC,LO,GO,NO> {
    public:

        typedef MultiVector<SC,LO,GO,NO> MultiVector_Type;
        typedef Teuchos::RCP<MultiVector_Type> MultiVectorPtr_Type;
        typedef Teuchos::RCP<const MultiVector_Type> MultiVectorConstPtr_Type;


      
        // Inherited Function from base abstract class 
        /*!
         \brief Implement a mapping description for evaluating output in dependence of given input and specified parameters
         @param[in] params Parameterlist as read from the xml file (maybe redundant)
         @param[in] x Independent variable
         @param[in,out] res Dependent variable
        */
        virtual void evaluateMapping(ParameterListPtr_Type params, MultiVectorConstPtr_Type input, MultiVectorPtr_Type &output) override { TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not yet implemented - HAS TO BE IMPLEMENTED."); };

         /*!
         \brief Implements a functional description for evaluating res in dependence of given dependent variables and specified parameters
                Here we overload the functional evaluation because we often can have the case that we have simple one dimensional function -
                but remember that MANY functional evaluations can be costly so maybe it is computationally more performant to give one array as
                an input and one array as an output
         @param[in] params Parameterlist as read from the xml file (maybe redundant)
         @param[in] x Independent variable
         @param[in,out] res Dependent variable
        */
        virtual void evaluateMapping(ParameterListPtr_Type params, double x, double &res) override { TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not yet implemented - HAS TO BE IMPLEMENTED."); };

        /*!
         \brief Computes value of derivative of defined function in evaluateMapping
         @param[in] params Parameterlist as read from the xml file (maybe redundant)
         @param[in] x Independent variable
         @param[in,out] res Dependent variable
        */
        virtual void evaluateDerivative(ParameterListPtr_Type params, double x, double &res) override { TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not yet implemented - HAS TO BE IMPLEMENTED."); };

        /*!
         \brief Computes value of derivative of defined function in evaluateMapping
         @param[in] params Parameterlist as read from the xml file (maybe redundant)
         @param[in] x Independent variable
         @param[in,out] res Dependent variable
        */
        virtual void evaluateDerivative(ParameterListPtr_Type params, MultiVectorConstPtr_Type x, MultiVectorPtr_Type &res) override { TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not yet implemented - HAS TO BE IMPLEMENTED."); };
    
        /*!
         \brief Function could include different parameters which will be specified in *.xml
                This function should set the needed parameters for defined function to the specified values.
        @param[in] ParameterList as read from the xml file (maybe redundant)
        */
        virtual void setParams(ParameterListPtr_Type params) override { TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not yet implemented - HAS TO BE IMPLEMENTED."); };

        /*!
        \brief Print parameter values used in model at runtime
        */
        virtual void echoInformationMapping() override { TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not yet implemented - HAS TO BE IMPLEMENTED."); };

        // Pure virtual functions inherited from abtsract TrainedMLModelClass
        /*!
        \brief In this method for a specific realisation of an object for example for a dense neuronal network the weight matrix, bias are initialized with values from an external e.g. csv file
        */
        virtual void initializeMatrices() override { TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not yet implemented - HAS TO BE IMPLEMENTED."); };


        // New virtual functions

        // Do we need a function like getOutput? The advantage would be if we have an complex Neuronal Network which is costly to evaluate we can save it but only for this one input 
        //virtual void getOutput() = 0;

    protected:

        /*!
         \brief Constructor
         @param[in] params Parameterlist for current problem
        */
        MultiLayerPerceptron(ParameterListPtr_Type parameters);


     

        /* Defined in the abstract base class
        ParameterListPtr_Type params_;
        int numberOfLayers;
        MultiVectorPtr_Type WeightMatrix;  // Should it be a pointer? 
        MultiVectorPtr_Type biasVector;
        MultiVectorPtr_Type output;

        InputToOutputMappingClassPtr_Type activationFunction;  // Let this be as general as possible
        */
    };
}
#endif
