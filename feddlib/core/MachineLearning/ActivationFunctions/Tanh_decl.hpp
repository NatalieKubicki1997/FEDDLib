#ifndef TANH_DECL_hpp
#define TANH_DECL_hpp

#include "feddlib/core/General/DifferentiableFuncClass.hpp"
//#include "feddlib/core/AceFemAssembly/Helper.hpp"
#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/AceFemAssembly/specific/AssembleFENavierStokesNonNewtonian.hpp"


namespace FEDD {

   /* template <class SC = default_sc,
              class LO = default_lo,
              class GO = default_go,
              class NO = default_no>
    class AssembleFENavierStokesNonNewtonian;
*/
    /*!
    \class Tanh
    \brief This class is derived from the abstract class DifferentiableFuncClass and should provide functionality to evaluate the viscosity function specified by Carreau-Yasuda model (see [1])
    \tparam SC The scalar type. So far, this is always double, but having it as a template parameter would allow flexibily, e.g., for using complex instead
    \tparam LO The local ordinal type. The is the index type for local indices
    \tparam GO The global ordinal type. The is the index type for global indices
    @todo This should actually be removed since the class should operate only on element level)
    \tparam NO The Kokkos Node type. This would allow for performance portibility when using Kokkos. Currently, this is not used.
    
    In general, there exist a variety of activation function - the basic idea is that they add a nonlinearity into the model in order to learn complex mappings.
    This is the concrete implementation of the tanh activation function.
    The activation is e.g. applied to the result of a weighted sum of inputs so we want to be able to apply a function to each entry of an input array
    */

    

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class Tanh : public DifferentiableFuncClass<SC,LO,GO,NO> {
  public:

        typedef MultiVector<SC,LO,GO,NO> MultiVector_Type;
        typedef Teuchos::RCP<MultiVector_Type> MultiVectorPtr_Type;
        typedef Teuchos::RCP<const MultiVector_Type> MultiVectorConstPtr_Type;


	typedef DifferentiableFuncClass<SC,LO,GO,NO> DifferentiableFuncClass_Type;


        // Inherited Function from base abstract class 
        /*!
         \brief Implement a mapping description for evaluating output in dependence of given input and specified parameters
         @param[in] params Parameterlist as read from the xml file (maybe redundant)
         @param[in] x Independent variable
         @param[in,out] res Dependent variable
        */
        virtual void evaluateMapping(ParameterListPtr_Type params, MultiVectorConstPtr_Type input, MultiVectorPtr_Type &output) override { TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not yet implemented - HAS TO BE IMPLEMENTED."); };
       
        /*!
         \brief This function should set the specififc needed parameters for each model to the defined values.
        @param[in] ParameterList as read from the xml file (maybe redundant)
        */
        virtual void setParams(ParameterListPtr_Type params) override;

   // Define functions abstract class DifferntiableFuncClass

       // Define functions from base abstract class InputOutputMapping

            /*!
         \brief Computes value of derivative of defined function in evaluateMapping
         @param[in] params Parameterlist as read from the xml file (maybe redundant)
         @param[in] x Independent variable
         @param[in,out] res Dependent variable
        */
       virtual void evaluateDerivative(ParameterListPtr_Type params, MultiVectorConstPtr_Type x, MultiVectorPtr_Type &res) override { TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not yet implemented - HAS TO BE IMPLEMENTED."); };


    /*! This two functions are not so important because we basically always want to operate on vectors but if there will occur a side we can implenent this
         \brief Update the viscosity according to a chosen shear thinning generalized newtonian constitutive equation. Viscosity depends on spatial coordinates due to its dependency on velocity gradients
         @param[in] params as read from the xml file (maybe redundant)
         @param[in] shearRate scalar value of computed shear rate
         @param[in,out] viscosity value of viscosity
        */
         virtual void evaluateMapping(ParameterListPtr_Type params, double shearRate, double &viscosity) override { TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Use instead the intern function std::tanh."); };
    /*!
         \brief For Newton method and NOX we need additional term in Jacobian considering Gateaux-derivative of our functional formulation.
                One part is depending on the derivative of the viscosity function and another term, i.e. $... \frac{\partial \eta}{\partial \Dot{\gamma}} *  \frac{\partial  \Dot{\gamma}}{\partial \Pi}.
                For each constitutive model the function looks different and will be defined inside this function
         @param[in] params as read from the xml file (maybe redundant)
         @param[in] shearRate scalar value of computed shear rate
         @param[in,out] res scalar value of \frac{\partial \eta}{\partial \Dot{\gamma}} *  \frac{\partial  \Dot{\gamma}}{\partial \Pi}
        */
      virtual void evaluateDerivative(ParameterListPtr_Type params, double shearRate, double &res) override { TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Use instead the intern function std::tanh."); };


        
        // New Added Functions


      /*!
        \brief Print parameter values used in model at runtime
        */
        virtual void echoInformationMapping() override;


   // protected: Why dould we protect it?

	/*!

	\brief Constructor for Tanh
	@param[in] parameters Parameterlist for current problem	*/
	Tanh(ParameterListPtr_Type parameters); 


   private:





 };

}
#endif

