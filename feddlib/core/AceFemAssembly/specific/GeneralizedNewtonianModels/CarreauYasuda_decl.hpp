#ifndef CARREAUYASUDA_DECL_hpp
#define CARREAUYASUDA_DECL_hpp

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
    \class CarreauYasuda
    \brief This class is derived from the abstract class DifferentiableFuncClass and should provide functionality to evaluate the viscosity function specified by Carreau-Yasuda model (see [1])
    \tparam SC The scalar type. So far, this is always double, but having it as a template parameter would allow flexibily, e.g., for using complex instead
    \tparam LO The local ordinal type. The is the index type for local indices
    \tparam GO The global ordinal type. The is the index type for global indices
    @todo This should actually be removed since the class should operate only on element level)
    \tparam NO The Kokkos Node type. This would allow for performance portibility when using Kokkos. Currently, this is not used.
    
    In general, there exist various constitutive equations for describing the material behaviour of blood. We will focus on generalized
    Newtonian constitutive equations capturing the shear thinning behaviour of blood. The chosen shear thinning model, e.g. Power-Law or
    Carreau-Yasuda etc., provides a function for updating the viscosity as the viscosity is no longer constant but instead
    depends on the shear rate and other parameters.

    In our original code we need therefore, depending on the chosen model, a update function for the viscosity (1). If we apply Newton method or NOX
    we also have to provide a derivative of the viscosity function which also depends on the chosen model (2). Lastly, we need functions to set
    the required parameters of the chosen model and additionally a function which prints the parameter values (3+4) 

    The material parameters can provided through a Teuchos::ParameterList object which will contain 
    all the parameters specified in the input file `ABC.xml`.
    The structure of the input file and, hence, of the resulting parameter list can be chosen freely. The FEDDLib will take care of reading the parameters 
    from the file and making them available.

    In this class we define the Carreau-Yasuda model.

    I decided to not generate another subclass which includes all viscosity models and then this class would be inherited from that because I decided it to not be that useful
    because the only thing they would have in common until now is getViscosity
    */

    

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class CarreauYasuda : public DifferentiableFuncClass<SC,LO,GO,NO> {
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
        virtual void evaluateMapping(ParameterListPtr_Type params, MultiVectorConstPtr_Type input, MultiVectorPtr_Type &output) override { TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not yet implemented."); };
       
        /*!
         \brief Each constitutive model includes different material parameters which will be specified in parametersProblem.xml
                This function should set the specififc needed parameters for each model to the defined values.
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
       virtual void evaluateDerivative(ParameterListPtr_Type params, MultiVectorConstPtr_Type x, MultiVectorPtr_Type &res) override { TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not yet implemented."); };


    /*!
         \brief Update the viscosity according to a chosen shear thinning generalized newtonian constitutive equation. Viscosity depends on spatial coordinates due to its dependency on velocity gradients
         @param[in] params as read from the xml file (maybe redundant)
         @param[in] shearRate scalar value of computed shear rate
         @param[in,out] viscosity value of viscosity
        */
         virtual void evaluateMapping(ParameterListPtr_Type params, double shearRate, double &viscosity) override;

  /*!
         \brief For Newton method and NOX we need additional term in Jacobian considering Gateaux-derivative of our functional formulation.
                One part is depending on the derivative of the viscosity function and another term, i.e. $... \frac{\partial \eta}{\partial \Dot{\gamma}} *  \frac{\partial  \Dot{\gamma}}{\partial \Pi}.
                For each constitutive model the function looks different and will be defined inside this function
         @param[in] params as read from the xml file (maybe redundant)
         @param[in] shearRate scalar value of computed shear rate
         @param[in,out] res scalar value of \frac{\partial \eta}{\partial \Dot{\gamma}} *  \frac{\partial  \Dot{\gamma}}{\partial \Pi}
        */
      virtual void evaluateDerivative(ParameterListPtr_Type params, double shearRate, double &res) override;



        
        // New Added Functions
        /*!
         \brief Get the current viscosity value
         \return the scalar value of viscosity
         */
        double getViscosity() {return viscosity_;};
        
        //string getShearThinningModel() {return shearThinningModel_;};
        

      /*!
        \brief Print parameter values used in model at runtime
        */
        virtual void echoInformationMapping() override;


   // protected: Why dould we protect it?

	/*!

	\brief Constructor for CarreauYasuda
	@param[in] parameters Parameterlist for current problem	*/
	CarreauYasuda(ParameterListPtr_Type parameters); 


   private:

	double viscosity_;
    std::string shearThinningModel_;// for printing out which model is actually used
    //! 
    double characteristicTime; // corresponds to \lambda in the formulas in the literature
    double fluid_index_n;      // corresponds to n in the formulas being the power-law index
    double nu_0;               // is the zero shear-rate viscosity
    double nu_infty;           // is the infnite shear-rate viscosity
    double inflectionPoint;    // corresponds to a in the formulas in the literature
    double shear_rate_limitZero;

    

  //  friend class AssembleFENavierStokesNonNewtonian<SC,LO,GO,NO>; why dit it not work?


 };

}
#endif

