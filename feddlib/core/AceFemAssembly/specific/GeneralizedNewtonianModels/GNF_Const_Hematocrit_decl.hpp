#ifndef GNF_CONST_HEMATOCRIT_DECL_hpp
#define GNF_CONST_HEMATOCRIT_DECL_hpp

#include "feddlib/core/General/DifferentiableFuncClass.hpp"
#include "feddlib/core/FEDDCore.hpp"


namespace FEDD
{

  /*!
        \class GNF_Const_Hematocrit_decl
        \brief This class is derived from the abstract class DifferentiableFuncClass and should provide functionality to evaluate the viscosity function specified by
               a certain model which not includes a dependence on the shear rate but also on the hematocrit
               For this model we assume that the hematocrit is spatially varying BUT constant - So basically we assume that we already know the correct hematocrit profile
               and we now want to compute the velocity and pressure field for this given hematocrit profile
        \tparam SC The scalar type. So far, this is always double, but having it as a template parameter would allow flexibily, e.g., for using complex instead
        \tparam LO The local ordinal type. The is the index type for local indices
        \tparam GO The global ordinal type. The is the index type for global indices
        \tparam NO The Kokkos Node type. This would allow for performance portibility when using Kokkos. Currently, this is not used.

    In general, there are several equations used to describe the material behavior of blood.
    Here (in this folder), we implement generalized Newtonian constitutive equations that capture the shear-thinning behavior of blood.
    The chosen shear-thinning model, such as the Power-Law or Carreau-Yasuda model, provides a function to update viscosity,
    which is no longer constant but depends on the shear rate and here also on the hematocrit (and other fixed constant parameters).

    In our FEM code, we need an update function for viscosity, depending on the chosen model.
    If we apply Newton's method or NOX, we also require the directional derivative of the viscosity function,
    which also depends on the chosen model. As we assume that we set the correct hematocrit solution it is not dependent on the velocity field.
    Such that we do not get an additional term in the Jacobian.

    Additionally, we need functions to set the required parameters
    of the chosen model and a function to print the parameter values.

    The material parameters can be provided through a Teuchos::ParameterList object,
    which contains all the parameters specified in the input file 'parametersProblem.xml'.
    The structure of the input file and, consequently,
    the resulting parameter list can be chosen freely. The FEDDLib will handle reading the parameters from the
    file and making them available.

    In order to have the set the correct hematocrit value one has to read in the hematocrit field from an external field and set it in each assembleFEElement
    (see FE_def.hpp, AssembleFEGeneralizedNewtonian_def.hpp)

    @IMPORTANT
    As this class is derived from the abstract class DifferentiableFuncClass, it has to implement the following functions:
    - setParams(ParameterListPtr_Type params)
    - evaluateMapping(ParameterListPtr_Type params, double shearRate, double &viscosity) 
      * This function has as inputs only one input which is the shear rate and the output is the viscosity, so this function is called if no external hematocrit field were provided
        and one assumes that the hematocrit is constant everywhere
    - evaluateDerivative(ParameterListPtr_Type params, double shearRate, double &res)
    - echoInformationMapping()

    We add the additiional function:
    - evaluateMapping(ParameterListPtr_Type params, double shearRate, double &viscosity, double localHematocrit) 
      * This function has as inputs the shear rate and the local hematocrit value and the output is the viscosity, so this function is called if an external hematocrit field were provided
        and one assumes that the hematocrit is spatially varying but constant in each element

   The spatially varying hematocrit will give us hopefully a better viscosity estimation and therefore in the end a better predictions of the velocity and pressure field

   */

  template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
  class GNF_Const_Hematocrit : public DifferentiableFuncClass<SC, LO, GO, NO>
  {
  public:
    typedef MultiVector<SC, LO, GO, NO> MultiVector_Type;
    typedef Teuchos::RCP<MultiVector_Type> MultiVectorPtr_Type;
    typedef Teuchos::RCP<const MultiVector_Type> MultiVectorConstPtr_Type;

    typedef DifferentiableFuncClass<SC, LO, GO, NO> DifferentiableFuncClass_Type;

    // Inherited Function from base abstract class
    /*!
     \brief Each constitutive model includes different material parameters which will be specified in parametersProblem.xml
            This function should set the specififc needed parameters for each model to the defined values.
    @param[in] ParameterList as read from the xml file (maybe redundant)
    */
    void setParams(ParameterListPtr_Type params) override;

    /*!
     \brief Update the viscosity according to a chosen shear thinning generalized newtonian constitutive equation. Viscosity depends on spatial coordinates due to its dependency on velocity gradients

     @param[in] params as read from the xml file (maybe redundant)
     @param[in] shearRate scalar value of computed shear rate
     @param[in,out] viscosity value of viscosity
    */
    void evaluateMapping(ParameterListPtr_Type params, double shearRate, double &viscosity) override;


    /*!
     \brief For Newton method and NOX we need additional term in Jacobian considering directional derivative of our functional formulation.
             IMPORTANT: Here we implement the contribution of this: d(eta)/ d(gamma_Dot) * d(gamma_Dot)*d(Pi_||).
            For each constitutive model the function looks different and will be defined inside this function
     @param[in] params as read from the xml file (maybe redundant)
     @param[in] shearRate scalar value of computed shear rate
     @param[in,out] res scalar value of   d(eta)/ d(gamma_Dot) * d(gamma_Dot)*d(Pi_||)
    */
    void evaluateDerivative(ParameterListPtr_Type params, double shearRate, double &res) override;
  

  
    /*!
    \brief Print parameter values used in model at runtime
    */
    void echoInformationMapping() override;
  
    // New Added Functions
    /*!
     \brief Get the current viscosity value
     \return the scalar value of viscosity
     */
    double getViscosity() { return viscosity_; };

    /*!
     \brief Set the local hematocrit value
     @param[in] localHematocrit scalar value of hematocrit
     */
    void setLocalHematocrit(double localHematocrit) { this->localHematocrit_ = localHematocrit; };



    /*!
    \brief Constructor for GNF_Const_Hematocrit
    @param[in] parameters Parameterlist for current problem	*/
    GNF_Const_Hematocrit(ParameterListPtr_Type parameters);

  private:
    double viscosity_;
    double localHematocrit_;
    std::string shearThinningModel_; // for printing out which model is actually used
    //!
    double characteristicTime;   // corresponds to \lambda in the formulas in the literature
    double fluid_index_n;        // corresponds to n in the formulas being the exponent
    double nu_0;                 // is the zero shear-rate viscosity
    double nu_infty;             // is the infinite shear-rate viscosity
    double inflectionPoint;      // corresponds to a in the formulas in the literature
    double shear_rate_limitZero; // In the formulas the shear rate is in the denominator so we have to ensure that it is

  };

}
#endif
