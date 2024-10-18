#ifndef GNF_CONST_HEMATOCRIT_DEF_hpp
#define GNF_CONST_HEMATOCRIT_DEF_hpp

#include "GNF_Const_Hematocrit_decl.hpp"

namespace FEDD
{

    template <class SC, class LO, class GO, class NO>
    GNF_Const_Hematocrit<SC, LO, GO, NO>::GNF_Const_Hematocrit(ParameterListPtr_Type params): 
    DifferentiableFuncClass<SC, LO, GO, NO>(params)
    {
        this->params_ = params;
        // Reading through parameterlist
        shearThinningModel_ = this->params_->sublist("Material").get("ShearThinningModel", "");
        characteristicTime = this->params_->sublist("Material").get("CharacteristicTime_Lambda", 0.); // corresponds to \lambda in the formulas in the literature
        fluid_index_n = this->params_->sublist("Material").get("FluidIndex_n", 1.);                   // corresponds to n in the formulas being the power-law index
        nu_0 = this->params_->sublist("Material").get("ZeroShearRateViscosity_eta0", 0.);             // is the zero shear-rate viscosity
        nu_infty = this->params_->sublist("Material").get("InftyShearRateViscosity_etaInfty", 0.);    // is the infnite shear-rate viscosity
        inflectionPoint = this->params_->sublist("Material").get("InflectionPoint_a", 1.);            // corresponds to a in the formulas in the literature
        shear_rate_limitZero = this->params_->sublist("Material").get("Numerical_ZeroValue_ShearRate", 1e-14);

        viscosity_ = 0.; // Initialize Viscsoity value to zero
        localHematocrit_= 0.; // Initialize Hematocrit value to zero
    }

  

    template <class SC, class LO, class GO, class NO>
    void GNF_Const_Hematocrit<SC, LO, GO, NO>::evaluateMapping(ParameterListPtr_Type params, double shearRate, double &viscosity)
    {
        /* So we first assume a simple functional form where the ZeroShearRateViscosity will depend on the hematocrit
           in that way we obtain an incrasing ZeroShearRateViscosity with increasing hematocrit 
           The functional form is a=1.18; b=-2.741 ; c=2.368; d=0.08088;
           eta_0 = @(phi) a*exp(b*phi) + c*exp(d*phi);
           local hematocrit will be a value between 0 and 1 so to obtain the percentage we have to multiply it by 100
        */
        double a, b, c, d = 0.0;
        a = 1.18;
        b = -2.741 ;
        c = 2.368;
        d = 0.08088;
        double eta_0 = a * exp(b * (this->localHematocrit_*100.0)) + c * exp(d * (this->localHematocrit_*100.0));
        this->nu_0 = eta_0;
        // So as first 
        viscosity = this->nu_infty + (eta_0 - this->nu_infty) * (pow(1.0 + pow(this->characteristicTime * shearRate, this->inflectionPoint), (this->fluid_index_n - 1.0) / this->inflectionPoint));
        this->viscosity_ = viscosity;
    }

 


    /*
     The function is composed of d(eta)/d(GammaDot) * d(GammaDot)/d(Tau) through chain rule
     while d(GammaDot) * d(GammaDot)/d(Tau)= - 2/GammaDot
     So a problematic case is if this->inflectionPoint-2.0 < 0 than the shear rate is in the denominator and because it can
     be zero we may get nan/inf values. Therefore we have to check these cases and catch them
    */
    template <class SC, class LO, class GO, class NO>
    void GNF_Const_Hematocrit<SC, LO, GO, NO>::evaluateDerivative(ParameterListPtr_Type params, double shearRate, double &res)
    {
                /* So we first assume a simple functional form where the ZeroShearRateViscosity will depend on the hematocrit
           in that way we obtain an incrasing ZeroShearRateViscosity with increasing hematocrit 
           The functional form is a=1.18; b=-2.741 ; c=2.368; d=0.08088;
           eta_0 = @(phi) a*exp(b*phi) + c*exp(d*phi);
           local hematocrit will be a value between 0 and 1 so to obtain the percentage we have to multiply it by 100
        */
        double a, b, c, d = 0.0;
        a = 1.18;
        b = -2.741 ;
        c = 2.368;
        d = 0.08088;
        double eta_0 = a * exp(b * (this->localHematocrit_*100.0)) + c * exp(d * (this->localHematocrit_*100.0));
        this->nu_0 = eta_0;
        // As the hematocrit is spatially varying but independent of the velocity we do not get a additional contribution for
        // the directional derivative of the viscosity with respect to the velocity therefore here we again have to only update the ZeroShearRateViscosity


        if (abs(this->inflectionPoint - 2.0) < std::numeric_limits<double>::epsilon()) // for a=2.0 we get gammaDot^{-0} which is 1
        {
            // So for a Carreau-like Fluid we should jump here because a=2.0
            res = (-2.0) * (eta_0 - this->nu_infty) * (this->fluid_index_n - 1.0) * pow(this->characteristicTime, this->inflectionPoint) * pow(1.0 + pow(this->characteristicTime * shearRate, this->inflectionPoint), ((this->fluid_index_n - 1.0 - this->inflectionPoint) / this->inflectionPoint));
        }
        else // in the other case we have to check that gammaDot is not zero because otherwise we get 1/0
        {
            if (abs(shearRate) <= shear_rate_limitZero) // How to choose epsilon?
            {
                shearRate = shear_rate_limitZero;
            }
            res = (-2.0) * (eta_0 - this->nu_infty) * (this->fluid_index_n - 1.0) * pow(this->characteristicTime, this->inflectionPoint) * pow(shearRate, this->inflectionPoint - 2.0) * pow(1.0 + pow(this->characteristicTime * shearRate, this->inflectionPoint), ((this->fluid_index_n - 1.0 - this->inflectionPoint) / this->inflectionPoint));
        }
    }

      
    template <class SC, class LO, class GO, class NO>
    void GNF_Const_Hematocrit<SC, LO, GO, NO>::setParams(ParameterListPtr_Type params)
    {
        this->params_ = params;
        // Reading through parameterlist
        shearThinningModel_ = this->params_->sublist("Material").get("ShearThinningModel", "");
        characteristicTime = this->params_->sublist("Material").get("CharacteristicTime_Lambda", 0.); // corresponds to \lambda in the formulas in the literature
        fluid_index_n = this->params_->sublist("Material").get("FluidIndex_n", 1.);                   // corresponds to n in the formulas being the power-law index
        nu_0 = this->params_->sublist("Material").get("ZeroShearRateViscosity_eta0", 0.);             // is the zero shear-rate viscosity
        nu_infty = this->params_->sublist("Material").get("InftyShearRateViscosity_etaInfty", 0.);    // is the infnite shear-rate viscosity
        inflectionPoint = this->params_->sublist("Material").get("InflectionPoint_a", 1.);
        shear_rate_limitZero = this->params_->sublist("Material").get("Numerical_ZeroValue_ShearRate", 1e-8);
    }

    template <class SC, class LO, class GO, class NO>
    void GNF_Const_Hematocrit<SC, LO, GO, NO>::echoInformationMapping()
    {
        std::cout << "************************************************************ " << std::endl;
        std::cout << "-- Chosen material model ..." << this->shearThinningModel_ << " --- " << std::endl;
        std::cout << "-- Specified material parameters:" << std::endl;
        std::cout << "-- eta_0:" << this->nu_0 << std::endl;
        std::cout << "-- eta_Infty:" << this->nu_infty << std::endl;
        std::cout << "-- Fluid index n:" << this->fluid_index_n << std::endl;
        std::cout << "-- Inflection point a:" << this->inflectionPoint << std::endl;
        std::cout << "-- Characteristic time lambda:" << this->characteristicTime << std::endl;
        std::cout << "-- Local Hematocrit:" << this->localHematocrit_ << std::endl;
        std::cout << "************************************************************ " << std::endl;
    }

}
#endif
