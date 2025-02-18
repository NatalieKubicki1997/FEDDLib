#ifndef Helper_hpp
#define Helper_hpp

//#include "AssembleFE_decl.hpp"
#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/LinearAlgebra/Matrix.hpp"
#include "feddlib/core/General/SmallMatrix.hpp"

namespace FEDD {

/*! Helper class that contains rudimental FE components
 * Currently it contains Basisfunctions, Quadrature rules, Transformation, and other stuff.
 *
 *
 */
class Helper {
  
public:
    typedef double SC;

    enum VarType {Std=0,Grad=1};


    /// @brief Compute surface normal of corresponding surface
    /// @param dim Dimension
    /// @param pointsRep List of all repeated nodes
    /// @param nodeList Ids of local surface points
    /// @param v_E output normale vector
    /// @param norm_v_E Normal vector length
    static void computeSurfaceNormal(int dim,
                                      vec2D_dbl_ptr_Type pointsRep,
                                      vec_int_Type nodeList,
                                      vec_dbl_Type &v_E,
                                      double &norm_v_E);

    /// @brief Compute the Quadrature Points and Weights defined in Physical Space on a surface (line (2D)/ face (3D))
    /// @param dim Dimension
    //  @param FEType FE Discretization
    //  @param QuadW Vector of quadrature weights
    //  @param QuadP Vector of quadrature coordinate vectors
    /// @param surfaceIDs Ids of local surface points
    /// @param pointsRep List of all repeated nodes
    static void getQuadraturePointsOnSurfaceInGlobalSpace(int dim, 	
    										std::string FEType, 
    										vec_dbl_Type &QuadW,
                                            vec2D_dbl_Type &QuadP,  
    										vec_LO_Type surfaceIDs, 
    										vec2D_dbl_ptr_Type points);

        
    /// @brief Build transformation of element to reference element depending on FEType
    /// @param element Finite element
    /// @param pointsRep List of repeated points
    /// @param B Resulting transformation matrix
    /// @param FEType FE Discretization 
    static void buildTransformation(const vec_int_Type& element,
                             vec2D_dbl_ptr_Type pointsRep,
                             SmallMatrix<SC>& B,
                             std::string FEType="P");
    
    /// @brief Build transformation of element to reference element depending on FEType
     /// @param element Finite element
    /// @param pointsRep List of repeated points
    /// @param B Resulting transformation matrix
    /// @param b Point to transform from
    /// @param FEType FE Discretization 
    static void buildTransformation(const vec_int_Type& element,
                             vec2D_dbl_ptr_Type pointsRep,
                             SmallMatrix<SC>& B,
                             vec_dbl_Type& b,
                             std::string FEType="P");

    
    /// @brief Transformation of a surface to the reference element
    /// @param element Finite element
    /// @param pointsRep List of repeated points
    /// @param B Resulting transformation matrix
    /// @param b Point to transform from
    /// @param FEType FE Discretization 
    static void buildTransformationSurface(const vec_int_Type& element,
                                    vec2D_dbl_ptr_Type pointsRep,
                                    SmallMatrix<SC>& B,
                                    vec_dbl_Type& b,
                                    std::string FEType="P");

	/// @brief Returning gradient of phi evaluated at the quadrature points
	/// @param Dimension Dimension
	/// @param intFE number corresponding to FE disc.
	/// @param i basisfunction i
	/// @param QuadPts quadpoints
	/// @param value vector including values
	static void gradPhi(	int Dimension,
                    int intFE,
                    int i,
                    vec_dbl_Type &QuadPts,
                    vec_dbl_ptr_Type &value);
    
    /*! Most of the quadrature formulas can be found in http://code-aster.org/doc/v11/en/man_r/r3/r3.01.01.pdf 01/2021  */
    static void getQuadratureValues(int Dimension,
                            int Degree,
                            vec2D_dbl_ptr_Type &QuadPts,
                            vec_dbl_ptr_Type &QuadW,
                            std::string FEType);
                            
    /// @brief Get quadrature values of surface
    /// @param dim Dimension
    /// @param FEType Finite element disc.
    /// @param QuadW return quadrature values
    /// @param surfaceIDs local suface node ids
    /// @param points points
    /// @return quadValues  probably
    static vec2D_dbl_Type getQuadratureValuesOnSurface(int dim, 	
    										std::string FEType, 
    										vec_dbl_Type &QuadW, 
    										vec_LO_Type surfaceIDs, 
    										vec2D_dbl_ptr_Type points);
    
    /// @brief Full matrix representation of grad phi per quadvalue
    /// @param DPhi grad Phi per quadpoint dim:(quadpoint,i,j)
    /// @param weightsDPhi Quadrature weights
    /// @param Dimension Dimension
    /// @param FEType Finite Element Type
    /// @param Degree Integration degree
    /// @return 
    static int getDPhi(	vec3D_dbl_ptr_Type &DPhi,
                	vec_dbl_ptr_Type &weightsDPhi,
                    int Dimension,
                    std::string FEType,
                    int Degree);

    // Natalie new function to get DPhi at specified quadrature points
    /// @param DPhi grad Phi per quadpoint dim:(quadpoint,i,j)
    /// @param weightsDPhi Quadrature weights
    //  @param QuadPts Specific quadrature points where we want to have DPhi evaluated
    /// @param Dimension Dimension
    /// @param FEType Finite Element Type
    /// @return 
    static int getDPhi(vec3D_dbl_ptr_Type &DPhi,
                     vec_dbl_Type weightsDPhi,
                     vec2D_dbl_Type QuadPts,
                     int dim,
		             std::string FEType);  

    //  @brief Natalie new function to get viscosity at center of mass
    /// @param DPhi grad Phi p
    /// @param Dimension Dimension
    /// @param FEType Finite Element Type
    /// @return 
    static int getDPhiAtCM(vec3D_dbl_ptr_Type &DPhi,
                     int dim,
		             std::string FEType);   


    /// @brief Applying the transformation matriX B to the gradient of phi, as is done in when transforming the gradient of phi to the reference element
    /// @param dPhiIn 
    /// @param dPhiOut 
    /// @param Binv 
    static void applyBTinv( vec3D_dbl_ptr_Type& dPhiIn,
                    vec3D_dbl_Type& dPhiOut,
                    const SmallMatrix<SC>& Binv);

	static UN determineDegree(UN dim, 
								std::string FEType1, 		
								std::string FEType2, 
								int type1,
								int type2, 
								UN extraDeg = 0);

    static UN determineDegree(UN dim,
                    std::string FEType,
                    UN degFunc);
                       
    static UN determineDegree(UN dim,
                       std::string FEType,
                       int type);               
                       

    /// @brief Get basisfunction phi per quadrature point
    /// @param Phi Basisfunction phi per quad point with (quadpoint,i)
    /// @param weightsPhi Quadrature weights
    /// @param dim dimension
    /// @param FEType Finite element discretization
    /// @param Degree Integration degree
    /// @param FETypeQuadPoints 
    /// @return 
    static int getPhi(vec2D_dbl_ptr_Type &Phi,
                            vec_dbl_ptr_Type &weightsPhi,
                            int dim,
                            std::string FEType,
                            int Degree,
               			    std::string FETypeQuadPoints="");


    /// @brief Get basisfunction phi at specific quadrature points
    /// @param Phi Basisfunction phi per quad point with (quadpoint,i)
    /// @param weightsPhi Quadrature weights
    //  @param QuadPts
    /// @param dim dimension
    /// @param FEType Finite element discretization
    /// @return 
    static int getPhi(vec2D_dbl_ptr_Type &Phi,
                            vec_dbl_Type weightsDPhi,
                            vec2D_dbl_Type QuadPts,
                            int dim,
                            std::string FEType);

    static int getFuncAtQuadNodes(vec_dbl_ptr_Type &funcVals, RhsFunc_Type &rhsFunc, int dim, std::string FEType,
                                  int Degree, std::string FETypeQuadPoints = "");

    /// @brief Get phi i
    /// @param dim
    /// @param intFE
    /// @param i
    /// @param p
    /// @param value
    static void phi(int dim, int intFE, int i, vec_dbl_Type &p, double *value);

    static int getPhiGlobal(vec2D_dbl_ptr_Type &Phi,
                            vec_dbl_ptr_Type &weightsPhi,
                            int dim,
                            std::string FEType,
                            int Degree)
    { TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "getPhiGlobal not implemented yet.");};


private:
	
	Helper(){};

};
}
#endif
