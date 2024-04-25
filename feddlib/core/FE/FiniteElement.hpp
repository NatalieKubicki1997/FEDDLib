#ifndef FiniteElement_hpp
#define FiniteElement_hpp

#include "feddlib/core/FEDDCore.hpp"
#include "Elements.hpp"
/*!
 Declaration of FiniteElement
 
 @brief  FiniteElement
 @author Christian Hochmuth
 @version 1.0
 @copyright CH
 */

namespace FEDD {
class Elements;
class FiniteElement {
    
  public:
    typedef default_lo LO;
    typedef default_go GO;
    typedef default_no NO;
    
    typedef std::vector<int> vec_int_Type;
    typedef std::vector<long long> vec_long_Type;
    typedef std::vector<std::vector<int> > vec2D_int_Type;
    
    typedef Elements Elements_Type;
    typedef Teuchos::RCP<Elements_Type> ElementsPtr_Type;
    
    typedef Teuchos::RCP<const Map<LO,GO,NO> > MapConstPtr_Type;
    
    FiniteElement();
    
    FiniteElement( vec_LO_Type& localNodeList );
    
    FiniteElement( vec_LO_Type& localNodeList, LO elementFlag );
    
    bool operator==(const FiniteElement &other);
    
    FiniteElement& operator=(const FiniteElement& in);
    
    void setElement( vec_int_Type& localNodeList );
    
    int getFlag() const {return flag_;};

	void setFlag( int elementFlag );
    
    const vec_LO_Type& getVectorNodeList() const { return localNodeIDs_; };
    
    vec_LO_Type getVectorNodeListNonConst(){ return localNodeIDs_; };

    int size() { return localNodeIDs_.size(); };    
    
    int getNode( int i ) const;
    
    int numSubElements();
    
    bool subElementsInitialized();
    
    void initializeSubElements( std::string feType, int dim ) ;
    
    void addSubElement( FiniteElement& fe );
    
    ElementsPtr_Type getSubElements(){ return subElements_; };
    
    void setSubElements( ElementsPtr_Type& subElements );
    
    void globalToLocalIDs( MapConstPtr_Type map );
    
    /*! We go through all possible permutations and check wether ids is equal to one of these permutations*/
    void addSubElementIfPart( FiniteElement& feSub, const vec2D_int_Type& permutation, std::string& feType, int dim );
    
    void findEdgeFlagInSubElements( const vec_LO_Type& edgeIDs, vec_int_Type& flags, bool isSubElement, const vec2D_int_Type& permutation, bool& foundLineSegment );
    
    bool findEdgeInElement( const vec_LO_Type& edgeIDs, vec_int_Type& flags, const vec2D_int_Type& permutation );
    
    void print(MapConstPtr_Type mapRepeated=Teuchos::null);

    void setFiniteElementRefinementType( std::string reType ){ refinementType_ = reType; }; // assigning a certain refinement typ, i.e. red, blue, green

    std::string getFiniteElementRefinementType( ){ return refinementType_; }; 

	void tagForRefinement(){taggedForRefinement_ = true; }; // assigning simply the information whether element is tagged for refinement
	
	void untagForRefinement(){taggedForRefinement_ = false; }; // untagging previously tagged element

	bool isTaggedForRefinement(){ return taggedForRefinement_; }; 

	bool isInterfaceElement(){return isInterfaceElement_; };

	void setInterfaceElement( bool interface){ isInterfaceElement_ = interface; };

    void setPredecessorElement(GO id) {predecessorElement_ = id; };

	GO getPredecessorElement(){return predecessorElement_; };

	void setRefinementEdge(LO id){refinementEdge_=id;};

	LO getRefinementEdge(){return refinementEdge_;};

	void setMarkedEdges(LO id){markedEdges_.push_back(id);};

	vec_LO_Type getMarkedEdges(){return markedEdges_;};

	void markEdge(){markedEdge_ = true;};

	bool isMarkedEdge(){return markedEdge_;};

    // Computed in Domain_def.hpp -> Save in each (Sub) Finite Element the outward normal if the Finite Element is laying on the outer boundary
    void setSurfaceNormal(vec_dbl_Type  computedNormal){ this->surfaceNormal_ =  computedNormal; };

    // Return the vector saved for the surface Normal of the edge/ face
    vec_dbl_Type getSurfaceNormal()
        { if (this->surfaceNormal_.size() == 0){ TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"SurfaceNormal is not initialized")};
        return this->surfaceNormal_;};

    // Getter and setter functions for a boundary surface element 
    void setNeumannBCElement(bool flag){ this->neumannBCElement_ = flag ; };   
    bool getNeumannBCElement(){ return this->neumannBCElement_; };   
    void setElementScaling(double elscaling){ this->elScaling_  = elscaling ; };   
    double getElementScaling(){ return this->elScaling_  ; };   
    void setQuadratureWeightsReference(vec_dbl_Type weights){this->neumannBCQuadratureWeightsReference_ = weights;};
    void setQuadraturePointsGlobal(vec2D_dbl_Type qpoints){this->neumannBCQuadraturePointsGlobal_ = qpoints; };
    vec_dbl_Type getQuadratureWeightsReference()
        { if (this->neumannBCQuadratureWeightsReference_.size() == 0){ TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"neumannBC_QuadratureWeightsRef is not initialized")};
        return this->neumannBCQuadratureWeightsReference_;};
    vec2D_dbl_Type getQuadraturePointsGlobal()
        { if (this->neumannBCQuadraturePointsGlobal_.size() == 0){ TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"neumannBC__QuadraturePointsPhy is not initialized")};
        return this->neumannBCQuadraturePointsGlobal_;};
    
private:
    
    vec_LO_Type localNodeIDs_; /*! Node IDs that define this element. */
    int flag_;
    ElementsPtr_Type subElements_;
    int numSubElements_;
    bool taggedForRefinement_ = false;
    bool markedEdge_ = false;
    std::string refinementType_;  // Tag of finite Element
	bool isInterfaceElement_ = false;
	GO predecessorElement_ = -1;
	LO refinementEdge_=-1;
	vec_LO_Type markedEdges_;

    // Defined variables for boundary elements
    vec_dbl_Type surfaceNormal_;                           // As we do not have dim here as variable we cannot initialize the vector 
    bool neumannBCElement_ = false;                        // Check if the element is an surface element where we want to add an additional contribution - only for boundary elements where we want to add surface integral
    vec_dbl_Type neumannBCQuadratureWeightsReference_;     // These are the Quadrature weights in terms of specified global quadrature points defined in reference coordinate system so here 2D: line [0 1] 3D: Triangle with (0,0)-(0,1)-(1,0) corners 
    vec2D_dbl_Type neumannBCQuadraturePointsGlobal_;        // 2D: Quadrature points on line, 3D: Quadrature points on face, have to be mapped on reference element with mapping in specificAssemblyClasses
    double elScaling_ = 0.0;                                // 2D: change of length , 3D: change of area - Surface Element scaling
    
    
public:    

};
}
#endif
