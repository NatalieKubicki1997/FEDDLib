@Natalie Kubicki 04.09.2024 nkubicki@uni-koeln.de

main_generateTrainingData
Here we want to try out what if we select as material law the artifical eta ( gamma, phi) function 
from the file dummyMaterialLaw.m and use the GNF_Constant_Hematocrit model and set an artifical density function in each element
one converges in the center to a velocity profile. At the beginning and at the end we do not have correct v

1. Specify flow rate Q
2. Generate suitable density profile along cross-section
3. Extrapolate density onto Mesh
4. Set in each element the density and specify GNF_Constant_Hematocrit as model
   and set as inlet velocity constant profile with specified Q
5. Let simulation run