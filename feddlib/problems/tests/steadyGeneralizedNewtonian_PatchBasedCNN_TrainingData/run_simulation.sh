#!/bin/sh

#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM --mpi proc=4
#PJM -L elapse=0:05:00

#PJM -g pz0530

#PJM -j


source $HOME/Installation/load_env.sh 


mpiexec ./problems_steadyGeneralizedNewtonian_PatchBasedCNN_TrainingData.exe --problemfile=parametersProblem.xml --precfile=parametersPrec.xml