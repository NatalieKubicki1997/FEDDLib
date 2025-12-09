#! /bin/bash -l

#SBATCH -N 64
#SBATCH --ntasks=4608
#SBATCH -t 00:15:00
#SBATCH --output=4608_MonoIPOU__Paper_MOS_2025_table_6_5.out
#SBATCH --error=4608_MonoIPOU__Paper_MOS_2025_table_6_5.err
#SBATCH --switches=1
# SBATCH --cpu-freq=2400000-2400000:performance

export OMP_NUM_THREADS=1

unset SLURM_EXPORT_ENV

source ~/Installation/load_updated_intel.sh


srun ./problems_steadyGeneralizedNewtonianAssFE_BFS_Scalability.exe --problemfile=Monolithic_IPOU/parametersProblemP1.xml --precfile=Monolithic_IPOU/parametersPrec_G_R_CB.xml
srun ./problems_steadyGeneralizedNewtonianAssFE_BFS_Scalability.exe --problemfile=Monolithic_IPOU/parametersProblemP1.xml --precfile=Monolithic_IPOU/parametersPrec_GS_R_CB.xml
srun ./problems_steadyGeneralizedNewtonianAssFE_BFS_Scalability.exe --problemfile=Monolithic_IPOU/parametersProblemP1.xml --precfile=Monolithic_IPOU/parametersPrec_R_R_CB.xml



srun ./problems_steadyGeneralizedNewtonianAssFE_BFS_Scalability.exe --problemfile=Monolithic_IPOU/parametersProblemP2.xml --precfile=Monolithic_IPOU/parametersPrec_R_R_CB.xml
srun ./problems_steadyGeneralizedNewtonianAssFE_BFS_Scalability.exe --problemfile=Monolithic_IPOU/parametersProblemP2.xml --precfile=Monolithic_IPOU/parametersPrec_GS_R_CB.xml
srun ./problems_steadyGeneralizedNewtonianAssFE_BFS_Scalability.exe --problemfile=Monolithic_IPOU/parametersProblemP2.xml --precfile=Monolithic_IPOU/parametersPrec_G_R_CB.xml 



##### This script should reproduce the results from Table 6.5 for the stationary BFS test case of the paper "MONOLITHIC AND BLOCK OVERLAPPING SCHWARZ PRECONDITIONERS FOR THE INCOMPRESSIBLE NAVIER–STOKES EQUATIONS" by Heinlein, Klawonn, Knepper and Saßmannshause (2025) #####
## Using also the corresponding parameter files from the Monolithic_IPOU folder ###
## Importantly, the settings in the bash script can have an effect on the performance, so for example if you see that linear iteration counts are ~same but the GMRES times differ than check the setting maybe it helps to remove exports etc. ##

# export OMP_NUM_THREADS=1



#SBATCH --cpus-per-task=1
#export OMP_NUM_THREADS=1
#export SRUN_CPUS_PER_TASK=1
