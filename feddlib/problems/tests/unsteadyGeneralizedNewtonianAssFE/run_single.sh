#! /bin/bash -l

#SBATCH -N 1
#SBATCH --ntasks=36
#SBATCH -t 00:10:00
#SBATCH --output=single_sim.out
#SBATCH --error=single_sim.err
#SBATCH --switches=1
#SBATCH --cpu-freq=2400000-2400000:performance

export OMP_NUM_THREADS=1


unset SLURM_EXPORT_ENV

source ~/Installation/load_updated_intel.sh

srun ./problems_unsteadyGeneralizedNewtonianAssFE.exe 

