#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --time=01:00:00
###########################

bash
source ~/.bashrc
source "/home/jwu558/anaconda3/etc/profile.d/conda.sh"
set -ex
conda activate rare_entity_3.7
echo $(date '+%Y_%m_%d_%H_%M') - $SLURM_JOB_NAME - $SLURM_JOBID - `hostname`
$@
