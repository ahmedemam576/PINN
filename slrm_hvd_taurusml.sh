#!/bin/bash -l

#SBATCH -p ml 
#SBATCH -t 01:00:00
#SBATCH --nodes=3
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=28
#SBATCH -o hostname_%j.out
#SBATCH --gres=gpu:6
#SBATCH --mem=0
#SBATCH --job-name=HVD_HEQN

module load modenv/ml
module load OpenMPI/3.1.4-gcccuda-2018b
module load TensorFlow/1.10.0-PythonAnaconda-3.6
module load cuDNN/7.1.4.18-fosscuda-2018b

source activate torch

echo "JOBID: $SLURM_JOB_ID"
echo "NNODES: $SLURM_NNODES"
echo "NTASKS: $SLURM_NTASKS"
echo "MPIRANK: $SLURM_PROVID"

cd /home/hoffmnic/hzdr

#srun python3 HelmholtzEquation_torch.py -ia 100000 -il 0 -f 100 -d 4 -lra 0.0001 -pd h5/ -ssim
srun --output="logs/train-$SLURM_JOB_ID.log" python3 -u HelmholtzEquation_torch.py -ia 100000 -il 0 -f 100 -d 4 -lra 0.0001 -pd h5/ -ssim 
#~/.local/bin/horovodrun -np 12 python3 HelmholtzEquation_torch.py -ia 100000 -il 0 -f 100 -d 8 -lra 0.0001 -pd h5/ -ssim
