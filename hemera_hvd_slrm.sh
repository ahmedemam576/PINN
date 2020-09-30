#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 08:00:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6 #workaround for slurm/openmpi issues on hemera. Enforce that tasks are evenly distributed among nodes. 
#SBATCH --gres=gpu:4
#SBATCH -o hostname_%j.out
#SBATCH --job-name=HVD_HELM

module load cuda/9.2
module load python/3.6.5
module load gcc/5.5.0
module load openmpi/3.1.2

echo "JOBID: $SLURM_JOB_ID"
echo "NNODES: $SLURM_NNODES"
echo "NTASKS: $SLURM_NTASKS"
echo "MPIRANK: $SLURM_PROVID"
echo "SOFT ACTIVATION" 

cd /home/hoffma83/Code/AIPP/3D_LaserPropagation

#srun --output="logs/train-$SLURM_JOB_ID.log" python -u HelmholtzEquation_torch.py -ia 100000 -il 0 -f 100 -d 4 -lra 0.0001 -pd h5/ -ssim

mpirun -output-filename logs/train-$SLURM_JOB_ID.log -tag-output \
	-bind-to none -map-by slot \
       -mca pml ob1 -mca btl ^openib \
        python -u startTraining.py -ia 100000 -il 0 -f 100 -d 4 -lra 0.0001 -ssim
