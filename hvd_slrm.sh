#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6 #workaround for slurm/openmpi issues on hemera. Enforce that tasks are evenly distributed among nodes. 
#SBATCH --gres=gpu:4
#SBATCH -o hostname_%j.out


module load cuda/9.2
module load python/3.6.5
module load gcc/5.5.0
module load openmpi/3.1.2

echo "JOBID: $SLURM_JOB_ID"
echo "NNODES: $SLURM_NNODES"
echo "NTASKS: $$SLURM_NTASKS"
echo "MPIRANK: $$SLURM_PROVID"

cd /home/hoffma83/Code/AIPP/3D_LaserPropagation

mpirun -output-filename logs/train-$SLURM_JOB_ID.log -tag-output \
	-bind-to none -map-by slot \
        -mca pml ob1 -mca btl ^openib \
        python HelmholtzEquation_torch.py -is 1000 -ia 0 -il 0
