#!/bin/bash

#SBATCH -N 1          # number of requested nodes
#SBATCH -t 30:00:00   # Walltime
#SBATCH -n 64         # ppn in PBS : for 48 tasks 
#SBATCH -p gpu2
#SBATCH --gres=gpu:2
#SBATCH -A ????? 
#SBATCH -o /home/????/QB4_code/out_NB_NT.out
#SBATCH --job-name NB_NT_rand10s # name to display in queue
export HOME_DIR=/home/????
export WORK_DIR=/work/????

## Make sure the WORK_DIR exists:
mkdir -p $WORK_DIR
cp -r $HOME_DIR $WORK_DIR
cd $WORK_DIR
cd QB4

JOBNAME=$SLURM_JOB_NAME            # re-use the job-name specified above
# Run 1 job per task
N_JOB=$SLURM_NTASKS                # create as many jobs as tasks


## run the code
# Load OpenMPI
module load openmpi
##iteration=$1


orterun -np 2 -npernode 2\
    singularity exec --nv -B /work,/project --pwd $PWD /work/????/horovod_tf2_24.simg \
                 python function_B_NT_gpu.py 


## Mark the time it finishes
date
## exit the job
exit 0
## And we're out'a here!
 







