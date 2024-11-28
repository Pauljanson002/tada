#!/bin/bash
#SBATCH --job-name=tada
#SBATCH --output=~/scratch/slurm/%j.out
#SBATCH --error=~/scratch/slurm/%j.err
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64Gb
#SBATCH --short-unkillable

source bin/mila/setup.sh

bash bin/mila/install_env.sh

# sync back the outputs
python -m apps.run +experiment=tmp

WORKSPACE=$SLURM_TMPDIR/workspace

rsync -avhz --progress $WORKSPACE/outputs $HOME/scratch/TADA/outputs