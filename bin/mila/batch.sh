#!/bin/bash
#SBATCH --job-name=tada
#SBATCH --output=/network/scratch/p/paul.janson/slurm/%j.out
#SBATCH --error=/network/scratch/p/paul.janson/slurm/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:59:00
#SBATCH --gres=gpu:80gb:1
##SBATCH --constraint=80gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pauljanson002@gmail.com
#SBATCH --mem=32G
#SBATCH --partition=unkillable

source bin/mila/setup.sh

bash bin/mila/install_env.sh

# sync back the outputs
python -m apps.run +experiment=tmp

WORKSPACE=$SLURM_TMPDIR/workspace

rsync -avhz --progress $WORKSPACE/outputs $HOME/scratch/TADA/outputs
