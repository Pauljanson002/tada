module purge
module load python/3.10
module load cuda/11.8

export ENV_DIR=$PWD/env

export CUDA_HOME=/encs/pkg/cuda-11.8/root/
# python -m venv $ENV_DIR

source $ENV_DIR/bin/activate