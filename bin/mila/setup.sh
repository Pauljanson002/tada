source /cvmfs/ai.mila.quebec/apps/x86_64/debian/miniconda/3/bin/deactivate 
module purge
module load python/3.10
module load cuda/11.8

ENV_DIR=env

# python -m venv $ENV_DIR

source $ENV_DIR/bin/activate