source /cvmfs/ai.mila.quebec/apps/x86_64/debian/miniconda/3/bin/deactivate 
module purge
module load python/3.10
module load cuda/11.8
WORKSPACE=$SLURM_TMPDIR/workspace
mkdir -p $WORKSPACE

rsync -avhz --progress ./ $WORKSPACE
cd $WORKSPACE

ENV_DIR=$SLURM_TMPDIR/env

python -m venv $ENV_DIR

source $ENV_DIR/bin/activate