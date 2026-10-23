#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=128g
#SBATCH -p qTRDGPUH
#SBATCH -t 24:00:00
#SBATCH --exclude=arctrddgxa001
#SBATCH --gres=gpu:A100:1
#SBATCH -J satellite
#SBATCH -D /data/users2/ppopov1/circRNA/trainer   # submit from anywhere #TODO: I think if we put . here, then it will run it from the folder where it got submitted. we can put it here, and leave a comment to run it from the trainer root
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH -A psy53c17

set -e
sleep 5s
echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID" >&2

export TMPDIR=/tmp
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MKL_THREADING_LAYER=GNU   # conda MKL vs torch's OpenMP

source /data/users2/ppopov1/miniconda/bin/activate circrna
echo "Using python from: $(which python)"

# Multi-GPU: raise --gres=gpu:A100:N above; SLURM_GPUS_ON_NODE drives the
# world size, nothing here needs to change.
# One fold (run) per array task: add --array=0-N and experiment.target_folds=[$SLURM_ARRAY_TASK_ID].
python train.py \
    experiment.name=spectral_spatial_run1
    # TODO: list the default configs here (not all, only important, so that I can see what I can change). ALso, make a script like that for baseline

sleep 5s
echo "Job $SLURM_JOB_ID completed"
