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
# -D . runs in the submit folder: submit from the trainer root, `sbatch slurm/train.sh`
#SBATCH -D .
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
# Several runs in parallel: add `#SBATCH --array=0-4` and set
# experiment.run_idx=$SLURM_ARRAY_TASK_ID; all land in one experiment folder.
# The main options, at their conf/config.yaml defaults:
python train.py \
    experiment.name=spectral_spatial \
    model.name=spectral_spatial \
    experiment.run_idx=0 \
    experiment.epochs=100 \
    experiment.batch_size=64 \
    optimizer.base_lr=0.001 \
    data.params.n_train=10000 \
    data.params.n_valid=1000 \
    data.params.n_test=1000 \
    data.params.seed_train=0 \
    data.params.seed_valid=1 \
    data.params.seed_test=2 \
    experiment.bf16=true \
    experiment.compile=true \
    experiment.channels_last=true
    # generator settings take a +, e.g. +data.params.scene=satellite,
    # +data.params.sim.sensor_noise_std=0.05 (all in src/data_generation/config.py)

sleep 5s
echo "Job $SLURM_JOB_ID completed"
