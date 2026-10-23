#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=256g
#SBATCH -p qTRDGPUH
#SBATCH -t 24:00:00
#SBATCH --exclude=arctrddgxa001
#SBATCH --gres=gpu:A100:2
#SBATCH -J repro_more_scenes
# 10 runs; qTRDGPUH allows 8 GPUs per user, so 4 jobs run at a time
#SBATCH --array=0-9
# -D . runs in the submit folder: submit from the trainer root, `sbatch slurm/train.sh`
#SBATCH -D .
#SBATCH --output=slurm/logs/%x_%A_%a.out
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

# Multi-GPU: --gres=gpu:A100:N above; SLURM_GPUS_ON_NODE drives the world size.
# batch_size is per GPU (2 x 64 per step) and max_lr = base_lr * sqrt(2).
# Each array task trains one run; all land in logs/repro_more_scenes/run_<i>.
# The main options, at their conf/config.yaml defaults unless noted:
python train.py \
    experiment.name=repro_more_scenes \
    model.name=spectral_spatial \
    experiment.run_idx=$SLURM_ARRAY_TASK_ID \
    experiment.epochs=200 \
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
    # epochs and run_idx differ from the config; generator settings take a +, e.g.
    # +data.params.scene=satellite, +data.params.sim.sensor_noise_std=0.05

sleep 5s
echo "Job $SLURM_JOB_ID completed"
