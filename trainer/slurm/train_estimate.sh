#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=256g
#SBATCH -p qTRDGPUH
#SBATCH -t 24:00:00
#SBATCH --exclude=arctrddgxa001
#SBATCH --gres=gpu:A100:2
#SBATCH -J estimate_10band
# 10 runs; qTRDGPUH allows 8 GPUs per user, so 4 jobs run at a time
#SBATCH --array=0-9
# -D . runs in the submit folder: submit from the trainer root, `sbatch slurm/train_estimate.sh`
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

# Models for assets/new_input/estimate.fits: its 10 bands (message.txt), in ascending
# order. The FITS cube is stored red to blue, so flip its band axis before inference.
# Bigger batches, more fresh scenes and more train workers than train.sh, to keep the
# GPUs busier: 2 x 128 scenes per step, 20000 per epoch.
python train.py \
    experiment.name=estimate_10band \
    model.name=spectral_spatial \
    experiment.run_idx=$SLURM_ARRAY_TASK_ID \
    experiment.epochs=200 \
    experiment.batch_size=128 \
    optimizer.base_lr=0.001 \
    data.params.n_train=20000 \
    data.params.n_valid=1000 \
    data.params.n_test=1000 \
    +data.params.sim.n_bands=10 \
    "+data.params.sim.wavelengths=[400.0,426.667,457.143,492.308,533.333,581.818,640.0,711.111,800.0,914.286]" \
    loader.train.num_workers=20 \
    loader.eval.num_workers=4 \
    experiment.bf16=true \
    experiment.compile=true \
    experiment.channels_last=true

sleep 5s
echo "Job $SLURM_JOB_ID completed"
