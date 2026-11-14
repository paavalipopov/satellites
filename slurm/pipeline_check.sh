#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16g
#SBATCH -t 00:30:00
#SBATCH -J pipeline_check
#SBATCH -A psy53c17
#SBATCH -p qTRD
# -D . runs in the submit folder: submit from the project root
#SBATCH -D .
#SBATCH --output=slurm/logs/%x_%j.out
# -A psy53c17 -p qTRD: the lab account and the CPU partition (the GPU jobs use qTRDGPUH). No GPU line on purpose: a plain CPU job.

# The self-test of every brightness / shape option (about a minute, no training). slurm/run_pipeline.sh makes the
# training jobs wait for this one, so that a broken option stops the run before any GPU time is spent.
set -e
echo "Running on host: $HOSTNAME" >&2
[ -f scripts/check_brightness_options.py ] || { echo "run this from the project root" >&2; exit 1; }
mkdir -p slurm/logs
export TMPDIR=/tmp
export MKL_THREADING_LAYER=GNU
source ~/.bashrc
conda activate satellite
echo "Using python from: $(which python)"
python scripts/check_brightness_options.py
