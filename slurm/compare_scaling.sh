#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32g
#SBATCH -t 04:00:00
#SBATCH -J scaling_compare
#SBATCH -A psy53c17
#SBATCH -p qTRD
# -D . runs in the submit folder: submit from the project root
#SBATCH -D .
#SBATCH --output=slurm/logs/%x_%j.out
# -A psy53c17 -p qTRD: the lab account and the CPU partition (the GPU jobs use qTRDGPUH). No GPU line on purpose (same as compare_brightness.sh): the models are tiny, a plain CPU job is enough.

# Compares ways of handling brightness: the original model, the two "originals + brightest = 1" trainings
# (pair_max1, pair_max1_aug) and the in-model max normalization (aug_norm), each with three scalings in front of the
# model (none | single | per_object, scripts/auto_scale.py), on estimate.fits and on synthetic scenes, including
# scenes with a bright and a dim object in one image. It trains nothing: run it after the training jobs have finished
# (bash slurm/submit_brightness_grid.sh pair_max1 pair_max1_aug).
#
#   sbatch slurm/compare_scaling.sh        # or:  bash slurm/compare_scaling.sh
#
# Environment variables:
#   EXPERIMENTS  space-separated folders under LOGDIR   (default: original, pair_max1, pair_max1_aug, aug_norm; missing ones are skipped)
#   READINGS     how estimate.fits is read              (default "solar radiometry"; see scripts/fits_reading.py)
#   LOGDIR       where the experiment folders are       (default logs)
#   OUT          where the results go                   (default outputs/scaling_compare)
#   N_SCENES     synthetic scenes per level             (default 100)
#   MAX_RUNS     use only the first N runs of each experiment (default: all)
#
# Results: OUT/scaling_summary.md (the tables), scaling_*.csv, fits_maps_<reading>.png

set -e
echo "Running on host: $HOSTNAME" >&2
[ -f scripts/compare_scaling.py ] || { echo "run this from the project root (scripts/compare_scaling.py not found)" >&2; exit 1; }
mkdir -p slurm/logs

export TMPDIR=/tmp
export MKL_THREADING_LAYER=GNU
export MPLBACKEND=Agg

source ~/.bashrc
conda activate satellite
echo "Using python from: $(which python)"

ARGS=(--logdir "${LOGDIR:-logs}" --out "${OUT:-outputs/scaling_compare}" --n-scenes "${N_SCENES:-100}")
[ -n "$MAX_RUNS" ] && ARGS+=(--max-runs "$MAX_RUNS")
[ -n "$READINGS" ] && ARGS+=(--readings $READINGS)
[ -n "$EXPERIMENTS" ] && ARGS+=(--experiments $EXPERIMENTS)

python scripts/compare_scaling.py "${ARGS[@]}"
echo "Done: see ${OUT:-outputs/scaling_compare}/scaling_summary.md"
