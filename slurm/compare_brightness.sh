#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32g
#SBATCH -p qTRDGPUH
#SBATCH -t 04:00:00
#SBATCH --exclude=arctrddgxa001
#SBATCH --gres=gpu:A100:1
#SBATCH -J brightness_compare
# -D . runs in the submit folder: submit from the project root
#SBATCH -D .
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH -A psy53c17

# Compares the new experiments with the previous (original) model: accuracy tables AND the notebook-06
# maps (true materials, models' labels, share of models: an object / its material), side by side.
# It needs no GPU (the models are tiny), so the GPU lines above can go if your queue wants CPU jobs.
#
#   sbatch slurm/compare_brightness.sh        # or:  bash slurm/compare_brightness.sh
#
# Run it after the training jobs have finished. By default it compares logs/estimate_10band (the
# previous model) with every logs/estimate_10band_<variant> folder it finds. Environment variables:
#   BASELINE     the previous model's folder under LOGDIR      (default estimate_10band)
#   EXPERIMENTS  space-separated new folders to compare         (default: every variant found)
#   LOGDIR       where the experiment folders are              (default logs)
#   OUT          where the results go                          (default outputs/brightness_compare)
#   N_SCENES     synthetic scenes per test level               (default 200)
#   MAX_RUNS     use only the first N runs of each experiment  (default: all)
# e.g.  EXPERIMENTS="estimate_10band_aug_norm estimate_10band_full_max" sbatch slurm/compare_brightness.sh
#
# Results: OUT/comparison_summary.md and .csv, OUT/maps_side_by_side.png, OUT/maps_<experiment>.png, ...

set -e
echo "Running on host: $HOSTNAME" >&2
[ -f scripts/brightness_compare.py ] || { echo "run this from the project root (scripts/brightness_compare.py not found)" >&2; exit 1; }
mkdir -p slurm/logs

export TMPDIR=/tmp
export MKL_THREADING_LAYER=GNU
export MPLBACKEND=Agg

source ~/.bashrc
conda activate satellite
echo "Using python from: $(which python)"

ARGS=(--baseline "${BASELINE:-estimate_10band}" --logdir "${LOGDIR:-logs}" --out "${OUT:-outputs/brightness_compare}" --n-scenes "${N_SCENES:-200}")
[ -n "$MAX_RUNS" ] && ARGS+=(--max-runs "$MAX_RUNS")
[ -n "$EXPERIMENTS" ] && ARGS+=(--experiments $EXPERIMENTS)

python scripts/brightness_compare.py "${ARGS[@]}"
echo "Done: see ${OUT:-outputs/brightness_compare}/comparison_summary.md and maps_side_by_side.png"
