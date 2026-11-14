#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32g
#SBATCH -t 12:00:00
#SBATCH -J pipeline_compare
#SBATCH -A psy53c17
#SBATCH -p qTRD
# -D . runs in the submit folder: submit from the project root
#SBATCH -D .
#SBATCH --output=slurm/logs/%x_%j.out
# -A psy53c17 -p qTRD: the lab account and the CPU partition (the GPU jobs use qTRDGPUH). No GPU line on purpose: the models are tiny, a plain CPU job is enough.

# Scores every trained experiment with every test-time choice (scripts/compare_pipeline.py: scaling x outline x
# cleanup x vote, on estimate.fits) and then the synthetic brightness tests (scripts/compare_scaling.py: one object at
# many brightnesses, a bright and a dim object in one image). Trains nothing.
#
#   sbatch slurm/compare_pipeline.sh        # or:  bash slurm/compare_pipeline.sh
#
# Environment variables:
#   EXPERIMENTS  space-separated folders under LOGDIR (default: every variant of scripts/brightness_presets.py, including the
#                original and the earlier rounds such as aug_norm, gain_flux, full_p99; folders that do not exist are skipped)
#   READINGS     how estimate.fits is read (default "solar radiometry"; scripts/fits_reading.py)
#   LOGDIR       where the experiment folders are (default logs)
#   OUT          where the results go (default outputs/pipeline_compare)
#   N_SCENES     synthetic scenes per level (default 100)
#   MAX_RUNS     use only the first N runs of each experiment (default 3: runs 0-2, the same seeds for every model)
#   SKIP_SYNTHETIC=1   only the estimate.fits part
#
# Results: OUT/pipeline_summary.md, pipeline_grid.csv, pipeline_maps_<reading>.png and OUT/synthetic/scaling_summary.md
set -e
echo "Running on host: $HOSTNAME" >&2
[ -f scripts/compare_pipeline.py ] || { echo "run this from the project root (scripts/compare_pipeline.py not found)" >&2; exit 1; }
mkdir -p slurm/logs
export TMPDIR=/tmp
export MKL_THREADING_LAYER=GNU
export MPLBACKEND=Agg
source ~/.bashrc
conda activate satellite
echo "Using python from: $(which python)"

OUT=${OUT:-outputs/pipeline_compare}
COMMON=(--logdir "${LOGDIR:-logs}")
COMMON+=(--max-runs "${MAX_RUNS:-3}")
[ -n "$READINGS" ] && COMMON+=(--readings $READINGS)
[ -n "$EXPERIMENTS" ] && COMMON+=(--experiments $EXPERIMENTS)

python scripts/compare_pipeline.py "${COMMON[@]}" --out "$OUT"
if [ -z "$SKIP_SYNTHETIC" ]; then
    python scripts/compare_scaling.py "${COMMON[@]}" --skip-fits --n-scenes "${N_SCENES:-100}" --out "$OUT/synthetic"
fi
echo "Done: see $OUT/pipeline_summary.md"
