#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2g
#SBATCH -p qTRDGPUH
#SBATCH -t 00:10:00
#SBATCH -J submit_grid
# -D . runs in the submit folder: submit from the project root
#SBATCH -D .
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH -A psy53c17

# Submit several variants of slurm/train_brightness.sh in one go (each becomes its own 10-run array job).
# It only calls sbatch a few times, so it can be run either way, from the project root:
#
#   bash slurm/submit_brightness_grid.sh                     # right now, in your shell
#   sbatch slurm/submit_brightness_grid.sh                   # as a short job that does the submitting
#
# Which variants: none given = the nine NEW ones (not the round-1 three); or name them:
#
#   sbatch slurm/submit_brightness_grid.sh full_max full_p99
#
# Environment variables (sbatch passes them on):
#   ARRAY=0-4     5 runs of each instead of 10
#   COMPILE=false train without torch.compile (if a job dies inside it)
#   GAIN_RANGE, GAIN_STAGE, EXTRA   as in slurm/train_brightness.sh
# e.g.  ARRAY=0-4 COMPILE=false sbatch slurm/submit_brightness_grid.sh gain_p99 full_max
#
# The variants are explained by: python scripts/brightness_presets.py list
# The training jobs' own logs are slurm/logs/b_<variant>_<jobid>_<task>.out; this job's is
# slurm/logs/submit_grid_<jobid>.out, and says which job ids it submitted.
set -e

# under sbatch this script runs from a copy in the spool folder, so find the project root from where it was submitted
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[ -f train.py ] || { echo "run this from the project root (train.py not found in $PWD)" >&2; exit 1; }
mkdir -p slurm/logs

# this job's own SLURM_* settings (memory, cpus, ...) must not leak into the jobs it submits
for v in $(env | grep -o '^SLURM_[A-Za-z0-9_]*' ); do unset "$v"; done

# scripts/brightness_presets.py uses only the standard library: any python will do
PY=$(command -v python3 || command -v python) || { echo "no python found to read scripts/brightness_presets.py" >&2; exit 1; }

VARIANTS=("$@")
[ ${#VARIANTS[@]} -gt 0 ] || VARIANTS=(gain_p99 gain_flux gain_bandflux gain_pixel curve_max empty_max full_max full_p99 full_raw)
for v in "${VARIANTS[@]}"; do
    "$PY" scripts/brightness_presets.py hydra "$v" > /dev/null   # fails on a misspelt variant, before anything is submitted
done
for v in "${VARIANTS[@]}"; do
    if [ -n "$ARRAY" ]; then
        VARIANT=$v sbatch --array=$ARRAY -J "b_$v" slurm/train_brightness.sh
    else
        VARIANT=$v sbatch -J "b_$v" slurm/train_brightness.sh
    fi
done
echo "submitted ${#VARIANTS[@]} variants; watch with: squeue -u \$USER"
