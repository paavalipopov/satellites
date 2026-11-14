#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32g
#SBATCH -t 08:00:00
#SBATCH -J stage1_compare
#SBATCH -A psy53c17
#SBATCH -p qTRD
# -D . runs in the submit folder: submit from the project root
#SBATCH -D .
#SBATCH --output=slurm/logs/%x_%j.out
# -A psy53c17 -p qTRD: the lab account and the CPU partition (the GPU jobs use qTRDGPUH). No GPU line on purpose: the models are tiny, a plain CPU job is enough.

# Stage 1 (brightness) comparison; trains nothing. slurm/run_brightness_stage.sh submits it after the trainings.
#   sbatch slurm/compare_brightness_stage.sh
# Every model with every test-time scaling; the rest fixed (model outline, no cleanup, vote 0.5).
#   EXPERIMENTS  folders under LOGDIR (default: python scripts/brightness_presets.py group stage1_compare --folders;
#                missing ones are skipped)
#   READINGS (default "solar radiometry"), LOGDIR (logs), OUT (outputs/stage1_brightness), N_SCENES (100),
#   MAX_RUNS (3: runs 0-2 of every model, the same seeds)
set -e
echo "Running on host: $HOSTNAME" >&2
[ -f scripts/compare_pipeline.py ] || { echo "run this from the project root" >&2; exit 1; }
mkdir -p slurm/logs
export TMPDIR=/tmp MKL_THREADING_LAYER=GNU MPLBACKEND=Agg
source ~/.bashrc
conda activate satellite
echo "Using python from: $(which python)"

OUT=${OUT:-outputs/stage1_brightness}
EXPERIMENTS=${EXPERIMENTS:-$(python scripts/brightness_presets.py group stage1_compare --folders)}
COMMON=(--logdir "${LOGDIR:-logs}" --max-runs "${MAX_RUNS:-3}" --experiments $EXPERIMENTS)
[ -n "$READINGS" ] && COMMON+=(--readings $READINGS)

python scripts/compare_pipeline.py "${COMMON[@]}" --outlines model --cleanups off --votes 0.5 \
    --title "Stage 1: brightness handling (model outline, no cleanup, vote 0.5)" --out "$OUT"
python scripts/compare_scaling.py "${COMMON[@]}" --skip-fits --n-scenes "${N_SCENES:-100}" --out "$OUT/synthetic"
echo "Done: see $OUT/pipeline_summary.md and $OUT/synthetic/scaling_summary.md"
