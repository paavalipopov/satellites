#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32g
#SBATCH -t 08:00:00
#SBATCH -J stage2_compare
#SBATCH -A psy53c17
#SBATCH -p qTRD
# -D . runs in the submit folder: submit from the project root
#SBATCH -D .
#SBATCH --output=slurm/logs/%x_%j.out
# -A psy53c17 -p qTRD: the lab account and the CPU partition (the GPU jobs use qTRDGPUH). No GPU line on purpose: the models are tiny, a plain CPU job is enough.

# Stage 2 (shape) comparison; trains nothing. slurm/run_shape_stage.sh submits it after the trainings.
#   sbatch slurm/compare_shape_stage.sh
# The brightness handling is fixed: automatic max scaling (single) on estimate.fits; every outline x cleanup x vote.
# Then the synthetic robustness tests (scripts/brightness_compare.py, against the original model). The synthetic scenes are
# at the training brightness, so they are given to the models as they are.
#   EXPERIMENTS  folders under LOGDIR (default: python scripts/brightness_presets.py group stage2_compare --folders;
#                missing ones are skipped)
#   READINGS (default "solar radiometry"), LOGDIR (logs), OUT (outputs/stage2_shape), N_SCENES (100),
#   MAX_RUNS (3: runs 0-2 of every model, the same seeds)
set -e
echo "Running on host: $HOSTNAME" >&2
[ -f scripts/compare_pipeline.py ] || { echo "run this from the project root" >&2; exit 1; }
mkdir -p slurm/logs
export TMPDIR=/tmp MKL_THREADING_LAYER=GNU MPLBACKEND=Agg
source ~/.bashrc
conda activate satellite
echo "Using python from: $(which python)"

OUT=${OUT:-outputs/stage2_shape}
LOGDIR=${LOGDIR:-logs}
EXPERIMENTS=${EXPERIMENTS:-$(python scripts/brightness_presets.py group stage2_compare --folders)}
COMMON=(--logdir "$LOGDIR" --max-runs "${MAX_RUNS:-3}")
[ -n "$READINGS" ] && COMMON+=(--readings $READINGS)

python scripts/compare_pipeline.py "${COMMON[@]}" --experiments $EXPERIMENTS --scalings single \
    --title "Stage 2: shape (original brightness handling: automatic max scaling)" --out "$OUT"

# the reference for the robustness tests: the original retrained here (same setup and seeds), else the earlier original
BASE=estimate_10band_original_retrained
[ -f "$LOGDIR/$BASE/config.yaml" ] || BASE=estimate_10band
NEW=()                                       # every other trained experiment, for the robustness tests
for n in $EXPERIMENTS; do
    [ "$n" != "$BASE" ] && [ -f "$LOGDIR/$n/config.yaml" ] && NEW+=("$n")
done
echo "robustness tests: reference $BASE; compared: ${NEW[*]}"
if [ ${#NEW[@]} -gt 0 ] && [ -f "$LOGDIR/$BASE/config.yaml" ]; then
    python scripts/brightness_compare.py --baseline "$BASE" --experiments "${NEW[@]}" --logdir "$LOGDIR" \
        --max-runs "${MAX_RUNS:-3}" --n-scenes "${N_SCENES:-100}" --reading radiometry --out "$OUT/robustness"
else
    echo "robustness tests skipped: no trained shape variant (or no reference model) yet"
fi
echo "Done: see $OUT/pipeline_summary.md and $OUT/robustness/comparison_summary.md"
