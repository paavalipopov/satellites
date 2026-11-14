#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32g
#SBATCH -t 08:00:00
#SBATCH -J stage3_compare
#SBATCH -A psy53c17
#SBATCH -p qTRD
# -D . runs in the submit folder: submit from the project root
#SBATCH -D .
#SBATCH --output=slurm/logs/%x_%j.out
# -A psy53c17 -p qTRD: the lab account and the CPU partition. No GPU line on purpose: the models are tiny.

# Stage 3 (combined) comparison; trains nothing. slurm/run_combined_stage.sh submits it after the trainings.
#   sbatch slurm/compare_combined_stage.sh
#   EXPERIMENTS  folders under LOGDIR (default: python scripts/brightness_presets.py group stage3_compare --folders;
#                missing ones are skipped)
#   READINGS (default radiometry), LOGDIR (logs), OUT (outputs/stage3_combined), N_SCENES (100), MAX_RUNS (3: runs 0-2)
set -e
echo "Running on host: $HOSTNAME" >&2
[ -f scripts/compare_pipeline.py ] || { echo "run this from the project root" >&2; exit 1; }
mkdir -p slurm/logs
export TMPDIR=/tmp MKL_THREADING_LAYER=GNU MPLBACKEND=Agg
source ~/.bashrc
conda activate satellite
echo "Using python from: $(which python)"

OUT=${OUT:-outputs/stage3_combined}
LOGDIR=${LOGDIR:-logs}
EXPERIMENTS=${EXPERIMENTS:-$(python scripts/brightness_presets.py group stage3_compare --folders)}
COMMON=(--logdir "$LOGDIR" --max-runs "${MAX_RUNS:-3}" --readings ${READINGS:-radiometry})

# 1. estimate.fits: every model x the test-time steps that mattered in stages 1 and 2
python scripts/compare_pipeline.py "${COMMON[@]}" --experiments $EXPERIMENTS \
    --scalings single per_object per_object_faint --outlines model mask@0.01 mask@0.01+faint fill@0.01 \
    --cleanups off tidy --votes 0.3 0.5 \
    --title "Stage 3: combined (brightness + shape)" --out "$OUT"

# 2. synthetic scenes at many brightnesses, and a bright and a dim object together
python scripts/compare_scaling.py "${COMMON[@]}" --experiments $EXPERIMENTS --skip-fits \
    --modes none single per_object per_object_faint --n-scenes "${N_SCENES:-100}" --out "$OUT/synthetic"

# 3. robustness (noise, halo, empty scenes, a curve over the bands), against the fair original
BASE=estimate_10band_original_retrained
NEW=()
for n in $EXPERIMENTS; do
    [ "$n" != "$BASE" ] && [ -f "$LOGDIR/$n/config.yaml" ] && NEW+=("$n")
done
if [ ${#NEW[@]} -gt 0 ] && [ -f "$LOGDIR/$BASE/config.yaml" ]; then
    python scripts/brightness_compare.py --baseline "$BASE" --experiments "${NEW[@]}" --logdir "$LOGDIR" \
        --max-runs "${MAX_RUNS:-3}" --n-scenes "${N_SCENES:-100}" --reading radiometry --out "$OUT/robustness"
else
    echo "robustness tests skipped: no trained variant or no $BASE yet"
fi
echo "Done: see $OUT/pipeline_summary.md, $OUT/synthetic/scaling_summary.md and $OUT/robustness/comparison_summary.md"
