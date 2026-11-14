#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2g
#SBATCH -t 00:10:00
#SBATCH -J run_pipeline
#SBATCH -A psy53c17
#SBATCH -p qTRD
# -D . runs in the submit folder: submit from the project root
#SBATCH -D .
#SBATCH --output=slurm/logs/%x_%j.out
# -A psy53c17 -p qTRD: the lab account and the CPU partition; no GPU (like submit_brightness_grid.sh): this job only calls sbatch.

# THE WHOLE PIPELINE in one go, 3 runs of every variant with the SAME 3 random seeds for all of them:
#
#   1. a self-test of every option (slurm/pipeline_check.sh, CPU, about a minute);
#   2. if it passes, the training of each variant (slurm/train_brightness.sh, 3 runs each, GPU);
#   3. when the trainings have ended, the comparison (slurm/compare_pipeline.sh, CPU): every model (the new ones and every
#      earlier one that is already trained, all scored on runs 0-2) x every test-time choice (scaling, outline from the
#      data, cleanup, vote) on estimate.fits, and the synthetic brightness tests.
#
#   bash slurm/run_pipeline.sh                     # right now, in your shell (it only calls sbatch a few times)
#   sbatch slurm/run_pipeline.sh                   # or as a short job that does the submitting
#
# The variants (python scripts/brightness_presets.py list; the default set: python scripts/brightness_presets.py pipeline):
#   pair_<max1|mean|p99>[_aug][_halo][_light]   24 variants: every training scene twice, as generated and divided by its max /
#                             the mean / the 99th percentile of its object pixels; with or without random brightness on the
#                             first copy; with or without the halo and the weaker restoration-like noise
#   fg3 fg10 fg3_dice         heavier loss on object pixels (3x, 10x; 3x + a Dice term)
#   lightnoise                weaker restoration-like noise
#   bank                      real noise patches: only if NOISE_BANK=/path/to/bank.npy is set (scripts/make_noise_bank.py)
# 28 variants x 3 runs = 84 GPU jobs by default (each variant is one array job of 3 tasks).
# They train on the original brightness; the brightness is handled in front of the model at test time (scripts/auto_scale.py).
# The reference models (logs/estimate_10band, logs/estimate_10band_aug_norm) must already be trained; one that is missing is
# skipped by the comparison.
#
# Environment variables (sbatch passes them on):
#   ARRAY=0-2            the runs of each variant (default 0-2 = 3 runs). The seeds are the same for every variant: run i
#                        starts its model from init_seed + i (42, 43, 44) and draws the training scenes of stream i; the
#                        validation and test scenes are the same for all. The comparison scores runs 0-2 of every model
#                        (MAX_RUNS, set from ARRAY), so the earlier models (10 runs) are compared on the same 3 seeds
#   VARIANTS="fg3 fg10"  train only these (default: the 28 above)
#   NOISE_BANK=...       add the `bank` variant
#   COMPILE=false        train without torch.compile (if a job dies inside it)
#   PAIR_N_TRAIN=10000   scenes per epoch of the pair_* variants (they use every scene twice, so an epoch costs the same as the
#                        20000 scenes of the others)
#   STAGE=brightness|shape|all   which stage (default all = every variant at once); the two stage scripts set it:
#                        sbatch slurm/run_brightness_stage.sh   /   sbatch slurm/run_shape_stage.sh
#   NO_TRAIN=1           skip steps 1 and 2: only submit the comparison (e.g. to compare models that are already trained)
#   SKIP_CHECK=1         no self-test
#   CPU_PARTITION=name   the partition for the CPU jobs (self-test, comparison); default qTRD (set in their headers).
#                        All jobs use the account psy53c17. Check one first:  sbatch --test-only slurm/pipeline_check.sh
#   OUT=...              where the comparison writes (default outputs/pipeline_compare)
# e.g.  VARIANTS="pair_mean pair_p99 fg3" sbatch slurm/run_pipeline.sh
set -e

# under sbatch this script runs from a copy in the spool folder, so find the project root from where it was submitted
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[ -f train.py ] || { echo "run this from the project root (train.py not found in $PWD)" >&2; exit 1; }
mkdir -p slurm/logs
for v in $(env | grep -o '^SLURM_[A-Za-z0-9_]*'); do unset "$v"; done      # this job's own settings must not leak into the jobs it submits

PY=$(command -v python3 || command -v python) || { echo "no python found" >&2; exit 1; }
ARRAY=${ARRAY:-0-2}
STAGE=${STAGE:-all}                      # brightness | shape | combined | all (the slurm/run_*_stage.sh scripts set it)
case "$STAGE" in
    brightness) DEFAULT_VARIANTS=$("$PY" scripts/brightness_presets.py group stage1_train); COMPARE=slurm/compare_brightness_stage.sh ;;
    shape)      DEFAULT_VARIANTS=$("$PY" scripts/brightness_presets.py group stage2_train); COMPARE=slurm/compare_shape_stage.sh ;;
    combined)   DEFAULT_VARIANTS=$("$PY" scripts/brightness_presets.py group stage3_train); COMPARE=slurm/compare_combined_stage.sh ;;
    all)        DEFAULT_VARIANTS=$("$PY" scripts/brightness_presets.py pipeline);           COMPARE=slurm/compare_pipeline.sh ;;
    *) echo "STAGE must be brightness, shape, combined or all (got '$STAGE')" >&2; exit 1 ;;
esac
VARIANTS=${VARIANTS:-$DEFAULT_VARIANTS}
echo "stage: $STAGE   variants: $VARIANTS"
if [ -z "$MAX_RUNS" ] && [[ "$ARRAY" =~ ^([0-9]+)-([0-9]+)$ ]] && [ "${BASH_REMATCH[1]}" = 0 ]; then
    MAX_RUNS=$((BASH_REMATCH[2] + 1))                                       # runs 0..N-1 of every model, the same seeds
fi
export MAX_RUNS
if { [ "$STAGE" = shape ] || [ "$STAGE" = all ]; } && [ -n "$NOISE_BANK" ] && [[ " $VARIANTS " != *" bank "* ]]; then VARIANTS="$VARIANTS bank"; fi
PAIR_N_TRAIN=${PAIR_N_TRAIN:-10000}

CPU_OPTS=()                                                                 # the CPU jobs (self-test, comparison)
[ -n "$CPU_PARTITION" ] && CPU_OPTS+=(-p "$CPU_PARTITION")
DEP=()
TRAIN_IDS=()
if [ -z "$NO_TRAIN" ]; then
    for v in $VARIANTS; do                                                  # a misspelt variant fails here, before anything is submitted
        "$PY" scripts/brightness_presets.py hydra "$v" > /dev/null
    done
    if [ -z "$SKIP_CHECK" ]; then
        CHECK=$(sbatch --parsable "${CPU_OPTS[@]}" slurm/pipeline_check.sh)
        echo "self-test: job $CHECK"
        DEP=(--dependency=afterok:"$CHECK" --kill-on-invalid-dep=yes)
    fi
    for v in $VARIANTS; do
        N=20000
        [[ "$v" == pair_* ]] && N=$PAIR_N_TRAIN
        ID=$(N_TRAIN=$N VARIANT=$v sbatch --parsable --array="$ARRAY" -J "b_$v" "${DEP[@]}" slurm/train_brightness.sh)
        echo "training $v ($ARRAY, $N scenes per epoch): job $ID"
        TRAIN_IDS+=("$ID")
    done
fi

CMP_DEP=()
if [ ${#TRAIN_IDS[@]} -gt 0 ]; then
    CMP_DEP=(--dependency=afterany:$(IFS=:; echo "${TRAIN_IDS[*]}"))        # run on whatever has finished, even if a run failed
fi
CMP=$(sbatch --parsable "${CPU_OPTS[@]}" "${CMP_DEP[@]}" "$COMPARE")
echo "comparison: job $CMP ($COMPARE; its header says where the results go)"
echo "watch with: squeue -u \$USER"
