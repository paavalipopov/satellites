#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2g
#SBATCH -t 00:10:00
#SBATCH -J run_combined_stage
#SBATCH -A psy53c17
#SBATCH -p qTRD
# -D . runs in the submit folder: submit from the project root
#SBATCH -D .
#SBATCH --output=slurm/logs/%x_%j.out
# -A psy53c17 -p qTRD: the lab account and the CPU partition; no GPU: this job only calls sbatch.

# STAGE 3: COMBINED. The best of stage 1 (brightness) and stage 2 (shape) together:
#
#   bash slurm/run_combined_stage.sh            # or:  sbatch slurm/run_combined_stage.sh
#
#   1. self-test (CPU, a minute or two);
#   2. trains 8 variants x 3 runs = 24 GPU jobs (about 20-25 min each on an A100), the same 3 seeds as every other stage:
#        aug_fg3            aug_fg3_halo            aug_fg3_light            aug_fg3_halo_light
#        pair_max1_aug_fg3  pair_max1_aug_fg3_halo  pair_max1_aug_fg3_light  pair_max1_aug_fg3_halo_light
#      aug = random brightness in training; pair_max1_aug = random brightness + every scene twice (the copy divided by its
#      brightest value); fg3 = object pixels count 3x in the loss; halo = a faint glow round the object, labelled background;
#      light = weaker restoration-like noise. No scaling inside the model.
#   3. compares them with original_retrained, aug, pair_max1_aug, fg3, halo and lightnoise (already trained; runs 0-2),
#      each with automatic scaling (single, per_object, per_object_faint), the object mask from brightness (model / mask /
#      mask+faint / fill), cleanup (off / tidy) and vote (0.3 / 0.5) on estimate.fits, plus the synthetic brightness tests
#      and robustness tests: slurm/compare_combined_stage.sh.
#
# Results: outputs/stage3_combined/pipeline_summary.md, outputs/stage3_combined/synthetic/scaling_summary.md and
# outputs/stage3_combined/robustness/comparison_summary.md.
# The same environment variables as slurm/run_pipeline.sh (VARIANTS, ARRAY, COMPILE, NO_TRAIN, SKIP_CHECK, CPU_PARTITION, ...).
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
STAGE=combined exec bash slurm/run_pipeline.sh
