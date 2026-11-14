#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2g
#SBATCH -t 00:10:00
#SBATCH -J run_shape_stage
#SBATCH -A psy53c17
#SBATCH -p qTRD
# -D . runs in the submit folder: submit from the project root
#SBATCH -D .
#SBATCH --output=slurm/logs/%x_%j.out
# -A psy53c17 -p qTRD: the lab account and the CPU partition; no GPU (like submit_brightness_grid.sh): this job only calls sbatch.

# STAGE 2: SHAPE ONLY. The brightness handling stays fixed: the original training, and automatic max scaling at test
# time (scripts/auto_scale.py, mode single). Only the shape choices change.
#
#   sbatch slurm/run_shape_stage.sh             # or:  bash slurm/run_shape_stage.sh
#
#   1. self-test (CPU, a minute or two);
#   2. trains 5 variants x 3 runs = 15 GPU jobs (about 20-25 min each), the same 3 seeds for all:
#        fg3 fg10 fg3_dice    every object pixel counts 3x / 10x in the loss (fg3_dice: 3x + a Dice term)
#        halo                 the halo (a faint skirt round the object, labelled background)
#        lightnoise           weaker restoration-like noise
#        bank                 real noise patches: only with NOISE_BANK=/path/to/bank.npy
#   3. compares them with the original model (already trained; runs 0-2), each with every outline (model, mask, fill),
#      cleanup (off, tidy, tidy+smooth) and vote (0.3, 0.5, 0.7) on estimate.fits, and on the synthetic robustness tests
#      (restoration-like noise, halo, empty scenes, a curve over the bands): slurm/compare_shape_stage.sh.
#      The outline / cleanup / vote part needs no training: NO_TRAIN=1 sbatch slurm/run_shape_stage.sh runs it on the
#      original model right away.
#
# Results: outputs/stage2_shape/pipeline_summary.md (estimate.fits) and outputs/stage2_shape/robustness/comparison_summary.md.
# The same environment variables as slurm/run_pipeline.sh (VARIANTS, ARRAY, NOISE_BANK, COMPILE, NO_TRAIN, SKIP_CHECK, ...).
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
STAGE=shape exec bash slurm/run_pipeline.sh
