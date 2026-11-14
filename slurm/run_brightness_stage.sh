#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2g
#SBATCH -t 00:10:00
#SBATCH -J run_brightness_stage
#SBATCH -A psy53c17
#SBATCH -p qTRD
# -D . runs in the submit folder: submit from the project root
#SBATCH -D .
#SBATCH --output=slurm/logs/%x_%j.out
# -A psy53c17 -p qTRD: the lab account and the CPU partition; no GPU (like submit_brightness_grid.sh): this job only calls sbatch.

# STAGE 1: BRIGHTNESS ONLY. Everything except the brightness handling stays fixed (no halo, no extra noise, the plain
# loss; at test time the model's own outline, no cleanup, vote 0.5).
#
#   sbatch slurm/run_brightness_stage.sh        # or:  bash slurm/run_brightness_stage.sh
#
#   1. self-test (CPU, a minute or two);
#   2. trains 6 variants x 3 runs = 18 GPU jobs (about 20-25 min each), the same 3 seeds for all:
#        pair_max1  pair_mean  pair_p99           every scene twice: as generated + divided by its max / the mean / the
#        pair_max1_aug pair_mean_aug pair_p99_aug   99th percentile of its object pixels (_aug: random brightness on the first copy)
#   3. compares them with the earlier brightness models that are already trained (original, aug, norm, aug_norm, gain_p99,
#      gain_flux, gain_bandflux, gain_pixel; runs 0-2, the same seeds), each with all 7 test-time scalings (none, single by
#      max / mean / p99, per object by max / mean / p99), on estimate.fits and on synthetic scenes (many brightnesses; a
#      bright and a dim object together): slurm/compare_brightness_stage.sh.
#
# Results: outputs/stage1_brightness/pipeline_summary.md (estimate.fits) and outputs/stage1_brightness/synthetic/scaling_summary.md.
# The same environment variables as slurm/run_pipeline.sh (VARIANTS, ARRAY, COMPILE, NO_TRAIN, SKIP_CHECK, ...).
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
STAGE=brightness exec bash slurm/run_pipeline.sh
