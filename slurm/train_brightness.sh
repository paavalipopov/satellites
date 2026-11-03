#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=128g
#SBATCH -p qTRDGPUH
#SBATCH -t 24:00:00
#SBATCH --exclude=arctrddgxa001
#SBATCH --gres=gpu:A100:1
#SBATCH -J brightness
# 10 runs of ONE variant per submission, like estimate_10band; for fewer: sbatch --array=0-4 ...
#SBATCH --array=0-9
# -D . runs in the submit folder: submit from the project root
#SBATCH -D .
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH -A psy53c17

# The 10-band setup of train_estimate.sh (the bands of assets/new_input/estimate.fits), plus one named
# way to make the model cope with brightness, the response of the system and empty sky:
#
#   VARIANT=aug_norm sbatch slurm/train_brightness.sh        (the default)
#   VARIANT=full_max sbatch slurm/train_brightness.sh
#
# The variants are defined in ONE place, scripts/brightness_presets.py; list them with
#   python scripts/brightness_presets.py list
# (aug, norm, aug_norm, gain_p99, gain_flux, gain_bandflux, gain_pixel, curve_max, empty_max,
#  full_max, full_p99, full_raw). To submit several at once: bash slurm/submit_brightness_grid.sh
#
# Results land in logs/estimate_10band_<variant>/run_<i>, next to logs/estimate_10band.
#
# GAIN_RANGE   log10 of the overall brightness factor, "lo,hi": -1.5,0.5 means 0.03x to 3x (default)
# GAIN_STAGE   before_noise (a dimmer object against the same sensor noise) | after_noise (a change of units)
# COMPILE      true (default) | false: set false if a job dies in torch.compile (p99 uses kthvalue)
# EXTRA        more Hydra overrides, e.g. EXTRA="+data.params.sim.noise_bank=/path/to/bank.npy"

set -e
sleep 5s
echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID" >&2

VARIANT=${VARIANT:-aug_norm}
GAIN_RANGE=${GAIN_RANGE:--1.5,0.5}
GAIN_STAGE=${GAIN_STAGE:-before_noise}
COMPILE=${COMPILE:-true}
EXTRA=${EXTRA:-}

export TMPDIR=/tmp
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MKL_THREADING_LAYER=GNU   # conda MKL vs torch's OpenMP

source ~/.bashrc
conda activate satellite
echo "Using python from: $(which python)"

# the Hydra overrides of this variant (a list of +key=value words, no spaces inside)
VARIANT_ARGS=($(python scripts/brightness_presets.py hydra "$VARIANT" --gain-range="$GAIN_RANGE" --gain-stage="$GAIN_STAGE"))
if [ ${#VARIANT_ARGS[@]} -eq 0 ]; then
    echo "VARIANT='$VARIANT' gave no overrides (unknown, or 'base', which is plain logs/estimate_10band)" >&2
    exit 1
fi
echo "Variant: $VARIANT  ${VARIANT_ARGS[*]}" >&2

python train.py \
    experiment.name=estimate_10band_$VARIANT \
    model.name=spectral_spatial \
    experiment.run_idx=$SLURM_ARRAY_TASK_ID \
    experiment.epochs=200 \
    experiment.batch_size=128 \
    optimizer.base_lr=0.001 \
    data.params.n_train=20000 \
    data.params.n_valid=1000 \
    data.params.n_test=1000 \
    +data.params.sim.n_bands=10 \
    "+data.params.sim.wavelengths=[400.0,426.667,457.143,492.308,533.333,581.818,640.0,711.111,800.0,914.286]" \
    loader.train.num_workers=20 \
    loader.eval.num_workers=4 \
    experiment.bf16=true \
    experiment.compile=$COMPILE \
    experiment.channels_last=true \
    "${VARIANT_ARGS[@]}" \
    $EXTRA

sleep 5s
echo "Job $SLURM_JOB_ID completed"
