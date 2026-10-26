# Per-pixel material classification of synthetic satellite HSI

Each scene is a [36, 15, 15] cube with a [15, 15] material mask (0 = background, 1-46 =
materials), generated on the fly by `src/data_generation` (a copy of the project's `src/data_generation`).
Training scenes are fresh every epoch; valid and test are fixed sets, chosen by their seeds.
Two test sets by default (`data.params.tests`): `test` (the scene mix) and `test_satellite`.
Catalyst runner, Hydra config, single- or multi-GPU on SLURM.

## Environment

A conda env on Python 3.12, from this folder:

```bash
conda create -n satellite python=3.12 -y
conda activate satellite
pip install -r requirements.txt
# pip keeps an installed catalyst when only the pinned commit changes (same version), so force it
pip install --force-reinstall --no-deps "$(grep '^catalyst' requirements.txt)"
```

The SLURM scripts activate `circrna`, which works once it also has `opencv-python-headless`,
`pandas` and `openpyxl`; change the `source ... activate` line to use `satellite` instead.

## Quick start

```bash

python train.py experiment.name=my_run                            # conf/config.yaml defaults
python train.py experiment.name=my_run model.name=basic_3dcnn      # the 3D-CNN baseline
sbatch slurm/train.sh                                             # from this folder; or train_baseline.sh
```

The material sheets are read from `assets/materials` in the project root (one level above
this folder); `+data.params.material_root=...` points elsewhere. On CPU, set
`experiment.bf16=false` (bf16 is emulated and ~50x slower) and `loader.*.prefetches=0`.

Any config key can be overridden on the command line, e.g.
`experiment.epochs=100 experiment.batch_size=4 experiment.run_idx=1`.

## Layout

```
train.py                  the entry point: run loop + Catalyst runner
requirements.txt          pinned deps, Catalyst from the neuroneural fork
conf/                     the config
slurm/                    train.sh (spectral_spatial), train_baseline.sh (basic_3dcnn)
src/models/               models: spectral_spatial (the paper's), basic_3dcnn (baseline)
src/datasets/             datasets (hsi), and their own README
src/data_generation/      the scene generator, copied from the project's src/
```

## Config sections breakdown

| section | what it sets |
|---|---|
| `data` | dataset `name` (a module in `src/datasets/`) and `params`: split sizes and seeds, generator settings under `sim` |
| `experiment` | experiment `name`; experiment params like `epochs`, `batch_size` (per rank), `run_idx` (the run or runs to train), `selection`, `test_every_epoch`; random seeds; utility params like `resume`, `save_state`, `cudnn_benchmark`, `bf16`, `channels_last`, `compile`; `paths` (`logdir`, `init_weights`: a checkpoint to start the model from) |
| `model` | model `name` (a module in `src/models/`), `init_seed`, `params` |
| `optimizer` | `base_lr` (per rank), and `scale_lr` for `base_lr * sqrt(world_size)` |
| `loader` | `num_workers`, `prefetch_factor`, `persistent_workers`, `prefetches` for `train` and `eval` |
| `wandb` | `project` / `entity` / `name`; off unless you uncomment the logger |
| `runtime` | filled in by `train.py` at startup — run index, world size, the scaled lr, the resume checkpoint, and the dataset's channel and class counts. Not for hand editing; it is there so the logged config says what ran |
| `stream_log` | config for stdout logger that fills each run's `more/run.log` |


## Logs

Everything lands under `logs/<experiment.name>/`.

| file | what |
|---|---|
| `runs.csv` | one row per finished run: split sizes, best epoch, valid and **test** metrics (`test_*`, `test_satellite_*`), seeds, wall time |
| `summary.json` | mean, std, median and a 95% t interval across runs |
| `config.yaml` | the resolved config of the latest job |
| `curves.png` | every run's training curves in one figure, one line per run |

And per run, in `run_<i>/`:

| file | what |
|---|---|
| `_epoch_.csv` | one row per epoch: `lr`, `momentum` and each loader's `loss` and pixel metrics (`train_loss`, `valid_miou`, ...; `test*_` with `experiment.test_every_epoch=true`) |
| `run_summary.json` | best epoch, best metrics, seeds, wall time |
| `<test>_predictions.npz` | per test loader: `index`, `kind` (0 satellite, 1 shapes, 2 patches), predicted and target masks of every scene, and the `epoch` of the weights |
| `curves.png` | per-epoch LR, loss, mIoU and accuracy, with the run's info |
| `model.best.pth` | weights from the best epoch |
| `model.best_<metric>.pth` | weights from the best epoch of each extra selection |
| `state.last.pth` | weights, optimizer, scheduler and best epochs after the latest epoch, for resuming; deleted once the run finishes |

and the detail, in `run_<i>/more/`:

| file | what |
|---|---|
| `run_config.yaml` | the config plus every seed, enough to reproduce this run |
| `<test>_preds_rank<r>.npz` | each rank's part of `<test>_predictions.npz` |
| `run.log` | everything the run printed, including from spawned DDP ranks |

Rerunning a run clears its directory first — nothing is deleted, the
old artefacts move to `backup_<timestamp>/`. With `experiment.resume=true` it instead continues from an interrupted run, or restarts it if `experiment.save_state=false`.

## Seeds

Five knobs, one per thing that can vary:

| seed | fixes | null means |
|---|---|---|
| `data.params.seed_train` | the training scenes, with the run and epoch: `scene_seed(seed_train, run, epoch, index)` | — |
| `data.params.seed_valid` | the valid set, `scene_seed(seed_valid, index)`: the same scenes every epoch, run and rerun | — |
| `data.params.seed_test` | every test set, likewise | — |
| `experiment.sampler_seed` | the train data order within an epoch (valid and test keep index order), and the base Catalyst reseeds the global RNG from (dropout, augmentation) | — |
| `model.init_seed` | the model initialisation, offset by the run: `init_seed + run_idx` | a random init each run |

All five are recorded in `run_config.yaml`, `run_summary.json` and `runs.csv`.
