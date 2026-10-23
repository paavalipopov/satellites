# Per-pixel material classification of synthetic satellite HSI

Each scene is a [36, 15, 15] cube with a [15, 15] material mask (0 = background, 1-46 =
materials), generated on the fly by `src/data_generation` (a copy of the project's `src/data_generation`).
Training scenes are fresh every epoch; valid and test are fixed sets, chosen by their seeds.
Catalyst runner, Hydra config, single- or multi-GPU on SLURM.

## Quick start

```bash
pip install -r requirements.txt
# pip keeps an installed catalyst when only the pinned commit changes (same version), so force it
pip install --force-reinstall --no-deps "$(grep '^catalyst' requirements.txt)"

python train.py experiment.name=my_run                            # conf/config.yaml defaults
python train.py experiment.name=my_run model.name=basic_3dcnn      # the 3D-CNN baseline
python train.py experiment.name=my_run +data.params.sim.sensor_noise_std=0.05  # any generator setting
python train.py experiment.name=my_run experiment.resume=true     # continue where it stopped
```

The material sheets are read from `assets/materials` in the project root (one level above
this folder); `+data.params.material_root=...` points elsewhere. On CPU, set
`experiment.bf16=false` (bf16 is emulated and ~50x slower) and `loader.*.prefetches=0`.

Any config key can be overridden on the command line, e.g.
`experiment.epochs=100 experiment.batch_size=4 experiment.target_folds=[0]`.

## Layout

```
train.py                  the entry point: CV loop + Catalyst runner
requirements.txt          pinned deps, Catalyst from the neuroneural fork
conf/                     run configs
src/models/               models: spectral_spatial (the paper's), basic_3dcnn (baseline)
src/datasets/             datasets (hsi), and their own README
src/data_generation/      the scene generator, copied from the project's src/
```

## Config sections breakdown

| section | what it sets |
|---|---|
| `data` | dataset `name` (a module in `src/datasets/`) and `params`: split sizes and seeds, generator settings under `sim` |
| `experiment` | experiment `name`; experiment params like `epochs`, `batch_size` (per rank), `cv_folds` (number of runs), `target_folds`, `selection`, `test_every_epoch`; random seeds; utility params like `resume`, `save_state`, `cudnn_benchmark`, `bf16`, `channels_last`, `compile`; `paths` (`logdir`, `init_weights`: a checkpoint to start the model from) |
| `model` | model `name` (a module in `src/models/`), `init_seed`, `params` |
| `optimizer` | `base_lr` (per rank), and `scale_lr` for `base_lr * sqrt(world_size)` |
| `loader` | `num_workers`, `prefetch_factor`, `persistent_workers`, `prefetches` for `train` and `eval` |
| `wandb` | `project` / `entity` / `name`; off unless you uncomment the logger |
| `runtime` | filled in by `train.py` at startup — fold index, world size, the scaled lr, the resume checkpoint, and the dataset's channel and class counts. Not for hand editing; it is there so the logged config says what ran |
| `stream_log` | config for stdout logger that fills `run.log` |

`experiment.name` is the log folder name *and* the wandb run name, so a run is easy to
trace in either direction. `experiment.selection` lists the valid metrics that each pick a
best epoch (lowest loss, highest of the others). The first one picks the checkpoint that is
tested and reported; the others only save their weights (`model.best_<metric>.pth`) and best
epoch. `experiment.selection=[miou,loss]` makes the highest valid mIoU the tested model.

A "fold" is one training run: its own stream of training scenes, the same valid and test
sets. `cv_folds=5` gives five runs and a mean and 95% interval across them. #TODO le'ts fix 'fold'

Metrics are per pixel, from the loader's confusion matrix: `accuracy`, `miou` and
`macro_f1` (over classes present in target or prediction, background included),
`fg_accuracy` (material accuracy on foreground pixels) and `fg_f1` (foreground vs background).
`experiment.test_every_epoch=true` also runs test after every epoch (`test.csv`), which gives
every selection its own test scores (`test_miou@miou`, ... in `CV_records.csv`).

## Logs

Everything lands under `logs/<experiment.name>/`.

| file | what |
|---|---|
| `CV_records.csv` | one row per fold: split sizes, best epoch, valid and **test** metrics, seeds, wall time |
| `summary.json` | mean, std, median and a 95% t interval across folds |
| `config.yaml` | the resolved config that actually ran |
| `run.log` | everything printed, including from spawned DDP ranks; `run_fold<i>.log` for a job given `target_folds` |
| `curves.png` | every fold's training curves in one figure, one line per fold |

And per fold, in `fold_<i>/`:

| file | what |
|---|---|
| `train.csv` `valid.csv` | per-epoch `loss` and the pixel metrics |
| `test.csv` | the same for test, with `experiment.test_every_epoch=true` |
| `_epoch_.csv` | per-epoch `lr` and `momentum` |
| `fold_summary.json` | best epoch, best metrics, seeds, wall time |
| `test_predictions.npz` | `index`, predicted and target masks of every test scene, and the `epoch` of the weights |
| `curves.png` | per-epoch LR, loss, mIoU and accuracy, with the run's info |
| `model.best.pth` | weights from the best epoch |
| `model.best_<metric>.pth` | weights from the best epoch of each extra selection |
| `state.last.pth` | weights, optimizer, scheduler and best epochs after the latest epoch, for resuming; deleted once the fold finishes |

and the detail, in `fold_<i>/more/`:

| file | what |
|---|---|
| `fold_config.yaml` | the config plus every seed, enough to reproduce this fold |
| `test_preds_rank<r>.npz` | each rank's part of `test_predictions.npz` |

Rerunning an experiment name clears the fold directory first — nothing is deleted, the
old artefacts move to `backup_<timestamp>/`. With `experiment.resume=true` it instead continues from an interrupted fold, or restarts it if `experiment.save_state=false`.

## Seeds

Five knobs, one per thing that can vary:

| seed | fixes | null means |
|---|---|---|
| `data.params.seed_train` | the training scenes, with the fold and epoch: `scene_seed(seed_train, fold, epoch, index)` | — |
| `data.params.seed_valid` | the valid set, `scene_seed(seed_valid, index)`: the same scenes every epoch, fold and rerun | — |
| `data.params.seed_test` | the test set, likewise | — |
| `experiment.sampler_seed` | the data order within an epoch, and the base Catalyst reseeds the global RNG from (dropout, augmentation) | — |
| `model.init_seed` | the model initialisation | a different init each fold |

All five are recorded in `fold_config.yaml`, `fold_summary.json` and `CV_records.csv`.
