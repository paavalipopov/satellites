# Datasets

`data.name` picks a module in this folder; `data.params` overrides its `default_HPs()`.

## The contract

A `DATASET(params)` object that `train.py` uses through:

- `n_channels`, `n_classes` — `train.py` sizes the model's `in_channels` and `num_classes` from them
- `loaders` — the loader names: train, valid and one per test set
- `split(name, run)` for each of them — a torch `Dataset` per loader

Each split's `__getitem__` receives what `SeededBatchSampler` (`base.py`) yields, `(epoch, indices)`,
and returns `{index: {"input": [C,H,W] float, "label": [H,W] long, "index": index, "kind": int}}`;
`collate_volumes` stacks that into `(inputs [B,C,H,W], masks [B,H,W], indices [B], kinds [B])`;
`kind` is the scene type's position in `SCENE_KINDS` (0 satellite, 1 shapes, 2 patches).
Inputs go to the model as they are: no normalisation.

## hsi.py

Synthetic scenes from `src/data_generation`, generated in the loader workers (~2 ms each on CPU).
A scene is fixed by its seed:

| split | seed | changes with |
|---|---|---|
| train | `scene_seed(seed_train, run, epoch, index)` | every epoch and run |
| valid | `scene_seed(seed_valid, index)` | nothing: the same set every epoch, run and rerun |
| test* | `scene_seed(seed_test, index)`, one loader per `tests` entry | nothing |

`n_train` is the number of scenes per epoch, `n_valid` and `n_test` the fixed set sizes.
`scene` picks the train/valid scene type (`random` draws from the generator's satellite/shapes/patches
mix); `tests` maps each test loader to its own.
`sim` holds every `SimConfig` field, so `config.yaml` records the full generator setting;
override one with e.g. `+data.params.sim.sensor_noise_std=0.05`.
