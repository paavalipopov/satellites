# Stage 1: brightness handling (model outline, no cleanup, vote 0.5)

Models: `original`, `original_retrained`, `aug`, `norm`, `aug_norm`, `gain_p99`, `gain_flux`, `gain_bandflux`, `gain_pixel`, `pair_max1`, `pair_mean`, `pair_p99`, `pair_max1_aug`, `pair_mean_aug`, `pair_p99_aug` (3 runs in the first). Choices: see `python scripts/compare_pipeline.py --help`. Higher is better for iou, ari, found and faint_found; lower for false_px. The true companion has 18 px. By hand = the notebook's reference (a person chose the object and its scale).

## Reading `solar`

### Each model: no step, automatic max scaling, by hand, and its best setting (best = highest iou + ari; a heuristic, look at the other columns)

| model | row | setting | outline_iou | ari_materials | false_px | satellite_px_found | companion_px_found | faint_found |
|---|---|---|---|---|---|---|---|---|
| original | today (no step) |  | 0.02 | 0.15 | 1634 | 33 | 1 | 1 |
| original | automatic max scaling only |  | 0.6 | 0.04 | 0 | 27 | 0 | 0 |
| original | by hand |  | 0.6 | 0.04 | 0 | 27 | 0 |  |
| original | best setting | single_p99 / model / off / vote 0.5 | 0.58 | 0.11 | 0 | 26 | 0 | 0 |
| original_retrained | today (no step) |  | 0.07 | 0.09 | 612 | 44 | 0 | 0 |
| original_retrained | automatic max scaling only |  | 0.4 | 0.08 | 0 | 18 | 0 | 0 |
| original_retrained | by hand |  | 0.4 | 0.08 | 0 | 18 | 0 |  |
| original_retrained | best setting | per_object_mean / model / off / vote 0.5 | 0.2 | 0.35 | 35 | 16 | 0 | 0 |
| aug | today (no step) |  | 0.01 | 0.29 | 1355 | 18 | 1 | 1 |
| aug | automatic max scaling only |  | 0.53 | 0.13 | 0 | 24 | 0 | 0 |
| aug | by hand |  | 0.53 | 0.13 | 0 | 24 | 0 |  |
| aug | best setting | single_mean / model / off / vote 0.5 | 0.82 | 0.1 | 0 | 37 | 0 | 0 |
| norm | today (no step) |  | 0.58 | 0.26 | 0 | 26 | 0 | 0 |
| norm | automatic max scaling only |  | 0.58 | 0.26 | 0 | 26 | 0 | 0 |
| norm | by hand |  | 0.58 | 0.26 | 0 | 26 | 0 |  |
| norm | best setting | none / model / off / vote 0.5 | 0.58 | 0.26 | 0 | 26 | 0 | 0 |
| aug_norm | today (no step) |  | 0.56 | 0.13 | 0 | 25 | 0 | 0 |
| aug_norm | automatic max scaling only |  | 0.56 | 0.13 | 0 | 25 | 0 | 0 |
| aug_norm | by hand |  | 0.56 | 0.13 | 0 | 25 | 0 |  |
| aug_norm | best setting | none / model / off / vote 0.5 | 0.56 | 0.13 | 0 | 25 | 0 | 0 |
| gain_p99 | today (no step) |  | 0.38 | 0.13 | 0 | 17 | 0 | 0 |
| gain_p99 | automatic max scaling only |  | 0.38 | 0.13 | 0 | 17 | 0 | 0 |
| gain_p99 | by hand |  | 0.38 | 0.13 | 0 | 17 | 0 |  |
| gain_p99 | best setting | per_object_p99 / model / off / vote 0.5 | 0.38 | 0.13 | 0 | 17 | 0 | 0 |
| gain_flux | today (no step) |  | 0.73 | 0.23 | 0 | 33 | 0 | 0 |
| gain_flux | automatic max scaling only |  | 0.73 | 0.23 | 0 | 33 | 0 | 0 |
| gain_flux | by hand |  | 0.73 | 0.23 | 0 | 33 | 0 |  |
| gain_flux | best setting | none / model / off / vote 0.5 | 0.73 | 0.23 | 0 | 33 | 0 | 0 |
| gain_bandflux | today (no step) |  | 0.65 | 0.18 | 1 | 30 | 0 | 0 |
| gain_bandflux | automatic max scaling only |  | 0.65 | 0.18 | 1 | 30 | 0 | 0 |
| gain_bandflux | by hand |  | 0.65 | 0.18 | 1 | 30 | 0 |  |
| gain_bandflux | best setting | none / model / off / vote 0.5 | 0.65 | 0.18 | 1 | 30 | 0 | 0 |
| gain_pixel | today (no step) |  | 0.39 | 0.34 | 31 | 29 | 1 | 1 |
| gain_pixel | automatic max scaling only |  | 0.39 | 0.34 | 31 | 29 | 1 | 1 |
| gain_pixel | by hand |  | 0.39 | 0.34 | 31 | 29 | 1 |  |
| gain_pixel | best setting | per_object_p99 / model / off / vote 0.5 | 0.64 | 0.35 | 0 | 29 | 0 | 0 |
| pair_max1 | today (no step) |  | 0.01 | -0.18 | 1575 | 17 | 1 | 1 |
| pair_max1 | automatic max scaling only |  | 0.29 | -0.04 | 0 | 13 | 0 | 0 |
| pair_max1 | by hand |  | 0.29 | -0.04 | 0 | 13 | 0 |  |
| pair_max1 | best setting | per_object_mean / model / off / vote 0.5 | 0.65 | 0.1 | 6 | 33 | 0 | 0 |
| pair_mean | today (no step) |  | 0 | -0.04 | 1238 | 0 | 1 | 1 |
| pair_mean | automatic max scaling only |  | 0.22 | -0.16 | 0 | 10 | 0 | 0 |
| pair_mean | by hand |  | 0.22 | -0.16 | 0 | 10 | 0 |  |
| pair_mean | best setting | single_mean / model / off / vote 0.5 | 0.02 | 0.13 | 0 | 1 | 0 | 0 |
| pair_p99 | today (no step) |  | 0.03 | 0.24 | 1624 | 42 | 1 | 1 |
| pair_p99 | automatic max scaling only |  | 0.29 | -0.02 | 0 | 13 | 0 | 0 |
| pair_p99 | by hand |  | 0.29 | -0.02 | 0 | 13 | 0 |  |
| pair_p99 | best setting | per_object_mean / model / off / vote 0.5 | 0.65 | 0.06 | 4 | 32 | 0 | 0 |
| pair_max1_aug | today (no step) |  | 0.31 | 0.29 | 98 | 44 | 0 | 0 |
| pair_max1_aug | automatic max scaling only |  | 0.31 | 0.28 | 0 | 14 | 0 | 0 |
| pair_max1_aug | by hand |  | 0.31 | 0.28 | 0 | 14 | 0 |  |
| pair_max1_aug | best setting | single_mean / model / off / vote 0.5 | 0.49 | 0.27 | 0 | 22 | 0 | 0 |
| pair_mean_aug | today (no step) |  | 0.38 | 0.6 | 67 | 43 | 0 | 0 |
| pair_mean_aug | automatic max scaling only |  | 0.16 | 0.08 | 0 | 7 | 0 | 0 |
| pair_mean_aug | by hand |  | 0.16 | 0.08 | 0 | 7 | 0 |  |
| pair_mean_aug | best setting | none / model / off / vote 0.5 | 0.38 | 0.6 | 67 | 43 | 0 | 0 |
| pair_p99_aug | today (no step) |  | 0.25 | 0.65 | 117 | 40 | 0 | 0 |
| pair_p99_aug | automatic max scaling only |  | 0.31 | 0.11 | 0 | 14 | 0 | 0 |
| pair_p99_aug | by hand |  | 0.31 | 0.11 | 0 | 14 | 0 |  |
| pair_p99_aug | best setting | none / model / off / vote 0.5 | 0.25 | 0.65 | 117 | 40 | 0 | 0 |

### What each choice does (mean over all models and all other choices)

`scaling`:

| scaling | outline_iou | ari_materials | false_px | satellite_px_found | companion_px_found | faint_found |
|---|---|---|---|---|---|---|
| none | 0.29 | 0.22 | 556.8 | 29.4 | 0.4 | 0.4 |
| per_object | 0.44 | 0.12 | 0.07 | 20 | 0 | 0 |
| per_object_mean | 0.49 | 0.2 | 6.8 | 24.2 | 0 | 0 |
| per_object_p99 | 0.43 | 0.14 | 0.07 | 19.27 | 0 | 0 |
| single | 0.43 | 0.12 | 2.13 | 20 | 0.07 | 0.07 |
| single_mean | 0.35 | 0.2 | 361.8 | 24.2 | 0.33 | 0.33 |
| single_p99 | 0.41 | 0.14 | 2.13 | 19.27 | 0.07 | 0.07 |

### Every model with every scaling (model outline, no cleanup, vote 0.5)

`outline_iou`:

| model | none | per_object | per_object_mean | per_object_p99 | single | single_mean | single_p99 |
|---|---|---|---|---|---|---|---|
| aug | 0.01 | 0.53 | 0.82 | 0.53 | 0.53 | 0.82 | 0.53 |
| aug_norm | 0.56 | 0.56 | 0.56 | 0.56 | 0.56 | 0.56 | 0.56 |
| gain_bandflux | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 |
| gain_flux | 0.73 | 0.73 | 0.73 | 0.73 | 0.73 | 0.73 | 0.73 |
| gain_p99 | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 |
| gain_pixel | 0.39 | 0.64 | 0.64 | 0.64 | 0.39 | 0.39 | 0.39 |
| norm | 0.58 | 0.58 | 0.58 | 0.58 | 0.58 | 0.58 | 0.58 |
| original | 0.02 | 0.6 | 0.38 | 0.58 | 0.6 | 0.02 | 0.58 |
| original_retrained | 0.07 | 0.4 | 0.2 | 0.4 | 0.4 | 0.01 | 0.4 |
| pair_max1 | 0.01 | 0.29 | 0.65 | 0.29 | 0.29 | 0.02 | 0.29 |
| pair_max1_aug | 0.31 | 0.31 | 0.49 | 0.29 | 0.31 | 0.49 | 0.29 |
| pair_mean | 0 | 0.22 | 0.02 | 0.07 | 0.22 | 0.02 | 0.07 |
| pair_mean_aug | 0.38 | 0.16 | 0.11 | 0.09 | 0.16 | 0.11 | 0.09 |
| pair_p99 | 0.03 | 0.29 | 0.65 | 0.29 | 0.29 | 0.03 | 0.29 |
| pair_p99_aug | 0.25 | 0.31 | 0.41 | 0.33 | 0.31 | 0.41 | 0.33 |

`ari_materials`:

| model | none | per_object | per_object_mean | per_object_p99 | single | single_mean | single_p99 |
|---|---|---|---|---|---|---|---|
| aug | 0.29 | 0.13 | 0.1 | 0.15 | 0.13 | 0.1 | 0.15 |
| aug_norm | 0.13 | 0.13 | 0.13 | 0.13 | 0.13 | 0.13 | 0.13 |
| gain_bandflux | 0.18 | 0.18 | 0.18 | 0.18 | 0.18 | 0.18 | 0.18 |
| gain_flux | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 |
| gain_p99 | 0.13 | 0.13 | 0.13 | 0.13 | 0.13 | 0.13 | 0.13 |
| gain_pixel | 0.34 | 0.35 | 0.35 | 0.35 | 0.34 | 0.34 | 0.34 |
| norm | 0.26 | 0.26 | 0.26 | 0.26 | 0.26 | 0.26 | 0.26 |
| original | 0.15 | 0.04 | 0.18 | 0.11 | 0.04 | 0.18 | 0.11 |
| original_retrained | 0.09 | 0.08 | 0.35 | 0.14 | 0.08 | 0.31 | 0.14 |
| pair_max1 | -0.18 | -0.04 | 0.1 | 0.03 | -0.04 | 0.1 | 0.03 |
| pair_max1_aug | 0.29 | 0.28 | 0.27 | 0.31 | 0.28 | 0.27 | 0.31 |
| pair_mean | -0.04 | -0.16 | 0.13 | -0.09 | -0.16 | 0.13 | -0.09 |
| pair_mean_aug | 0.6 | 0.08 | 0.27 | 0.17 | 0.08 | 0.27 | 0.17 |
| pair_p99 | 0.24 | -0.02 | 0.06 | -0.02 | -0.02 | 0.07 | -0.02 |
| pair_p99_aug | 0.65 | 0.11 | 0.23 | 0.08 | 0.11 | 0.23 | 0.08 |

`faint_found`:

| model | none | per_object | per_object_mean | per_object_p99 | single | single_mean | single_p99 |
|---|---|---|---|---|---|---|---|
| aug | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| aug_norm | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_bandflux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_p99 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_pixel | 1 | 0 | 0 | 0 | 1 | 1 | 1 |
| norm | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| original | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| original_retrained | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| pair_max1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| pair_max1_aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_mean | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_mean_aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_p99 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| pair_p99_aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

`companion_px_found`:

| model | none | per_object | per_object_mean | per_object_p99 | single | single_mean | single_p99 |
|---|---|---|---|---|---|---|---|
| aug | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| aug_norm | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_bandflux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_p99 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_pixel | 1 | 0 | 0 | 0 | 1 | 1 | 1 |
| norm | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| original | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| original_retrained | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| pair_max1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| pair_max1_aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_mean | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_mean_aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_p99 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| pair_p99_aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Every run on its own (mean and std over the runs; no voting)

| model | scaling | outline_iou | outline_iou_std | ari_materials | ari_materials_std |
|---|---|---|---|---|---|
| original | none | 0.02 | 0.01 | 0.16 | 0.16 |
| original | single | 0.53 | 0.08 | 0.19 | 0.06 |
| original | single_mean | 0.02 | 0 | 0.3 | 0.15 |
| original | single_p99 | 0.49 | 0.1 | 0.22 | 0.02 |
| original_retrained | none | 0.06 | 0.02 | 0.13 | 0.09 |
| original_retrained | single | 0.41 | 0.06 | 0.04 | 0.07 |
| original_retrained | single_mean | 0.03 | 0.03 | 0.35 | 0.12 |
| original_retrained | single_p99 | 0.43 | 0.08 | 0.08 | 0.06 |
| aug | none | 0.02 | 0.01 | 0.29 | 0.1 |
| aug | single | 0.59 | 0.14 | 0.18 | 0.07 |
| aug | single_mean | 0.46 | 0.32 | 0.19 | 0.09 |
| aug | single_p99 | 0.6 | 0.16 | 0.18 | 0.06 |
| norm | none | 0.53 | 0.18 | 0.2 | 0.23 |
| norm | single | 0.53 | 0.18 | 0.2 | 0.23 |
| norm | single_mean | 0.53 | 0.18 | 0.2 | 0.23 |
| norm | single_p99 | 0.53 | 0.18 | 0.2 | 0.23 |
| aug_norm | none | 0.54 | 0.1 | 0.22 | 0.05 |
| aug_norm | single | 0.54 | 0.1 | 0.22 | 0.05 |
| aug_norm | single_mean | 0.54 | 0.1 | 0.22 | 0.05 |
| aug_norm | single_p99 | 0.54 | 0.1 | 0.22 | 0.05 |
| gain_p99 | none | 0.24 | 0.13 | 0.14 | 0.15 |
| gain_p99 | single | 0.24 | 0.13 | 0.14 | 0.15 |
| gain_p99 | single_mean | 0.24 | 0.13 | 0.14 | 0.15 |
| gain_p99 | single_p99 | 0.24 | 0.13 | 0.14 | 0.15 |
| gain_flux | none | 0.68 | 0.15 | 0.15 | 0.08 |
| gain_flux | single | 0.68 | 0.15 | 0.15 | 0.08 |
| gain_flux | single_mean | 0.68 | 0.15 | 0.15 | 0.08 |
| gain_flux | single_p99 | 0.68 | 0.15 | 0.15 | 0.08 |
| gain_bandflux | none | 0.63 | 0.19 | 0.21 | 0.06 |
| gain_bandflux | single | 0.63 | 0.19 | 0.21 | 0.06 |
| gain_bandflux | single_mean | 0.63 | 0.19 | 0.21 | 0.06 |
| gain_bandflux | single_p99 | 0.63 | 0.19 | 0.21 | 0.06 |
| gain_pixel | none | 0.36 | 0.15 | 0.32 | 0.08 |
| gain_pixel | single | 0.36 | 0.15 | 0.32 | 0.08 |
| gain_pixel | single_mean | 0.36 | 0.15 | 0.32 | 0.08 |
| gain_pixel | single_p99 | 0.36 | 0.15 | 0.32 | 0.08 |
| pair_max1 | none | 0.01 | 0.01 | 0 | 0.03 |
| pair_max1 | single | 0.33 | 0.12 | -0.01 | 0.04 |
| pair_max1 | single_mean | 0.02 | 0.01 | 0.28 | 0.1 |
| pair_max1 | single_p99 | 0.31 | 0.16 | 0.09 | 0.1 |
| pair_mean | none | 0 | 0 | -0.02 | 0.02 |
| pair_mean | single | 0.3 | 0.22 | -0.01 | 0.21 |
| pair_mean | single_mean | 0.13 | 0.16 | 0.04 | 0.13 |
| pair_mean | single_p99 | 0.16 | 0.19 | -0.04 | 0.04 |
| pair_p99 | none | 0.02 | 0 | 0.3 | 0.16 |
| pair_p99 | single | 0.19 | 0.12 | 0.08 | 0.11 |
| pair_p99 | single_mean | 0.18 | 0.22 | 0.15 | 0 |
| pair_p99 | single_p99 | 0.2 | 0.13 | 0.03 | 0.05 |
| pair_max1_aug | none | 0.2 | 0.13 | 0.33 | 0.03 |
| pair_max1_aug | single | 0.38 | 0.13 | 0.24 | 0.06 |
| pair_max1_aug | single_mean | 0.3 | 0.23 | 0.29 | 0.24 |
| pair_max1_aug | single_p99 | 0.36 | 0.11 | 0.23 | 0.07 |
| pair_mean_aug | none | 0.18 | 0.14 | 0.6 | 0.19 |
| pair_mean_aug | single | 0.21 | 0.05 | 0.11 | 0.28 |
| pair_mean_aug | single_mean | 0.18 | 0.14 | 0.16 | 0.16 |
| pair_mean_aug | single_p99 | 0.19 | 0.08 | 0.12 | 0.26 |
| pair_p99_aug | none | 0.12 | 0.09 | 0.26 | 0.33 |
| pair_p99_aug | single | 0.41 | 0.11 | 0.13 | 0.09 |
| pair_p99_aug | single_mean | 0.34 | 0.29 | 0.21 | 0.02 |
| pair_p99_aug | single_p99 | 0.4 | 0.1 | 0.14 | 0.1 |

### A window with nothing in it: labelled pixels (all false)

| model | none | per_object | per_object_mean | per_object_p99 | single | single_mean | single_p99 |
|---|---|---|---|---|---|---|---|
| aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| aug_norm | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_bandflux | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_flux | 29 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_p99 | 1647 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_pixel | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| norm | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| original | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| original_retrained | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_max1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_max1_aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_mean | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_mean_aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_p99 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_p99_aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Reading `radiometry`

### Each model: no step, automatic max scaling, by hand, and its best setting (best = highest iou + ari; a heuristic, look at the other columns)

| model | row | setting | outline_iou | ari_materials | false_px | satellite_px_found | companion_px_found | faint_found |
|---|---|---|---|---|---|---|---|---|
| original | today (no step) |  | 0.67 | 0.13 | 0 | 30 | 0 | 0 |
| original | automatic max scaling only |  | 0.03 | 0.71 | 1458 | 38 | 1 | 1 |
| original | by hand |  | 0.03 | 0.71 | 1458 | 38 | 1 |  |
| original | best setting | per_object_p99 / model / off / vote 0.5 | 0.83 | 0.77 | 2 | 39 | 0 | 0 |
| original_retrained | today (no step) |  | 0.71 | 0.38 | 0 | 32 | 0 | 0 |
| original_retrained | automatic max scaling only |  | 0.82 | 0.64 | 0 | 37 | 0 | 0 |
| original_retrained | by hand |  | 0.82 | 0.64 | 0 | 37 | 0 |  |
| original_retrained | best setting | single / model / off / vote 0.5 | 0.82 | 0.64 | 0 | 37 | 0 | 0 |
| aug | today (no step) |  | 0.76 | 0.54 | 0 | 34 | 0 | 0 |
| aug | automatic max scaling only |  | 0.89 | 0.9 | 0 | 40 | 0 | 0 |
| aug | by hand |  | 0.89 | 0.9 | 0 | 40 | 0 |  |
| aug | best setting | single / model / off / vote 0.5 | 0.89 | 0.9 | 0 | 40 | 0 | 0 |
| norm | today (no step) |  | 0.57 | 0.09 | 6 | 29 | 0 | 0 |
| norm | automatic max scaling only |  | 0.57 | 0.09 | 6 | 29 | 0 | 0 |
| norm | by hand |  | 0.57 | 0.09 | 6 | 29 | 0 |  |
| norm | best setting | per_object_p99 / model / off / vote 0.5 | 0.64 | 0.09 | 0 | 29 | 0 | 0 |
| aug_norm | today (no step) |  | 0.8 | 0.55 | 0 | 36 | 0 | 0 |
| aug_norm | automatic max scaling only |  | 0.8 | 0.55 | 0 | 36 | 0 | 0 |
| aug_norm | by hand |  | 0.8 | 0.55 | 0 | 36 | 0 |  |
| aug_norm | best setting | none / model / off / vote 0.5 | 0.8 | 0.55 | 0 | 36 | 0 | 0 |
| gain_p99 | today (no step) |  | 0.44 | 0.22 | 0 | 20 | 0 | 0 |
| gain_p99 | automatic max scaling only |  | 0.44 | 0.22 | 0 | 20 | 0 | 0 |
| gain_p99 | by hand |  | 0.44 | 0.22 | 0 | 20 | 0 |  |
| gain_p99 | best setting | none / model / off / vote 0.5 | 0.44 | 0.22 | 0 | 20 | 0 | 0 |
| gain_flux | today (no step) |  | 0.66 | 0.13 | 20 | 43 | 0 | 0 |
| gain_flux | automatic max scaling only |  | 0.66 | 0.13 | 20 | 43 | 0 | 0 |
| gain_flux | by hand |  | 0.66 | 0.13 | 20 | 43 | 0 |  |
| gain_flux | best setting | none / model / off / vote 0.5 | 0.66 | 0.13 | 20 | 43 | 0 | 0 |
| gain_bandflux | today (no step) |  | 0.65 | 0.18 | 1 | 30 | 0 | 0 |
| gain_bandflux | automatic max scaling only |  | 0.65 | 0.18 | 1 | 30 | 0 | 0 |
| gain_bandflux | by hand |  | 0.65 | 0.18 | 1 | 30 | 0 |  |
| gain_bandflux | best setting | none / model / off / vote 0.5 | 0.65 | 0.18 | 1 | 30 | 0 | 0 |
| gain_pixel | today (no step) |  | 0.19 | 0.07 | 41 | 15 | 1 | 1 |
| gain_pixel | automatic max scaling only |  | 0.19 | 0.07 | 41 | 15 | 1 | 1 |
| gain_pixel | by hand |  | 0.19 | 0.07 | 41 | 15 | 1 |  |
| gain_pixel | best setting | per_object_p99 / model / off / vote 0.5 | 0.31 | 0.11 | 0 | 14 | 0 | 0 |
| pair_max1 | today (no step) |  | 0.31 | -0.12 | 0 | 14 | 0 | 0 |
| pair_max1 | automatic max scaling only |  | 0.58 | 0.23 | 0 | 26 | 0 | 0 |
| pair_max1 | by hand |  | 0.58 | 0.23 | 0 | 26 | 0 |  |
| pair_max1 | best setting | single_p99 / model / off / vote 0.5 | 0.69 | 0.32 | 0 | 31 | 0 | 0 |
| pair_mean | today (no step) |  | 0.73 | 0.29 | 0 | 33 | 0 | 0 |
| pair_mean | automatic max scaling only |  | 0 | -0.04 | 1089 | 0 | 1 | 1 |
| pair_mean | by hand |  | 0 | -0.04 | 1089 | 0 | 1 |  |
| pair_mean | best setting | none / model / off / vote 0.5 | 0.73 | 0.29 | 0 | 33 | 0 | 0 |
| pair_p99 | today (no step) |  | 0.62 | 0.03 | 0 | 28 | 0 | 0 |
| pair_p99 | automatic max scaling only |  | 0.71 | 0.12 | 0 | 32 | 0 | 0 |
| pair_p99 | by hand |  | 0.71 | 0.12 | 0 | 32 | 0 |  |
| pair_p99 | best setting | per_object_mean / model / off / vote 0.5 | 0.67 | 0.27 | 3 | 32 | 0 | 0 |
| pair_max1_aug | today (no step) |  | 0.82 | 0.49 | 0 | 37 | 0 | 0 |
| pair_max1_aug | automatic max scaling only |  | 0.87 | 0.44 | 0 | 39 | 0 | 0 |
| pair_max1_aug | by hand |  | 0.87 | 0.44 | 0 | 39 | 0 |  |
| pair_max1_aug | best setting | none / model / off / vote 0.5 | 0.82 | 0.49 | 0 | 37 | 0 | 0 |
| pair_mean_aug | today (no step) |  | 0.6 | 0.29 | 0 | 27 | 0 | 0 |
| pair_mean_aug | automatic max scaling only |  | 0.33 | -0.1 | 0 | 15 | 0 | 0 |
| pair_mean_aug | by hand |  | 0.33 | -0.1 | 0 | 15 | 0 |  |
| pair_mean_aug | best setting | none / model / off / vote 0.5 | 0.6 | 0.29 | 0 | 27 | 0 | 0 |
| pair_p99_aug | today (no step) |  | 0.8 | 0.39 | 0 | 36 | 0 | 0 |
| pair_p99_aug | automatic max scaling only |  | 0.82 | 0.36 | 0 | 37 | 0 | 0 |
| pair_p99_aug | by hand |  | 0.82 | 0.36 | 0 | 37 | 0 |  |
| pair_p99_aug | best setting | single_p99 / model / off / vote 0.5 | 0.82 | 0.37 | 0 | 37 | 0 | 0 |

### What each choice does (mean over all models and all other choices)

`scaling`:

| scaling | outline_iou | ari_materials | false_px | satellite_px_found | companion_px_found | faint_found |
|---|---|---|---|---|---|---|
| none | 0.62 | 0.24 | 4.53 | 29.6 | 0.07 | 0.07 |
| per_object | 0.62 | 0.3 | 1.8 | 29.07 | 0 | 0 |
| per_object_mean | 0.57 | 0.29 | 5.93 | 28.6 | 0 | 0 |
| per_object_p99 | 0.62 | 0.3 | 1.53 | 29 | 0 | 0 |
| single | 0.56 | 0.3 | 174.33 | 29.13 | 0.2 | 0.2 |
| single_mean | 0.45 | 0.28 | 286.87 | 28.67 | 0.27 | 0.27 |
| single_p99 | 0.55 | 0.3 | 166.2 | 29.07 | 0.2 | 0.2 |

### Every model with every scaling (model outline, no cleanup, vote 0.5)

`outline_iou`:

| model | none | per_object | per_object_mean | per_object_p99 | single | single_mean | single_p99 |
|---|---|---|---|---|---|---|---|
| aug | 0.76 | 0.89 | 0.85 | 0.89 | 0.89 | 0.85 | 0.89 |
| aug_norm | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| gain_bandflux | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 |
| gain_flux | 0.66 | 0.66 | 0.66 | 0.66 | 0.66 | 0.66 | 0.66 |
| gain_p99 | 0.44 | 0.44 | 0.44 | 0.44 | 0.44 | 0.44 | 0.44 |
| gain_pixel | 0.19 | 0.31 | 0.31 | 0.31 | 0.19 | 0.19 | 0.19 |
| norm | 0.57 | 0.64 | 0.64 | 0.64 | 0.57 | 0.57 | 0.57 |
| original | 0.67 | 0.75 | 0.44 | 0.83 | 0.03 | 0.03 | 0.03 |
| original_retrained | 0.71 | 0.82 | 0.21 | 0.73 | 0.82 | 0.01 | 0.73 |
| pair_max1 | 0.31 | 0.58 | 0.72 | 0.69 | 0.58 | 0.03 | 0.69 |
| pair_max1_aug | 0.82 | 0.87 | 0.82 | 0.87 | 0.87 | 0.82 | 0.87 |
| pair_mean | 0.73 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_mean_aug | 0.6 | 0.33 | 0.44 | 0.29 | 0.33 | 0.44 | 0.29 |
| pair_p99 | 0.62 | 0.71 | 0.67 | 0.69 | 0.71 | 0.37 | 0.69 |
| pair_p99_aug | 0.8 | 0.82 | 0.82 | 0.82 | 0.82 | 0.82 | 0.82 |

`ari_materials`:

| model | none | per_object | per_object_mean | per_object_p99 | single | single_mean | single_p99 |
|---|---|---|---|---|---|---|---|
| aug | 0.54 | 0.9 | 0.77 | 0.9 | 0.9 | 0.77 | 0.9 |
| aug_norm | 0.55 | 0.55 | 0.55 | 0.55 | 0.55 | 0.55 | 0.55 |
| gain_bandflux | 0.18 | 0.18 | 0.18 | 0.18 | 0.18 | 0.18 | 0.18 |
| gain_flux | 0.13 | 0.13 | 0.13 | 0.13 | 0.13 | 0.13 | 0.13 |
| gain_p99 | 0.22 | 0.22 | 0.22 | 0.22 | 0.22 | 0.22 | 0.22 |
| gain_pixel | 0.07 | 0.11 | 0.11 | 0.11 | 0.07 | 0.07 | 0.07 |
| norm | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 |
| original | 0.13 | 0.7 | 0.84 | 0.77 | 0.71 | 0.85 | 0.78 |
| original_retrained | 0.38 | 0.64 | 0.4 | 0.45 | 0.64 | 0.36 | 0.45 |
| pair_max1 | -0.12 | 0.23 | 0.2 | 0.32 | 0.23 | 0.2 | 0.32 |
| pair_max1_aug | 0.49 | 0.44 | 0.35 | 0.44 | 0.44 | 0.35 | 0.44 |
| pair_mean | 0.29 | 0 | 0 | 0 | -0.04 | 0 | -0.04 |
| pair_mean_aug | 0.29 | -0.1 | -0.01 | -0.13 | -0.1 | -0.01 | -0.13 |
| pair_p99 | 0.03 | 0.12 | 0.27 | 0.16 | 0.12 | 0.27 | 0.16 |
| pair_p99_aug | 0.39 | 0.36 | 0.2 | 0.37 | 0.36 | 0.2 | 0.37 |

`faint_found`:

| model | none | per_object | per_object_mean | per_object_p99 | single | single_mean | single_p99 |
|---|---|---|---|---|---|---|---|
| aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| aug_norm | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_bandflux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_p99 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_pixel | 1 | 0 | 0 | 0 | 1 | 1 | 1 |
| norm | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| original | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| original_retrained | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| pair_max1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| pair_max1_aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_mean | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| pair_mean_aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_p99 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_p99_aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

`companion_px_found`:

| model | none | per_object | per_object_mean | per_object_p99 | single | single_mean | single_p99 |
|---|---|---|---|---|---|---|---|
| aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| aug_norm | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_bandflux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_flux | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_p99 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_pixel | 1 | 0 | 0 | 0 | 1 | 1 | 1 |
| norm | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| original | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| original_retrained | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| pair_max1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| pair_max1_aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_mean | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| pair_mean_aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_p99 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_p99_aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Every run on its own (mean and std over the runs; no voting)

| model | scaling | outline_iou | outline_iou_std | ari_materials | ari_materials_std |
|---|---|---|---|---|---|
| original | none | 0.61 | 0.09 | 0.21 | 0.13 |
| original | single | 0.02 | 0.01 | 0.54 | 0.27 |
| original | single_mean | 0.02 | 0.01 | 0.52 | 0.23 |
| original | single_p99 | 0.02 | 0.01 | 0.56 | 0.2 |
| original_retrained | none | 0.73 | 0.05 | 0.39 | 0.08 |
| original_retrained | single | 0.47 | 0.32 | 0.35 | 0.25 |
| original_retrained | single_mean | 0.02 | 0.02 | 0.45 | 0.11 |
| original_retrained | single_p99 | 0.45 | 0.3 | 0.3 | 0.27 |
| aug | none | 0.59 | 0.29 | 0.43 | 0.09 |
| aug | single | 0.78 | 0.11 | 0.56 | 0.24 |
| aug | single_mean | 0.41 | 0.26 | 0.53 | 0.05 |
| aug | single_p99 | 0.77 | 0.11 | 0.56 | 0.24 |
| norm | none | 0.36 | 0.29 | 0.2 | 0.13 |
| norm | single | 0.36 | 0.29 | 0.2 | 0.13 |
| norm | single_mean | 0.36 | 0.29 | 0.2 | 0.13 |
| norm | single_p99 | 0.36 | 0.29 | 0.2 | 0.13 |
| aug_norm | none | 0.64 | 0.28 | 0.45 | 0.18 |
| aug_norm | single | 0.64 | 0.28 | 0.45 | 0.18 |
| aug_norm | single_mean | 0.64 | 0.28 | 0.45 | 0.18 |
| aug_norm | single_p99 | 0.64 | 0.28 | 0.45 | 0.18 |
| gain_p99 | none | 0.46 | 0.29 | 0.34 | 0.2 |
| gain_p99 | single | 0.46 | 0.29 | 0.34 | 0.2 |
| gain_p99 | single_mean | 0.46 | 0.29 | 0.34 | 0.2 |
| gain_p99 | single_p99 | 0.46 | 0.29 | 0.34 | 0.2 |
| gain_flux | none | 0.53 | 0.38 | 0.36 | 0.04 |
| gain_flux | single | 0.53 | 0.38 | 0.36 | 0.04 |
| gain_flux | single_mean | 0.53 | 0.38 | 0.36 | 0.04 |
| gain_flux | single_p99 | 0.53 | 0.38 | 0.36 | 0.04 |
| gain_bandflux | none | 0.63 | 0.19 | 0.21 | 0.06 |
| gain_bandflux | single | 0.63 | 0.19 | 0.21 | 0.06 |
| gain_bandflux | single_mean | 0.63 | 0.19 | 0.21 | 0.06 |
| gain_bandflux | single_p99 | 0.63 | 0.19 | 0.21 | 0.06 |
| gain_pixel | none | 0.21 | 0.1 | 0.16 | 0.1 |
| gain_pixel | single | 0.21 | 0.1 | 0.16 | 0.1 |
| gain_pixel | single_mean | 0.21 | 0.1 | 0.16 | 0.1 |
| gain_pixel | single_p99 | 0.21 | 0.1 | 0.16 | 0.1 |
| pair_max1 | none | 0.42 | 0.17 | 0.09 | 0.25 |
| pair_max1 | single | 0.51 | 0.33 | 0.31 | 0.28 |
| pair_max1 | single_mean | 0.03 | 0.02 | 0.21 | 0.09 |
| pair_max1 | single_p99 | 0.53 | 0.35 | 0.25 | 0.23 |
| pair_mean | none | 0.66 | 0.2 | 0.23 | 0.21 |
| pair_mean | single | 0 | 0 | -0.02 | 0.02 |
| pair_mean | single_mean | 0 | 0 | 0 | 0 |
| pair_mean | single_p99 | 0 | 0 | -0.02 | 0.02 |
| pair_p99 | none | 0.33 | 0.26 | 0.12 | 0.09 |
| pair_p99 | single | 0.51 | 0.35 | 0.07 | 0.14 |
| pair_p99 | single_mean | 0.21 | 0.15 | 0.16 | 0.05 |
| pair_p99 | single_p99 | 0.51 | 0.35 | 0.06 | 0.16 |
| pair_max1_aug | none | 0.79 | 0.03 | 0.49 | 0.09 |
| pair_max1_aug | single | 0.82 | 0.04 | 0.44 | 0.19 |
| pair_max1_aug | single_mean | 0.74 | 0.07 | 0.41 | 0.19 |
| pair_max1_aug | single_p99 | 0.82 | 0.04 | 0.44 | 0.19 |
| pair_mean_aug | none | 0.47 | 0.33 | 0.26 | 0.2 |
| pair_mean_aug | single | 0.34 | 0.28 | 0.05 | 0.15 |
| pair_mean_aug | single_mean | 0.4 | 0.3 | 0.14 | 0.11 |
| pair_mean_aug | single_p99 | 0.32 | 0.27 | 0.02 | 0.13 |
| pair_p99_aug | none | 0.77 | 0.07 | 0.35 | 0.17 |
| pair_p99_aug | single | 0.74 | 0.02 | 0.37 | 0.19 |
| pair_p99_aug | single_mean | 0.74 | 0.05 | 0.33 | 0.22 |
| pair_p99_aug | single_p99 | 0.74 | 0.01 | 0.36 | 0.16 |

### A window with nothing in it: labelled pixels (all false)

| model | none | per_object | per_object_mean | per_object_p99 | single | single_mean | single_p99 |
|---|---|---|---|---|---|---|---|
| aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| aug_norm | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_bandflux | 7 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_flux | 18 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_p99 | 1643 | 0 | 0 | 0 | 0 | 0 | 0 |
| gain_pixel | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| norm | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| original | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| original_retrained | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_max1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_max1_aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_mean | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_mean_aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_p99 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_p99_aug | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
