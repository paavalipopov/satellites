# Stage 2: shape (original brightness handling: automatic max scaling)

Models: `original`, `original_retrained`, `fg3`, `fg10`, `fg3_dice`, `halo`, `lightnoise` (3 runs in the first). Choices: see `python scripts/compare_pipeline.py --help`. Higher is better for iou, ari, found and faint_found; lower for false_px. The true companion has 18 px. By hand = the notebook's reference (a person chose the object and its scale).

## Reading `solar`

### Each model: no step, automatic max scaling, by hand, and its best setting (best = highest iou + ari; a heuristic, look at the other columns)

| model | row | setting | outline_iou | ari_materials | false_px | satellite_px_found | companion_px_found | faint_found |
|---|---|---|---|---|---|---|---|---|
| original | automatic max scaling only |  | 0.6 | 0.04 | 0 | 27 | 0 | 0 |
| original | by hand |  | 0.6 | 0.04 | 0 | 27 | 0 |  |
| original | best setting | single / fill@0.01 / tidy+smooth / vote 0.7 | 0.81 | 0.4 | 9 | 44 | 0 | 0 |
| original_retrained | automatic max scaling only |  | 0.4 | 0.08 | 0 | 18 | 0 | 0 |
| original_retrained | by hand |  | 0.4 | 0.08 | 0 | 18 | 0 |  |
| original_retrained | best setting | single / fill@0.01 / off / vote 0.7 | 0.81 | 0.54 | 9 | 44 | 0 | 0 |
| fg3 | automatic max scaling only |  | 0.36 | -0.03 | 0 | 16 | 0 | 0 |
| fg3 | by hand |  | 0.36 | -0.03 | 0 | 16 | 0 |  |
| fg3 | best setting | single / fill@0.01 / off / vote 0.7 | 0.81 | 0.44 | 9 | 44 | 0 | 0 |
| fg10 | automatic max scaling only |  | 0.42 | -0.03 | 0 | 19 | 0 | 0 |
| fg10 | by hand |  | 0.42 | -0.03 | 0 | 19 | 0 |  |
| fg10 | best setting | single / fill@0.01 / tidy+smooth / vote 0.7 | 0.81 | 0.44 | 9 | 44 | 0 | 0 |
| fg3_dice | automatic max scaling only |  | 0.33 | 0.12 | 0 | 15 | 0 | 0 |
| fg3_dice | by hand |  | 0.33 | 0.12 | 0 | 15 | 0 |  |
| fg3_dice | best setting | single / fill@0.01 / off / vote 0.5 | 0.81 | 0.53 | 9 | 44 | 0 | 0 |
| halo | automatic max scaling only |  | 0.51 | 0.15 | 0 | 23 | 0 | 0 |
| halo | by hand |  | 0.51 | 0.15 | 0 | 23 | 0 |  |
| halo | best setting | single / fill@0.01 / tidy+smooth / vote 0.7 | 0.81 | 0.75 | 9 | 44 | 0 | 0 |
| lightnoise | automatic max scaling only |  | 0.42 | -0.04 | 0 | 19 | 0 | 0 |
| lightnoise | by hand |  | 0.42 | -0.04 | 0 | 19 | 0 |  |
| lightnoise | best setting | single / fill@0.01 / tidy+smooth / vote 0.5 | 0.81 | 0.42 | 9 | 44 | 0 | 0 |

### What each choice does (mean over all models and all other choices)

`outline`:

| outline | outline_iou | ari_materials | false_px | satellite_px_found | companion_px_found | faint_found |
|---|---|---|---|---|---|---|
| fill@0.001 | 0.51 | 0.38 | 42 | 44 | 0 | 0 |
| fill@0.01 | 0.81 | 0.39 | 9 | 44 | 0 | 0 |
| mask@0.001 | 0.41 | 0.02 | 0 | 18.46 | 0 | 0 |
| mask@0.01 | 0.41 | 0.02 | 0 | 18.46 | 0 | 0 |
| model | 0.41 | 0.02 | 0 | 18.46 | 0 | 0 |

`cleanup`:

| cleanup | outline_iou | ari_materials | false_px | satellite_px_found | companion_px_found | faint_found |
|---|---|---|---|---|---|---|
| off | 0.52 | 0.18 | 10.2 | 29.17 | 0 | 0 |
| tidy | 0.5 | 0.14 | 10.2 | 28.43 | 0 | 0 |
| tidy+smooth | 0.5 | 0.17 | 10.2 | 28.43 | 0 | 0 |

`vote`:

| vote | outline_iou | ari_materials | false_px | satellite_px_found | companion_px_found | faint_found |
|---|---|---|---|---|---|---|
| 0.3 | 0.6 | 0.18 | 10.2 | 32.83 | 0 | 0 |
| 0.5 | 0.52 | 0.16 | 10.2 | 29 | 0 | 0 |
| 0.7 | 0.41 | 0.15 | 10.2 | 24.2 | 0 | 0 |

### Every run on its own (mean and std over the runs; no voting)

| model | scaling | outline_iou | outline_iou_std | ari_materials | ari_materials_std |
|---|---|---|---|---|---|
| original | single | 0.53 | 0.08 | 0.19 | 0.06 |
| original_retrained | single | 0.41 | 0.06 | 0.04 | 0.07 |
| fg3 | single | 0.33 | 0.11 | -0.01 | 0.04 |
| fg10 | single | 0.45 | 0.1 | 0.02 | 0.04 |
| fg3_dice | single | 0.36 | 0.05 | 0.09 | 0.02 |
| halo | single | 0.48 | 0.11 | 0.18 | 0.03 |
| lightnoise | single | 0.44 | 0.07 | 0.02 | 0.03 |

### A window with nothing in it: labelled pixels (all false)

| model | single |
|---|---|
| fg10 | 0 |
| fg3 | 0 |
| fg3_dice | 0 |
| halo | 0 |
| lightnoise | 0 |
| original | 0 |
| original_retrained | 0 |

## Reading `radiometry`

### Each model: no step, automatic max scaling, by hand, and its best setting (best = highest iou + ari; a heuristic, look at the other columns)

| model | row | setting | outline_iou | ari_materials | false_px | satellite_px_found | companion_px_found | faint_found |
|---|---|---|---|---|---|---|---|---|
| original | automatic max scaling only |  | 0.03 | 0.71 | 1458 | 38 | 1 | 1 |
| original | by hand |  | 0.03 | 0.71 | 1458 | 38 | 1 |  |
| original | best setting | single / mask@0.01 / tidy+smooth / vote 0.3 | 0.89 | 0.89 | 0 | 40 | 0 | 0 |
| original_retrained | automatic max scaling only |  | 0.82 | 0.64 | 0 | 37 | 0 | 0 |
| original_retrained | by hand |  | 0.82 | 0.64 | 0 | 37 | 0 |  |
| original_retrained | best setting | single / mask@0.01 / tidy+smooth / vote 0.3 | 0.87 | 0.82 | 0 | 39 | 0 | 0 |
| fg3 | automatic max scaling only |  | 0.71 | 0.57 | 0 | 32 | 0 | 0 |
| fg3 | by hand |  | 0.71 | 0.57 | 0 | 32 | 0 |  |
| fg3 | best setting | single / mask@0.01 / tidy / vote 0.3 | 0.89 | 0.91 | 0 | 40 | 0 | 0 |
| fg10 | automatic max scaling only |  | 0.03 | 0.34 | 1377 | 37 | 1 | 1 |
| fg10 | by hand |  | 0.03 | 0.34 | 1377 | 37 | 1 |  |
| fg10 | best setting | single / mask@0.01 / tidy+smooth / vote 0.3 | 0.96 | 0.38 | 0 | 43 | 0 | 0 |
| fg3_dice | automatic max scaling only |  | 0.07 | 0.54 | 507 | 35 | 1 | 1 |
| fg3_dice | by hand |  | 0.07 | 0.54 | 507 | 35 | 1 |  |
| fg3_dice | best setting | single / mask@0.01 / tidy+smooth / vote 0.3 | 0.84 | 0.69 | 0 | 38 | 0 | 0 |
| halo | automatic max scaling only |  | 0.04 | 0.18 | 814 | 33 | 1 | 1 |
| halo | by hand |  | 0.04 | 0.18 | 814 | 33 | 1 |  |
| halo | best setting | single / mask@0.01 / tidy+smooth / vote 0.3 | 0.91 | 0.28 | 0 | 41 | 0 | 0 |
| lightnoise | automatic max scaling only |  | 0.13 | 0.67 | 0 | 6 | 0 | 0 |
| lightnoise | by hand |  | 0.13 | 0.67 | 0 | 6 | 0 |  |
| lightnoise | best setting | single / fill@0.01 / tidy / vote 0.5 | 0.8 | 0.02 | 10 | 44 | 0 | 0 |

### What each choice does (mean over all models and all other choices)

`outline`:

| outline | outline_iou | ari_materials | false_px | satellite_px_found | companion_px_found | faint_found |
|---|---|---|---|---|---|---|
| fill@0.001 | 0.49 | 0.48 | 44 | 44 | 0 | 0 |
| fill@0.01 | 0.8 | 0.48 | 10 | 44 | 0 | 0 |
| mask@0.001 | 0.63 | 0.47 | 0 | 28.33 | 0 | 0 |
| mask@0.01 | 0.63 | 0.47 | 0 | 28.33 | 0 | 0 |
| model | 0.2 | 0.44 | 701.65 | 30.94 | 0.57 | 0.57 |

`cleanup`:

| cleanup | outline_iou | ari_materials | false_px | satellite_px_found | companion_px_found | faint_found |
|---|---|---|---|---|---|---|
| off | 0.55 | 0.47 | 139.37 | 34.77 | 0.11 | 0.11 |
| tidy | 0.55 | 0.46 | 157.01 | 35.3 | 0.11 | 0.11 |
| tidy+smooth | 0.55 | 0.46 | 157.01 | 35.3 | 0.11 | 0.11 |

`vote`:

| vote | outline_iou | ari_materials | false_px | satellite_px_found | companion_px_found | faint_found |
|---|---|---|---|---|---|---|
| 0.3 | 0.58 | 0.56 | 237.82 | 39.07 | 0.17 | 0.17 |
| 0.5 | 0.59 | 0.51 | 141.16 | 36.63 | 0.11 | 0.11 |
| 0.7 | 0.49 | 0.33 | 74.41 | 29.67 | 0.06 | 0.06 |

### Every run on its own (mean and std over the runs; no voting)

| model | scaling | outline_iou | outline_iou_std | ari_materials | ari_materials_std |
|---|---|---|---|---|---|
| original | single | 0.02 | 0.01 | 0.54 | 0.27 |
| original_retrained | single | 0.47 | 0.32 | 0.35 | 0.25 |
| fg3 | single | 0.4 | 0.3 | 0.42 | 0.21 |
| fg10 | single | 0.03 | 0 | 0.57 | 0.21 |
| fg3_dice | single | 0.28 | 0.32 | 0.4 | 0.14 |
| halo | single | 0.27 | 0.34 | 0.17 | 0.08 |
| lightnoise | single | 0.13 | 0 | 0.67 | 0 |

### A window with nothing in it: labelled pixels (all false)

| model | single |
|---|---|
| fg10 | 0 |
| fg3 | 0 |
| fg3_dice | 0 |
| halo | 0 |
| lightnoise | 0 |
| original | 0 |
| original_retrained | 0 |
