# Stage 3: combined (brightness + shape)

Models: `original_retrained`, `aug`, `pair_max1_aug`, `fg3`, `halo`, `lightnoise`, `aug_fg3`, `aug_fg3_halo`, `aug_fg3_light`, `aug_fg3_halo_light`, `pair_max1_aug_fg3`, `pair_max1_aug_fg3_halo`, `pair_max1_aug_fg3_light`, `pair_max1_aug_fg3_halo_light` (3 runs in the first). Choices: see `python scripts/compare_pipeline.py --help`. Higher is better for iou, ari, found and faint_found; lower for false_px. The true companion is 1 pixel, 18 px right of the satellite (companion_px_found is 0 or 1). By hand = the notebook's reference (a person chose the object and its scale).

## Reading `radiometry`

### Each model: no step, automatic max scaling, by hand, and its best setting (best = highest iou + ari; a heuristic, look at the other columns)

| model | row | setting | outline_iou | ari_materials | false_px | satellite_px_found | companion_px_found | faint_found |
|---|---|---|---|---|---|---|---|---|
| original_retrained | automatic max scaling only |  | 0.82 | 0.64 | 0 | 37 | 0 | 0 |
| original_retrained | by hand |  | 0.82 | 0.64 | 0 | 37 | 0 |  |
| original_retrained | best setting | single / mask@0.01 / tidy / vote 0.3 | 0.87 | 0.76 | 0 | 39 | 0 | 0 |
| aug | automatic max scaling only |  | 0.89 | 0.9 | 0 | 40 | 0 | 0 |
| aug | by hand |  | 0.89 | 0.9 | 0 | 40 | 0 |  |
| aug | best setting | per_object_faint / mask@0.01+faint / off / vote 0.5 | 0.91 | 0.91 | 0 | 40 | 1 | 1 |
| pair_max1_aug | automatic max scaling only |  | 0.87 | 0.44 | 0 | 39 | 0 | 0 |
| pair_max1_aug | by hand |  | 0.87 | 0.44 | 0 | 39 | 0 |  |
| pair_max1_aug | best setting | single / mask@0.01 / tidy / vote 0.3 | 0.91 | 0.53 | 0 | 41 | 0 | 0 |
| fg3 | automatic max scaling only |  | 0.71 | 0.57 | 0 | 32 | 0 | 0 |
| fg3 | by hand |  | 0.71 | 0.57 | 0 | 32 | 0 |  |
| fg3 | best setting | single / mask@0.01 / tidy / vote 0.3 | 0.89 | 0.91 | 0 | 40 | 0 | 0 |
| halo | automatic max scaling only |  | 0.04 | 0.18 | 814 | 33 | 1 | 1 |
| halo | by hand |  | 0.04 | 0.18 | 814 | 33 | 1 |  |
| halo | best setting | single / mask@0.01 / tidy / vote 0.3 | 0.91 | 0.26 | 0 | 41 | 0 | 0 |
| lightnoise | automatic max scaling only |  | 0.13 | 0.67 | 0 | 6 | 0 | 0 |
| lightnoise | by hand |  | 0.13 | 0.67 | 0 | 6 | 0 |  |
| lightnoise | best setting | single / fill@0.01 / off / vote 0.3 | 0.8 | 0.02 | 10 | 44 | 0 | 0 |
| aug_fg3 | automatic max scaling only |  | 0.71 | 0.84 | 13 | 41 | 0 | 0 |
| aug_fg3 | by hand |  | 0.71 | 0.84 | 13 | 41 | 0 |  |
| aug_fg3 | best setting | per_object_faint / mask@0.01+faint / off / vote 0.3 | 0.91 | 0.85 | 2 | 42 | 1 | 1 |
| aug_fg3_halo | automatic max scaling only |  | 0.85 | 0.19 | 3 | 41 | 0 | 0 |
| aug_fg3_halo | by hand |  | 0.85 | 0.19 | 3 | 41 | 0 |  |
| aug_fg3_halo | best setting | per_object_faint / mask@0.01+faint / off / vote 0.3 | 0.91 | 0.22 | 2 | 42 | 1 | 1 |
| aug_fg3_light | automatic max scaling only |  | 0.84 | 0.25 | 0 | 38 | 0 | 0 |
| aug_fg3_light | by hand |  | 0.84 | 0.25 | 0 | 38 | 0 |  |
| aug_fg3_light | best setting | single / mask@0.01 / tidy / vote 0.3 | 0.93 | 0.24 | 0 | 42 | 0 | 0 |
| aug_fg3_halo_light | automatic max scaling only |  | 0.33 | 0.28 | 0 | 15 | 0 | 0 |
| aug_fg3_halo_light | by hand |  | 0.33 | 0.28 | 0 | 15 | 0 |  |
| aug_fg3_halo_light | best setting | single / fill@0.01 / off / vote 0.5 | 0.8 | 0.34 | 10 | 44 | 0 | 0 |
| pair_max1_aug_fg3 | automatic max scaling only |  | 0.91 | 0.26 | 1 | 42 | 0 | 0 |
| pair_max1_aug_fg3 | by hand |  | 0.91 | 0.26 | 1 | 42 | 0 |  |
| pair_max1_aug_fg3 | best setting | single / mask@0.01 / tidy / vote 0.5 | 0.93 | 0.26 | 0 | 42 | 0 | 0 |
| pair_max1_aug_fg3_halo | automatic max scaling only |  | 0.67 | 0.25 | 0 | 30 | 0 | 0 |
| pair_max1_aug_fg3_halo | by hand |  | 0.67 | 0.25 | 0 | 30 | 0 |  |
| pair_max1_aug_fg3_halo | best setting | single / fill@0.01 / off / vote 0.3 | 0.8 | 0.52 | 10 | 44 | 0 | 0 |
| pair_max1_aug_fg3_light | automatic max scaling only |  | 0.47 | 0.26 | 0 | 21 | 0 | 0 |
| pair_max1_aug_fg3_light | by hand |  | 0.47 | 0.26 | 0 | 21 | 0 |  |
| pair_max1_aug_fg3_light | best setting | single / fill@0.01 / off / vote 0.5 | 0.8 | 0.33 | 10 | 44 | 0 | 0 |
| pair_max1_aug_fg3_halo_light | automatic max scaling only |  | 0.22 | 0.45 | 0 | 10 | 0 | 0 |
| pair_max1_aug_fg3_halo_light | by hand |  | 0.22 | 0.45 | 0 | 10 | 0 |  |
| pair_max1_aug_fg3_halo_light | best setting | single / fill@0.01 / off / vote 0.5 | 0.8 | 0.65 | 10 | 44 | 0 | 0 |

### What each choice does (mean over all models and all other choices)

`scaling`:

| scaling | outline_iou | ari_materials | false_px | satellite_px_found | companion_px_found | faint_found |
|---|---|---|---|---|---|---|
| per_object | 0.73 | 0.45 | 3.79 | 35.76 | 0 | 0 |
| per_object_faint | 0.71 | 0.45 | 5.78 | 35.76 | 0.4 | 0.4 |
| single | 0.7 | 0.45 | 46.91 | 35.87 | 0.07 | 0.07 |

`outline`:

| outline | outline_iou | ari_materials | false_px | satellite_px_found | companion_px_found | faint_found |
|---|---|---|---|---|---|---|
| fill@0.01 | 0.8 | 0.43 | 10 | 44 | 0 | 0 |
| mask@0.01 | 0.73 | 0.45 | 0.43 | 33.02 | 0 | 0 |
| mask@0.01+faint | 0.72 | 0.45 | 1.15 | 33.02 | 0.29 | 0.29 |
| model | 0.61 | 0.45 | 63.74 | 33.15 | 0.35 | 0.35 |

`cleanup`:

| cleanup | outline_iou | ari_materials | false_px | satellite_px_found | companion_px_found | faint_found |
|---|---|---|---|---|---|---|
| off | 0.72 | 0.44 | 17.41 | 35.86 | 0.19 | 0.19 |
| tidy | 0.71 | 0.45 | 20.25 | 35.74 | 0.12 | 0.12 |

`vote`:

| vote | outline_iou | ari_materials | false_px | satellite_px_found | companion_px_found | faint_found |
|---|---|---|---|---|---|---|
| 0.3 | 0.74 | 0.45 | 28.94 | 38.01 | 0.19 | 0.19 |
| 0.5 | 0.69 | 0.45 | 8.72 | 33.59 | 0.13 | 0.13 |

### Every model with every scaling (model outline, no cleanup, vote 0.5)

`outline_iou`:

| model | per_object | per_object_faint | single |
|---|---|---|---|
| aug | 0.89 | 0.89 | 0.89 |
| aug_fg3 | 0.71 | 0.69 | 0.71 |
| aug_fg3_halo | 0.85 | 0.82 | 0.85 |
| aug_fg3_halo_light | 0.33 | 0.34 | 0.33 |
| aug_fg3_light | 0.84 | 0.76 | 0.84 |
| fg3 | 0.71 | 0.69 | 0.71 |
| halo | 0.73 | 0.71 | 0.04 |
| lightnoise | 0.13 | 0.15 | 0.13 |
| original_retrained | 0.82 | 0.78 | 0.82 |
| pair_max1_aug | 0.87 | 0.85 | 0.87 |
| pair_max1_aug_fg3 | 0.91 | 0.86 | 0.91 |
| pair_max1_aug_fg3_halo | 0.67 | 0.66 | 0.67 |
| pair_max1_aug_fg3_halo_light | 0.22 | 0.24 | 0.22 |
| pair_max1_aug_fg3_light | 0.47 | 0.44 | 0.47 |

`ari_materials`:

| model | per_object | per_object_faint | single |
|---|---|---|---|
| aug | 0.9 | 0.91 | 0.9 |
| aug_fg3 | 0.84 | 0.85 | 0.84 |
| aug_fg3_halo | 0.19 | 0.19 | 0.19 |
| aug_fg3_halo_light | 0.28 | 0.25 | 0.28 |
| aug_fg3_light | 0.25 | 0.25 | 0.25 |
| fg3 | 0.57 | 0.58 | 0.57 |
| halo | 0.19 | 0.18 | 0.18 |
| lightnoise | 0.67 | 0.61 | 0.67 |
| original_retrained | 0.64 | 0.64 | 0.64 |
| pair_max1_aug | 0.44 | 0.44 | 0.44 |
| pair_max1_aug_fg3 | 0.26 | 0.26 | 0.26 |
| pair_max1_aug_fg3_halo | 0.25 | 0.24 | 0.25 |
| pair_max1_aug_fg3_halo_light | 0.45 | 0.41 | 0.45 |
| pair_max1_aug_fg3_light | 0.26 | 0.23 | 0.26 |

`faint_found`:

| model | per_object | per_object_faint | single |
|---|---|---|---|
| aug | 0 | 1 | 0 |
| aug_fg3 | 0 | 1 | 0 |
| aug_fg3_halo | 0 | 1 | 0 |
| aug_fg3_halo_light | 0 | 1 | 0 |
| aug_fg3_light | 0 | 1 | 0 |
| fg3 | 0 | 1 | 0 |
| halo | 0 | 1 | 1 |
| lightnoise | 0 | 1 | 0 |
| original_retrained | 0 | 1 | 0 |
| pair_max1_aug | 0 | 1 | 0 |
| pair_max1_aug_fg3 | 0 | 1 | 0 |
| pair_max1_aug_fg3_halo | 0 | 1 | 0 |
| pair_max1_aug_fg3_halo_light | 0 | 1 | 0 |
| pair_max1_aug_fg3_light | 0 | 1 | 0 |

`companion_px_found`:

| model | per_object | per_object_faint | single |
|---|---|---|---|
| aug | 0 | 1 | 0 |
| aug_fg3 | 0 | 1 | 0 |
| aug_fg3_halo | 0 | 1 | 0 |
| aug_fg3_halo_light | 0 | 1 | 0 |
| aug_fg3_light | 0 | 1 | 0 |
| fg3 | 0 | 1 | 0 |
| halo | 0 | 1 | 1 |
| lightnoise | 0 | 1 | 0 |
| original_retrained | 0 | 1 | 0 |
| pair_max1_aug | 0 | 1 | 0 |
| pair_max1_aug_fg3 | 0 | 1 | 0 |
| pair_max1_aug_fg3_halo | 0 | 1 | 0 |
| pair_max1_aug_fg3_halo_light | 0 | 1 | 0 |
| pair_max1_aug_fg3_light | 0 | 1 | 0 |

### Every run on its own (mean and std over the runs; no voting)

| model | scaling | outline_iou | outline_iou_std | ari_materials | ari_materials_std |
|---|---|---|---|---|---|
| original_retrained | single | 0.47 | 0.32 | 0.35 | 0.25 |
| aug | single | 0.78 | 0.11 | 0.56 | 0.24 |
| pair_max1_aug | single | 0.82 | 0.04 | 0.44 | 0.19 |
| fg3 | single | 0.4 | 0.3 | 0.42 | 0.21 |
| halo | single | 0.27 | 0.34 | 0.17 | 0.08 |
| lightnoise | single | 0.13 | 0 | 0.67 | 0 |
| aug_fg3 | single | 0.64 | 0.08 | 0.62 | 0.25 |
| aug_fg3_halo | single | 0.74 | 0.06 | 0.3 | 0.06 |
| aug_fg3_light | single | 0.66 | 0.16 | 0.35 | 0.19 |
| aug_fg3_halo_light | single | 0.34 | 0.17 | 0.32 | 0.18 |
| pair_max1_aug_fg3 | single | 0.83 | 0.05 | 0.38 | 0.07 |
| pair_max1_aug_fg3_halo | single | 0.53 | 0.27 | 0.36 | 0.19 |
| pair_max1_aug_fg3_light | single | 0.47 | 0.07 | 0.27 | 0.12 |
| pair_max1_aug_fg3_halo_light | single | 0.27 | 0.08 | 0.43 | 0.16 |

### A window with nothing in it: labelled pixels (all false)

| model | per_object | per_object_faint | single |
|---|---|---|---|
| aug | 0 | 0 | 0 |
| aug_fg3 | 0 | 0 | 0 |
| aug_fg3_halo | 0 | 0 | 0 |
| aug_fg3_halo_light | 0 | 0 | 0 |
| aug_fg3_light | 0 | 0 | 0 |
| fg3 | 0 | 0 | 0 |
| halo | 0 | 0 | 0 |
| lightnoise | 0 | 0 | 0 |
| original_retrained | 0 | 0 | 0 |
| pair_max1_aug | 0 | 0 | 0 |
| pair_max1_aug_fg3 | 0 | 0 | 0 |
| pair_max1_aug_fg3_halo | 0 | 0 | 0 |
| pair_max1_aug_fg3_halo_light | 0 | 0 | 0 |
| pair_max1_aug_fg3_light | 0 | 0 | 0 |
