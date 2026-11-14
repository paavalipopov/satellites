# Comparison with the original model

Mean over each experiment's runs. (higher) / (lower) says which way is better. The change column is new minus original.
estimate.fits was read as: `radiometry` (solar = notebook 06, the previous results; radiometry = the collaborators' recipe, see scripts/fits_reading.py).

## synthetic: material acc at the trained brightness (higher)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.846 |  |
| original | 0.735 | -0.110 |
| fg3 | 0.855 | +0.009 |
| fg10 | 0.869 | +0.023 |
| fg3_dice | 0.856 | +0.010 |
| halo | 0.842 | -0.004 |
| lightnoise | 0.842 | -0.004 |

## synthetic: material acc, whole image 10x brighter (higher)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.030 |  |
| original | 0.054 | +0.025 |
| fg3 | 0.024 | -0.005 |
| fg10 | 0.035 | +0.006 |
| fg3_dice | 0.038 | +0.008 |
| halo | 0.020 | -0.010 |
| lightnoise | 0.058 | +0.029 |

## synthetic: material acc, whole image 10x dimmer (higher)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.004 |  |
| original | 0.003 | -0.001 |
| fg3 | 0.006 | +0.002 |
| fg10 | 0.009 | +0.005 |
| fg3_dice | 0.007 | +0.003 |
| halo | 0.005 | +0.001 |
| lightnoise | 0.004 | -0.000 |

## synthetic: material acc, object 1/3 as bright, same noise (higher)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.019 |  |
| original | 0.015 | -0.003 |
| fg3 | 0.031 | +0.012 |
| fg10 | 0.043 | +0.024 |
| fg3_dice | 0.029 | +0.010 |
| halo | 0.017 | -0.001 |
| lightnoise | 0.016 | -0.003 |

## synthetic: material acc, object 1/10 as bright, same noise (higher)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.003 |  |
| original | 0.003 | -0.001 |
| fg3 | 0.005 | +0.002 |
| fg10 | 0.008 | +0.005 |
| fg3_dice | 0.005 | +0.002 |
| halo | 0.004 | +0.000 |
| lightnoise | 0.003 | -0.001 |

## band curve sd 0.2: material acc (higher)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.307 |  |
| original | 0.229 | -0.078 |
| fg3 | 0.319 | +0.012 |
| fg10 | 0.331 | +0.024 |
| fg3_dice | 0.317 | +0.011 |
| halo | 0.294 | -0.013 |
| lightnoise | 0.306 | -0.000 |

## halo at 1% of the object: background given a material (lower)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.044 |  |
| original | 0.038 | -0.006 |
| fg3 | 0.096 | +0.053 |
| fg10 | 0.176 | +0.132 |
| fg3_dice | 0.083 | +0.039 |
| halo | 0.006 | -0.038 |
| lightnoise | 0.024 | -0.020 |

## restoration-like noise sd 0.03: background given a material (lower)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.083 |  |
| original | 0.136 | +0.052 |
| fg3 | 0.112 | +0.029 |
| fg10 | 0.156 | +0.073 |
| fg3_dice | 0.109 | +0.026 |
| halo | 0.125 | +0.042 |
| lightnoise | 0.030 | -0.054 |

## empty scene, training noise: pixels given a material (lower)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.005 |  |
| original | 0.002 | -0.003 |
| fg3 | 0.021 | +0.016 |
| fg10 | 0.086 | +0.081 |
| fg3_dice | 0.013 | +0.009 |
| halo | 0.008 | +0.003 |
| lightnoise | 0.001 | -0.003 |

## empty scene, 1/100 of that noise: pixels given a material (lower)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.000 |  |
| original | 0.000 | +0.000 |
| fg3 | 0.000 | +0.000 |
| fg10 | 0.000 | +0.000 |
| fg3_dice | 0.000 | +0.000 |
| halo | 0.000 | +0.000 |
| lightnoise | 0.000 | +0.000 |

## estimate.fits, NO scale chosen: outline IoU (higher)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.711 |  |
| original | 0.667 | -0.044 |
| fg3 | 0.711 | +0.000 |
| fg10 | 0.800 | +0.089 |
| fg3_dice | 0.778 | +0.067 |
| halo | 0.600 | -0.111 |
| lightnoise | 0.622 | -0.089 |

## estimate.fits, NO scale chosen: materials ARI (higher)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.385 |  |
| original | 0.133 | -0.252 |
| fg3 | 0.251 | -0.134 |
| fg10 | 0.625 | +0.241 |
| fg3_dice | 0.349 | -0.036 |
| halo | 0.266 | -0.119 |
| lightnoise | 0.105 | -0.280 |

## estimate.fits, NO scale chosen: false pixels (lower)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.000 |  |
| original | 0.000 | +0.000 |
| fg3 | 0.000 | +0.000 |
| fg10 | 0.000 | +0.000 |
| fg3_dice | 0.000 | +0.000 |
| halo | 0.000 | +0.000 |
| lightnoise | 0.000 | +0.000 |

## estimate.fits, by hand (max = 1): outline IoU (higher)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.822 |  |
| original | 0.026 | -0.796 |
| fg3 | 0.711 | -0.111 |
| fg10 | 0.027 | -0.795 |
| fg3_dice | 0.065 | -0.757 |
| halo | 0.040 | -0.783 |
| lightnoise | 0.133 | -0.689 |

## estimate.fits, by hand (max = 1): materials ARI (higher)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.641 |  |
| original | 0.708 | +0.067 |
| fg3 | 0.572 | -0.069 |
| fg10 | 0.338 | -0.304 |
| fg3_dice | 0.538 | -0.104 |
| halo | 0.185 | -0.457 |
| lightnoise | 0.666 | +0.025 |

## estimate.fits, blank window: pixels given a material (lower)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.000 |  |
| original | 0.000 | +0.000 |
| fg3 | 0.000 | +0.000 |
| fg10 | 0.000 | +0.000 |
| fg3_dice | 0.000 | +0.000 |
| halo | 0.000 | +0.000 |
| lightnoise | 0.000 | +0.000 |
