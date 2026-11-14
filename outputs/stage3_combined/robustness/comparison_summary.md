# Comparison with the original model

Mean over each experiment's runs. (higher) / (lower) says which way is better. The change column is new minus original.
estimate.fits was read as: `radiometry` (solar = notebook 06, the previous results; radiometry = the collaborators' recipe, see scripts/fits_reading.py).

## synthetic: material acc at the trained brightness (higher)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.846 |  |
| aug | 0.598 | -0.247 |
| pair_max1_aug | 0.755 | -0.091 |
| fg3 | 0.855 | +0.009 |
| halo | 0.842 | -0.004 |
| lightnoise | 0.842 | -0.004 |
| aug_fg3 | 0.764 | -0.082 |
| aug_fg3_halo | 0.763 | -0.083 |
| aug_fg3_light | 0.768 | -0.077 |
| aug_fg3_halo_light | 0.766 | -0.080 |
| pair_max1_aug_fg3 | 0.771 | -0.075 |
| pair_max1_aug_fg3_halo | 0.769 | -0.077 |
| pair_max1_aug_fg3_light | 0.776 | -0.070 |
| pair_max1_aug_fg3_halo_light | 0.763 | -0.083 |

## synthetic: material acc, whole image 10x brighter (higher)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.030 |  |
| aug | 0.272 | +0.243 |
| pair_max1_aug | 0.222 | +0.192 |
| fg3 | 0.024 | -0.005 |
| halo | 0.020 | -0.010 |
| lightnoise | 0.058 | +0.029 |
| aug_fg3 | 0.295 | +0.265 |
| aug_fg3_halo | 0.326 | +0.297 |
| aug_fg3_light | 0.299 | +0.269 |
| aug_fg3_halo_light | 0.324 | +0.294 |
| pair_max1_aug_fg3 | 0.269 | +0.240 |
| pair_max1_aug_fg3_halo | 0.278 | +0.249 |
| pair_max1_aug_fg3_light | 0.261 | +0.232 |
| pair_max1_aug_fg3_halo_light | 0.237 | +0.208 |

## synthetic: material acc, whole image 10x dimmer (higher)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.004 |  |
| aug | 0.317 | +0.313 |
| pair_max1_aug | 0.480 | +0.477 |
| fg3 | 0.006 | +0.002 |
| halo | 0.005 | +0.001 |
| lightnoise | 0.004 | -0.000 |
| aug_fg3 | 0.518 | +0.514 |
| aug_fg3_halo | 0.507 | +0.503 |
| aug_fg3_light | 0.470 | +0.466 |
| aug_fg3_halo_light | 0.472 | +0.468 |
| pair_max1_aug_fg3 | 0.522 | +0.518 |
| pair_max1_aug_fg3_halo | 0.524 | +0.521 |
| pair_max1_aug_fg3_light | 0.509 | +0.505 |
| pair_max1_aug_fg3_halo_light | 0.516 | +0.512 |

## synthetic: material acc, object 1/3 as bright, same noise (higher)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.019 |  |
| aug | 0.445 | +0.426 |
| pair_max1_aug | 0.597 | +0.579 |
| fg3 | 0.031 | +0.012 |
| halo | 0.017 | -0.001 |
| lightnoise | 0.016 | -0.003 |
| aug_fg3 | 0.625 | +0.606 |
| aug_fg3_halo | 0.626 | +0.608 |
| aug_fg3_light | 0.624 | +0.606 |
| aug_fg3_halo_light | 0.622 | +0.603 |
| pair_max1_aug_fg3 | 0.624 | +0.606 |
| pair_max1_aug_fg3_halo | 0.618 | +0.599 |
| pair_max1_aug_fg3_light | 0.619 | +0.600 |
| pair_max1_aug_fg3_halo_light | 0.623 | +0.604 |

## synthetic: material acc, object 1/10 as bright, same noise (higher)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.003 |  |
| aug | 0.236 | +0.233 |
| pair_max1_aug | 0.341 | +0.338 |
| fg3 | 0.005 | +0.002 |
| halo | 0.004 | +0.000 |
| lightnoise | 0.003 | -0.001 |
| aug_fg3 | 0.383 | +0.379 |
| aug_fg3_halo | 0.386 | +0.382 |
| aug_fg3_light | 0.382 | +0.378 |
| aug_fg3_halo_light | 0.380 | +0.377 |
| pair_max1_aug_fg3 | 0.379 | +0.375 |
| pair_max1_aug_fg3_halo | 0.371 | +0.367 |
| pair_max1_aug_fg3_light | 0.376 | +0.373 |
| pair_max1_aug_fg3_halo_light | 0.372 | +0.369 |

## band curve sd 0.2: material acc (higher)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.307 |  |
| aug | 0.160 | -0.147 |
| pair_max1_aug | 0.215 | -0.092 |
| fg3 | 0.319 | +0.012 |
| halo | 0.294 | -0.013 |
| lightnoise | 0.306 | -0.000 |
| aug_fg3 | 0.212 | -0.095 |
| aug_fg3_halo | 0.220 | -0.087 |
| aug_fg3_light | 0.229 | -0.078 |
| aug_fg3_halo_light | 0.235 | -0.072 |
| pair_max1_aug_fg3 | 0.237 | -0.070 |
| pair_max1_aug_fg3_halo | 0.228 | -0.078 |
| pair_max1_aug_fg3_light | 0.248 | -0.059 |
| pair_max1_aug_fg3_halo_light | 0.236 | -0.070 |

## halo at 1% of the object: background given a material (lower)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.044 |  |
| aug | 0.042 | -0.001 |
| pair_max1_aug | 0.037 | -0.006 |
| fg3 | 0.096 | +0.053 |
| halo | 0.006 | -0.038 |
| lightnoise | 0.024 | -0.020 |
| aug_fg3 | 0.081 | +0.037 |
| aug_fg3_halo | 0.016 | -0.028 |
| aug_fg3_light | 0.053 | +0.010 |
| aug_fg3_halo_light | 0.016 | -0.028 |
| pair_max1_aug_fg3 | 0.079 | +0.035 |
| pair_max1_aug_fg3_halo | 0.015 | -0.028 |
| pair_max1_aug_fg3_light | 0.056 | +0.013 |
| pair_max1_aug_fg3_halo_light | 0.016 | -0.027 |

## restoration-like noise sd 0.03: background given a material (lower)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.083 |  |
| aug | 0.099 | +0.015 |
| pair_max1_aug | 0.052 | -0.031 |
| fg3 | 0.112 | +0.029 |
| halo | 0.125 | +0.042 |
| lightnoise | 0.030 | -0.054 |
| aug_fg3 | 0.094 | +0.010 |
| aug_fg3_halo | 0.107 | +0.024 |
| aug_fg3_light | 0.029 | -0.054 |
| aug_fg3_halo_light | 0.043 | -0.041 |
| pair_max1_aug_fg3 | 0.082 | -0.001 |
| pair_max1_aug_fg3_halo | 0.083 | -0.001 |
| pair_max1_aug_fg3_light | 0.026 | -0.058 |
| pair_max1_aug_fg3_halo_light | 0.030 | -0.054 |

## empty scene, training noise: pixels given a material (lower)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.005 |  |
| aug | 0.000 | -0.004 |
| pair_max1_aug | 0.000 | -0.004 |
| fg3 | 0.021 | +0.016 |
| halo | 0.008 | +0.003 |
| lightnoise | 0.001 | -0.003 |
| aug_fg3 | 0.003 | -0.002 |
| aug_fg3_halo | 0.002 | -0.002 |
| aug_fg3_light | 0.001 | -0.004 |
| aug_fg3_halo_light | 0.001 | -0.004 |
| pair_max1_aug_fg3 | 0.003 | -0.002 |
| pair_max1_aug_fg3_halo | 0.003 | -0.001 |
| pair_max1_aug_fg3_light | 0.001 | -0.004 |
| pair_max1_aug_fg3_halo_light | 0.000 | -0.004 |

## empty scene, 1/100 of that noise: pixels given a material (lower)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.000 |  |
| aug | 0.000 | +0.000 |
| pair_max1_aug | 0.000 | +0.000 |
| fg3 | 0.000 | +0.000 |
| halo | 0.000 | +0.000 |
| lightnoise | 0.000 | +0.000 |
| aug_fg3 | 0.000 | +0.000 |
| aug_fg3_halo | 0.000 | +0.000 |
| aug_fg3_light | 0.000 | +0.000 |
| aug_fg3_halo_light | 0.000 | +0.000 |
| pair_max1_aug_fg3 | 0.000 | +0.000 |
| pair_max1_aug_fg3_halo | 0.000 | +0.000 |
| pair_max1_aug_fg3_light | 0.000 | +0.000 |
| pair_max1_aug_fg3_halo_light | 0.000 | +0.000 |

## estimate.fits, NO scale chosen: outline IoU (higher)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.711 |  |
| aug | 0.756 | +0.044 |
| pair_max1_aug | 0.822 | +0.111 |
| fg3 | 0.711 | +0.000 |
| halo | 0.600 | -0.111 |
| lightnoise | 0.622 | -0.089 |
| aug_fg3 | 0.756 | +0.044 |
| aug_fg3_halo | 0.733 | +0.022 |
| aug_fg3_light | 0.778 | +0.067 |
| aug_fg3_halo_light | 0.467 | -0.244 |
| pair_max1_aug_fg3 | 0.867 | +0.156 |
| pair_max1_aug_fg3_halo | 0.778 | +0.067 |
| pair_max1_aug_fg3_light | 0.622 | -0.089 |
| pair_max1_aug_fg3_halo_light | 0.444 | -0.267 |

## estimate.fits, NO scale chosen: materials ARI (higher)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.385 |  |
| aug | 0.542 | +0.157 |
| pair_max1_aug | 0.487 | +0.102 |
| fg3 | 0.251 | -0.134 |
| halo | 0.266 | -0.119 |
| lightnoise | 0.105 | -0.280 |
| aug_fg3 | 0.476 | +0.091 |
| aug_fg3_halo | 0.100 | -0.285 |
| aug_fg3_light | 0.232 | -0.152 |
| aug_fg3_halo_light | 0.159 | -0.225 |
| pair_max1_aug_fg3 | 0.402 | +0.017 |
| pair_max1_aug_fg3_halo | 0.433 | +0.048 |
| pair_max1_aug_fg3_light | 0.231 | -0.154 |
| pair_max1_aug_fg3_halo_light | 0.242 | -0.142 |

## estimate.fits, NO scale chosen: false pixels (lower)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.000 |  |
| aug | 0.000 | +0.000 |
| pair_max1_aug | 0.000 | +0.000 |
| fg3 | 0.000 | +0.000 |
| halo | 0.000 | +0.000 |
| lightnoise | 0.000 | +0.000 |
| aug_fg3 | 0.000 | +0.000 |
| aug_fg3_halo | 0.000 | +0.000 |
| aug_fg3_light | 0.000 | +0.000 |
| aug_fg3_halo_light | 0.000 | +0.000 |
| pair_max1_aug_fg3 | 0.000 | +0.000 |
| pair_max1_aug_fg3_halo | 0.000 | +0.000 |
| pair_max1_aug_fg3_light | 0.000 | +0.000 |
| pair_max1_aug_fg3_halo_light | 0.000 | +0.000 |

## estimate.fits, by hand (max = 1): outline IoU (higher)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.822 |  |
| aug | 0.889 | +0.067 |
| pair_max1_aug | 0.867 | +0.044 |
| fg3 | 0.711 | -0.111 |
| halo | 0.040 | -0.783 |
| lightnoise | 0.133 | -0.689 |
| aug_fg3 | 0.707 | -0.115 |
| aug_fg3_halo | 0.854 | +0.032 |
| aug_fg3_light | 0.844 | +0.022 |
| aug_fg3_halo_light | 0.333 | -0.489 |
| pair_max1_aug_fg3 | 0.913 | +0.091 |
| pair_max1_aug_fg3_halo | 0.667 | -0.156 |
| pair_max1_aug_fg3_light | 0.467 | -0.356 |
| pair_max1_aug_fg3_halo_light | 0.222 | -0.600 |

## estimate.fits, by hand (max = 1): materials ARI (higher)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.641 |  |
| aug | 0.900 | +0.258 |
| pair_max1_aug | 0.437 | -0.204 |
| fg3 | 0.572 | -0.069 |
| halo | 0.185 | -0.457 |
| lightnoise | 0.666 | +0.025 |
| aug_fg3 | 0.841 | +0.199 |
| aug_fg3_halo | 0.187 | -0.454 |
| aug_fg3_light | 0.249 | -0.393 |
| aug_fg3_halo_light | 0.277 | -0.364 |
| pair_max1_aug_fg3 | 0.255 | -0.386 |
| pair_max1_aug_fg3_halo | 0.253 | -0.388 |
| pair_max1_aug_fg3_light | 0.258 | -0.384 |
| pair_max1_aug_fg3_halo_light | 0.453 | -0.189 |

## estimate.fits, blank window: pixels given a material (lower)

| experiment | value | change from original |
|---|---|---|
| original_retrained | 0.000 |  |
| aug | 0.000 | +0.000 |
| pair_max1_aug | 0.000 | +0.000 |
| fg3 | 0.000 | +0.000 |
| halo | 0.000 | +0.000 |
| lightnoise | 0.000 | +0.000 |
| aug_fg3 | 0.000 | +0.000 |
| aug_fg3_halo | 0.000 | +0.000 |
| aug_fg3_light | 0.000 | +0.000 |
| aug_fg3_halo_light | 0.000 | +0.000 |
| pair_max1_aug_fg3 | 0.000 | +0.000 |
| pair_max1_aug_fg3_halo | 0.000 | +0.000 |
| pair_max1_aug_fg3_light | 0.000 | +0.000 |
| pair_max1_aug_fg3_halo_light | 0.000 | +0.000 |
