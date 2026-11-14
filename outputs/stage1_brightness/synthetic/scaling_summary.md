# Brightness: training x scaling

Models: `original`, `original_retrained`, `aug`, `norm`, `aug_norm`, `gain_p99`, `gain_flux`, `gain_bandflux`, `gain_pixel`, `pair_max1`, `pair_mean`, `pair_p99`, `pair_max1_aug`, `pair_mean_aug`, `pair_p99_aug`. Modes (scripts/auto_scale.py): `none` = the image as it is, `single` = one scale for the image, `per_object` = every object its own scale. The by-hand rows are the notebook's reference (a person picked the objects and the scales).

## One object, overall brightness changed (material accuracy; flat rows = brightness does not matter)

`before_noise`, columns = log10 of the gain:

| model | mode | -2.0 | -1.5 | -1.0 | -0.5 | 0.0 |
|---|---|---|---|---|---|---|
| aug | none | 0 | 0.094 | 0.236 | 0.445 | 0.598 |
| aug | per_object | 0 | 0 | 0.01 | 0.192 | 0.406 |
| aug | per_object_mean | 0 | 0 | 0.01 | 0.168 | 0.374 |
| aug | per_object_p99 | 0 | 0 | 0.01 | 0.188 | 0.405 |
| aug | single | 0 | 0 | 0.017 | 0.249 | 0.529 |
| aug | single_mean | 0 | 0 | 0.016 | 0.217 | 0.482 |
| aug | single_p99 | 0 | 0 | 0.017 | 0.242 | 0.527 |
| aug_norm | none | 0.002 | 0.079 | 0.196 | 0.427 | 0.569 |
| aug_norm | per_object | 0 | 0 | 0.024 | 0.253 | 0.408 |
| aug_norm | per_object_mean | 0 | 0 | 0.024 | 0.253 | 0.408 |
| aug_norm | per_object_p99 | 0 | 0 | 0.024 | 0.253 | 0.408 |
| aug_norm | single | 0 | 0 | 0.068 | 0.361 | 0.514 |
| aug_norm | single_mean | 0 | 0 | 0.068 | 0.361 | 0.514 |
| aug_norm | single_p99 | 0 | 0 | 0.068 | 0.361 | 0.514 |
| gain_bandflux | none | 0 | 0.077 | 0.266 | 0.466 | 0.599 |
| gain_bandflux | per_object | 0 | 0 | 0.005 | 0.175 | 0.373 |
| gain_bandflux | per_object_mean | 0 | 0 | 0.005 | 0.175 | 0.373 |
| gain_bandflux | per_object_p99 | 0 | 0 | 0.005 | 0.175 | 0.373 |
| gain_bandflux | single | 0 | 0 | 0.092 | 0.392 | 0.542 |
| gain_bandflux | single_mean | 0 | 0 | 0.092 | 0.392 | 0.542 |
| gain_bandflux | single_p99 | 0 | 0 | 0.092 | 0.392 | 0.542 |
| gain_flux | none | 0 | 0.119 | 0.325 | 0.581 | 0.741 |
| gain_flux | per_object | 0 | 0 | 0.008 | 0.289 | 0.52 |
| gain_flux | per_object_mean | 0 | 0 | 0.008 | 0.289 | 0.52 |
| gain_flux | per_object_p99 | 0 | 0 | 0.008 | 0.289 | 0.52 |
| gain_flux | single | 0 | 0 | 0.12 | 0.497 | 0.672 |
| gain_flux | single_mean | 0 | 0 | 0.12 | 0.497 | 0.672 |
| gain_flux | single_p99 | 0 | 0 | 0.12 | 0.497 | 0.672 |
| gain_p99 | none | 0 | 0.135 | 0.355 | 0.604 | 0.77 |
| gain_p99 | per_object | 0 | 0 | 0.022 | 0.334 | 0.551 |
| gain_p99 | per_object_mean | 0 | 0 | 0.022 | 0.334 | 0.551 |
| gain_p99 | per_object_p99 | 0 | 0 | 0.022 | 0.334 | 0.551 |
| gain_p99 | single | 0 | 0 | 0.136 | 0.515 | 0.697 |
| gain_p99 | single_mean | 0 | 0 | 0.136 | 0.515 | 0.697 |
| gain_p99 | single_p99 | 0 | 0 | 0.136 | 0.515 | 0.697 |
| gain_pixel | none | 0 | 0.12 | 0.313 | 0.528 | 0.7 |
| gain_pixel | per_object | 0 | 0 | 0.04 | 0.294 | 0.472 |
| gain_pixel | per_object_mean | 0 | 0 | 0.04 | 0.294 | 0.472 |
| gain_pixel | per_object_p99 | 0 | 0 | 0.04 | 0.294 | 0.472 |
| gain_pixel | single | 0 | 0 | 0.113 | 0.451 | 0.631 |
| gain_pixel | single_mean | 0 | 0 | 0.113 | 0.451 | 0.631 |
| gain_pixel | single_p99 | 0 | 0 | 0.113 | 0.451 | 0.631 |
| norm | none | 0.003 | 0.005 | 0.007 | 0.037 | 0.681 |
| norm | per_object | 0 | 0 | 0.002 | 0.05 | 0.348 |
| norm | per_object_mean | 0 | 0 | 0.002 | 0.05 | 0.348 |
| norm | per_object_p99 | 0 | 0 | 0.002 | 0.05 | 0.348 |
| norm | single | 0 | 0 | 0.005 | 0.037 | 0.62 |
| norm | single_mean | 0 | 0 | 0.005 | 0.037 | 0.62 |
| norm | single_p99 | 0 | 0 | 0.005 | 0.037 | 0.62 |
| original | none | 0.001 | 0.002 | 0.003 | 0.015 | 0.735 |
| original | per_object | 0 | 0 | 0.008 | 0.108 | 0.186 |
| original | per_object_mean | 0 | 0 | 0.014 | 0.052 | 0.064 |
| original | per_object_p99 | 0 | 0 | 0.01 | 0.095 | 0.163 |
| original | single | 0 | 0 | 0.013 | 0.147 | 0.262 |
| original | single_mean | 0 | 0 | 0.015 | 0.065 | 0.088 |
| original | single_p99 | 0 | 0 | 0.013 | 0.131 | 0.23 |
| original_retrained | none | 0.002 | 0.001 | 0.003 | 0.019 | 0.846 |
| original_retrained | per_object | 0 | 0 | 0.01 | 0.109 | 0.216 |
| original_retrained | per_object_mean | 0 | 0 | 0.012 | 0.046 | 0.055 |
| original_retrained | per_object_p99 | 0 | 0 | 0.01 | 0.093 | 0.19 |
| original_retrained | single | 0 | 0 | 0.011 | 0.143 | 0.307 |
| original_retrained | single_mean | 0 | 0 | 0.014 | 0.054 | 0.075 |
| original_retrained | single_p99 | 0 | 0 | 0.012 | 0.123 | 0.27 |
| pair_max1 | none | 0.001 | 0.001 | 0.003 | 0.021 | 0.759 |
| pair_max1 | per_object | 0 | 0 | 0.001 | 0.052 | 0.337 |
| pair_max1 | per_object_mean | 0 | 0 | 0.001 | 0.032 | 0.278 |
| pair_max1 | per_object_p99 | 0 | 0 | 0.001 | 0.05 | 0.333 |
| pair_max1 | single | 0 | 0 | 0.001 | 0.029 | 0.666 |
| pair_max1 | single_mean | 0 | 0 | 0.001 | 0.02 | 0.469 |
| pair_max1 | single_p99 | 0 | 0 | 0.001 | 0.027 | 0.662 |
| pair_max1_aug | none | 0.001 | 0.126 | 0.341 | 0.597 | 0.755 |
| pair_max1_aug | per_object | 0 | 0 | 0.023 | 0.321 | 0.533 |
| pair_max1_aug | per_object_mean | 0 | 0 | 0.02 | 0.301 | 0.522 |
| pair_max1_aug | per_object_p99 | 0 | 0 | 0.023 | 0.321 | 0.534 |
| pair_max1_aug | single | 0 | 0 | 0.127 | 0.505 | 0.683 |
| pair_max1_aug | single_mean | 0 | 0 | 0.123 | 0.472 | 0.668 |
| pair_max1_aug | single_p99 | 0 | 0 | 0.127 | 0.505 | 0.682 |
| pair_mean | none | 0.003 | 0.003 | 0.002 | 0.015 | 0.674 |
| pair_mean | per_object | 0 | 0 | 0.002 | 0.087 | 0.231 |
| pair_mean | per_object_mean | 0 | 0 | 0.001 | 0.068 | 0.316 |
| pair_mean | per_object_p99 | 0 | 0 | 0.002 | 0.079 | 0.222 |
| pair_mean | single | 0 | 0 | 0.003 | 0.102 | 0.327 |
| pair_mean | single_mean | 0 | 0 | 0.002 | 0.083 | 0.488 |
| pair_mean | single_p99 | 0 | 0 | 0.003 | 0.097 | 0.323 |
| pair_mean_aug | none | 0 | 0.097 | 0.277 | 0.537 | 0.697 |
| pair_mean_aug | per_object | 0 | 0 | 0.009 | 0.249 | 0.476 |
| pair_mean_aug | per_object_mean | 0 | 0 | 0.014 | 0.259 | 0.466 |
| pair_mean_aug | per_object_p99 | 0 | 0 | 0.009 | 0.249 | 0.473 |
| pair_mean_aug | single | 0 | 0 | 0.03 | 0.351 | 0.615 |
| pair_mean_aug | single_mean | 0 | 0 | 0.052 | 0.365 | 0.602 |
| pair_mean_aug | single_p99 | 0 | 0 | 0.031 | 0.352 | 0.611 |
| pair_p99 | none | 0.002 | 0.001 | 0.002 | 0.019 | 0.83 |
| pair_p99 | per_object | 0 | 0 | 0.004 | 0.048 | 0.313 |
| pair_p99 | per_object_mean | 0 | 0 | 0.003 | 0.043 | 0.328 |
| pair_p99 | per_object_p99 | 0 | 0 | 0.003 | 0.048 | 0.31 |
| pair_p99 | single | 0 | 0 | 0.001 | 0.024 | 0.739 |
| pair_p99 | single_mean | 0 | 0 | 0.001 | 0.021 | 0.529 |
| pair_p99 | single_p99 | 0 | 0 | 0.001 | 0.023 | 0.739 |
| pair_p99_aug | none | 0 | 0.125 | 0.334 | 0.592 | 0.753 |
| pair_p99_aug | per_object | 0 | 0 | 0.019 | 0.318 | 0.53 |
| pair_p99_aug | per_object_mean | 0 | 0 | 0.024 | 0.317 | 0.525 |
| pair_p99_aug | per_object_p99 | 0 | 0 | 0.02 | 0.317 | 0.53 |
| pair_p99_aug | single | 0 | 0 | 0.11 | 0.497 | 0.68 |
| pair_p99_aug | single_mean | 0 | 0 | 0.156 | 0.52 | 0.677 |
| pair_p99_aug | single_p99 | 0 | 0 | 0.112 | 0.5 | 0.68 |

`after_noise`, columns = log10 of the gain:

| model | mode | -3.0 | -2.0 | -1.0 | 0.0 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|---|
| aug | none | 0 | 0 | 0.317 | 0.598 | 0.272 | 0.024 |
| aug | per_object | 0.406 | 0.406 | 0.406 | 0.406 | 0.406 | 0.406 |
| aug | per_object_mean | 0.374 | 0.374 | 0.374 | 0.374 | 0.374 | 0.374 |
| aug | per_object_p99 | 0.405 | 0.405 | 0.405 | 0.405 | 0.405 | 0.405 |
| aug | single | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 |
| aug | single_mean | 0.482 | 0.482 | 0.482 | 0.482 | 0.482 | 0.482 |
| aug | single_p99 | 0.527 | 0.527 | 0.527 | 0.527 | 0.527 | 0.527 |
| aug_norm | none | 0.569 | 0.569 | 0.569 | 0.569 | 0.569 | 0.569 |
| aug_norm | per_object | 0.408 | 0.408 | 0.408 | 0.408 | 0.408 | 0.408 |
| aug_norm | per_object_mean | 0.408 | 0.408 | 0.408 | 0.408 | 0.408 | 0.408 |
| aug_norm | per_object_p99 | 0.408 | 0.408 | 0.408 | 0.408 | 0.408 | 0.408 |
| aug_norm | single | 0.514 | 0.514 | 0.514 | 0.514 | 0.514 | 0.514 |
| aug_norm | single_mean | 0.514 | 0.514 | 0.514 | 0.514 | 0.514 | 0.514 |
| aug_norm | single_p99 | 0.514 | 0.514 | 0.514 | 0.514 | 0.514 | 0.514 |
| gain_bandflux | none | 0.599 | 0.599 | 0.599 | 0.599 | 0.599 | 0.599 |
| gain_bandflux | per_object | 0.373 | 0.373 | 0.373 | 0.373 | 0.373 | 0.373 |
| gain_bandflux | per_object_mean | 0.373 | 0.373 | 0.373 | 0.373 | 0.373 | 0.373 |
| gain_bandflux | per_object_p99 | 0.373 | 0.373 | 0.373 | 0.373 | 0.373 | 0.373 |
| gain_bandflux | single | 0.542 | 0.542 | 0.542 | 0.542 | 0.542 | 0.542 |
| gain_bandflux | single_mean | 0.542 | 0.542 | 0.542 | 0.542 | 0.542 | 0.542 |
| gain_bandflux | single_p99 | 0.542 | 0.542 | 0.542 | 0.542 | 0.542 | 0.542 |
| gain_flux | none | 0.741 | 0.741 | 0.741 | 0.741 | 0.741 | 0.741 |
| gain_flux | per_object | 0.52 | 0.52 | 0.52 | 0.52 | 0.52 | 0.52 |
| gain_flux | per_object_mean | 0.52 | 0.52 | 0.52 | 0.52 | 0.52 | 0.52 |
| gain_flux | per_object_p99 | 0.52 | 0.52 | 0.52 | 0.52 | 0.52 | 0.52 |
| gain_flux | single | 0.672 | 0.672 | 0.672 | 0.672 | 0.672 | 0.672 |
| gain_flux | single_mean | 0.672 | 0.672 | 0.672 | 0.672 | 0.672 | 0.672 |
| gain_flux | single_p99 | 0.672 | 0.672 | 0.672 | 0.672 | 0.672 | 0.672 |
| gain_p99 | none | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 |
| gain_p99 | per_object | 0.551 | 0.551 | 0.551 | 0.551 | 0.551 | 0.551 |
| gain_p99 | per_object_mean | 0.551 | 0.551 | 0.551 | 0.551 | 0.551 | 0.551 |
| gain_p99 | per_object_p99 | 0.551 | 0.551 | 0.551 | 0.551 | 0.551 | 0.551 |
| gain_p99 | single | 0.697 | 0.697 | 0.697 | 0.697 | 0.697 | 0.697 |
| gain_p99 | single_mean | 0.697 | 0.697 | 0.697 | 0.697 | 0.697 | 0.697 |
| gain_p99 | single_p99 | 0.697 | 0.697 | 0.697 | 0.697 | 0.697 | 0.697 |
| gain_pixel | none | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 |
| gain_pixel | per_object | 0.472 | 0.472 | 0.472 | 0.472 | 0.472 | 0.472 |
| gain_pixel | per_object_mean | 0.472 | 0.472 | 0.472 | 0.472 | 0.472 | 0.472 |
| gain_pixel | per_object_p99 | 0.472 | 0.472 | 0.472 | 0.472 | 0.472 | 0.472 |
| gain_pixel | single | 0.631 | 0.631 | 0.631 | 0.631 | 0.631 | 0.631 |
| gain_pixel | single_mean | 0.631 | 0.631 | 0.631 | 0.631 | 0.631 | 0.631 |
| gain_pixel | single_p99 | 0.631 | 0.631 | 0.631 | 0.631 | 0.631 | 0.631 |
| norm | none | 0.681 | 0.681 | 0.681 | 0.681 | 0.681 | 0.681 |
| norm | per_object | 0.348 | 0.348 | 0.348 | 0.348 | 0.348 | 0.348 |
| norm | per_object_mean | 0.348 | 0.348 | 0.348 | 0.348 | 0.348 | 0.348 |
| norm | per_object_p99 | 0.348 | 0.348 | 0.348 | 0.348 | 0.348 | 0.348 |
| norm | single | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 |
| norm | single_mean | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 |
| norm | single_p99 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 |
| original | none | 0 | 0 | 0.003 | 0.735 | 0.054 | 0.019 |
| original | per_object | 0.186 | 0.186 | 0.186 | 0.186 | 0.186 | 0.186 |
| original | per_object_mean | 0.064 | 0.064 | 0.064 | 0.064 | 0.064 | 0.064 |
| original | per_object_p99 | 0.163 | 0.163 | 0.163 | 0.163 | 0.163 | 0.163 |
| original | single | 0.262 | 0.262 | 0.262 | 0.262 | 0.262 | 0.262 |
| original | single_mean | 0.088 | 0.088 | 0.088 | 0.088 | 0.088 | 0.088 |
| original | single_p99 | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 |
| original_retrained | none | 0 | 0.001 | 0.004 | 0.846 | 0.03 | 0.028 |
| original_retrained | per_object | 0.216 | 0.216 | 0.216 | 0.216 | 0.216 | 0.216 |
| original_retrained | per_object_mean | 0.055 | 0.055 | 0.055 | 0.055 | 0.055 | 0.055 |
| original_retrained | per_object_p99 | 0.19 | 0.19 | 0.19 | 0.19 | 0.19 | 0.19 |
| original_retrained | single | 0.307 | 0.307 | 0.307 | 0.307 | 0.307 | 0.307 |
| original_retrained | single_mean | 0.075 | 0.075 | 0.075 | 0.075 | 0.075 | 0.075 |
| original_retrained | single_p99 | 0.27 | 0.27 | 0.27 | 0.27 | 0.27 | 0.27 |
| pair_max1 | none | 0 | 0 | 0.004 | 0.759 | 0.109 | 0.025 |
| pair_max1 | per_object | 0.337 | 0.337 | 0.337 | 0.337 | 0.337 | 0.337 |
| pair_max1 | per_object_mean | 0.278 | 0.278 | 0.278 | 0.278 | 0.278 | 0.278 |
| pair_max1 | per_object_p99 | 0.333 | 0.333 | 0.333 | 0.333 | 0.333 | 0.333 |
| pair_max1 | single | 0.666 | 0.666 | 0.666 | 0.666 | 0.666 | 0.666 |
| pair_max1 | single_mean | 0.469 | 0.469 | 0.469 | 0.469 | 0.469 | 0.469 |
| pair_max1 | single_p99 | 0.662 | 0.662 | 0.662 | 0.662 | 0.662 | 0.662 |
| pair_max1_aug | none | 0 | 0.002 | 0.48 | 0.755 | 0.222 | 0.03 |
| pair_max1_aug | per_object | 0.533 | 0.533 | 0.533 | 0.533 | 0.533 | 0.533 |
| pair_max1_aug | per_object_mean | 0.522 | 0.522 | 0.522 | 0.522 | 0.522 | 0.522 |
| pair_max1_aug | per_object_p99 | 0.534 | 0.534 | 0.534 | 0.534 | 0.534 | 0.534 |
| pair_max1_aug | single | 0.683 | 0.683 | 0.683 | 0.683 | 0.683 | 0.683 |
| pair_max1_aug | single_mean | 0.668 | 0.668 | 0.668 | 0.668 | 0.668 | 0.668 |
| pair_max1_aug | single_p99 | 0.682 | 0.682 | 0.682 | 0.682 | 0.682 | 0.682 |
| pair_mean | none | 0 | 0 | 0.002 | 0.674 | 0.275 | 0.015 |
| pair_mean | per_object | 0.231 | 0.231 | 0.231 | 0.231 | 0.231 | 0.231 |
| pair_mean | per_object_mean | 0.316 | 0.316 | 0.316 | 0.316 | 0.316 | 0.316 |
| pair_mean | per_object_p99 | 0.222 | 0.222 | 0.222 | 0.222 | 0.222 | 0.222 |
| pair_mean | single | 0.327 | 0.327 | 0.327 | 0.327 | 0.327 | 0.327 |
| pair_mean | single_mean | 0.488 | 0.488 | 0.488 | 0.488 | 0.488 | 0.488 |
| pair_mean | single_p99 | 0.323 | 0.323 | 0.323 | 0.323 | 0.323 | 0.323 |
| pair_mean_aug | none | 0 | 0 | 0.383 | 0.697 | 0.466 | 0.051 |
| pair_mean_aug | per_object | 0.476 | 0.476 | 0.476 | 0.476 | 0.476 | 0.476 |
| pair_mean_aug | per_object_mean | 0.466 | 0.466 | 0.466 | 0.466 | 0.466 | 0.466 |
| pair_mean_aug | per_object_p99 | 0.473 | 0.473 | 0.473 | 0.473 | 0.473 | 0.473 |
| pair_mean_aug | single | 0.615 | 0.615 | 0.615 | 0.615 | 0.615 | 0.615 |
| pair_mean_aug | single_mean | 0.602 | 0.602 | 0.602 | 0.602 | 0.602 | 0.602 |
| pair_mean_aug | single_p99 | 0.611 | 0.611 | 0.611 | 0.611 | 0.611 | 0.611 |
| pair_p99 | none | 0.006 | 0.001 | 0.009 | 0.83 | 0.139 | 0.021 |
| pair_p99 | per_object | 0.313 | 0.313 | 0.313 | 0.313 | 0.313 | 0.313 |
| pair_p99 | per_object_mean | 0.328 | 0.328 | 0.328 | 0.328 | 0.328 | 0.328 |
| pair_p99 | per_object_p99 | 0.31 | 0.31 | 0.31 | 0.31 | 0.31 | 0.31 |
| pair_p99 | single | 0.739 | 0.739 | 0.739 | 0.739 | 0.739 | 0.739 |
| pair_p99 | single_mean | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 |
| pair_p99 | single_p99 | 0.739 | 0.739 | 0.739 | 0.739 | 0.739 | 0.739 |
| pair_p99_aug | none | 0 | 0.002 | 0.499 | 0.753 | 0.268 | 0.024 |
| pair_p99_aug | per_object | 0.53 | 0.53 | 0.53 | 0.53 | 0.53 | 0.53 |
| pair_p99_aug | per_object_mean | 0.525 | 0.525 | 0.525 | 0.525 | 0.525 | 0.525 |
| pair_p99_aug | per_object_p99 | 0.53 | 0.53 | 0.53 | 0.53 | 0.53 | 0.53 |
| pair_p99_aug | single | 0.68 | 0.68 | 0.68 | 0.68 | 0.68 | 0.68 |
| pair_p99_aug | single_mean | 0.677 | 0.677 | 0.677 | 0.677 | 0.677 | 0.677 |
| pair_p99_aug | single_p99 | 0.68 | 0.68 | 0.68 | 0.68 | 0.68 | 0.68 |

## A bright and a dim object in one image (columns = log10 of the dim object's brightness)

`acc_bright`:

| model | mode | -1.5 | -1.0 | -0.5 | 0.0 |
|---|---|---|---|---|---|
| aug | none | 0.504 | 0.516 | 0.532 | 0.578 |
| aug | per_object | 0.492 | 0.483 | 0.461 | 0.387 |
| aug | per_object_mean | 0.458 | 0.454 | 0.437 | 0.367 |
| aug | per_object_p99 | 0.492 | 0.482 | 0.46 | 0.386 |
| aug | single | 0.507 | 0.51 | 0.522 | 0.577 |
| aug | single_mean | 0.452 | 0.455 | 0.445 | 0.539 |
| aug | single_p99 | 0.506 | 0.508 | 0.518 | 0.575 |
| aug_norm | none | 0.532 | 0.549 | 0.573 | 0.569 |
| aug_norm | per_object | 0.51 | 0.505 | 0.483 | 0.407 |
| aug_norm | per_object_mean | 0.51 | 0.505 | 0.483 | 0.407 |
| aug_norm | per_object_p99 | 0.51 | 0.505 | 0.483 | 0.407 |
| aug_norm | single | 0.532 | 0.549 | 0.573 | 0.569 |
| aug_norm | single_mean | 0.532 | 0.549 | 0.573 | 0.569 |
| aug_norm | single_p99 | 0.532 | 0.549 | 0.573 | 0.569 |
| gain_bandflux | none | 0.31 | 0.341 | 0.401 | 0.508 |
| gain_bandflux | per_object | 0.282 | 0.274 | 0.249 | 0.184 |
| gain_bandflux | per_object_mean | 0.282 | 0.274 | 0.249 | 0.184 |
| gain_bandflux | per_object_p99 | 0.282 | 0.274 | 0.249 | 0.184 |
| gain_bandflux | single | 0.31 | 0.341 | 0.401 | 0.508 |
| gain_bandflux | single_mean | 0.31 | 0.341 | 0.401 | 0.508 |
| gain_bandflux | single_p99 | 0.31 | 0.341 | 0.401 | 0.508 |
| gain_flux | none | 0.561 | 0.586 | 0.636 | 0.682 |
| gain_flux | per_object | 0.548 | 0.537 | 0.509 | 0.422 |
| gain_flux | per_object_mean | 0.548 | 0.537 | 0.509 | 0.422 |
| gain_flux | per_object_p99 | 0.548 | 0.537 | 0.509 | 0.422 |
| gain_flux | single | 0.561 | 0.586 | 0.636 | 0.682 |
| gain_flux | single_mean | 0.561 | 0.586 | 0.636 | 0.682 |
| gain_flux | single_p99 | 0.561 | 0.586 | 0.636 | 0.682 |
| gain_p99 | none | 0.608 | 0.651 | 0.681 | 0.709 |
| gain_p99 | per_object | 0.58 | 0.571 | 0.545 | 0.466 |
| gain_p99 | per_object_mean | 0.58 | 0.571 | 0.545 | 0.466 |
| gain_p99 | per_object_p99 | 0.58 | 0.571 | 0.545 | 0.466 |
| gain_p99 | single | 0.608 | 0.651 | 0.681 | 0.709 |
| gain_p99 | single_mean | 0.608 | 0.651 | 0.681 | 0.709 |
| gain_p99 | single_p99 | 0.608 | 0.651 | 0.681 | 0.709 |
| gain_pixel | none | 0.514 | 0.564 | 0.6 | 0.621 |
| gain_pixel | per_object | 0.491 | 0.481 | 0.452 | 0.388 |
| gain_pixel | per_object_mean | 0.491 | 0.481 | 0.452 | 0.388 |
| gain_pixel | per_object_p99 | 0.491 | 0.481 | 0.452 | 0.388 |
| gain_pixel | single | 0.514 | 0.564 | 0.6 | 0.621 |
| gain_pixel | single_mean | 0.514 | 0.564 | 0.6 | 0.621 |
| gain_pixel | single_p99 | 0.514 | 0.564 | 0.6 | 0.621 |
| norm | none | 0.67 | 0.678 | 0.682 | 0.662 |
| norm | per_object | 0.383 | 0.371 | 0.332 | 0.263 |
| norm | per_object_mean | 0.383 | 0.371 | 0.332 | 0.263 |
| norm | per_object_p99 | 0.383 | 0.371 | 0.332 | 0.263 |
| norm | single | 0.67 | 0.678 | 0.682 | 0.662 |
| norm | single_mean | 0.67 | 0.678 | 0.682 | 0.662 |
| norm | single_p99 | 0.67 | 0.678 | 0.682 | 0.662 |
| original | none | 0.712 | 0.713 | 0.726 | 0.714 |
| original | per_object | 0.351 | 0.344 | 0.318 | 0.247 |
| original | per_object_mean | 0.079 | 0.084 | 0.09 | 0.09 |
| original | per_object_p99 | 0.283 | 0.281 | 0.261 | 0.208 |
| original | single | 0.37 | 0.368 | 0.366 | 0.422 |
| original | single_mean | 0.079 | 0.084 | 0.091 | 0.136 |
| original | single_p99 | 0.298 | 0.3 | 0.294 | 0.343 |
| original_retrained | none | 0.766 | 0.781 | 0.79 | 0.789 |
| original_retrained | per_object | 0.325 | 0.317 | 0.29 | 0.222 |
| original_retrained | per_object_mean | 0.057 | 0.065 | 0.074 | 0.078 |
| original_retrained | per_object_p99 | 0.246 | 0.246 | 0.227 | 0.178 |
| original_retrained | single | 0.363 | 0.369 | 0.377 | 0.449 |
| original_retrained | single_mean | 0.056 | 0.065 | 0.076 | 0.137 |
| original_retrained | single_p99 | 0.276 | 0.284 | 0.284 | 0.363 |
| pair_max1 | none | 0.704 | 0.715 | 0.724 | 0.716 |
| pair_max1 | per_object | 0.348 | 0.333 | 0.3 | 0.241 |
| pair_max1 | per_object_mean | 0.24 | 0.238 | 0.222 | 0.187 |
| pair_max1 | per_object_p99 | 0.331 | 0.316 | 0.287 | 0.234 |
| pair_max1 | single | 0.701 | 0.71 | 0.717 | 0.712 |
| pair_max1 | single_mean | 0.26 | 0.299 | 0.324 | 0.425 |
| pair_max1 | single_p99 | 0.692 | 0.699 | 0.706 | 0.704 |
| pair_max1_aug | none | 0.565 | 0.594 | 0.635 | 0.693 |
| pair_max1_aug | per_object | 0.548 | 0.537 | 0.517 | 0.442 |
| pair_max1_aug | per_object_mean | 0.55 | 0.547 | 0.526 | 0.448 |
| pair_max1_aug | per_object_p99 | 0.552 | 0.54 | 0.52 | 0.445 |
| pair_max1_aug | single | 0.581 | 0.608 | 0.644 | 0.698 |
| pair_max1_aug | single_mean | 0.557 | 0.594 | 0.618 | 0.688 |
| pair_max1_aug | single_p99 | 0.585 | 0.611 | 0.645 | 0.699 |
| pair_mean | none | 0.671 | 0.673 | 0.679 | 0.657 |
| pair_mean | per_object | 0.4 | 0.393 | 0.365 | 0.294 |
| pair_mean | per_object_mean | 0.339 | 0.328 | 0.303 | 0.25 |
| pair_mean | per_object_p99 | 0.356 | 0.351 | 0.33 | 0.267 |
| pair_mean | single | 0.45 | 0.448 | 0.444 | 0.464 |
| pair_mean | single_mean | 0.524 | 0.533 | 0.562 | 0.563 |
| pair_mean | single_p99 | 0.431 | 0.429 | 0.422 | 0.442 |
| pair_mean_aug | none | 0.601 | 0.615 | 0.657 | 0.676 |
| pair_mean_aug | per_object | 0.575 | 0.567 | 0.539 | 0.453 |
| pair_mean_aug | per_object_mean | 0.571 | 0.562 | 0.534 | 0.446 |
| pair_mean_aug | per_object_p99 | 0.575 | 0.567 | 0.539 | 0.452 |
| pair_mean_aug | single | 0.594 | 0.604 | 0.64 | 0.671 |
| pair_mean_aug | single_mean | 0.586 | 0.596 | 0.619 | 0.663 |
| pair_mean_aug | single_p99 | 0.592 | 0.602 | 0.633 | 0.667 |
| pair_p99 | none | 0.736 | 0.753 | 0.764 | 0.751 |
| pair_p99 | per_object | 0.292 | 0.276 | 0.242 | 0.183 |
| pair_p99 | per_object_mean | 0.204 | 0.2 | 0.182 | 0.146 |
| pair_p99 | per_object_p99 | 0.266 | 0.248 | 0.218 | 0.168 |
| pair_p99 | single | 0.744 | 0.758 | 0.763 | 0.75 |
| pair_p99 | single_mean | 0.292 | 0.319 | 0.337 | 0.464 |
| pair_p99 | single_p99 | 0.743 | 0.755 | 0.764 | 0.747 |
| pair_p99_aug | none | 0.582 | 0.608 | 0.647 | 0.705 |
| pair_p99_aug | per_object | 0.563 | 0.55 | 0.525 | 0.447 |
| pair_p99_aug | per_object_mean | 0.558 | 0.554 | 0.532 | 0.457 |
| pair_p99_aug | per_object_p99 | 0.567 | 0.555 | 0.527 | 0.449 |
| pair_p99_aug | single | 0.595 | 0.619 | 0.655 | 0.712 |
| pair_p99_aug | single_mean | 0.577 | 0.613 | 0.642 | 0.7 |
| pair_p99_aug | single_p99 | 0.598 | 0.621 | 0.656 | 0.714 |

`acc_dim`:

| model | mode | -1.5 | -1.0 | -0.5 | 0.0 |
|---|---|---|---|---|---|
| aug | none | 0 | 0.002 | 0.049 | 0.575 |
| aug | per_object | 0 | 0.003 | 0.12 | 0.4 |
| aug | per_object_mean | 0 | 0.003 | 0.111 | 0.385 |
| aug | per_object_p99 | 0 | 0.003 | 0.118 | 0.399 |
| aug | single | 0 | 0.003 | 0.052 | 0.576 |
| aug | single_mean | 0.002 | 0.005 | 0.051 | 0.536 |
| aug | single_p99 | 0 | 0.004 | 0.053 | 0.572 |
| aug_norm | none | 0 | 0.002 | 0.015 | 0.569 |
| aug_norm | per_object | 0 | 0.002 | 0.159 | 0.423 |
| aug_norm | per_object_mean | 0 | 0.002 | 0.159 | 0.423 |
| aug_norm | per_object_p99 | 0 | 0.002 | 0.159 | 0.423 |
| aug_norm | single | 0 | 0.002 | 0.015 | 0.569 |
| aug_norm | single_mean | 0 | 0.002 | 0.015 | 0.569 |
| aug_norm | single_p99 | 0 | 0.002 | 0.015 | 0.569 |
| gain_bandflux | none | 0.002 | 0.002 | 0.015 | 0.535 |
| gain_bandflux | per_object | 0 | 0.001 | 0.042 | 0.209 |
| gain_bandflux | per_object_mean | 0 | 0.001 | 0.042 | 0.209 |
| gain_bandflux | per_object_p99 | 0 | 0.001 | 0.042 | 0.209 |
| gain_bandflux | single | 0.002 | 0.002 | 0.015 | 0.535 |
| gain_bandflux | single_mean | 0.002 | 0.002 | 0.015 | 0.535 |
| gain_bandflux | single_p99 | 0.002 | 0.002 | 0.015 | 0.535 |
| gain_flux | none | 0.001 | 0.001 | 0.021 | 0.683 |
| gain_flux | per_object | 0 | 0.001 | 0.104 | 0.464 |
| gain_flux | per_object_mean | 0 | 0.001 | 0.104 | 0.464 |
| gain_flux | per_object_p99 | 0 | 0.001 | 0.104 | 0.464 |
| gain_flux | single | 0.001 | 0.001 | 0.021 | 0.683 |
| gain_flux | single_mean | 0.001 | 0.001 | 0.021 | 0.683 |
| gain_flux | single_p99 | 0.001 | 0.001 | 0.021 | 0.683 |
| gain_p99 | none | 0.001 | 0.002 | 0.015 | 0.702 |
| gain_p99 | per_object | 0 | 0.002 | 0.154 | 0.512 |
| gain_p99 | per_object_mean | 0 | 0.002 | 0.154 | 0.512 |
| gain_p99 | per_object_p99 | 0 | 0.002 | 0.154 | 0.512 |
| gain_p99 | single | 0.001 | 0.002 | 0.015 | 0.702 |
| gain_p99 | single_mean | 0.001 | 0.002 | 0.015 | 0.702 |
| gain_p99 | single_p99 | 0.001 | 0.002 | 0.015 | 0.702 |
| gain_pixel | none | 0.001 | 0.007 | 0.143 | 0.615 |
| gain_pixel | per_object | 0 | 0.004 | 0.146 | 0.413 |
| gain_pixel | per_object_mean | 0 | 0.004 | 0.146 | 0.413 |
| gain_pixel | per_object_p99 | 0 | 0.004 | 0.146 | 0.413 |
| gain_pixel | single | 0.001 | 0.007 | 0.143 | 0.615 |
| gain_pixel | single_mean | 0.001 | 0.007 | 0.143 | 0.615 |
| gain_pixel | single_p99 | 0.001 | 0.007 | 0.143 | 0.615 |
| norm | none | 0.001 | 0.002 | 0.013 | 0.651 |
| norm | per_object | 0 | 0.001 | 0.077 | 0.274 |
| norm | per_object_mean | 0 | 0.001 | 0.077 | 0.274 |
| norm | per_object_p99 | 0 | 0.001 | 0.077 | 0.274 |
| norm | single | 0.001 | 0.002 | 0.013 | 0.651 |
| norm | single_mean | 0.001 | 0.002 | 0.013 | 0.651 |
| norm | single_p99 | 0.001 | 0.002 | 0.013 | 0.651 |
| original | none | 0.001 | 0.001 | 0.008 | 0.721 |
| original | per_object | 0 | 0.003 | 0.099 | 0.238 |
| original | per_object_mean | 0 | 0.009 | 0.056 | 0.088 |
| original | per_object_p99 | 0 | 0.004 | 0.091 | 0.209 |
| original | single | 0.003 | 0.005 | 0.036 | 0.38 |
| original | single_mean | 0.015 | 0.019 | 0.11 | 0.125 |
| original | single_p99 | 0.005 | 0.007 | 0.056 | 0.315 |
| original_retrained | none | 0 | 0.002 | 0.01 | 0.802 |
| original_retrained | per_object | 0 | 0.003 | 0.099 | 0.226 |
| original_retrained | per_object_mean | 0 | 0.007 | 0.051 | 0.078 |
| original_retrained | per_object_p99 | 0 | 0.003 | 0.087 | 0.193 |
| original_retrained | single | 0.001 | 0.004 | 0.037 | 0.41 |
| original_retrained | single_mean | 0.015 | 0.032 | 0.156 | 0.121 |
| original_retrained | single_p99 | 0.001 | 0.005 | 0.052 | 0.344 |
| pair_max1 | none | 0.001 | 0.001 | 0.009 | 0.729 |
| pair_max1 | per_object | 0 | 0 | 0.083 | 0.265 |
| pair_max1 | per_object_mean | 0 | 0.001 | 0.068 | 0.213 |
| pair_max1 | per_object_p99 | 0 | 0 | 0.082 | 0.259 |
| pair_max1 | single | 0 | 0.001 | 0.017 | 0.719 |
| pair_max1 | single_mean | 0.001 | 0.002 | 0.017 | 0.461 |
| pair_max1 | single_p99 | 0 | 0.001 | 0.018 | 0.706 |
| pair_max1_aug | none | 0.001 | 0.002 | 0.022 | 0.696 |
| pair_max1_aug | per_object | 0 | 0.002 | 0.147 | 0.485 |
| pair_max1_aug | per_object_mean | 0 | 0.002 | 0.152 | 0.498 |
| pair_max1_aug | per_object_p99 | 0 | 0.002 | 0.148 | 0.488 |
| pair_max1_aug | single | 0.001 | 0.002 | 0.023 | 0.699 |
| pair_max1_aug | single_mean | 0 | 0.002 | 0.017 | 0.692 |
| pair_max1_aug | single_p99 | 0.001 | 0.002 | 0.023 | 0.701 |
| pair_mean | none | 0.002 | 0.002 | 0.009 | 0.67 |
| pair_mean | per_object | 0 | 0.001 | 0.095 | 0.289 |
| pair_mean | per_object_mean | 0 | 0.001 | 0.074 | 0.261 |
| pair_mean | per_object_p99 | 0 | 0.001 | 0.09 | 0.267 |
| pair_mean | single | 0.002 | 0.003 | 0.029 | 0.455 |
| pair_mean | single_mean | 0 | 0.001 | 0.02 | 0.581 |
| pair_mean | single_p99 | 0.001 | 0.004 | 0.037 | 0.425 |
| pair_mean_aug | none | 0 | 0.002 | 0.026 | 0.676 |
| pair_mean_aug | per_object | 0 | 0.002 | 0.132 | 0.482 |
| pair_mean_aug | per_object_mean | 0 | 0.003 | 0.144 | 0.485 |
| pair_mean_aug | per_object_p99 | 0 | 0.002 | 0.133 | 0.481 |
| pair_mean_aug | single | 0 | 0.002 | 0.027 | 0.671 |
| pair_mean_aug | single_mean | 0 | 0.002 | 0.023 | 0.661 |
| pair_mean_aug | single_p99 | 0 | 0.002 | 0.026 | 0.666 |
| pair_p99 | none | 0.001 | 0.002 | 0.009 | 0.764 |
| pair_p99 | per_object | 0 | 0.002 | 0.043 | 0.211 |
| pair_p99 | per_object_mean | 0 | 0.003 | 0.034 | 0.162 |
| pair_p99 | per_object_p99 | 0 | 0.002 | 0.042 | 0.195 |
| pair_p99 | single | 0 | 0.002 | 0.011 | 0.759 |
| pair_p99 | single_mean | 0.002 | 0.002 | 0.011 | 0.474 |
| pair_p99 | single_p99 | 0 | 0.001 | 0.011 | 0.756 |
| pair_p99_aug | none | 0.001 | 0.002 | 0.024 | 0.697 |
| pair_p99_aug | per_object | 0 | 0.003 | 0.15 | 0.484 |
| pair_p99_aug | per_object_mean | 0 | 0.003 | 0.155 | 0.496 |
| pair_p99_aug | per_object_p99 | 0 | 0.003 | 0.152 | 0.486 |
| pair_p99_aug | single | 0.001 | 0.002 | 0.023 | 0.702 |
| pair_p99_aug | single_mean | 0.001 | 0.002 | 0.02 | 0.699 |
| pair_p99_aug | single_p99 | 0.001 | 0.002 | 0.024 | 0.703 |

`false_fg_share`:

| model | mode | -1.5 | -1.0 | -0.5 | 0.0 |
|---|---|---|---|---|---|
| aug | none | 0.011 | 0.011 | 0.007 | 0.005 |
| aug | per_object | 0.003 | 0.005 | 0.012 | 0.005 |
| aug | per_object_mean | 0.007 | 0.009 | 0.019 | 0.01 |
| aug | per_object_p99 | 0.004 | 0.005 | 0.013 | 0.006 |
| aug | single | 0.016 | 0.013 | 0.009 | 0.007 |
| aug | single_mean | 0.025 | 0.023 | 0.02 | 0.011 |
| aug | single_p99 | 0.017 | 0.014 | 0.01 | 0.007 |
| aug_norm | none | 0.006 | 0.011 | 0.006 | 0.003 |
| aug_norm | per_object | 0.002 | 0.002 | 0.003 | 0.003 |
| aug_norm | per_object_mean | 0.002 | 0.002 | 0.003 | 0.003 |
| aug_norm | per_object_p99 | 0.002 | 0.002 | 0.003 | 0.003 |
| aug_norm | single | 0.006 | 0.011 | 0.006 | 0.003 |
| aug_norm | single_mean | 0.006 | 0.011 | 0.006 | 0.003 |
| aug_norm | single_p99 | 0.006 | 0.011 | 0.006 | 0.003 |
| gain_bandflux | none | 0.024 | 0.012 | 0.006 | 0.005 |
| gain_bandflux | per_object | 0.003 | 0.003 | 0.005 | 0.004 |
| gain_bandflux | per_object_mean | 0.003 | 0.003 | 0.005 | 0.004 |
| gain_bandflux | per_object_p99 | 0.003 | 0.003 | 0.005 | 0.004 |
| gain_bandflux | single | 0.024 | 0.012 | 0.006 | 0.005 |
| gain_bandflux | single_mean | 0.024 | 0.012 | 0.006 | 0.005 |
| gain_bandflux | single_p99 | 0.024 | 0.012 | 0.006 | 0.005 |
| gain_flux | none | 0.024 | 0.013 | 0.01 | 0.004 |
| gain_flux | per_object | 0.002 | 0.003 | 0.003 | 0.003 |
| gain_flux | per_object_mean | 0.002 | 0.003 | 0.003 | 0.003 |
| gain_flux | per_object_p99 | 0.002 | 0.003 | 0.003 | 0.003 |
| gain_flux | single | 0.024 | 0.013 | 0.01 | 0.004 |
| gain_flux | single_mean | 0.024 | 0.013 | 0.01 | 0.004 |
| gain_flux | single_p99 | 0.024 | 0.013 | 0.01 | 0.004 |
| gain_p99 | none | 0.023 | 0.01 | 0.009 | 0.003 |
| gain_p99 | per_object | 0.001 | 0.001 | 0.002 | 0.002 |
| gain_p99 | per_object_mean | 0.001 | 0.001 | 0.002 | 0.002 |
| gain_p99 | per_object_p99 | 0.001 | 0.001 | 0.002 | 0.002 |
| gain_p99 | single | 0.023 | 0.01 | 0.009 | 0.003 |
| gain_p99 | single_mean | 0.023 | 0.01 | 0.009 | 0.003 |
| gain_p99 | single_p99 | 0.023 | 0.01 | 0.009 | 0.003 |
| gain_pixel | none | 0.025 | 0.012 | 0.006 | 0.004 |
| gain_pixel | per_object | 0.001 | 0.001 | 0.002 | 0.002 |
| gain_pixel | per_object_mean | 0.001 | 0.001 | 0.002 | 0.002 |
| gain_pixel | per_object_p99 | 0.001 | 0.001 | 0.002 | 0.002 |
| gain_pixel | single | 0.025 | 0.012 | 0.006 | 0.004 |
| gain_pixel | single_mean | 0.025 | 0.012 | 0.006 | 0.004 |
| gain_pixel | single_p99 | 0.025 | 0.012 | 0.006 | 0.004 |
| norm | none | 0.025 | 0.02 | 0.009 | 0.007 |
| norm | per_object | 0.008 | 0.009 | 0.017 | 0.021 |
| norm | per_object_mean | 0.008 | 0.009 | 0.017 | 0.021 |
| norm | per_object_p99 | 0.008 | 0.009 | 0.017 | 0.021 |
| norm | single | 0.025 | 0.02 | 0.009 | 0.007 |
| norm | single_mean | 0.025 | 0.02 | 0.009 | 0.007 |
| norm | single_p99 | 0.025 | 0.02 | 0.009 | 0.007 |
| original | none | 0.024 | 0.018 | 0.007 | 0.007 |
| original | per_object | 0.006 | 0.008 | 0.017 | 0.011 |
| original | per_object_mean | 0.112 | 0.102 | 0.11 | 0.126 |
| original | per_object_p99 | 0.01 | 0.011 | 0.021 | 0.015 |
| original | single | 0.042 | 0.031 | 0.021 | 0.019 |
| original | single_mean | 0.313 | 0.276 | 0.262 | 0.174 |
| original | single_p99 | 0.057 | 0.042 | 0.033 | 0.031 |
| original_retrained | none | 0.021 | 0.016 | 0.004 | 0.003 |
| original_retrained | per_object | 0.004 | 0.005 | 0.012 | 0.008 |
| original_retrained | per_object_mean | 0.067 | 0.06 | 0.065 | 0.072 |
| original_retrained | per_object_p99 | 0.007 | 0.008 | 0.016 | 0.012 |
| original_retrained | single | 0.021 | 0.021 | 0.011 | 0.009 |
| original_retrained | single_mean | 0.151 | 0.134 | 0.132 | 0.101 |
| original_retrained | single_p99 | 0.025 | 0.025 | 0.017 | 0.015 |
| pair_max1 | none | 0.021 | 0.017 | 0.004 | 0.004 |
| pair_max1 | per_object | 0.007 | 0.008 | 0.014 | 0.016 |
| pair_max1 | per_object_mean | 0.005 | 0.006 | 0.012 | 0.016 |
| pair_max1 | per_object_p99 | 0.007 | 0.008 | 0.014 | 0.018 |
| pair_max1 | single | 0.02 | 0.015 | 0.005 | 0.004 |
| pair_max1 | single_mean | 0.015 | 0.013 | 0.007 | 0.004 |
| pair_max1 | single_p99 | 0.019 | 0.015 | 0.005 | 0.004 |
| pair_max1_aug | none | 0.029 | 0.013 | 0.01 | 0.004 |
| pair_max1_aug | per_object | 0.002 | 0.002 | 0.003 | 0.003 |
| pair_max1_aug | per_object_mean | 0.002 | 0.002 | 0.003 | 0.003 |
| pair_max1_aug | per_object_p99 | 0.002 | 0.002 | 0.003 | 0.003 |
| pair_max1_aug | single | 0.027 | 0.013 | 0.01 | 0.005 |
| pair_max1_aug | single_mean | 0.021 | 0.012 | 0.008 | 0.004 |
| pair_max1_aug | single_p99 | 0.026 | 0.013 | 0.01 | 0.005 |
| pair_mean | none | 0.03 | 0.016 | 0.01 | 0.009 |
| pair_mean | per_object | 0.003 | 0.003 | 0.006 | 0.005 |
| pair_mean | per_object_mean | 0.005 | 0.005 | 0.005 | 0.007 |
| pair_mean | per_object_p99 | 0.003 | 0.003 | 0.005 | 0.005 |
| pair_mean | single | 0.021 | 0.015 | 0.01 | 0.008 |
| pair_mean | single_mean | 0.003 | 0.006 | 0.003 | 0.002 |
| pair_mean | single_p99 | 0.016 | 0.013 | 0.008 | 0.006 |
| pair_mean_aug | none | 0.015 | 0.011 | 0.006 | 0.004 |
| pair_mean_aug | per_object | 0.002 | 0.002 | 0.003 | 0.003 |
| pair_mean_aug | per_object_mean | 0.002 | 0.001 | 0.002 | 0.002 |
| pair_mean_aug | per_object_p99 | 0.002 | 0.002 | 0.003 | 0.003 |
| pair_mean_aug | single | 0.011 | 0.008 | 0.004 | 0.003 |
| pair_mean_aug | single_mean | 0.008 | 0.007 | 0.004 | 0.002 |
| pair_mean_aug | single_p99 | 0.01 | 0.008 | 0.004 | 0.003 |
| pair_p99 | none | 0.023 | 0.015 | 0.004 | 0.003 |
| pair_p99 | per_object | 0.009 | 0.012 | 0.031 | 0.025 |
| pair_p99 | per_object_mean | 0.051 | 0.05 | 0.072 | 0.073 |
| pair_p99 | per_object_p99 | 0.012 | 0.015 | 0.036 | 0.03 |
| pair_p99 | single | 0.021 | 0.015 | 0.004 | 0.003 |
| pair_p99 | single_mean | 0.064 | 0.047 | 0.039 | 0.036 |
| pair_p99 | single_p99 | 0.021 | 0.015 | 0.004 | 0.003 |
| pair_p99_aug | none | 0.025 | 0.013 | 0.01 | 0.003 |
| pair_p99_aug | per_object | 0.002 | 0.002 | 0.003 | 0.003 |
| pair_p99_aug | per_object_mean | 0.003 | 0.003 | 0.004 | 0.004 |
| pair_p99_aug | per_object_p99 | 0.002 | 0.002 | 0.003 | 0.003 |
| pair_p99_aug | single | 0.024 | 0.013 | 0.011 | 0.004 |
| pair_p99_aug | single_mean | 0.027 | 0.017 | 0.014 | 0.005 |
| pair_p99_aug | single_p99 | 0.024 | 0.013 | 0.011 | 0.004 |
