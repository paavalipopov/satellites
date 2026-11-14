# Brightness: training x scaling

Models: `original_retrained`, `aug`, `pair_max1_aug`, `fg3`, `halo`, `lightnoise`, `aug_fg3`, `aug_fg3_halo`, `aug_fg3_light`, `aug_fg3_halo_light`, `pair_max1_aug_fg3`, `pair_max1_aug_fg3_halo`, `pair_max1_aug_fg3_light`, `pair_max1_aug_fg3_halo_light`. Modes (scripts/auto_scale.py): `none` = the image as it is, `single` = one scale for the image, `per_object` = every object its own scale. The by-hand rows are the notebook's reference (a person picked the objects and the scales).

## One object, overall brightness changed (material accuracy; flat rows = brightness does not matter)

`before_noise`, columns = log10 of the gain:

| model | mode | -2.0 | -1.5 | -1.0 | -0.5 | 0.0 |
|---|---|---|---|---|---|---|
| aug | none | 0 | 0.094 | 0.236 | 0.445 | 0.598 |
| aug | per_object | 0 | 0 | 0.01 | 0.192 | 0.406 |
| aug | per_object_faint | 0 | 0 | 0.01 | 0.192 | 0.406 |
| aug | single | 0 | 0 | 0.017 | 0.249 | 0.529 |
| aug_fg3 | none | 0.003 | 0.18 | 0.383 | 0.625 | 0.764 |
| aug_fg3 | per_object | 0 | 0 | 0.007 | 0.178 | 0.522 |
| aug_fg3 | per_object_faint | 0 | 0 | 0.007 | 0.178 | 0.522 |
| aug_fg3 | single | 0 | 0 | 0.012 | 0.244 | 0.678 |
| aug_fg3_halo | none | 0.003 | 0.174 | 0.386 | 0.626 | 0.763 |
| aug_fg3_halo | per_object | 0 | 0 | 0.006 | 0.174 | 0.529 |
| aug_fg3_halo | per_object_faint | 0 | 0 | 0.006 | 0.174 | 0.529 |
| aug_fg3_halo | single | 0 | 0 | 0.014 | 0.235 | 0.677 |
| aug_fg3_halo_light | none | 0.002 | 0.171 | 0.38 | 0.622 | 0.766 |
| aug_fg3_halo_light | per_object | 0 | 0 | 0.008 | 0.197 | 0.536 |
| aug_fg3_halo_light | per_object_faint | 0 | 0 | 0.008 | 0.197 | 0.536 |
| aug_fg3_halo_light | single | 0 | 0 | 0.014 | 0.27 | 0.686 |
| aug_fg3_light | none | 0.002 | 0.167 | 0.382 | 0.624 | 0.768 |
| aug_fg3_light | per_object | 0 | 0 | 0.007 | 0.182 | 0.531 |
| aug_fg3_light | per_object_faint | 0 | 0 | 0.007 | 0.182 | 0.531 |
| aug_fg3_light | single | 0 | 0 | 0.014 | 0.27 | 0.684 |
| fg3 | none | 0.007 | 0.005 | 0.005 | 0.031 | 0.855 |
| fg3 | per_object | 0 | 0 | 0.007 | 0.111 | 0.223 |
| fg3 | per_object_faint | 0 | 0 | 0.007 | 0.111 | 0.223 |
| fg3 | single | 0 | 0 | 0.011 | 0.145 | 0.319 |
| halo | none | 0.003 | 0.002 | 0.004 | 0.017 | 0.842 |
| halo | per_object | 0 | 0 | 0.01 | 0.104 | 0.214 |
| halo | per_object_faint | 0 | 0 | 0.009 | 0.104 | 0.214 |
| halo | single | 0 | 0 | 0.013 | 0.126 | 0.308 |
| lightnoise | none | 0.001 | 0.001 | 0.003 | 0.016 | 0.842 |
| lightnoise | per_object | 0 | 0 | 0.008 | 0.111 | 0.22 |
| lightnoise | per_object_faint | 0 | 0 | 0.008 | 0.111 | 0.22 |
| lightnoise | single | 0 | 0 | 0.011 | 0.15 | 0.318 |
| original_retrained | none | 0.002 | 0.001 | 0.003 | 0.019 | 0.846 |
| original_retrained | per_object | 0 | 0 | 0.01 | 0.109 | 0.216 |
| original_retrained | per_object_faint | 0 | 0 | 0.01 | 0.109 | 0.216 |
| original_retrained | single | 0 | 0 | 0.011 | 0.143 | 0.307 |
| pair_max1_aug | none | 0.001 | 0.126 | 0.341 | 0.597 | 0.755 |
| pair_max1_aug | per_object | 0 | 0 | 0.023 | 0.321 | 0.533 |
| pair_max1_aug | per_object_faint | 0 | 0 | 0.023 | 0.321 | 0.533 |
| pair_max1_aug | single | 0 | 0 | 0.127 | 0.505 | 0.683 |
| pair_max1_aug_fg3 | none | 0.003 | 0.164 | 0.379 | 0.624 | 0.771 |
| pair_max1_aug_fg3 | per_object | 0 | 0 | 0.025 | 0.348 | 0.547 |
| pair_max1_aug_fg3 | per_object_faint | 0 | 0 | 0.025 | 0.348 | 0.547 |
| pair_max1_aug_fg3 | single | 0 | 0 | 0.146 | 0.538 | 0.7 |
| pair_max1_aug_fg3_halo | none | 0.004 | 0.158 | 0.371 | 0.618 | 0.769 |
| pair_max1_aug_fg3_halo | per_object | 0 | 0 | 0.028 | 0.34 | 0.544 |
| pair_max1_aug_fg3_halo | per_object_faint | 0 | 0 | 0.028 | 0.34 | 0.544 |
| pair_max1_aug_fg3_halo | single | 0 | 0 | 0.143 | 0.53 | 0.695 |
| pair_max1_aug_fg3_halo_light | none | 0.002 | 0.158 | 0.372 | 0.623 | 0.763 |
| pair_max1_aug_fg3_halo_light | per_object | 0 | 0 | 0.027 | 0.343 | 0.547 |
| pair_max1_aug_fg3_halo_light | per_object_faint | 0 | 0 | 0.027 | 0.343 | 0.547 |
| pair_max1_aug_fg3_halo_light | single | 0 | 0 | 0.137 | 0.531 | 0.691 |
| pair_max1_aug_fg3_light | none | 0.003 | 0.161 | 0.376 | 0.619 | 0.776 |
| pair_max1_aug_fg3_light | per_object | 0 | 0 | 0.027 | 0.347 | 0.553 |
| pair_max1_aug_fg3_light | per_object_faint | 0 | 0 | 0.027 | 0.347 | 0.553 |
| pair_max1_aug_fg3_light | single | 0 | 0 | 0.143 | 0.528 | 0.704 |

`after_noise`, columns = log10 of the gain:

| model | mode | -3.0 | -2.0 | -1.0 | 0.0 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|---|
| aug | none | 0 | 0 | 0.317 | 0.598 | 0.272 | 0.024 |
| aug | per_object | 0.406 | 0.406 | 0.406 | 0.406 | 0.406 | 0.406 |
| aug | per_object_faint | 0.406 | 0.406 | 0.406 | 0.406 | 0.406 | 0.406 |
| aug | single | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 |
| aug_fg3 | none | 0 | 0.002 | 0.518 | 0.764 | 0.295 | 0.022 |
| aug_fg3 | per_object | 0.522 | 0.522 | 0.522 | 0.522 | 0.522 | 0.522 |
| aug_fg3 | per_object_faint | 0.522 | 0.522 | 0.522 | 0.522 | 0.522 | 0.522 |
| aug_fg3 | single | 0.678 | 0.678 | 0.678 | 0.678 | 0.678 | 0.678 |
| aug_fg3_halo | none | 0 | 0.002 | 0.507 | 0.763 | 0.326 | 0.04 |
| aug_fg3_halo | per_object | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 |
| aug_fg3_halo | per_object_faint | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 |
| aug_fg3_halo | single | 0.677 | 0.677 | 0.677 | 0.677 | 0.677 | 0.677 |
| aug_fg3_halo_light | none | 0 | 0.007 | 0.472 | 0.766 | 0.324 | 0.032 |
| aug_fg3_halo_light | per_object | 0.536 | 0.536 | 0.536 | 0.536 | 0.536 | 0.536 |
| aug_fg3_halo_light | per_object_faint | 0.536 | 0.536 | 0.536 | 0.536 | 0.536 | 0.536 |
| aug_fg3_halo_light | single | 0.686 | 0.686 | 0.686 | 0.686 | 0.686 | 0.686 |
| aug_fg3_light | none | 0 | 0.004 | 0.47 | 0.768 | 0.299 | 0.033 |
| aug_fg3_light | per_object | 0.531 | 0.531 | 0.531 | 0.531 | 0.531 | 0.531 |
| aug_fg3_light | per_object_faint | 0.531 | 0.531 | 0.531 | 0.531 | 0.531 | 0.531 |
| aug_fg3_light | single | 0.684 | 0.684 | 0.684 | 0.684 | 0.684 | 0.684 |
| fg3 | none | 0 | 0.003 | 0.006 | 0.855 | 0.024 | 0.019 |
| fg3 | per_object | 0.223 | 0.223 | 0.223 | 0.223 | 0.223 | 0.223 |
| fg3 | per_object_faint | 0.223 | 0.223 | 0.223 | 0.223 | 0.223 | 0.223 |
| fg3 | single | 0.319 | 0.319 | 0.319 | 0.319 | 0.319 | 0.319 |
| halo | none | 0 | 0.001 | 0.005 | 0.842 | 0.02 | 0.007 |
| halo | per_object | 0.214 | 0.214 | 0.214 | 0.214 | 0.214 | 0.214 |
| halo | per_object_faint | 0.214 | 0.214 | 0.214 | 0.214 | 0.214 | 0.214 |
| halo | single | 0.308 | 0.308 | 0.308 | 0.308 | 0.308 | 0.308 |
| lightnoise | none | 0 | 0.002 | 0.004 | 0.842 | 0.058 | 0.029 |
| lightnoise | per_object | 0.22 | 0.22 | 0.22 | 0.22 | 0.22 | 0.22 |
| lightnoise | per_object_faint | 0.22 | 0.22 | 0.22 | 0.22 | 0.22 | 0.22 |
| lightnoise | single | 0.318 | 0.318 | 0.318 | 0.318 | 0.318 | 0.318 |
| original_retrained | none | 0 | 0.001 | 0.004 | 0.846 | 0.03 | 0.028 |
| original_retrained | per_object | 0.216 | 0.216 | 0.216 | 0.216 | 0.216 | 0.216 |
| original_retrained | per_object_faint | 0.216 | 0.216 | 0.216 | 0.216 | 0.216 | 0.216 |
| original_retrained | single | 0.307 | 0.307 | 0.307 | 0.307 | 0.307 | 0.307 |
| pair_max1_aug | none | 0 | 0.002 | 0.48 | 0.755 | 0.222 | 0.03 |
| pair_max1_aug | per_object | 0.533 | 0.533 | 0.533 | 0.533 | 0.533 | 0.533 |
| pair_max1_aug | per_object_faint | 0.533 | 0.533 | 0.533 | 0.533 | 0.533 | 0.533 |
| pair_max1_aug | single | 0.683 | 0.683 | 0.683 | 0.683 | 0.683 | 0.683 |
| pair_max1_aug_fg3 | none | 0 | 0.009 | 0.522 | 0.771 | 0.269 | 0.032 |
| pair_max1_aug_fg3 | per_object | 0.547 | 0.547 | 0.547 | 0.547 | 0.547 | 0.547 |
| pair_max1_aug_fg3 | per_object_faint | 0.547 | 0.547 | 0.547 | 0.547 | 0.547 | 0.547 |
| pair_max1_aug_fg3 | single | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 |
| pair_max1_aug_fg3_halo | none | 0 | 0.005 | 0.524 | 0.769 | 0.278 | 0.02 |
| pair_max1_aug_fg3_halo | per_object | 0.544 | 0.544 | 0.544 | 0.544 | 0.544 | 0.544 |
| pair_max1_aug_fg3_halo | per_object_faint | 0.544 | 0.544 | 0.544 | 0.544 | 0.544 | 0.544 |
| pair_max1_aug_fg3_halo | single | 0.695 | 0.695 | 0.695 | 0.695 | 0.695 | 0.695 |
| pair_max1_aug_fg3_halo_light | none | 0 | 0.006 | 0.516 | 0.763 | 0.237 | 0.018 |
| pair_max1_aug_fg3_halo_light | per_object | 0.547 | 0.547 | 0.547 | 0.547 | 0.547 | 0.547 |
| pair_max1_aug_fg3_halo_light | per_object_faint | 0.547 | 0.547 | 0.547 | 0.547 | 0.547 | 0.547 |
| pair_max1_aug_fg3_halo_light | single | 0.691 | 0.691 | 0.691 | 0.691 | 0.691 | 0.691 |
| pair_max1_aug_fg3_light | none | 0 | 0.006 | 0.509 | 0.776 | 0.261 | 0.029 |
| pair_max1_aug_fg3_light | per_object | 0.553 | 0.553 | 0.553 | 0.553 | 0.553 | 0.553 |
| pair_max1_aug_fg3_light | per_object_faint | 0.553 | 0.553 | 0.553 | 0.553 | 0.553 | 0.553 |
| pair_max1_aug_fg3_light | single | 0.704 | 0.704 | 0.704 | 0.704 | 0.704 | 0.704 |

## A bright and a dim object in one image (columns = log10 of the dim object's brightness)

`acc_bright`:

| model | mode | -1.5 | -1.0 | -0.5 | 0.0 |
|---|---|---|---|---|---|
| aug | none | 0.504 | 0.516 | 0.532 | 0.578 |
| aug | per_object | 0.492 | 0.483 | 0.461 | 0.387 |
| aug | per_object_faint | 0.492 | 0.483 | 0.461 | 0.387 |
| aug | single | 0.507 | 0.51 | 0.522 | 0.577 |
| aug_fg3 | none | 0.594 | 0.624 | 0.665 | 0.723 |
| aug_fg3 | per_object | 0.565 | 0.554 | 0.528 | 0.443 |
| aug_fg3 | per_object_faint | 0.565 | 0.554 | 0.528 | 0.443 |
| aug_fg3 | single | 0.608 | 0.627 | 0.659 | 0.722 |
| aug_fg3_halo | none | 0.558 | 0.603 | 0.649 | 0.715 |
| aug_fg3_halo | per_object | 0.531 | 0.522 | 0.5 | 0.426 |
| aug_fg3_halo | per_object_faint | 0.531 | 0.522 | 0.5 | 0.426 |
| aug_fg3_halo | single | 0.576 | 0.611 | 0.648 | 0.722 |
| aug_fg3_halo_light | none | 0.559 | 0.596 | 0.652 | 0.709 |
| aug_fg3_halo_light | per_object | 0.532 | 0.525 | 0.501 | 0.424 |
| aug_fg3_halo_light | per_object_faint | 0.532 | 0.525 | 0.501 | 0.424 |
| aug_fg3_halo_light | single | 0.579 | 0.608 | 0.652 | 0.715 |
| aug_fg3_light | none | 0.599 | 0.631 | 0.664 | 0.716 |
| aug_fg3_light | per_object | 0.559 | 0.548 | 0.522 | 0.441 |
| aug_fg3_light | per_object_faint | 0.559 | 0.548 | 0.522 | 0.441 |
| aug_fg3_light | single | 0.61 | 0.631 | 0.652 | 0.715 |
| fg3 | none | 0.769 | 0.787 | 0.8 | 0.802 |
| fg3 | per_object | 0.321 | 0.31 | 0.284 | 0.22 |
| fg3 | per_object_faint | 0.321 | 0.31 | 0.284 | 0.22 |
| fg3 | single | 0.36 | 0.367 | 0.376 | 0.453 |
| halo | none | 0.749 | 0.766 | 0.778 | 0.78 |
| halo | per_object | 0.313 | 0.308 | 0.286 | 0.219 |
| halo | per_object_faint | 0.313 | 0.308 | 0.286 | 0.219 |
| halo | single | 0.368 | 0.369 | 0.373 | 0.45 |
| lightnoise | none | 0.748 | 0.763 | 0.776 | 0.773 |
| lightnoise | per_object | 0.312 | 0.306 | 0.281 | 0.212 |
| lightnoise | per_object_faint | 0.312 | 0.306 | 0.281 | 0.212 |
| lightnoise | single | 0.338 | 0.345 | 0.35 | 0.425 |
| original_retrained | none | 0.766 | 0.781 | 0.79 | 0.789 |
| original_retrained | per_object | 0.325 | 0.317 | 0.29 | 0.222 |
| original_retrained | per_object_faint | 0.325 | 0.317 | 0.29 | 0.222 |
| original_retrained | single | 0.363 | 0.369 | 0.377 | 0.449 |
| pair_max1_aug | none | 0.565 | 0.594 | 0.635 | 0.693 |
| pair_max1_aug | per_object | 0.548 | 0.537 | 0.517 | 0.442 |
| pair_max1_aug | per_object_faint | 0.548 | 0.537 | 0.517 | 0.442 |
| pair_max1_aug | single | 0.581 | 0.608 | 0.644 | 0.698 |
| pair_max1_aug_fg3 | none | 0.603 | 0.632 | 0.666 | 0.719 |
| pair_max1_aug_fg3 | per_object | 0.58 | 0.567 | 0.543 | 0.466 |
| pair_max1_aug_fg3 | per_object_faint | 0.58 | 0.567 | 0.543 | 0.466 |
| pair_max1_aug_fg3 | single | 0.618 | 0.642 | 0.672 | 0.726 |
| pair_max1_aug_fg3_halo | none | 0.564 | 0.591 | 0.639 | 0.698 |
| pair_max1_aug_fg3_halo | per_object | 0.547 | 0.538 | 0.517 | 0.444 |
| pair_max1_aug_fg3_halo | per_object_faint | 0.547 | 0.538 | 0.517 | 0.444 |
| pair_max1_aug_fg3_halo | single | 0.576 | 0.602 | 0.645 | 0.707 |
| pair_max1_aug_fg3_halo_light | none | 0.569 | 0.604 | 0.645 | 0.709 |
| pair_max1_aug_fg3_halo_light | per_object | 0.555 | 0.547 | 0.527 | 0.456 |
| pair_max1_aug_fg3_halo_light | per_object_faint | 0.555 | 0.547 | 0.527 | 0.456 |
| pair_max1_aug_fg3_halo_light | single | 0.585 | 0.615 | 0.656 | 0.72 |
| pair_max1_aug_fg3_light | none | 0.603 | 0.636 | 0.667 | 0.727 |
| pair_max1_aug_fg3_light | per_object | 0.581 | 0.57 | 0.542 | 0.471 |
| pair_max1_aug_fg3_light | per_object_faint | 0.581 | 0.57 | 0.542 | 0.471 |
| pair_max1_aug_fg3_light | single | 0.623 | 0.648 | 0.674 | 0.735 |

`acc_dim`:

| model | mode | -1.5 | -1.0 | -0.5 | 0.0 |
|---|---|---|---|---|---|
| aug | none | 0 | 0.002 | 0.049 | 0.575 |
| aug | per_object | 0 | 0.003 | 0.12 | 0.4 |
| aug | per_object_faint | 0 | 0.003 | 0.121 | 0.4 |
| aug | single | 0 | 0.003 | 0.052 | 0.576 |
| aug_fg3 | none | 0.001 | 0.003 | 0.028 | 0.716 |
| aug_fg3 | per_object | 0 | 0.001 | 0.098 | 0.476 |
| aug_fg3 | per_object_faint | 0 | 0.001 | 0.099 | 0.476 |
| aug_fg3 | single | 0.001 | 0.003 | 0.033 | 0.713 |
| aug_fg3_halo | none | 0.002 | 0.003 | 0.034 | 0.704 |
| aug_fg3_halo | per_object | 0 | 0.001 | 0.096 | 0.451 |
| aug_fg3_halo | per_object_faint | 0 | 0.001 | 0.097 | 0.451 |
| aug_fg3_halo | single | 0.002 | 0.004 | 0.037 | 0.703 |
| aug_fg3_halo_light | none | 0.001 | 0.003 | 0.024 | 0.7 |
| aug_fg3_halo_light | per_object | 0 | 0.002 | 0.102 | 0.459 |
| aug_fg3_halo_light | per_object_faint | 0 | 0.002 | 0.103 | 0.459 |
| aug_fg3_halo_light | single | 0.001 | 0.003 | 0.029 | 0.7 |
| aug_fg3_light | none | 0.001 | 0.003 | 0.024 | 0.706 |
| aug_fg3_light | per_object | 0 | 0.001 | 0.099 | 0.469 |
| aug_fg3_light | per_object_faint | 0 | 0.001 | 0.1 | 0.469 |
| aug_fg3_light | single | 0.001 | 0.003 | 0.028 | 0.708 |
| fg3 | none | 0.002 | 0.002 | 0.013 | 0.811 |
| fg3 | per_object | 0 | 0.003 | 0.1 | 0.213 |
| fg3 | per_object_faint | 0 | 0.003 | 0.1 | 0.213 |
| fg3 | single | 0.002 | 0.005 | 0.046 | 0.417 |
| halo | none | 0 | 0.001 | 0.009 | 0.792 |
| halo | per_object | 0 | 0.003 | 0.098 | 0.221 |
| halo | per_object_faint | 0 | 0.003 | 0.099 | 0.221 |
| halo | single | 0.002 | 0.003 | 0.036 | 0.416 |
| lightnoise | none | 0 | 0.001 | 0.007 | 0.793 |
| lightnoise | per_object | 0 | 0.003 | 0.097 | 0.217 |
| lightnoise | per_object_faint | 0 | 0.003 | 0.098 | 0.217 |
| lightnoise | single | 0 | 0.003 | 0.031 | 0.411 |
| original_retrained | none | 0 | 0.002 | 0.01 | 0.802 |
| original_retrained | per_object | 0 | 0.003 | 0.099 | 0.226 |
| original_retrained | per_object_faint | 0 | 0.003 | 0.099 | 0.226 |
| original_retrained | single | 0.001 | 0.004 | 0.037 | 0.41 |
| pair_max1_aug | none | 0.001 | 0.002 | 0.022 | 0.696 |
| pair_max1_aug | per_object | 0 | 0.002 | 0.147 | 0.485 |
| pair_max1_aug | per_object_faint | 0 | 0.002 | 0.15 | 0.485 |
| pair_max1_aug | single | 0.001 | 0.002 | 0.023 | 0.699 |
| pair_max1_aug_fg3 | none | 0.002 | 0.003 | 0.027 | 0.713 |
| pair_max1_aug_fg3 | per_object | 0 | 0.001 | 0.145 | 0.489 |
| pair_max1_aug_fg3 | per_object_faint | 0 | 0.001 | 0.147 | 0.489 |
| pair_max1_aug_fg3 | single | 0.002 | 0.003 | 0.029 | 0.72 |
| pair_max1_aug_fg3_halo | none | 0.002 | 0.002 | 0.027 | 0.695 |
| pair_max1_aug_fg3_halo | per_object | 0 | 0.001 | 0.137 | 0.475 |
| pair_max1_aug_fg3_halo | per_object_faint | 0 | 0.001 | 0.139 | 0.475 |
| pair_max1_aug_fg3_halo | single | 0.002 | 0.003 | 0.028 | 0.703 |
| pair_max1_aug_fg3_halo_light | none | 0.001 | 0.002 | 0.022 | 0.699 |
| pair_max1_aug_fg3_halo_light | per_object | 0 | 0.002 | 0.151 | 0.49 |
| pair_max1_aug_fg3_halo_light | per_object_faint | 0 | 0.002 | 0.154 | 0.49 |
| pair_max1_aug_fg3_halo_light | single | 0.001 | 0.002 | 0.024 | 0.703 |
| pair_max1_aug_fg3_light | none | 0.001 | 0.003 | 0.024 | 0.714 |
| pair_max1_aug_fg3_light | per_object | 0 | 0.003 | 0.157 | 0.493 |
| pair_max1_aug_fg3_light | per_object_faint | 0 | 0.003 | 0.161 | 0.493 |
| pair_max1_aug_fg3_light | single | 0.001 | 0.003 | 0.027 | 0.72 |

`false_fg_share`:

| model | mode | -1.5 | -1.0 | -0.5 | 0.0 |
|---|---|---|---|---|---|
| aug | none | 0.011 | 0.011 | 0.007 | 0.005 |
| aug | per_object | 0.003 | 0.005 | 0.012 | 0.005 |
| aug | per_object_faint | 0.003 | 0.005 | 0.012 | 0.005 |
| aug | single | 0.016 | 0.013 | 0.009 | 0.007 |
| aug_fg3 | none | 0.049 | 0.03 | 0.021 | 0.01 |
| aug_fg3 | per_object | 0.004 | 0.005 | 0.009 | 0.005 |
| aug_fg3 | per_object_faint | 0.004 | 0.005 | 0.009 | 0.005 |
| aug_fg3 | single | 0.042 | 0.029 | 0.022 | 0.011 |
| aug_fg3_halo | none | 0.051 | 0.034 | 0.022 | 0.007 |
| aug_fg3_halo | per_object | 0.003 | 0.005 | 0.011 | 0.005 |
| aug_fg3_halo | per_object_faint | 0.003 | 0.005 | 0.011 | 0.005 |
| aug_fg3_halo | single | 0.05 | 0.036 | 0.024 | 0.008 |
| aug_fg3_halo_light | none | 0.036 | 0.031 | 0.023 | 0.007 |
| aug_fg3_halo_light | per_object | 0.003 | 0.004 | 0.009 | 0.004 |
| aug_fg3_halo_light | per_object_faint | 0.003 | 0.004 | 0.009 | 0.004 |
| aug_fg3_halo_light | single | 0.034 | 0.03 | 0.025 | 0.008 |
| aug_fg3_light | none | 0.045 | 0.029 | 0.024 | 0.008 |
| aug_fg3_light | per_object | 0.003 | 0.004 | 0.008 | 0.004 |
| aug_fg3_light | per_object_faint | 0.003 | 0.004 | 0.008 | 0.004 |
| aug_fg3_light | single | 0.034 | 0.025 | 0.023 | 0.008 |
| fg3 | none | 0.044 | 0.031 | 0.009 | 0.007 |
| fg3 | per_object | 0.007 | 0.008 | 0.017 | 0.012 |
| fg3 | per_object_faint | 0.007 | 0.008 | 0.017 | 0.012 |
| fg3 | single | 0.043 | 0.036 | 0.022 | 0.019 |
| halo | none | 0.023 | 0.015 | 0.003 | 0.003 |
| halo | per_object | 0.003 | 0.005 | 0.017 | 0.008 |
| halo | per_object_faint | 0.003 | 0.006 | 0.017 | 0.008 |
| halo | single | 0.027 | 0.024 | 0.012 | 0.01 |
| lightnoise | none | 0.013 | 0.014 | 0.003 | 0.003 |
| lightnoise | per_object | 0.004 | 0.005 | 0.011 | 0.008 |
| lightnoise | per_object_faint | 0.004 | 0.005 | 0.011 | 0.008 |
| lightnoise | single | 0.011 | 0.015 | 0.007 | 0.007 |
| original_retrained | none | 0.021 | 0.016 | 0.004 | 0.003 |
| original_retrained | per_object | 0.004 | 0.005 | 0.012 | 0.008 |
| original_retrained | per_object_faint | 0.004 | 0.005 | 0.012 | 0.008 |
| original_retrained | single | 0.021 | 0.021 | 0.011 | 0.009 |
| pair_max1_aug | none | 0.029 | 0.013 | 0.01 | 0.004 |
| pair_max1_aug | per_object | 0.002 | 0.002 | 0.003 | 0.003 |
| pair_max1_aug | per_object_faint | 0.002 | 0.002 | 0.003 | 0.003 |
| pair_max1_aug | single | 0.027 | 0.013 | 0.01 | 0.005 |
| pair_max1_aug_fg3 | none | 0.051 | 0.033 | 0.027 | 0.009 |
| pair_max1_aug_fg3 | per_object | 0.006 | 0.006 | 0.008 | 0.007 |
| pair_max1_aug_fg3 | per_object_faint | 0.006 | 0.006 | 0.008 | 0.007 |
| pair_max1_aug_fg3 | single | 0.049 | 0.033 | 0.028 | 0.01 |
| pair_max1_aug_fg3_halo | none | 0.05 | 0.033 | 0.023 | 0.007 |
| pair_max1_aug_fg3_halo | per_object | 0.004 | 0.004 | 0.006 | 0.005 |
| pair_max1_aug_fg3_halo | per_object_faint | 0.004 | 0.004 | 0.006 | 0.005 |
| pair_max1_aug_fg3_halo | single | 0.051 | 0.033 | 0.024 | 0.008 |
| pair_max1_aug_fg3_halo_light | none | 0.035 | 0.027 | 0.025 | 0.007 |
| pair_max1_aug_fg3_halo_light | per_object | 0.003 | 0.003 | 0.005 | 0.005 |
| pair_max1_aug_fg3_halo_light | per_object_faint | 0.003 | 0.003 | 0.005 | 0.005 |
| pair_max1_aug_fg3_halo_light | single | 0.036 | 0.027 | 0.027 | 0.007 |
| pair_max1_aug_fg3_light | none | 0.045 | 0.033 | 0.033 | 0.012 |
| pair_max1_aug_fg3_light | per_object | 0.006 | 0.006 | 0.009 | 0.007 |
| pair_max1_aug_fg3_light | per_object_faint | 0.006 | 0.006 | 0.009 | 0.007 |
| pair_max1_aug_fg3_light | single | 0.043 | 0.032 | 0.034 | 0.013 |
