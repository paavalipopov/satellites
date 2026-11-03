# Brightness, system response and empty sky, without hand-set scales

This copy of the project adds opt-in ways to make the 10-band model cope with three things that the
manual steps of notebooks 03-06 ("max = 1", the companion's own scale, the solar division) do by hand:

1. **Overall brightness** of the image (distance, exposure, albedo).
2. **The response of the system**: sunlight x atmosphere x optics x detector multiply the reflectance by a
   smooth curve over wavelength that the simulator never had.
3. **Empty sky**: scenes with nothing in them, and the faint halo around a real object. A model that has
   only ever seen scenes WITH an object, and divides each input by its own maximum, blows up an empty
   window to full strength and finds "materials" in it.

With every new option off, the generator and the model are exactly what they were (checked: 402
generated arrays, 10 and 36 bands, all four scene types, bit-for-bit equal; the model gives the same
output to the last bit; `config.fingerprint()` is unchanged).

## Quick start (from the project root)

    python scripts/check_brightness_options.py      # 1 min, no GPU: every option does what it says
    python scripts/brightness_presets.py list       # the named variants

    bash slurm/submit_brightness_grid.sh            # train the nine new variants (10 runs each)
    VARIANT=full_max sbatch slurm/train_brightness.sh      # or just one

    sbatch slurm/compare_brightness.sh              # when they are done: tables + maps vs the original
    # or: python scripts/brightness_compare.py

Results of the comparison land in `outputs/brightness_compare/` (see "Comparing" below).

## The options

Generator (`SimConfig`, set with `+data.params.sim.<name>=...`; all off by default):

| option | what it does |
|---|---|
| `gain_log10_range=[lo,hi]`, `gain_stage` | each scene is multiplied by ONE random factor `10**U(lo,hi)`. `before_noise`: the object is dimmer against the same noise (low signal-to-noise). `after_noise`: the whole image is rescaled (a change of units) |
| `response_log10_std`, `response_knots` | each scene is multiplied by ONE random smooth curve over the bands: `log10(curve)` interpolates `response_knots` random values with this sd, mean 0 over the bands (so it changes the spectral shape, not the brightness). 4 knots, sd 0.2 means curves that vary by roughly a factor 2-3 across 400-914 nm |
| `empty_prob`, `empty_noise_log10_range` | with this probability a scene has NO object: the label is all background. Its noise is `sensor_noise_std x 10**U(lo,hi)`, from almost nothing (`estimate.fits`'s background is exactly 0) up to the full sensor noise |
| `halo_log10_range`, `halo_sigma_px` | a faint wide skirt around the object, labelled BACKGROUND: the object blurred by a Gaussian of sigma `U(halo_sigma_px)`, its peak `10**U(lo,hi)` times the object's peak. It keeps the object's spectrum. In `estimate.fits` the specks around the satellite are 1e-4 to 1e-2 of its peak |
| `artifact_std_range`, `artifact_sigma_px`, `artifact_band_corr`, `artifact_skew` | restoration-like noise that is not white Gaussian: smooth in space, correlated between neighbouring bands, skewed to the high side |
| `noise_bank`, `noise_bank_log10_scale` | real noise patches (a `.npy`) added after the sensor noise; build one with `scripts/make_noise_bank.py` |

Each option draws from its own random generator, so switching one on never changes the scene, its
spectra, its label or its sensor noise: a scene with an option off is the same scene.
`generate_sample(..., scene="empty")` makes an empty scene on request; `Sample.gain`, `.response`,
`.halo_amp` and `.artifact_std` record what was applied.

Model (`+model.params.input_norm=...`); every setting has the same layers and weights, so any checkpoint loads into all of them:

| `input_norm` | divides each input by | keeps |
|---|---|---|
| `none` | nothing (the paper's model) | everything, including absolute brightness |
| `max` | the brightest value of the input | the brightness differences between pixels |
| `p99` | its 99th percentile (`norm_quantile`): one hot pixel can no longer set the scale | the same |
| `flux` | the mean of all values (total flux per pixel) | the same; depends on how much of the window the object fills |
| `band_flux` | the mean of each band separately: also removes a smooth curve over the bands | pixel-to-pixel differences within a band; loses a spectral shape that is the same everywhere |
| `pixel` | the brightest band of each pixel: spectral SHAPE only | shape only: dark and bright materials with the same shape become identical, and every empty pixel becomes full strength |

`model.input_scale(x)` returns the number(s) used, and `model(x, scale=...)` divides by a number you give
instead, e.g. `model.input_scale(whole_image)`, so that a window with nothing in it is not blown up
(only meaningful for `max`; `brightness_eval` uses it for its "scale from whole image" rows).

## The variants (scripts/brightness_presets.py: the one place that defines them)

| variant | folder under `logs/` | what it is |
|---|---|---|
| (original) | `estimate_10band` | the original model and data |
| `aug` | `estimate_10band_aug` | random overall brightness; raw values |
| `norm` | `estimate_10band_norm` | divide by the max inside the model; ordinary training |
| `aug_norm` | `estimate_10band_aug_norm` | both |
| `gain_p99`, `gain_flux`, `gain_bandflux`, `gain_pixel` | `estimate_10band_<name>` | random brightness + the other ways to normalize |
| `curve_max` | `estimate_10band_curve_max` | `aug_norm` + a random smooth curve over the bands |
| `empty_max` | `estimate_10band_empty_max` | `aug_norm` + empty scenes, halo, restoration-like noise |
| `full_max`, `full_p99` | `estimate_10band_<name>` | all of it, normalized by max / by p99 |
| `full_raw` | `estimate_10band_full_raw` | all of the augmentation and NO normalization (can augmentation alone do it?) |

Same 10 bands and settings as `slurm/train_estimate.sh`. `GAIN_RANGE`, `GAIN_STAGE`, `COMPILE` and `EXTRA`
(more Hydra overrides) are environment variables of `train_brightness.sh`; `ARRAY=0-4` on the grid
script gives 5 runs each. Per-pixel normalization (`gain_pixel`) is in the list so that the claim "it
loses the brightness differences between materials" is measured, not assumed.

## Comparing with the previous results

    sbatch slurm/compare_brightness.sh          # or: bash slurm/compare_brightness.sh
    python scripts/brightness_compare.py [--baseline estimate_10band] [--experiments estimate_10band_full_max ...]

It compares `logs/estimate_10band` (the previous model) with every variant folder it finds. Outputs:

* `comparison_summary.md` / `.csv`: one row per experiment, the accuracies that matter, and the change from the original.
* `maps_side_by_side.png`: every experiment in a row: the true materials, the labels with the satellite's max set
  to 1 by hand, the labels with NO scale chosen, and the share of models that call a pixel an object.
* `maps_<experiment>.png`: notebook 06's figure for each experiment: true materials (object.fits); then, per row
  (by hand: satellite max = 1 / no scale chosen / companion on its own scale), the models' labels, the share of the
  models that call each pixel an object, and the share that give it its material. Green outline = the true objects.
* `synthetic_vs_gain.png`, `robustness.png` and the csv behind every number.

The accuracies in the table: material accuracy on synthetic satellite scenes (at the trained brightness, with the
whole image 10x brighter / dimmer, with the object dimmer against the same noise), with a smooth curve over the
bands, with a halo or restoration-like noise (how much background is wrongly given a material), on empty scenes, and
on `estimate.fits` (outline IoU and materials ARI against `object.fits`, as in notebooks 03 and 06; with the scale set by
hand and with none chosen; false pixels; a window with nothing in it).

## First results (a CPU proxy, NOT the cluster runs)

`outputs/brightness_compare_cpu_proxy/` holds what `scripts/brightness_compare.py` printed for the original model and the
twelve variants, each ONE model trained for 1000 steps on a CPU (material accuracy at the trained brightness is only 0.2-0.5
for all of them). Treat it as a check that the options and the comparison work and a first hint, not as a ranking: one model per
variant means the differences between variants are mostly noise. What looked solid enough to follow up on:

* **No scale chosen on `estimate.fits`**: the original labels 1574 of the 1681 pixels of the patch as material (IoU 0.00);
  `aug` alone is no better (1420 false pixels); every variant that divides by one number (max, p99, flux) or by band means labels 0-7 false pixels (per-pixel normalization: 72).
* **Dividing by the max keeps the background clean; the other scales do not always**: on a window with nothing in it but a
  speck, `gain_p99` labelled 1529 of 1681 pixels (a window that is 99% zeros has a 99th percentile of 0), `gain_bandflux` 40,
  `full_p99` 34, `gain_flux` 19, and the `max` variants 0-1. On the real file, whose background is exactly zero, `max` is the
  safer choice; `p99` needs a window that really is mostly object.
* **Per pixel (`gain_pixel`)**: lower accuracy (0.29 against 0.36 for `aug_norm`), and a blob of false material near the
  companion (72 false pixels). Your guess that it is not worth pursuing held up; it stays in the list only as the measured reference.
* **Halo, restoration-like noise, empty scenes**: training on them cuts the background wrongly given a material from 12.6%
  (original) to 0.3-0.7% at restoration-like noise of sd 0.03 (`empty_max`, `full_*`), and flattens the loss at a halo of 10% of the
  object's peak (0.50 for the original, 0.08-0.17 for `empty_max` and the `full_*` variants).
* **Smooth curve over the bands**: the original falls from 0.54 to 0.10 material accuracy as the curve's sd goes 0 -> 0.4;
  `curve_max` stays at 0.24-0.29 over the same range (but starts lower, and these models are undertrained).
* **Augmentation alone vs normalization**: `full_raw` (no normalization) is robust to the synthetic tests only inside the trained
  brightness range (0.08 at 10x brighter) and floods 95 background pixels on `estimate.fits`; the simple cutoff
  ("no scale chosen + cutoff": drop pixels below 1% of the image maximum) rescues it (IoU 0.80) and `aug` (0.73). Because the
  file's background is exactly zero, a cutoff is a strong, honest baseline to beat.
* **Real-file scores are not a ranking**: outline IoU on `estimate.fits` ranges 0.13-0.77 and the materials ARI -0.1 to 0.65 across
  the variants that normalize, with `empty_max`/`full_max` finding few pixels but giving them the right material (high ARI, low IoU) and `gain_flux`
  the reverse. The cluster runs (10 models, 200 epochs, a majority vote) are what will settle it.

## Things to know

* `estimate.fits` is not noisy in the way the simulator is: 99% of its pixels are exactly 0, and the specks
  around the satellite are 1e-4 to 1e-2 of its peak (training noise is 2e-2 of the object's brightness). That is
  why empty scenes use noise from nearly nothing up to the full sensor noise, and why real restoration noise is best
  supplied as a `noise_bank` (Fabien offered empty-region examples). A bank cut from `estimate.fits` itself makes
  that file a training source for the noise; score such a model on another restoration.
* Dividing by anything computed from a window amplifies a window with nothing in it. Training on empty scenes
  teaches the model what that looks like; taking the scale from the whole image (`input_scale` / `scale=`) avoids it;
  a cutoff on the data (the "no scale chosen + cutoff" row: drop pixels below 1% of the image maximum) is the blunt
  third option, reported separately.
* `flux` and `band_flux` divide by a mean that can be near zero or negative for a window of pure noise; they are
  the most fragile on empty windows, and `p99` uses `kthvalue`. If a job dies inside `torch.compile`, resubmit with
  `COMPILE=false`.
* The real `train.py` (Catalyst + Hydra) is not in the environment these were developed in: the generator, the model,
  the evaluation and a small CPU training were run end to end, the Hydra overrides were checked as strings
  (`python scripts/brightness_presets.py hydra full_max`), but not through `train.py` itself. If a job fails at start-up,
  the `.err` file will name the override.
* `scripts/brightness_quick_train.py` trains the same model for a few hundred steps on a CPU; it is a way to try a
  variant in ten minutes, not a replacement for the cluster runs.
