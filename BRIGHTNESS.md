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
    sbatch slurm/submit_brightness_grid.sh          # the same, as a short job that does the submitting
    VARIANT=full_max sbatch slurm/train_brightness.sh      # or just one

    sbatch slurm/compare_brightness.sh              # when they are done: tables + maps vs the original
    # or: python scripts/brightness_compare.py

Results of the comparison land in `outputs/brightness_compare/` (see "Comparing" below).

How `estimate.fits` is read before the models see it is a switch too (see "Two ways to read estimate.fits"):

    READING=radiometry sbatch slurm/compare_brightness.sh     # the collaborators' recipe (default: solar = the previous results)
    python scripts/compare_readings.py                        # original + curve_max + empty_max, both readings side by side

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
| `add_empty`, `add_halo`, `add_noise` | `estimate_10band_<name>` | `aug_norm` + ONLY one part of `empty_max`: the empty scenes, the halo, or the restoration-like noise (which of them makes the model drop faint real structure?) |
| `pair_max1`, `pair_max1_aug` | `estimate_10band_<name>` | the rescaling is done to the TRAINING DATA, not inside the model: every training scene appears twice, as generated and divided by its brightest value (brightest = 1). The model has no scaling step of its own (`input_norm` stays `none`). `pair_max1_aug` also gives the "as generated" copy a random overall brightness |
| `pair_<max1\|mean\|p99>[_aug][_halo][_light]` | `estimate_10band_<name>` | the 24 pair variants: the second copy divided by its max, or by the MEAN or the 99th PERCENTILE of its object pixels (`+data.params.pair_copies=mean\|p99`); `_aug` random brightness on the first copy; `_halo` the halo of `empty_max`; `_light` the weaker restoration-like noise of `lightnoise`. At test time a `mean` / `p99` model should get the matching scaling (`single_mean` / `single_p99`) |
| `fg3`, `fg10`, `fg3_dice` | `estimate_10band_<name>` | the ORIGINAL data and model, but every object pixel counts 3x / 10x in the loss (`fg3_dice`: 3x plus a Dice term for object vs background). Aimed at the faint panels that models drop. `src/losses.py`; `train.py` builds the loss from `+loss.fg_weight=..`, `+loss.dice=..` |
| `lightnoise` | `estimate_10band_lightnoise` | the original, plus restoration-like noise 4x weaker at the top than in `empty_max` |
| `bank` | `estimate_10band_bank` | the original, plus real noise patches from the file in the environment variable `NOISE_BANK` (`scripts/make_noise_bank.py`; a bank cut from `estimate.fits` itself leaks that file into training) |
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

## Two ways to read estimate.fits

The values in `estimate.fits` and `object.fits` are detected electrons per pixel, per slice, per 5 ms frame: a
material's reflectance times what the telescope and camera do to each colour. A *reading* divides that out,
one number per slice, before the model sees the cube. `scripts/fits_reading.py` has three, and `--reading`
(or `READING=` for the sbatch scripts) chooses between them:

| reading | divides by | status |
|---|---|---|
| `solar` (default) | sunlight x band width, `E(lam) * dlam` (notebook 06) | the previous results; unchanged, bit for bit |
| `radiometry` | what the simulator applied, `G(lam) * lam * dlam` (atmosphere x optics x camera, photon counting, slice width) | the collaborators' recipe (`scripts/radiometry.py`, `recover_reflectance.py`), with the sun NOT divided out |
| `radiometry_sun` | the same, and a 6000 K sun | the default of `recover_reflectance.py` (`TRUTH_INCLUDES_SUN = True`); kept so the switch can be tested |

`scripts/radiometry.py` is the file received from the collaborators with the New_Data folder, copied unchanged
(it names no author; the self-test checks that it has not been edited). `fits_reading.py` holds the logic of
`recover_reflectance.py` (the wavelength grid, the divisor, relative reflectance, the disk spectrum) and as a script
does what that file does, without touching `assets/`: `python scripts/fits_reading.py`.

Why `radiometry`, not `radiometry_sun` (checked by `check_brightness_options.py` on `assets/new_input`): the truth
pixels fall into five spectra, the five materials of the reflectance figure. With `radiometry` the 35-pixel group (solar
panel) has 68% / 3.5% = about 19 between 400 and 640 nm in the figure and 21 here, and the 4-pixel group (aluminized mylar,
flat at 58-70% in the figure) has max/min 1.17. With `solar` the panel ratio is 4.0 and mylar is 6.2; with
`radiometry_sun` mylar is 2.2. So the sun is not in the truth cube, and the old reading tilts the colours.

Two cautions. (1) The recipe gives RELATIVE reflectance, with one overall constant free; `--reading-mag` sets it
(default 6, the example satellite's magnitude, which makes a pixel read about its reflectance times its share of the
light, around 0.1). The old reading's values are about 3700 times larger. A model that divides by the input's own
maximum does not care; the original model does. (2) The collaborators' atmosphere, optics and camera curves are
placeholders, and the simulation used the same ones, so the recovery works here by construction; real data needs the
measured throughput.

    python scripts/compare_readings.py                        # solar vs radiometry on the original, curve_max, empty_max
    python scripts/compare_readings.py --experiments estimate_10band estimate_10band_full_max --readings solar radiometry radiometry:1

It writes `outputs/brightness_readings/reading_comparison.md` / `.csv` and `readings_side_by_side.png`. Read the "by hand"
rows first: the satellite's max is set to 1, so only the colours change. `radiometry:1` is the collaborators' own constant
(mag 1) and shows what a "no scale chosen" row does when only the constant moves.

### First look (CPU proxy: ONE weak model per variant, not a ranking)

`outputs/brightness_readings_cpu_proxy/`, from the same single 1000-step models as above:

* **By hand (satellite max = 1): the colour change alone.** Outline IoU on the 41 x 41 patch: original 0.76 -> 0.89, curve_max 0.40 -> 0.76,
  empty_max 0.13 -> 0.13 (it labels only 6 pixels either way). Materials ARI: original 0.14 -> 0.28, curve_max 0.25 -> 0.04,
  empty_max 0.65 -> 0.67. So the reading moves the scores by about as much as the differences between these models do.
* **No scale chosen, the original model:** `solar` floods the patch (1574 false pixels, IoU 0.00); `radiometry` (mag 6) labels
  34 pixels, all on the satellite (IoU 0.76, ARI 0.57); `radiometry:1` (values 100 times SMALLER, about 0.001 against training noise of 0.02) labels nothing. The original
  model has no scale of its own, so this row depends on the arbitrary constant. `max` models (curve_max, empty_max) give the same
  result at every constant.
* **The companion pixel:** found by no model under any reading (the one pixel the flooded original model "finds" is chance).
* The synthetic and robustness tests are not affected by the reading, and were not rerun.

### Which part of `empty_max` costs the faint solar panels? (CPU proxy: 3 weak models per variant)

`add_empty`, `add_halo` and `add_noise` are `aug_norm` plus one part of `empty_max` (empty scenes / halo / restoration-like
noise). Three single-seed 1000-step models each (run 0-2, majority vote), `outputs/brightness_ablation_cpu_proxy/`. In
`estimate.fits` the 29 solar-panel pixels are only 10-20% as bright as the brightest pixel, and `empty_max` labels just the
bright middle (it was my guess that the halo, which is labelled background, did this; the numbers do not support that).

Outline IoU on the patch, satellite max = 1 by hand (majority vote of the 3 models; the same file read two ways):

| model | `radiometry` | `solar` |
|---|---|---|
| `aug_norm` (no extra part) | 0.76 | 0.51 |
| `add_halo` | 0.73 | 0.71 |
| `add_empty` | 0.60 | 0.40 |
| `empty_max` (all three) | 0.42 | 0.20 |
| `add_noise` | 0.31 | 0.36 |

* The halo does not cost the panels; the restoration-like noise does most of it, with the empty scenes in between. But the spread
  between the 3 models of one variant is large (`add_empty`: 0.11, 0.60, 0.80 under `radiometry`), so read this as a hint to check on
  the cluster runs (10 models each), not a result.
* What each part buys (synthetic tests, `radiometry` run): background given a material at restoration noise sd 0.03 falls from 5.4%
  (`aug_norm`) to 0.6% (`add_noise`) and 0.5% (`empty_max`) but only to 3.7% (`add_halo`) and 3.1% (`add_empty`); a halo at 1% of the object:
  1.2% -> 0.3-0.6%. Empty scenes buy nothing measurable here: every model, including `aug_norm`, labels nothing in an empty scene.
  Material accuracy at the trained brightness: 0.35 (`aug_norm`), 0.36 (`add_halo`), 0.32 (`add_empty`), 0.30 (`add_noise`), 0.27 (`empty_max`).
* So the noise part is a trade: it is what protects against restoration-like noise, and it is the part that makes the model
  drop faint real structure. Ideas: a smaller noise sd range, or mixing in faint real objects (labelled as objects) during training.
  Train them on the cluster with `bash slurm/submit_brightness_grid.sh add_empty add_halo add_noise`.

## Training on the originals and a "brightest = 1" copy (`pair_max1`, `pair_max1_aug`)

The alternative to scaling inside the model. With `data.params.max_one_copies=true` (set by these two variants) every
training scene is used twice: item 2k is scene k as generated, item 2k+1 is the same scene (same pixels, same noise,
same label) divided by its brightest value. This is what Pavel did by hand to estimate.fits, put into the training data.

* The model itself does no scaling (`input_norm=none`), so at test time it takes whatever is given: an image left as
  it is looks like the first kind of copy, an image rescaled by hand to max = 1 looks like the second.
* Only the TRAINING split changes. Valid and test scenes are as generated, so these models are scored on the same
  scenes as every other variant.
* An epoch has twice as many items, so it takes about twice as long (`n_train` is the number of scenes, not items).
* Caution: with empty scenes (`empty_prob > 0`) the second copy would blow pure noise up to brightness 1. The two
  variants do not use empty scenes; if you combine the option with them, look at the result.
* Hydra: `+data.params.max_one_copies=true`; `python scripts/brightness_presets.py hydra pair_max1`.
* Train: `bash slurm/submit_brightness_grid.sh pair_max1 pair_max1_aug`.

## Comparing the ways of handling brightness (`scripts/compare_scaling.py`, `slurm/compare_scaling.sh`)

Two separate choices, compared together so they can be read separately:

* **How the model was trained:** the original; `pair_max1` / `pair_max1_aug` (every scene twice, as generated and
  divided by its max); `aug_norm` (the model divides by the max itself, as a reference).
* **How the image is scaled in front of the model** (`scripts/auto_scale.py`, nothing learned, the model is untouched):
  `none` (as it is), `single` (one number for the image: the brightest object value, which is Pavel's "set it to 1"
  with no person involved) and `per_object` (every separate object its own number and its own run, then the answers
  are put back together: what the notebook did by hand for the companion).

What counts as an object: pixels brighter than 1% of the image's brightest AND than 5 times the noise level
(0 for `estimate.fits`); touching pixels form one blob; blobs under 3 pixels are specks; each blob also owns the 3
pixels around it (its faint skirt), and the rest is background. An image with no object in it is never given to the model.
Limits: two objects that touch are one object and get one scale; an object dimmer than 5 times the noise is not found;
one hot pixel inside an object sets its scale unless `q` is lowered (0.99).

Tests: (1) `estimate.fits` against `object.fits` for each reading, next to the notebook's by-hand rows; (2) synthetic
scenes at many overall brightnesses (a model that copes gives a flat row); (3) synthetic scenes with a bright and a dim
object in one image (the dim one 1x, 0.3x, 0.1x, 0.03x as bright), scored for each object separately: the case one scale
cannot fit.

    python scripts/compare_scaling.py                      # default: original, pair_max1, pair_max1_aug, aug_norm
    sbatch slurm/compare_scaling.sh                        # EXPERIMENTS, READINGS, N_SCENES, MAX_RUNS, OUT as variables

It trains nothing: train first (`bash slurm/submit_brightness_grid.sh pair_max1 pair_max1_aug`). Results:
`outputs/scaling_compare/scaling_summary.md` (the tables), `scaling_*.csv`, `fits_maps_<reading>.png`. Missing
experiment folders are skipped with a message. Not yet run: the self-test (`python scripts/check_brightness_options.py`)
covers `auto_scale.py`, but neither it nor the comparison has been run against trained models.

## Two separate stages: brightness first, shape separately (`slurm/run_brightness_stage.sh`, `slurm/run_shape_stage.sh`)

Not every option against every other: each stage changes one thing and keeps the rest fixed. The best setup of each is
then combined in a later, small final round. Both stages use 3 runs per variant with the same 3 seeds (42, 43, 44), and
score the earlier models on their runs 0-2. They are independent and can run at the same time.

**Stage 1, brightness only** (`sbatch slurm/run_brightness_stage.sh`): trains `pair_max1`, `pair_mean`, `pair_p99` and their
`_aug` versions (6 x 3 = 18 GPU jobs); compares them with the earlier brightness models (original, `aug`, `norm`, `aug_norm`,
`gain_p99`, `gain_flux`, `gain_bandflux`, `gain_pixel`), each with all 7 test-time scalings. No halo, no extra noise, the plain
loss, and at test time the model's own outline, no cleanup, vote 0.5. Results: `outputs/stage1_brightness/pipeline_summary.md`
(estimate.fits; see "Every model with every scaling") and `outputs/stage1_brightness/synthetic/scaling_summary.md`.

**Stage 2, shape only** (`sbatch slurm/run_shape_stage.sh`): the brightness handling is fixed to the original training with
automatic max scaling at test time (`single`). Trains `fg3`, `fg10`, `fg3_dice`, `halo`, `lightnoise` (+ `bank` with
`NOISE_BANK`; 5 x 3 = 15 GPU jobs); compares them with the original, each with every outline x cleanup x vote, on estimate.fits,
plus the synthetic robustness tests. `NO_TRAIN=1 sbatch slurm/run_shape_stage.sh` scores the outline / cleanup / vote choices on
the original model right away (no GPU). Results: `outputs/stage2_shape/pipeline_summary.md` and `outputs/stage2_shape/robustness/`.

**The fair reference, `original_retrained`.** `logs/estimate_10band` was trained elsewhere on 2 GPUs (each step saw 2 x 128
scenes, with a higher learning rate), so it is not a fair reference for the new variants (1 GPU). `original_retrained` is the
original model and data, trained here with exactly their setup and seeds (3 runs, `logs/estimate_10band_original_retrained`).
Train it once, then run the comparisons again (both stages use it as the reference; the earlier original stays in the tables):

    VARIANTS=original_retrained bash slurm/run_shape_stage.sh

**Stage 3, combined** (`bash slurm/run_combined_stage.sh`): the best of each stage together. Two brightness bases (`aug`:
random brightness in training; `pair_max1_aug`: random brightness + every scene twice, the copy divided by its brightest value)
x the shape winner `fg3`, alone, with the halo, with the weaker noise, or with both: 8 variants x 3 runs = 24 GPU jobs
(`aug_fg3`, `aug_fg3_halo`, `aug_fg3_light`, `aug_fg3_halo_light`, and the same with `pair_max1_aug_`). The comparison scores
them with `original_retrained`, `aug`, `pair_max1_aug`, `fg3`, `halo` and `lightnoise`, each with automatic scaling (single,
per object, per object + the faint search), the object mask from brightness, cleanup and vote, on `estimate.fits`, plus the
synthetic brightness and robustness tests. Results: `outputs/stage3_combined/`.

**Finding the companion (`per_object_faint`, outlines `mask@0.01+faint` / `fill@0.01+faint`).** The companion in
`object.fits` is 1 pixel, 18 px right of the satellite; in `estimate.fits` it is about 0.1% of the satellite's brightest
value, below the 1% the object search uses, so no scaling mode gave it its own scale. `per_object_faint` adds a second
search (`scripts/auto_scale.py`, `find_faint`): the bright objects are set aside with 3 pixels round them (the specks that
hug them), and the rest is searched again; its brightest pixel becomes an object only if it is at least 2x brighter than
every other pixel left (on `estimate.fits` the companion is 2.5x brighter than any speck more than 3 px from the satellite).
It runs only next to a bright object, so an empty window stays empty. The `+faint` outlines keep such objects in the
outline taken from the data. Untested against trained models so far; to try it on models that are already trained:

    sbatch -A psy53c17 -p qTRD -c 8 --mem=32g -t 02:00:00 -J companion -o slurm/logs/%x_%j.out --wrap "source ~/.bashrc; conda activate satellite; \
      python scripts/compare_pipeline.py --max-runs 3 --readings radiometry \
        --experiments estimate_10band_original_retrained estimate_10band_fg3 estimate_10band_aug estimate_10band_pair_max1_aug estimate_10band_pair_p99_aug \
        --scalings single per_object per_object_faint --outlines model mask@0.01 mask@0.01+faint --cleanups off tidy --votes 0.3 0.5 \
        --title 'Companion test' --out outputs/companion_test"

The groups are in `scripts/brightness_presets.py` (`python scripts/brightness_presets.py group stage1_train`, etc.).
`slurm/run_pipeline.sh` with `STAGE=all` (its default) still trains everything at once (84 jobs).

## The whole pipeline: training choices x test-time choices (`slurm/run_pipeline.sh`)

Aim: get the shape (outline) and the materials of the object in `estimate.fits` right, with no hand-set scale.

**At training time** (each variant is 3 runs with the SAME 3 random seeds for all variants: run i starts its model from
`init_seed + i` = 42, 43, 44 and draws training-scene stream i; the validation and test scenes are the same for every run):
the 24 `pair_*` variants (every scene twice: as generated and divided by its max / object mean / object p99; with or without
random brightness, the halo, the weaker noise), `fg3`, `fg10`, `fg3_dice` (heavier loss on object pixels), `lightnoise`, `bank`
(less or real noise). No scaling inside the model. The comparison also scores every EARLIER model that is already trained
(the original, `aug_norm`, `gain_flux`, `full_p99`, ...) on its runs 0-2, i.e. the same 3 seeds (those were trained earlier, in
part with another number of GPUs, which can change results a little). 28 variants x 3 runs = 84 GPU jobs by default.

**At test time** (nothing trained; `scripts/auto_scale.py`, `scripts/postprocess.py`), every combination is scored:

| choice | settings | what it does |
|---|---|---|
| scaling | `none`, `single`, `single_mean`, `single_p99`, `per_object`, `per_object_mean`, `per_object_p99` | the image as it is / one scale for the image from the object's brightest value, mean or 99th percentile / the same per object, with a model run per object |
| outline | `model`, `mask@0.01`, `fill@0.01`, `mask@0.001`, `fill@0.001` | the model's own outline / labels outside the outline found in the data are dropped (the data outline = pixels brighter than that fraction of the image's brightest) / and background inside it gets the nearest material |
| cleanup | `off`, `tidy`, `tidy+smooth` | nothing / remove specks under 3 px and fill enclosed holes / also give each pixel the most common material of its 3x3 neighbourhood |
| vote | `0.3`, `0.5`, `0.7` | the share of runs that must call a pixel an object (0.5 = the notebooks' rule); also reported: every run on its own, with the spread |

Scores (`scripts/compare_pipeline.py`, on the 41 x 41 px window of `estimate.fits`, for both readings): outline IoU, materials
ARI, false pixels, satellite and companion pixels found (the true companion is 1 pixel, 18 px right of the satellite), and `faint_found`: the share of the FAINT
true pixels (estimate dimmer than 5% of its brightest) that were labelled. The by-hand row (a person chose the object and its
scale) is the reference. `scripts/compare_scaling.py` adds the synthetic tests (one object at many brightnesses; a bright and
a dim object in one image).

    sbatch slurm/run_pipeline.sh                # self-test -> train 28 variants x 3 runs -> compare (all as dependent jobs)
    bash slurm/run_pipeline.sh                  # the same, submitted from your shell
    VARIANTS="pair_mean pair_p99 fg3" sbatch slurm/run_pipeline.sh       # only some variants
    NO_TRAIN=1 bash slurm/run_pipeline.sh       # only the comparison, on models that are already trained
    NOISE_BANK=/path/bank.npy bash slurm/run_pipeline.sh     # also trains the `bank` variant

The training jobs wait for the self-test (`slurm/pipeline_check.sh`) and are cancelled if it fails; the comparison
(`slurm/compare_pipeline.sh`) waits for the trainings to end (even if a run failed: it scores what exists). The pair variants get
`N_TRAIN=10000` so that an epoch costs the same as the 20000 scenes of the others. `--max-runs` takes the first runs that
exist, so if run 0 of a model failed, that model is scored on other seeds: check the run folders. Results: `outputs/pipeline_compare/pipeline_summary.md`
(tables), `pipeline_grid.csv` (every cell), `pipeline_maps_<reading>.png`, and `synthetic/scaling_summary.md`.
The reference models `logs/estimate_10band` and `logs/estimate_10band_aug_norm` must already exist; a missing folder is skipped.

Not yet run, anywhere: the self-test and the comparison scripts have been compiled but not executed; `train.py` was changed in one
function (`get_criterion`) and has not been run with a `loss` section.

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
