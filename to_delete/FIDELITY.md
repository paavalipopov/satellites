# Fidelity audit — how this differs from the original

Short answer: **the default config is not a faithful recreation. `SimConfig.legacy()`
is, to within run-to-run noise.** Use `legacy()` to reproduce old results.

Bit-identical reproduction is impossible with any code, for reasons listed at the
bottom. The achievable target is *distributional* equivalence, measured here.

## Measured validation

The paper's own traced checkpoint (`yes_id/model_nozoom_with_shapes.pt`) fed with
150 satellite scenes at σ = 0.02, its training noise level:

| source | material acc | shape acc |
|---|---|---|
| **A** original `moduler_gen.py`, executed directly | **0.800** | 0.979 |
| **B** this rebuild, `SimConfig.legacy()` | **0.807** | 0.979 |
| **C** this rebuild, `SimConfig()` (fixes enabled) | 0.774 | 0.979 |
| — | | |
| paper Table 1 | 0.764 | 0.992 |

B tracks A to 0.007, far inside the original's own seed-to-seed spread. C is ~2.6
points lower because the fixes remove the band-0 shortcut feature the original was
exploiting.

Scene statistics, 200 scenes each:

| scene | A original | B rebuild `legacy()` |
|---|---|---|
| satellite (both layouts) | 20.0% fg · 12.3 px · 4.9 mats | 20.9% · 12.4 · 4.9 |
| shapes | 21.1% · 11.1 · 2.0 | 21.1% · 11.5 · 2.1 |
| patches | 33.4% · 14.8 · 11.7 | 33.7% · 14.8 · 11.9 |

## Material selection

The original put **every sheet with a wavelength and a pristine column** into the
library — 46 materials, irradiated trace optional — and drew uniformly from it:
3-from-bus / 1-from-antenna / 1-from-wings without replacement for layout 1
(layout 2 draws 2 bus halves without replacement plus an *independent* boom
material, so the boom may duplicate a half), and `randint(1, 47)` for the shape
and patch scenes, which ignore the groups entirely. No filter on data quality or
completeness anywhere.

`require_irradiated=True` (default False, so the original's behaviour is intact)
restricts to the 35 materials that actually have a measured irradiated trace.
This is not bug-compatible and will change results; it is offered because 11
materials silently opt out of the degradation model — `blend()` returns pristine
and ignores w for them — so a scene at w = 0.9 can contain a fully pristine
Titanium antenna. Material ids are not renumbered, so a 47-class checkpoint still
runs.

## Reversible — `SimConfig.legacy()` restores all of these

| flag | original behaviour |
|---|---|
| `clip_reflectance=False`, `on_invalid="ignore"` | detector artifacts (−9.8 … +15.6) fed into band 0, unannounced |
| `extrapolate_red="zero"` | zero-padded past the last measured λ |
| `legacy_drop_edge_faces=True` | faces touching the frame edge skipped, not clipped — **measured to never fire at image_size=15 / camera_height=17**, so the original's masks were not chopped |
| `legacy_flat_face_depth=True` | one depth per triangle |
| `legacy_triangle_bug=True` | "triangles" from 2 points, so every one was a line |
| `legacy_patch_label_bug=True` | discarded its own unique patch labels; unseeded global RNG |

## NOT reversible — unconditional differences

1. **RNG streams differ, so the same seed gives a different sample.** The original
   mixed a seeded `default_rng` with the *unseeded* `np.random.*` and the `random`
   module: `create_spectral_cube` used global `np.random.rand`/`randn`,
   `sample_material` used `random.random()`, `generate_sample` used global
   `np.random.randint`, and only the final sensor noise used a seeded generator.
   This rebuild threads one explicit `Generator` everywhere. Statistically
   equivalent; not sample-for-sample identical, and it cannot be made so without
   reintroducing the non-reproducibility.

2. **The PSF now follows the actual wavelength grid.** The original hard-coded
   `np.linspace(3.5e-7, 1.0e-6, C)` regardless of λ range. Identical for the
   350–1000 nm default (0.583 → 1.667 px). **Differs for `june_2026()`**: this
   rebuild gives 0.583 → 3.333 px, the original gave 0.583 → 1.667 px. If you are
   reproducing a June-2026 run you must account for this — the original's blur did
   not match its own wavelengths.

3. **Blur backend.** `cv2.GaussianBlur` with `BORDER_REFLECT_101` instead of
   `scipy.ndimage.gaussian_filter(mode="mirror")`. Agrees to 2 × 10⁻⁵; not
   bit-identical (different kernel truncation).

4. **Geometry primitives are mine, not trimesh's.** Vertex and face counts and
   bounding boxes match exactly (cylinder 42/80, box 8/12, icosphere 162/320), but
   vertex ordering and face winding may differ, which can change z-buffer
   tie-breaking on coincident surfaces.

5. **Scene type is drawn from probabilities, not from the storage index.** The
   original keyed it off the ring-buffer slot — `index < 300` satellite,
   `< 600` shapes, else patches — so the composition of a *split* was deterministic
   (validation indices 0–299 were all satellites). Same 30/30/40 marginal, different
   mechanism. If you need the original's exact split composition, select by index.

6. **No zoom path.** The original's `generate_sample` also rendered a mask and cube
   at `image_size * zoom_level` and returned the *zoomed* mask as the label. The
   generator ran with `zoom_level=1`, so the two coincide and nothing is lost for
   the configurations that were actually used — but the code path is absent.

7. **No file I/O**, as requested. Also not reproduced: `generate_distinct_color_list`
   (it only ever produced dictionary keys and never touched the output) and the
   `_match_len` decimation branch in `shape_simulator2` (unreachable in practice).

## Corrections to what I told you earlier in this session

Two mistakes, now fixed, both found by running the original directly rather than
trusting my reading of it:

- **`make_satellite_with_ids2` was missing.** The original picks between two
  spacecraft layouts with equal probability (`idx = np.random.randint(0,2)`), and I
  had only implemented the first. Variant 2 is a bigger object — bus radius 1.25
  instead of 0.8, a 6-unit boom, a 3.2-radius dish plus feed, and the whole assembly
  re-centred on its bounding box. Alone it gives 26.4% foreground against variant 1's
  13.5%. Now implemented as `build_satellite_v2`.
- **`shapes_scene` used `max_regions=5`; the original uses 3.** `shape_simulator`
  declares 3 as its default and `simulator_worker` never overrides it, even though
  `random_complex_region_mask` itself defaults to 5.

Consequently the "original" scene statistics I quoted before (13.6% / 29.9% / 34.2%
foreground) were from my own partial re-implementation, not the original. The true
values are 20.0% / 21.1% / 33.4%.

## Why bit-level reproduction is impossible regardless of the code

- The training data was a **1000-slot ring buffer overwritten at ~4800 writes/s** by
  192 concurrent workers. It no longer exists and cannot be regenerated.
- Those workers seeded from `time.time() * 1e6 + os.getpid()`, and much of the
  pipeline drew from the **unseeded global RNG**.
- For **20 of the 41 archived runs** (seeds 18–29) the generator driver that produced
  their data was overwritten. Its settings are unrecoverable.
- The original **does not run on numpy ≥ 1.25**: `angles[0] = np.random.randint(45,135,1)`
  raises, and `float(np.array([w]))` raises on numpy 2. It needs their `p38` env or
  two one-line shims. (`/home/claude/orig/moduler_gen.py` in the session container
  holds a shimmed copy, marked `# [shim]`.)

So "reproduce the old results" means reproducing the *distribution* and the reported
*metrics*, which row B above does. It does not mean regenerating their exact arrays.

## Recommended workflow

```python
# 1. reproduce
cfg = SimConfig.legacy()                  # + june_2026 params if targeting a June run
# 2. once the old numbers are confirmed, move to
cfg = SimConfig()                         # fixes on, band-0 shortcut removed
```
