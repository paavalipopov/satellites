# The original generator — notes

Draft, moved out of the walkthrough on 2026-10-22. Keep, edit or delete.
Source: `_remote_data/yao/src/moduler_gen.py` and `generator_realtime.py`.

## Behaviour the rebuild does not copy

- **Scene type by storage slot.** Slots 0–299 of a 1000-slot ring buffer held satellites, 300–599 shapes, 600–999 patches. The rebuild draws the type with probabilities 30/30/40.
- **Three separate .npy files per slot** (mask, cube, clean cube), overwritten at ~4800 writes/s by 192 workers, so a reader could pair a cube with another scene's mask. Shapes always matched, so this entered training as label noise.
- **"Triangles" were lines.** The triangle branch drew 2 random points, so `fillPoly` produced a ~8 px line instead of a ~26 px triangle. The paper lists triangles among the shapes.
- **Patch labels repeated.** It drew unique labels, then wrote a fresh `np.random.randint(1, 47)` into each patch: about 11.9 distinct materials per patch scene instead of ~13.5.
- **Faces touching the frame edge were skipped**, not clipped (0.57% of layout-2 poses).
- **Band-0 artifacts went into training.** Several sheets begin at 350 nm, where DHR is a ratio of two near-zero signals; values reach −9.8 … +15.5. The Kapton-family sheets show the same at their first point, 275 nm, below the grid.
- **Zeros past the scan.** `right=0.0` set every band past a sheet's last measurement to 0. BetaCloth ends at 999 nm, so its 1000 nm band was 0 (22 zero bands on the 350–2000 nm grid): a fingerprint no other material has.
- **Blur hard-coded to 350–1000 nm.** σ came from a fixed 350–1000 nm ramp, so the June 350–2000 nm data had blur that did not match its wavelengths (σ ended at 1.67 px instead of 3.33).
- **A second blend weight for the saved clean cube.** For satellites, the saved `spectral_cube` came from a second call with its own weight, so it is a different blend from the model input.
- **Not reproducible.** Clock-seeded global `np.random`, `default_rng(None)` (OS entropy) for sensor noise and shape/patch geometry, and Python `random` for shape types.

## Observations

- Brightness differs by an order of magnitude between materials (Si solar cell 0.03–0.08, bus materials 0.7–1.0). The trained models use absolute brightness, so they are not scale-invariant.
