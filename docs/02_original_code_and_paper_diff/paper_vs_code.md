# The paper vs the original code — notes

Draft, collected on 2026-10-22 for the resubmission. Keep, edit or delete.
Line numbers refer to `assets/acta_astra/Main.tex`.

## Discrepancies

1. **Grid** (L218): 350–1000 nm. The paper's models match; the surviving driver uses 350–2000 nm (the June runs).
2. **Blur** (L273–278): K_j = k·j/2.5 + b with random k ∈ [0.7, 1.2] and b ∈ [1, 3] (1.3–3.5 at band 1, 11–20 at band 36). The code uses a fixed σ of 0.58 → 1.67 px from a diffraction formula whose constants (D = 3.6 m, 5 µm pixels, f = 120 m) are not in the paper.
3. **Material noise** (L239–243): α ∈ (0, 1), with "weight and noise level independently sampled". The code fixes α = 0.5; only w is sampled.
4. **Sensor noise** (L280): called atmospheric; its level is never stated. The surviving driver uses σ = 0.2 (U(0.05, 0.08) commented out); Table 1 reproduces at 0.02–0.03.
5. **Camera** (L260): "multiple camera distances" and yaw ±45°. The code fixes the height at 17 and the yaw at 0 (both randomisations commented out); pitch and roll match.
6. **Components** (L249): four, one material each. Layout 1 has five materials (the bus is two halves); layout 2 has no connector but a boom, dish and feed.
7. **Figure caption** (L267): "in-plane rotations of a base satellite mask" and a spectrum per pixel. The code renders in 3D with a z-buffer and assigns one spectrum per material region.
8. **Shapes** (L387): eight types including triangles. The code adds an unlisted "line" type, and its "triangles" are lines.
9. **NISAR**: "unseen during training" (L393) against Fig. caption (c), "used for training" (L399). No NISAR generator code survives.
10. **Training** (L389): matches `main.py`, which crashes as stored (`get_dataset` is undefined).
11. **Not stated**: the 30/30/40 scene mix (satellites are only 30% of the data), the two layouts, the sensor-noise σ.

## For writing claims

- Noise destroys material identification much faster than shape. On the paper's More-Scenes model, material accuracy falls 0.874 → 0.028 from σ = 0 to 0.2, while shape accuracy falls 0.995 → 0.878. "We recover shape robustly" is the better-supported claim.
- The additive Gaussian is described as atmosphere; real atmosphere is multiplicative and spectrally structured.

## Checking the paper's checkpoint

The traced checkpoints reproduce Table 1 when evaluated at σ between 0.02 and 0.03 (More Scenes: 0.812 / 0.985 at 0.02 and 0.735 / 0.969 at 0.03; the paper reports 0.764 / 0.992). Code moved from the walkthrough (run it from `scripts/`, after its setup cells):

```python
CKPT = "/Users/ppopov1/_remote_data/satellite_project/resluts_n_model/yes_id/model_nozoom_with_shapes.pt"   # <-- edit

if os.path.exists(CKPT):
    import torch
    model = torch.jit.load(CKPT, map_location="cpu").eval()
    ma, sa = [], []
    for i in range(120):
        smp = generate_sample(cfg, 5000 + i, lib, scene="satellite")
        fg = smp.mask != 0
        if not fg.sum(): continue
        with torch.no_grad():
            p = model(torch.from_numpy(smp.cube).float()[None]).argmax(1)[0].numpy()
        ma.append((p[fg] == smp.mask[fg]).mean()); sa.append(((p != 0) == fg).mean())
    print(f"material accuracy {np.mean(ma):.3f}   shape accuracy {np.mean(sa):.3f}   (n={len(ma)})")
    print("paper Table 1:    0.764                0.992")
else:
    print("checkpoint not found — set CKPT above to run this cell")
```
