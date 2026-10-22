# The satellite at 15×15 — notes

Draft, moved out of the walkthrough on 2026-10-22. Keep, edit or delete.

- With a 60° field of view the focal length is 0.866 × 15 ≈ 13 px. At camera height 17 (layout 1): bus cylinder (r 0.8) ~1.2 px across, antenna (r 0.4) ~0.6 px, connectors ~0.23 px, solar panels ~3.1 × 1.5 px each, ~9 px apart.
- Bus, antenna and connectors appear only because the cv2 rule (vertices truncated to int32, then `fillConvexPoly`) fills slivers generously. With exact pixel-centre coverage, foreground drops from 13.5% to 5.7% (layout 1) and from 26.7% to 15.5% (layout 2). At 60×60 the two rules still differ (7.6% vs 5.6%; 17.7% vs 15.0%).
- Per part, seed 3, layout 1, each part rendered alone (cv2 / exact, in px): Melinex 5 / 2, Kapton-MT 5 / 2, AZ-3700 6 / 0, Copper 4 / 0, Si solar cell 16 / 10.
- The object spans ~81% of the frame at any `image_size`, because the field of view is fixed. To resolve it, raise `image_size` (e.g. `SimConfig(image_size=60, rasterizer="exact")`); moving the camera does not help.
- Layout 2 (half of the satellites) is larger: 26.7% foreground against 13.5% for layout 1.
- Reading: the training "satellite" is largely two panels plus a few stray pixels between them.

To reproduce the comparison (from the project root):

```python
import numpy as np
from src.data_generation import SimConfig
from src.data_generation.geometry import rasterize
from src.data_generation.pipeline import build_satellite

cfg = SimConfig()
rng = np.random.default_rng(3)
comps, chosen = build_satellite(cfg, rng)
angles = np.array([float(rng.integers(45, 135)), 0.0, float(rng.integers(-180, 180))])
for mode in ("cv2", "exact"):
    mask = rasterize(comps, cfg.with_(rasterizer=mode), angles)
    print(mode, f"{(mask != 0).mean():.1%} foreground")
```
