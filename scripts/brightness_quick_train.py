"""A short, plain-PyTorch training of the 10-band model, for trying the brightness options on a CPU.

NOT the real trainer: no Catalyst, a fixed pool of scenes instead of fresh ones every epoch, a few
hundred steps instead of 200 epochs. It builds the data and model through the project's own
registry, so a variant here is the same variant as in slurm/train_brightness.sh, and it writes
logs/<experiment>/config.yaml and run_<i>/model.best.pth in the layout scripts/brightness_eval.py reads.

    python scripts/brightness_quick_train.py --variant aug_norm --steps 600 --out logs_quick
    variants: see `python scripts/brightness_presets.py list` (base, aug, norm, aug_norm, gain_p99, ..., full_max)
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from brightness_presets import PRESETS, experiment_name, model_options, sim_options  # noqa: E402
from src.data_generation import SpectralLibraryWarning  # noqa: E402
from src.registry import build_dataset, build_model  # noqa: E402

warnings.simplefilter("ignore", SpectralLibraryWarning)

BANDS_10 = [400.0, 426.667, 457.143, 492.308, 533.333, 581.818, 640.0, 711.111, 800.0, 914.286]


def scenes(split, n):
    """The first n scenes of a split as tensors [n, bands, H, W] and [n, H, W]."""
    results = split[(0, list(range(n)))]
    xs = torch.stack([results[i]["input"] for i in range(n)])
    ys = torch.stack([results[i]["label"] for i in range(n)])
    return xs, ys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=list(PRESETS), required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--pool", type=int, default=3000, help="training scenes, generated once")
    ap.add_argument("--valid", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gain-range", default="-1.5,0.5", help="log10 gain 'lo,hi' for aug variants")
    ap.add_argument("--gain-stage", default="before_noise", choices=["before_noise", "after_noise"])
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "logs_quick"))
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    sim = {"n_bands": 10, "wavelengths": BANDS_10}
    sim.update(sim_options(args.variant, [float(v) for v in args.gain_range.split(",")], args.gain_stage))
    model_params = model_options(args.variant)
    name = experiment_name(args.variant)

    cfg = OmegaConf.create({
        "data": {"name": "hsi", "params": {"n_train": args.pool, "n_valid": args.valid, "n_test": 100, "sim": sim}},
        "model": {"name": "spectral_spatial", "init_seed": 42, "params": model_params},
        "runtime": {"n_channels": 10, "n_classes": 47, "run_idx": args.run},
    })
    dataset = build_dataset(cfg)           # writes the merged parameters back into cfg
    model = build_model(cfg)
    torch.manual_seed(cfg.model.init_seed + args.run)

    t0 = time.time()
    xt, yt = scenes(dataset.split("train", run=args.run), args.pool)
    xv, yv = scenes(dataset.split("valid"), args.valid)
    print(f"{name}: {len(xt)} train / {len(xv)} valid scenes in {time.time() - t0:.0f} s; "
          f"input range [{xt.min():.4f}, {xt.max():.2f}], model input_norm={model.input_norm}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(1234 + args.run)
    order, pos = torch.randperm(len(xt), generator=g), 0
    best, best_state, curve = float("inf"), None, []
    for step in range(1, args.steps + 1):
        if pos + args.batch > len(order):
            order, pos = torch.randperm(len(xt), generator=g), 0
        idx = order[pos:pos + args.batch]; pos += args.batch
        model.train()
        loss = F.cross_entropy(model(xt[idx]), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                out = model(xv)
                vloss = F.cross_entropy(out, yv).item()
                pred = out.argmax(1)
                fg = yv > 0
                acc = (pred == yv)[fg].float().mean().item()
            curve.append({"step": step, "train_loss": loss.item(), "valid_loss": vloss, "valid_material_acc": acc})
            print(f"  step {step:5d}  train {loss.item():.3f}  valid {vloss:.3f}  material acc {acc:.3f}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
            if vloss < best:
                best, best_state = vloss, {k: v.clone() for k, v in model.state_dict().items()}

    folder = Path(args.out) / name
    (folder / f"run_{args.run}").mkdir(parents=True, exist_ok=True)
    torch.save(best_state, folder / f"run_{args.run}" / "model.best.pth")
    OmegaConf.save(cfg, folder / "config.yaml")
    (folder / f"run_{args.run}" / "curve.json").write_text(json.dumps(curve))
    print("saved", folder)


if __name__ == "__main__":
    main()
