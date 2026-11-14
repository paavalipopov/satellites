"""Training losses. The default is the plain cross-entropy of the original training, unchanged.

Switched on from the command line (Hydra, `+loss.<key>=<value>`; see BRIGHTNESS.md):

    +loss.fg_weight=3.0   every OBJECT pixel (any material) counts 3x as much as a background pixel. Background is
                          most of every image, so a model can score well by calling doubtful pixels background;
                          this makes it expensive to drop faint real structure.
    +loss.dice=0.5        adds 0.5 x (1 - Dice) of "object vs background" over the batch. Dice compares the whole
                          object area, so a missing faint panel is not hidden by the many easy background pixels.
                          Soft: it uses the model's probability of "object" (1 - probability of class 0).

With neither key (or no `loss` section) `build_criterion` returns torch.nn.CrossEntropyLoss(), exactly what train.py used.
"""
import torch
import torch.nn.functional as F


class ObjectAwareLoss(torch.nn.Module):
    """Cross-entropy with a heavier weight on object pixels, plus an optional Dice term for object vs background."""

    def __init__(self, fg_weight=1.0, dice=0.0, eps=1e-6):
        super().__init__()
        if fg_weight < 1.0 or dice < 0.0:
            raise ValueError("loss: fg_weight must be >= 1 and dice >= 0")
        self.fg_weight, self.dice, self.eps = float(fg_weight), float(dice), eps

    def forward(self, logits, target):
        """logits [B, classes, H, W] (any float type), target [B, H, W] int, 0 = background."""
        logits = logits.float()
        weight = None
        if self.fg_weight != 1.0:
            weight = torch.ones(logits.shape[1], device=logits.device)
            weight[1:] = self.fg_weight                       # class 0 is the background
        loss = F.cross_entropy(logits, target, weight=weight)
        if self.dice > 0.0:
            p_object = 1.0 - logits.softmax(1)[:, 0]          # [B, H, W]
            t_object = (target > 0).float()
            dice = (2.0 * (p_object * t_object).sum() + self.eps) / (p_object.sum() + t_object.sum() + self.eps)
            loss = loss + self.dice * (1.0 - dice)
        return loss


def build_criterion(cfg=None):
    """cfg: the `loss` section of the config (a mapping with optional fg_weight and dice), or None."""
    cfg = {} if cfg is None else dict(cfg)
    unknown = set(cfg) - {"fg_weight", "dice"}
    if unknown:
        raise ValueError(f"loss: unknown keys {sorted(unknown)}; use fg_weight and/or dice")
    fg_weight, dice = float(cfg.get("fg_weight", 1.0)), float(cfg.get("dice", 0.0))
    if fg_weight == 1.0 and dice == 0.0:
        return torch.nn.CrossEntropyLoss()
    return ObjectAwareLoss(fg_weight, dice)
