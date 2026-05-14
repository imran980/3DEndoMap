"""
Test-time augmentation wrapper for depth backbones.

Runs the backbone N times per frame under invertible augmentations,
registers the predictions back to canonical pixel coordinates, and
returns per-pixel (mean_depth, sigma) where sigma is the std across
the N predictions. Feeds Paper 1's split-conformal calibration: the
score s = |pred - gt| / sigma is what gets calibrated.

Paper 1 (research.md) calls for N=4 under {hflip, brightness ±10%,
small crop}. We use {hflip, brightness +10%, brightness -10%,
horizontal 5-px shift} — the shift is a stand-in for "small crop"
that's exactly invertible (no resampling artifact) and the rest
match.

Wrap any DepthBackbone via:

    bb = make_backbone("endodac", repo_dir=..., weights_path=...)
    tta = TTAWrapper(bb, n=4)
    mean_mm, sigma_mm = tta.predict_with_uncertainty(rgb_u8)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np


# ---------- invertible augmentations ----------
#
# Each aug is two functions:
#   forward(rgb_u8)   -> augmented rgb_u8 (same H, W, 3 dtype)
#   inverse(depth)    -> un-augmented depth (same H, W float32)
#
# `inverse` must register the prediction back to canonical pixel
# coordinates so we can stack N predictions and take pixel-wise stats.


@dataclass
class _Aug:
    name: str
    forward: Callable[[np.ndarray], np.ndarray]
    inverse: Callable[[np.ndarray], np.ndarray]


def _aug_hflip() -> _Aug:
    return _Aug(
        name="hflip",
        forward=lambda rgb: np.ascontiguousarray(rgb[:, ::-1, :]),
        inverse=lambda d: np.ascontiguousarray(d[:, ::-1]),
    )


def _aug_brightness(delta: float) -> _Aug:
    # Multiplicative brightness shift in u8 space, clipped. delta=+0.1
    # = +10%. Spatially identity, so inverse is identity.
    def fwd(rgb):
        out = rgb.astype(np.float32) * (1.0 + delta)
        return np.clip(out, 0, 255).astype(np.uint8)

    return _Aug(name=f"bright{delta:+.2f}", forward=fwd, inverse=lambda d: d)


def _aug_hshift(px: int) -> _Aug:
    # Pure pixel shift (replicate-pad), exactly invertible by inverse
    # shift. No resampling, so no smoothing artefacts.
    def shift_x(arr: np.ndarray, dx: int) -> np.ndarray:
        if dx == 0:
            return arr
        out = np.empty_like(arr)
        if dx > 0:
            out[:, :dx] = arr[:, :1]
            out[:, dx:] = arr[:, :-dx]
        else:
            out[:, dx:] = arr[:, -1:]
            out[:, :dx] = arr[:, -dx:]
        return out

    return _Aug(
        name=f"hshift{px:+d}",
        forward=lambda rgb: shift_x(rgb, px),
        inverse=lambda d: shift_x(d, -px),
    )


def default_augs() -> List[_Aug]:
    """Paper 1's N=4 augmentation set."""
    return [
        _aug_hflip(),
        _aug_brightness(+0.10),
        _aug_brightness(-0.10),
        _aug_hshift(+5),
    ]


# ---------- wrapper ----------


class TTAWrapper:
    """Wrap a DepthBackbone with test-time augmentation.

    is_metric / variant attributes are passed through so the rest of
    the pipeline (scale calibration, dashboard) doesn't have to care
    whether TTA is on.
    """

    def __init__(self, backbone, n: int = 4,
                 augs: Optional[List[_Aug]] = None):
        self.backbone = backbone
        self.is_metric = bool(getattr(backbone, "is_metric", False))
        self.variant = getattr(backbone, "variant",
                               backbone.__class__.__name__)
        if augs is None:
            augs = default_augs()
        if n > len(augs):
            raise ValueError(
                f"n={n} > available augs ({len(augs)}). Extend "
                f"default_augs() or pass a longer list.")
        self._augs: List[_Aug] = augs[:n]

    @property
    def n(self) -> int:
        return len(self._augs)

    def predict(self, rgb_u8: np.ndarray) -> np.ndarray:
        """DepthBackbone-compatible scalar prediction (the TTA mean).

        Lets TTAWrapper drop in anywhere a backbone is expected; if
        the caller wants uncertainty too, they call
        predict_with_uncertainty().
        """
        mean, _ = self.predict_with_uncertainty(rgb_u8)
        return mean

    def predict_with_uncertainty(self, rgb_u8: np.ndarray
                                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, sigma), both H x W float32, aligned to rgb_u8.

        For non-metric backbones (DAv2 relative / EndoDAC), the values
        are disparity-like — std is in the same units. Downstream code
        handles the disparity->depth scale conversion; sigma is
        propagated through that conversion by the eval script.
        """
        preds = []
        for aug in self._augs:
            x = aug.forward(rgb_u8)
            p = self.backbone.predict(x).astype(np.float32)
            preds.append(aug.inverse(p))
        stack = np.stack(preds, axis=0)  # (N, H, W)
        mean = stack.mean(axis=0)
        # Unbiased std (ddof=1) when N>1; falls back to 0 for N=1 so
        # callers don't have to special-case N=1 (sigma is just zero).
        sigma = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 \
            else np.zeros_like(mean)
        return mean, sigma
