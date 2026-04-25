"""
Depth-Anything-V2 wrapper — produces metric (or scale-calibrated) depth
maps from RGB endoscopic frames.

We use the Hugging Face transformers port. Two flavors:

  * "metric"   — uses Depth-Anything-V2-Metric-Indoor-* which outputs depth
                 in meters directly. Good when you have no GT to calibrate
                 against; the indoor model is the closest match for a
                 close-range endoscope view.

  * "relative" — uses Depth-Anything-V2-* which outputs disparity-style
                 relative depth. We then fit a per-sequence linear scale
                 (depth = a / pred + b) against any frames where ground-
                 truth depth is available. For C3VD that's every frame.

Install once:
    pip install transformers accelerate
"""

import numpy as np
import torch
from PIL import Image


_HF_MODELS = {
    # Relative-depth (disparity-like) checkpoints
    "vits": "depth-anything/Depth-Anything-V2-Small-hf",
    "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
    "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
    # Metric-depth checkpoints (indoor, output is meters)
    "metric_indoor_s": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "metric_indoor_b": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "metric_indoor_l": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
}


class DepthAnythingV2:
    def __init__(self, variant="vitb", device=None):
        if variant not in _HF_MODELS:
            raise ValueError(
                f"Unknown variant {variant!r}. Choose from {list(_HF_MODELS)}.")
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.variant = variant
        self.is_metric = variant.startswith("metric_")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        repo = _HF_MODELS[variant]
        self.processor = AutoImageProcessor.from_pretrained(repo)
        self.model = AutoModelForDepthEstimation.from_pretrained(repo) \
            .to(self.device).eval()

    @torch.no_grad()
    def predict(self, rgb_u8):
        """rgb_u8: HxWx3 uint8 (RGB). Returns HxW float32.

        - For metric variants: meters.
        - For relative variants: a disparity-like value (larger = closer).
          Convert to depth via `depth = a / pred + b` after fitting (a, b).
        """
        H, W = rgb_u8.shape[:2]
        img = Image.fromarray(rgb_u8)
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        pred = outputs.predicted_depth  # (1, h', w')
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1) if pred.dim() == 3 else pred,
            size=(H, W), mode="bicubic", align_corners=False,
        ).squeeze(1).squeeze(0)
        return pred.float().cpu().numpy()


def fit_disparity_to_depth(pred, gt_depth, valid_mask=None):
    """Robustly fit `depth ≈ a / pred + b` (Depth-Anything style) on one frame.

    Returns (a, b). Using least squares on (1/pred, gt_depth) is fine; we
    also clip outliers with a 5-95th percentile trim before fitting.
    """
    if valid_mask is None:
        valid_mask = (gt_depth > 0) & np.isfinite(gt_depth) & np.isfinite(pred) \
            & (pred > 1e-6)
    p = pred[valid_mask].astype(np.float64)
    g = gt_depth[valid_mask].astype(np.float64)
    if p.size < 100:
        return None, None
    inv_p = 1.0 / p
    # Trim worst 5% on each side of inv_p to be robust
    lo, hi = np.percentile(inv_p, [5, 95])
    keep = (inv_p >= lo) & (inv_p <= hi)
    inv_p = inv_p[keep]
    g = g[keep]
    A = np.stack([inv_p, np.ones_like(inv_p)], axis=1)
    sol, *_ = np.linalg.lstsq(A, g, rcond=None)
    return float(sol[0]), float(sol[1])


def disparity_to_depth(pred, a, b, eps=1e-6):
    return a / np.clip(pred, eps, None) + b
