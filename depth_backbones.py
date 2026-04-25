"""
Pluggable depth backbones for the C3VD/endoscopy fusion pipelines.

A "backbone" takes an RGB frame and returns a per-pixel depth (or
disparity) prediction. Concrete implementations:

  - DAv2Backbone        : Hugging Face Depth-Anything-V2 (already wired up
                          via dav2_depth.py). Default.
  - EndoDACBackbone     : EndoDAC (MICCAI 2024) — endoscopy-finetuned
                          DepthAnything with DV-LoRA adapter. Released by
                          the same group as Endo-4DGS, so well-matched
                          to colonoscopy. Lazy-imported from a local
                          clone of https://github.com/BeileiCui/EndoDAC.

Use via the make_backbone() factory:

    bb = make_backbone("dav2", variant="vitb")
    bb = make_backbone("endodac",
                       repo_dir="external/EndoDAC",
                       weights_path="external/EndoDAC/checkpoints/endodac.pth")
    depth = bb.predict(rgb_u8)             # HxW float32
    is_metric = bb.is_metric               # bool
"""

import importlib
import os
import sys
import numpy as np
import torch


class DepthBackbone:
    """Common interface every backbone must satisfy."""

    is_metric: bool = False  # True if predict() returns metric meters

    def predict(self, rgb_u8: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DAv2Backbone(DepthBackbone):
    """Wraps dav2_depth.DepthAnythingV2 (HF transformers Depth-Anything-V2)."""

    def __init__(self, variant="vitb", device=None):
        from dav2_depth import DepthAnythingV2
        self._impl = DepthAnythingV2(variant=variant, device=device)
        self.is_metric = self._impl.is_metric
        self.variant = variant

    def predict(self, rgb_u8):
        return self._impl.predict(rgb_u8)


class EndoDACBackbone(DepthBackbone):
    """
    EndoDAC: Efficient Adapting Foundation Model for Self-Supervised
    Endoscopic Depth Estimation (Cui et al., MICCAI 2024).

    EndoDAC is not on PyPI/HuggingFace — install it manually:

        git clone https://github.com/BeileiCui/EndoDAC external/EndoDAC
        # follow README to download endodac.pth into external/EndoDAC/checkpoints/

    Then point this backbone at the clone:

        EndoDACBackbone(repo_dir="external/EndoDAC",
                        weights_path="external/EndoDAC/checkpoints/endodac.pth")

    Output is a *disparity-like* prediction (larger = closer), the same
    convention as DAv2's relative variants. Calibrate with
    fit_disparity_to_depth() if GT depth is available.
    """

    is_metric = False

    def __init__(self, repo_dir, weights_path, device=None,
                 input_size=(280, 350)):
        if not os.path.isdir(repo_dir):
            raise FileNotFoundError(
                f"EndoDAC repo not found at {repo_dir}. "
                f"Clone https://github.com/BeileiCui/EndoDAC first.")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"EndoDAC weights not found at {weights_path}.")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.repo_dir = os.path.abspath(repo_dir)
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

        # EndoDAC ships its model definition under networks/. We import
        # lazily so users without the repo can still import this module.
        try:
            networks = importlib.import_module("networks")
            self.depth_anything = importlib.import_module(
                "networks.depth_anything").DepthAnything
        except Exception as e:
            raise ImportError(
                f"Could not import EndoDAC's `networks` package from "
                f"{self.repo_dir}. Make sure the clone is intact.\n  {e}")

        # Model construction follows EndoDAC/test_simple.py at the time of
        # writing. If the upstream repo changes, mirror it here.
        self.model = self.depth_anything(
            {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
        ).to(self.device)
        state = torch.load(weights_path, map_location=self.device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    @torch.no_grad()
    def predict(self, rgb_u8):
        H, W = rgb_u8.shape[:2]
        ih, iw = self.input_size
        x = rgb_u8.astype(np.float32) / 255.0
        # ImageNet normalization expected by DepthAnything backbone
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))[None]  # 1x3xHxW
        x = torch.from_numpy(x).to(self.device)
        x = torch.nn.functional.interpolate(
            x, size=(ih, iw), mode="bilinear", align_corners=False)
        pred = self.model(x)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1) if pred.dim() == 3 else pred,
            size=(H, W), mode="bicubic", align_corners=False,
        ).squeeze(1).squeeze(0)
        return pred.float().cpu().numpy()


def make_backbone(name, **kwargs):
    """Factory: name='dav2' or 'endodac' (case-insensitive)."""
    name = name.lower()
    if name == "dav2":
        return DAv2Backbone(**kwargs)
    if name == "endodac":
        return EndoDACBackbone(**kwargs)
    raise ValueError(f"Unknown depth backbone: {name!r}. "
                     f"Choose 'dav2' or 'endodac'.")
