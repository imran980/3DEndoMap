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

    Install (one-time):
        git clone https://github.com/BeileiCui/EndoDAC external/EndoDAC
        # download backbone:
        #   https://drive.google.com/file/d/163ILZcnz_-IUoIgy1UF_r7PAQBqgDbll
        # → external/EndoDAC/pretrained_model/depth_anything_vitb14.pth
        # download EndoDAC adapter checkpoints (folder):
        #   https://drive.google.com/file/d/1qzAYBtwYJDN7hEi6pApqBOOz6pUhyY70
        # → external/EndoDAC/EndoDAC_fullmodel/depth_model.pth (and others)

    Then:
        EndoDACBackbone(
            repo_dir="external/EndoDAC",
            weights_path="external/EndoDAC/EndoDAC_fullmodel/depth_model.pth",
        )

    Output is a *disparity-like* prediction (larger = closer), the same
    convention as DAv2's relative variants. Calibrate with
    fit_disparity_to_depth() if GT depth is available.
    """

    is_metric = False

    def __init__(self, repo_dir, weights_path, device=None,
                 input_size=(280, 350), encoder="vitb"):
        if not os.path.isdir(repo_dir):
            raise FileNotFoundError(
                f"EndoDAC repo not found at {repo_dir}. "
                f"Clone https://github.com/BeileiCui/EndoDAC first.")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"EndoDAC depth_model not found at {weights_path}.")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.repo_dir = os.path.abspath(repo_dir)
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

        # The class lives at networks.depth_anything.dpt.DepthAnything in the
        # public EndoDAC repo. Import it lazily so the rest of this file
        # remains usable on systems without EndoDAC installed.
        try:
            mod = importlib.import_module("networks.depth_anything.dpt")
            DepthAnything = getattr(mod, "DepthAnything", None) or \
                getattr(mod, "DPT_DINOv2", None)
            if DepthAnything is None:
                raise AttributeError(
                    "Could not find DepthAnything/DPT_DINOv2 class in "
                    "networks.depth_anything.dpt")
        except Exception as e:
            raise ImportError(
                f"Could not import EndoDAC's DepthAnything from "
                f"{self.repo_dir}/networks/depth_anything/dpt.py.\n  {e}")

        # vitb14 config used by EndoDAC's released checkpoint
        cfg = {
            "encoder": encoder,
            "features": 128,
            "out_channels": [96, 192, 384, 768],
            "use_bn": False,
            "use_clstoken": False,
        }
        try:
            self.model = DepthAnything(cfg)
        except TypeError:
            # Older signature accepts kwargs
            self.model = DepthAnything(**cfg)
        self.model = self.model.to(self.device)

        state = torch.load(weights_path, map_location=self.device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        # Strip a "module." prefix if the checkpoint was saved with DDP
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[EndoDAC] {len(missing)} missing keys (first 3): "
                  f"{missing[:3]}")
        if unexpected:
            print(f"[EndoDAC] {len(unexpected)} unexpected keys (first 3): "
                  f"{unexpected[:3]}")
        self.model.eval()

    @torch.no_grad()
    def predict(self, rgb_u8):
        H, W = rgb_u8.shape[:2]
        ih, iw = self.input_size
        # DepthAnything expects ImageNet-normalized float in [0, 1]
        x = rgb_u8.astype(np.float32) / 255.0
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
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        pred = torch.nn.functional.interpolate(
            pred, size=(H, W), mode="bicubic", align_corners=False,
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
