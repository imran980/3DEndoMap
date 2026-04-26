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

    Mirrors the inference flow in EndoDAC/test_simple.py:
        depther = models.endodac.endodac(
            backbone_size="base", r=4, lora_type="dvlora",
            image_shape=(feed_h, feed_w),
            pretrained_path="<repo>/pretrained_model",
            residual_block_indexes=[2,5,8,11], include_cls_token=True)
        depther.load_state_dict({k: v for k, v in depth_model.items()
                                 if k in depther.state_dict()})
        out = depther(image_tensor)["disp", 0]    # disparity-like

    Install:
        git clone https://github.com/BeileiCui/EndoDAC.git external/EndoDAC
        # backbone (one .pth, ~390 MB):
        # https://drive.google.com/file/d/163ILZcnz_-IUoIgy1UF_r7PAQBqgDbll
        # → external/EndoDAC/pretrained_model/depth_anything_vitb14.pth
        # adapter (folder with depth_model.pth):
        # https://drive.google.com/file/d/1qzAYBtwYJDN7hEi6pApqBOOz6pUhyY70
        # → external/EndoDAC/EndoDAC_fullmodel/depth_model.pth

    Output is a *disparity-like* prediction (larger = closer), same
    convention as DAv2 relative variants. Use fit_disparity_to_depth().
    """

    is_metric = False

    def __init__(self, repo_dir, weights_path, device=None,
                 pretrained_dir=None, lora_rank=4, lora_type="dvlora",
                 residual_block_indexes=(2, 5, 8, 11),
                 include_cls_token=True):
        if not os.path.isdir(repo_dir):
            raise FileNotFoundError(
                f"EndoDAC repo not found at {repo_dir}.")

        # weights_path may be a folder containing depth_model.pth, or the
        # .pth file directly.
        depth_model_path = (
            os.path.join(weights_path, "depth_model.pth")
            if os.path.isdir(weights_path) else weights_path)
        if not os.path.isfile(depth_model_path):
            raise FileNotFoundError(
                f"EndoDAC depth_model.pth not found at {depth_model_path}.")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.repo_dir = os.path.abspath(repo_dir)
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

        if pretrained_dir is None:
            pretrained_dir = os.path.join(self.repo_dir, "pretrained_model")
        if not os.path.isdir(pretrained_dir):
            raise FileNotFoundError(
                f"EndoDAC pretrained_model dir not found at {pretrained_dir}. "
                f"Place depth_anything_vitb14.pth inside it.")

        # Lazy-import the EndoDAC `endodac` factory function
        try:
            endodac_mod = importlib.import_module("models.endodac")
        except Exception as e:
            raise ImportError(
                f"Could not import models.endodac from {self.repo_dir}.\n"
                f"Did you copy the EndoDAC source into the repo dir? "
                f"Expected path: {self.repo_dir}/models/endodac/endodac.py\n  {e}")

        # The released checkpoint stores its training resolution
        state = torch.load(depth_model_path, map_location=self.device)
        feed_h = int(state.get("height", 224))
        feed_w = int(state.get("width", 280))
        self.feed_size = (feed_h, feed_w)

        self.model = endodac_mod.endodac(
            backbone_size="base",
            r=lora_rank,
            lora_type=lora_type,
            image_shape=(feed_h, feed_w),
            pretrained_path=pretrained_dir,
            residual_block_indexes=list(residual_block_indexes),
            include_cls_token=include_cls_token,
        ).to(self.device)

        model_dict = self.model.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_dict}
        n_total = len(model_dict)
        n_loaded = len(filtered)
        unexpected = [k for k in state.keys() if k not in model_dict
                      and k not in ("height", "width")]
        self.model.load_state_dict(filtered, strict=False)
        print(f"[EndoDAC] loaded {n_loaded}/{n_total} model params from "
              f"depth_model.pth (feed {feed_h}x{feed_w})")
        if unexpected:
            print(f"[EndoDAC] {len(unexpected)} non-model keys ignored "
                  f"(first 3: {unexpected[:3]})")
        self.model.eval()

    @torch.no_grad()
    def predict(self, rgb_u8):
        from PIL import Image
        H, W = rgb_u8.shape[:2]
        feed_h, feed_w = self.feed_size
        # Match test_simple.py exactly: PIL resize + ToTensor (no ImageNet
        # normalization — EndoDAC's adapter expects raw [0, 1] tensors).
        img = Image.fromarray(rgb_u8).resize((feed_w, feed_h), Image.LANCZOS)
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]
        x = torch.from_numpy(x).to(self.device)
        out = self.model(x)
        disp = out[("disp", 0)] if isinstance(out, dict) else out
        if disp.dim() == 3:
            disp = disp.unsqueeze(1)
        disp = torch.nn.functional.interpolate(
            disp, size=(H, W), mode="bilinear", align_corners=False)
        return disp.squeeze(1).squeeze(0).float().cpu().numpy()


def make_backbone(name, **kwargs):
    """Factory: name='dav2' or 'endodac' (case-insensitive)."""
    name = name.lower()
    if name == "dav2":
        return DAv2Backbone(**kwargs)
    if name == "endodac":
        return EndoDACBackbone(**kwargs)
    raise ValueError(f"Unknown depth backbone: {name!r}. "
                     f"Choose 'dav2' or 'endodac'.")
