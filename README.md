# 3DEndoMap (bronchoscopy + Endo-2DTAM)

Single-pipeline scaffold: **endoscopic video → Endo-2DTAM tracking +
2D-Gaussian map → dashboard.** No ORB+RANSAC fallback, no TSDF, no
subprocess hand-off — the dashboard imports and calls Endo-2DTAM's
`rgbd_slam()` directly.

> **Status:** proof-of-concept on bronchoscopy. Endo-2DTAM is the
> tracking + mapping engine. Patient-specific airway CT is optional;
> if absent, a procedural bronchial atlas is used as the GPS canvas
> (HUD discloses "approximate, atlas-based").

---

## 1. Pipeline

```
video.mp4
   │
   ├── frame extraction          (cv2.VideoCapture)
   ├── per-frame depth           (EndoDAC | DAv2 fallback)  → mm depth
   ├── Endo-2DTAM rgbd_slam()    (in-process)
   │     ├── camera tracking      (refined poses, mm scale)
   │     └── 2D-Gaussian mapping  (.ply of Gaussian centers)
   ├── procedural atlas           (bronchus_atlas, optional canvas)
   └── dashboard composite        (render_navigation_c3vd)
        ├── endoscopic panel
        ├── depth panel
        └── 3D GPS panel          (atlas or GS map + camera trajectory)
                              (HUD: time / speed / coverage / disclaimer)
```

## 2. Files

| File | Purpose |
|---|---|
| `run_video_dashboard.py` | one-shot entry point: video → dashboard |
| `endo2dtam_runner.py` | stages data, builds config, calls `rgbd_slam()`, parses outputs |
| `depth_backbones.py` + `dav2_depth.py` | EndoDAC / DAv2 depth wrappers |
| `bronchus_atlas.py` | procedural bronchial-tree atlas |
| `render_navigation_c3vd.py` | per-frame dashboard orchestrator |
| `dashboard_common.py` | shared rendering helpers (panels, HUD, GPS) |

## 3. Install

**One env, theirs.** Follow Endo-2DTAM's README exactly, then add our
small set of deps on top:

```bash
conda create -n endo2dtam python=3.10 -y && conda activate endo2dtam

# Endo-2DTAM stack (their README)
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/lastbasket/Endo-2DTAM external/Endo-2DTAM
pip install -r external/Endo-2DTAM/requirements.txt
git clone https://github.com/hbb1/diff-surfel-rasterization.git /tmp/dsr
pip install /tmp/dsr
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git /tmp/sknn
pip install /tmp/sknn

# Our small additions (compatible with the env above)
pip install numpy opencv-python matplotlib tqdm open3d transformers accelerate

# (one-time, optional) EndoDAC for endoscopy-tuned depth
git clone https://github.com/BeileiCui/EndoDAC.git external/EndoDAC
pip install fvcore timm einops
# Backbone (~390 MB):
#   https://drive.google.com/file/d/163ILZcnz_-IUoIgy1UF_r7PAQBqgDbll
#   → external/EndoDAC/pretrained_model/depth_anything_vitb14.pth
# Adapter:
#   https://drive.google.com/file/d/1qzAYBtwYJDN7hEi6pApqBOOz6pUhyY70
#   → external/EndoDAC/EndoDAC_fullmodel/depth_model.pth
```

If you skip EndoDAC, run with `--backbone dav2` (downloads from
HuggingFace at first call).

## 4. Run

```bash
python run_video_dashboard.py \
    --video your_bronchoscopy.mp4 \
    --output_dir output/bronchus \
    --hfov 90
```

Auto-detects `external/Endo-2DTAM` and `external/EndoDAC`. Pass
`--endo2dtam_repo` / `--endodac_repo` / `--endodac_weights` if they
live elsewhere.

Outputs under `output/bronchus/`:
- `navigation_dashboard.mp4` — composite video.
- `dashboard_frames/*.png` — keyframes.
- `atlas_airway.ply` — procedural canvas (when no `--organ_mesh`).
- `video_<timestamp>/params.npz` — Endo-2DTAM Gaussian + pose state.
- `video_<timestamp>/means3D.npy` — Gaussian centers.
- `video_<timestamp>/endo2dtam_gs_map.ply` — same centers, Open3D format.

Useful flags:

| Flag | Purpose |
|---|---|
| `--max_frames 200` | smoke test on long videos |
| `--skip_every 2` | take every Nth frame |
| `--organ_mesh airway.ply` | use a real CT-segmented airway, `--mode reveal` |
| `--mode {coverage, reveal, dynamic}` | dashboard GPS mode (default reveal with atlas) |
| `--hfov 110` | match your bronchoscope (Olympus BF-H190 ≈ 110°) |
| `--endo2dtam_tracking_iters 30` | longer per-frame tracking optimization |

## 5. What's known to be wrong / left to do

- **Atlas is generic, not patient-specific.** HUD discloses this.
- **No segmentation overlays** (polyps, lesions, tools, blood).
- **No anomaly detection** vs. a learned "normal airway" prior.
- **No calibrated confidence channel** on top of the ML overlays.
- **Endo-2DTAM scale anchoring**: the depth we feed in (EndoDAC or DAv2)
  is calibrated against an assumed median scene depth (`20 mm` default
  for bronchoscopy); if your scope geometry differs, pass
  `--assumed_median_depth_mm`.

These live as future modules in this repo's history; they're omitted
from the codebase until they earn their place by being demonstrably
useful on real footage.

## 6. Citations / acknowledgements

Built on:
- **Endo-2DTAM** (Huang, Cui, Bai et al., ICRA 2025) — tracking + mapping. [GitHub](https://github.com/lastbasket/Endo-2DTAM)
- **EndoDAC** (Cui et al., MICCAI 2024) — endoscopy depth. [GitHub](https://github.com/BeileiCui/EndoDAC)
- **Depth-Anything-V2** (Yang et al., NeurIPS 2024) — generic depth fallback.
- **Open3D** for mesh I/O.
