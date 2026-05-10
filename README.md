# 3DEndoMap (bronchoscopy + Endo-2DTAM)

Endoscopic video → tracking + per-frame depth + 3D map + dashboard, in
one command. Bronchoscopy-tuned, single-env.

> **Status (honest):** end-to-end runs and produces an MP4 + GS map +
> per-keyframe PNGs. The numerical quality is research-grade — drift,
> small mesh, generic atlas — not clinical. See *§5 Known limits*.

---

## 1. Pipeline

```
video.mp4
   │
   ├── frame extraction          (cv2.VideoCapture)
   ├── per-frame depth           (EndoDAC | DAv2)            mm
   ├── Endo-2DTAM rgbd_slam()    (in-process, ICRA 2025)
   │     ├── refined poses
   │     └── 2D-Gaussian map     → endo2dtam_gs_map.ply
   ├── procedural bronchial atlas (generic, anatomically credible)
   └── dashboard composite        (render_navigation_c3vd)
        ├── endoscopic panel
        ├── depth panel
        └── 3D GPS panel          (atlas + cyan trajectory)
                              (HUD: time / speed / coverage)
```

## 2. Files

| File | Purpose |
|---|---|
| `run_video_dashboard.py` | one-shot entry point: video → dashboard |
| `endo2dtam_runner.py` | stages data, calls Endo-2DTAM `rgbd_slam()`, parses outputs, **auto-cleans staged inputs + run dirs** (only the GS map .ply survives) |
| `depth_backbones.py` + `dav2_depth.py` | EndoDAC / DAv2 wrappers |
| `bronchus_atlas.py` | procedural 5-generation bronchial-tree atlas |
| `render_navigation_c3vd.py` | per-frame dashboard orchestrator |
| `dashboard_common.py` | shared helpers (panels, HUD, GPS) |

## 3. Install

One conda env. Endo-2DTAM's pins (python 3.10, torch 1.13.1, custom CUDA
extensions) drive the env; everything else fits on top.

```bash
conda create -n e2dtam python=3.10 -y && conda activate e2dtam
conda install -y -c "nvidia/label/cuda-11.7.1" cuda-nvcc cuda-cudart-dev cuda-libraries-dev
conda install -y -c conda-forge gxx_linux-64=9 gcc_linux-64=9 binutils_linux-64

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117
pip install ninja "setuptools<81" "numpy<2"

# Endo-2DTAM + its CUDA extensions (clone with --recursive for glm)
git clone --recursive https://github.com/lastbasket/Endo-2DTAM external/Endo-2DTAM
git clone --recursive https://github.com/hbb1/diff-surfel-rasterization /tmp/dsr
git clone https://gitlab.inria.fr/bkerbl/simple-knn /tmp/sknn
export CUDA_HOME=$CONDA_PREFIX
pip install --no-build-isolation /tmp/dsr /tmp/sknn

pip install -r external/Endo-2DTAM/requirements.txt
pip install opencv-python matplotlib tqdm open3d transformers accelerate \
    fvcore timm einops imageio[ffmpeg] natsort kornia pyyaml plotly lpips \
    trimesh pytorch-msssim torchmetrics

# (optional) EndoDAC for endoscopy-tuned depth
git clone https://github.com/BeileiCui/EndoDAC.git external/EndoDAC
# follow EndoDAC README to download:
#   external/EndoDAC/pretrained_model/depth_anything_vitb14.pth
#   external/EndoDAC/EndoDAC_fullmodel/depth_model.pth

# Pin the env so future pip installs don't break it
pip freeze > endo2dtam_env.txt
```

When adding deps later, `pip install --no-deps <name>` to keep the env stable.

## 4. Run

```bash
python run_video_dashboard.py \
    --video your_bronchoscopy.mp4 \
    --output_dir output/bronchus \
    --hfov 90 \
    --max_frames 100 --skip_every 3 \
    --endo2dtam_tracking_iters 8 --endo2dtam_mapping_iters 8
```

What stays on disk per run (everything else is auto-cleaned):
- `output/bronchus/navigation_dashboard.mp4`
- `output/bronchus/dashboard_frames/*.png` (keyframes)
- `output/bronchus/atlas_airway.ply` (procedural canvas)
- `output/bronchus/<seq>_gs_map.ply` (Endo-2DTAM Gaussian centers)

Useful flags:

| Flag | Effect |
|---|---|
| `--max_frames N` `--skip_every K` | smoke-test on long videos |
| `--organ_mesh patient_airway.ply` `--mode reveal` | use a real CT-segmented airway as the GPS canvas |
| `--atlas none` | drop the procedural atlas |
| `--backbone dav2` | skip EndoDAC, use HuggingFace Depth-Anything-V2 |
| `--keep_endo2dtam_artifacts` | keep params.npz / staged inputs for debugging |

## 5. Known limits (what's left to make this real)

| # | Gap | Why it matters | Effort |
|---|---|---|---|
| **L1** | **GPS canvas is a generic atlas, not patient-specific** | The dot can only roughly map onto a population-average shape — `--organ_mesh patient_airway.ply` is supported but you have to segment a CT yourself (3D Slicer + Lung CT Analyzer / `lungmask`) | tooling: ~1 wk |
| **L2** | **Endo-2DTAM GS map is small / drifty on short clips** | At ~30 frames it's under-converged; the .ply has hundreds of points instead of tens of thousands | tune iters + frame count, ~hours |
| **L3** | **Dashboard ignores the GS map** | The procedural atlas is shown instead. Wiring `--mode dynamic` to render the GS-map .ply directly is a small change | ~1 day |
| **L4** | **No segmentation overlays** | Polyps / tools / blood / lobar landmarks aren't flagged | SAM2 + heads, ~1 wk |
| **L5** | **No anomaly detection vs a normal-airway prior** | "Deviation from normal" — the clinical signal — isn't computed | PointNet++/DeepSDF on ATM'22, 4-8 wks |
| **L6** | **No calibrated confidence / trust channel** | Every overlay is presented with equal certainty | TTA std-dev + temperature scaling, 1-2 wks |
| **L7** | **No clinical UI** | PDF report, missed-region alerts, replay, polyp-size measurement | 6+ wks (MVP in 2) |

## 6. Citations / acknowledgements

Built on:
- **Endo-2DTAM** (Huang, Cui, Bai et al., ICRA 2025) — tracking + 2D-Gaussian mapping. [GitHub](https://github.com/lastbasket/Endo-2DTAM)
- **EndoDAC** (Cui et al., MICCAI 2024) — endoscopy depth. [GitHub](https://github.com/BeileiCui/EndoDAC)
- **Depth-Anything-V2** (Yang et al., NeurIPS 2024) — generic depth fallback.
- **Open3D** for mesh / point-cloud I/O.
