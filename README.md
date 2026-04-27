# 3DEndoMap — Surgical Navigation GPS for Endoscopy

3DEndoMap turns a monocular endoscopic video into a live **surgical
navigation dashboard**: the camera's position inside the organ, a 3D
map of inspected anatomy, and clinical metrics (withdrawal time,
pullback speed, coverage percentage) composited into a single MP4.

The project began as a fork of **Endo-4DGS** (MICCAI 2024) for 4D
Gaussian Splatting reconstruction, then grew into a standalone
pipeline that:

- ingests raw **C3VD** colonoscopy phantom data (or any sequence with
  RGB + per-frame poses),
- runs an off-the-shelf **monocular depth backbone** (EndoDAC, the
  endoscopy-specific depth model from MICCAI 2024 — or DAv2 as a
  generic fallback),
- fuses depth + poses into a **growing 3D mesh** of the organ via
  TSDF, and / or uses a pre-operative CT mesh painted in by the
  camera's frustum, and
- composites everything onto a clinician-style dashboard.

**Full project scope** (today + planned):
The end goal is a clinician-facing tool that, given just a live
endoscopic video, shows (1) **where** the camera is in 3D anatomy,
(2) **what** is in front of it (anatomy, polyps, tools, bleeding —
auto-segmented), (3) **how much** has been inspected vs. missed, and
(4) **how trustworthy** every ML-derived overlay is. Today only
1 + 3 are implemented (on phantom data with GT poses). The roadmap
adds monocular SLAM (drops the GT-pose requirement), online 4D
Gaussian fusion, anatomical shape priors, **per-frame semantic
segmentation overlays**, and a **calibrated confidence / trust
layer** that surfaces "how much to rely on this" alongside every
ML output — the safety property that separates a research demo
from a clinical tool.

> Status. **Research-grade demo on phantom data.** It works
> end-to-end on C3VD because C3VD ships ground-truth poses and an
> exact CT mesh of the phantom. Both of those disappear in the OR;
> see the **Roadmap** for what's needed to take this clinical.

---

## 1. What works today

| Capability | Module | Status |
|---|---|---|
| C3VD → EndoNeRF data prep (correct hFOV, mask convention, near/far) | `prepare_c3vd.py` | ✓ |
| Endo-4DGS training & rendering (legacy path) | `train.py`, `render.py` | works on EndoNeRF; on C3VD needs careful tuning |
| Depth-Anything pseudo-depth generator | `scripts/pre_dam_dep.py` | ✓ |
| **TSDF fusion from C3VD GT depth + poses** | `build_colon_from_c3vd.py` | ✓ ICP fitness 0.64 vs phantom |
| **Pluggable monocular depth backbones** | `depth_backbones.py`, `dav2_depth.py` | DAv2 + EndoDAC |
| **TSDF fusion from learned depth + GT poses** | `build_colon_from_dav2.py` | ✓ EndoDAC achieves ICP fitness 0.66 vs phantom (beats GT-depth) |
| **Standalone surgical-GPS dashboard** | `render_navigation_c3vd.py` | ✓ Three modes: `coverage`, `reveal`, `dynamic` |
| Endo-4DGS-based dashboard (older) | `render_navigation.py` | works when the trained Gaussian model is converged |
| Trajectory ↔ organ-mesh ICP alignment | `_align_trajectory_to_organ` in `render_navigation_c3vd.py` | ✓ |
| Coverage heatmap + reveal-as-you-go + live TSDF growth | dashboard | ✓ |
| Withdrawal-time / pullback-speed HUD | `dashboard_common.py:draw_hud` | ✓ |

### Architecture

```mermaid
flowchart LR
    A[C3VD or any RGB+poses] --> B(prepare_c3vd.py)
    A --> C(render_navigation_c3vd.py)
    A --> D(build_colon_from_dav2.py)
    A --> E(build_colon_from_c3vd.py)
    B --> F(scripts/pre_dam_dep.py)
    F --> G(train.py – Endo-4DGS)
    G --> H(render.py / render_navigation.py)
    D -. depth backbone .-> I[depth_backbones.py: DAv2 / EndoDAC]
    C -. depth backbone .-> I
    E -- GT depth --> J[TSDF fusion]
    D -- learned depth --> J
    C -- learned depth --> J
    C --> K[navigation_dashboard.mp4]
    J --> L[colon_mesh.ply]
```

---

## 2. End-to-end on C3VD (the only thing you need most days)

```bash
# 0. Install
conda create -n ED4DGS python=3.8 && conda activate ED4DGS
pip install -r requirements.txt
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -e submodules/diff-gaussian-rasterization-depth
pip install -e submodules/simple-knn

# 1. Set up EndoDAC (one-time, ~800 MB downloads)
git clone https://github.com/BeileiCui/EndoDAC.git external/EndoDAC
cd external/EndoDAC && pip install fvcore timm einops && cd ../..
# Backbone (one .pth)  → external/EndoDAC/pretrained_model/depth_anything_vitb14.pth
#   https://drive.google.com/file/d/163ILZcnz_-IUoIgy1UF_r7PAQBqgDbll
# Adapter (folder)     → external/EndoDAC/EndoDAC_fullmodel/depth_model.pth
#   https://drive.google.com/file/d/1qzAYBtwYJDN7hEi6pApqBOOz6pUhyY70

# 2. Run the dashboard — three modes, no Endo-4DGS training required
python render_navigation_c3vd.py \
    --c3vd_dir dataset/trans_t1_b \
    --output_dir output/c3vd_dash_reveal \
    --backbone endodac \
    --endodac_repo external/EndoDAC \
    --endodac_weights external/EndoDAC/EndoDAC_fullmodel/depth_model.pth \
    --organ_mesh dataset/trans_model.obj \
    --mode reveal     # also: coverage  |  dynamic
```

**What you get:**
- `navigation_dashboard.mp4` — composited per-frame video (1769×896 by default)
- `dashboard_frames/frame_*.png` — one keyframe every ~5%
- `dynamic_organ_mesh.ply` (dynamic mode only) — final fused mesh

**HUD:**
- `Withdrawal MM:SS` (target ≥ 6:00 per colonoscopy guidelines, color-coded)
- `Speed mm/s` (1–6 mm/s green, otherwise red)
- `Path mm` cumulative
- `Revealed % | Coverage % | Mesh NN,NNN pts` depending on mode

### Verifying a fused mesh

```bash
python check_alignment.py \
    --ours output/c3vd_dash_dynamic/dynamic_organ_mesh.ply \
    --gt   dataset/trans_model.obj
```
Sweeps 32 axis conventions + ICP, reports top-5 fitness. EndoDAC + GT
poses currently scores ~0.66 on `trans_t1_b`.

---

## 3. Repo layout

```
prepare_c3vd.py             C3VD → EndoNeRF format
scripts/pre_dam_dep.py      Depth-Anything pseudo-depth generator

train.py / render.py        Endo-4DGS training + offline rendering (legacy)

# Surface + organ reconstruction
build_colon_from_c3vd.py    TSDF fusion from C3VD GT depth + poses
build_colon_from_dav2.py    TSDF fusion from learned depth (DAv2 / EndoDAC) + poses
dynamic_organ.py            Open3D ScalableTSDFVolume wrapper
check_alignment.py          ICP sweep against a GT mesh

# Depth backbones
depth_backbones.py          Pluggable backbone interface (DAv2 / EndoDAC)
dav2_depth.py               Hugging Face Depth-Anything-V2 wrapper

# Dashboards
render_navigation_c3vd.py   Standalone C3VD dashboard (recommended path)
render_navigation.py        Endo-4DGS-based dashboard (needs trained model)
dashboard_common.py         Shared GPS panel, depth viz, HUD helpers

# Plus the full Endo-4DGS infrastructure inherited from upstream
arguments/  scene/  gaussian_renderer/  utils/  submodules/
```

---

## 4. Honest assessment — does it work for a real surgeon?

**No.** Not yet. The C3VD demo works because C3VD provides:

1. **Perfect ground-truth camera poses** from an external optical tracker.
2. **An exact CT mesh** of the same physical phantom the camera moves through.
3. **Static, rigid, non-deforming anatomy.**

In the operating room, all three vanish:
- A real colonoscope has no external tracker.
- The pre-operative CT is days/weeks old, of a deformed organ.
- Tissue moves with peristalsis and scope pressure.

What we have today is the right **scaffold** — depth backbone, fusion,
visualization, and HUD all work — but several large research components
are still missing before this is clinically useful.

---

## 5. Roadmap to a clinical tool

The list below is ordered by build cost. Each phase produces a working
demo at the end; each unlocks the next.

### Phase 0 (current state)
- Static EndoDAC depth → C3VD GT poses → TSDF / pre-op CT reveal.
- Useful as a teaching tool, demo, and integration shell. Not clinical.

### Phase 1 — Drop the GT-pose dependency: monocular SLAM
- **Replace C3VD `pose.txt` with live visual odometry.**
- **SOTA option (best fit):** **Endo-2DTAM** — Gaussian-Splatting SLAM
  for endoscopic scenes (the same authors as Endo-4DGS).
  https://github.com/lastbasket/Endo-2DTAM
- **Alternates:** DROID-SLAM, ORB-SLAM3, MonoGS / RTG-SLAM. All
  general-purpose; expect drift on long featureless tubes.
- **Deliverable:** dashboard works on a sequence with **only RGB**
  (no `pose.txt`), poses estimated live. Acceptance: trajectory ICP
  fitness ≥ 0.5 vs C3VD GT poses on `trans_t1_b`.
- **Effort:** 4–6 weeks of integration.

### Phase 2 — Drop the static-TSDF assumption: online Gaussian fusion
- **Replace `Open3D.ScalableTSDFVolume` with an online 3D Gaussian
  Splatting map** that grows / refines per frame instead of
  averaging discrete voxels.
- **SOTA options:**
  - **MonoGS** (CVPR 2024) — monocular GS-SLAM with live map.
  - **RTG-SLAM** (SIGGRAPH 2024) — real-time GS-SLAM on a single GPU.
  - Or extend Endo-2DTAM's GS map directly.
- **Deliverable:** the `dynamic` dashboard mode renders smooth 3DGS
  splats in place of voxel mesh; sub-mm local detail.
- **Effort:** 4–8 weeks; needs GPU at inference (≥ 16 GB VRAM).

### Phase 3 — Drop the exact-mesh assumption: anatomical shape prior
- **Add a Statistical Shape Model (SSM) of the colon as a soft
  Bayesian prior.** Where the camera has seen tissue → trust the
  reconstruction; where it hasn't → fall back to the prior with a
  visible **uncertainty channel** (translucent / hatched in the GPS
  panel).
- **Approaches, ranked by clinical safety:**
  1. **Patient-specific pre-op CT segmentation as the prior**
     (requires a CT per patient; safest because the prior is
     literally the patient's anatomy).
  2. **PCA-based SSM** trained on a colon-population dataset
     (smooth, deterministic, conservative — the preferred
     research-grade path).
  3. **Generative diffusion / NeRF prior** (most expressive,
     **most dangerous** clinically because it hallucinates
     plausible-looking pockets that may not exist).
- **SOTA references:**
  - **ColonNeRF** (ICCV 2023 W) — neural colon shape prior.
  - **AnatomyDiff** family — diffusion priors for organs.
  - **OrganMNIST / classical SSM toolkits** for the PCA path.
- **Deliverable:** organ surface beyond the camera frustum is
  filled in by the prior, with a colored uncertainty overlay
  (e.g. low-α teal = predicted, opaque pink = observed).
- **Effort:** 6–12 weeks for #1, 3+ months for #2, research project for #3.

### Phase 4 — Drop the rigid-organ assumption: deformable / 4D model
- **Tissue deforms** under scope pressure and peristalsis. A static
  fused mesh smears those changes away.
- **Approach:** time-aware 4D Gaussians (Endo-4DGS already supports
  this offline) integrated with the online SLAM map.
- **SOTA references:** Endo-4DGS (offline), Deformable-3DGS
  (CVPR 2024), 4DGS variants.
- **Deliverable:** GPS panel shows organ shape *changing* over time,
  not just growing.
- **Effort:** research project; expected at the end of this roadmap.

### Phase 5 — Per-frame semantic segmentation overlays
- **What:** add a pluggable **segmentation backbone** (mirroring
  `depth_backbones.py`) that produces per-pixel masks for clinically
  relevant classes — **polyps, anatomy landmarks (lumen / haustra /
  ileocecal valve), surgical tools, bleeding, image-quality
  (clear / bubbles / stool / blur)**.
- **Where it shows up:**
  1. **Endo panel**: colored translucent masks overlaid on the live
     frame.
  2. **3D map**: back-project mask centroids through depth + pose
     onto the organ surface — polyps become persistent 3D markers
     you can revisit.
  3. **HUD**: per-finding counters ("polyps detected: 3"), and an
     image-quality % that drives the existing withdrawal-quality
     metric.
- **SOTA options (off-the-shelf, no training):**
  - **SAM2 / SAM2-Video** (Meta, 2024) — class-agnostic, prompt-trackable across frames.
  - **MedSAM** (Nature 2024) — medical-tuned SAM.
  - **Polyp-PVT / FCBFormer / TransNetR** — colonoscopy polyp segmentation.
  - **Surgical-SAM** family — surgical-tool segmentation.
- **Deliverable:** dashboard with a fourth "findings" overlay; every
  mask clickable to a 3D location; report at the end listing all
  flagged findings with thumbnails + GPS positions.
- **Effort:** ~250 LOC for SAM2 + a polyp head, ~6 hours integration.
  Calibrating against a labeled dataset (Kvasir-SEG, CVC-ClinicDB) is
  another 2–3 days for evaluation.

### Phase 6 — Calibrated confidence / trust layer
- **Why:** stock NN confidences are overconfident. A clinical tool
  needs every ML-derived overlay to come with a defensible
  "trust this how much?" signal. This phase adds a **unified
  uncertainty channel** spanning depth, segmentation, pose, mesh,
  and (eventually) the SSM prior.
- **Per-output uncertainty sources:**
  - **Depth (EndoDAC / DAv2):** test-time augmentation variance
    (predict 4× with horizontal flip / brightness jitter, take
    per-pixel std-dev). Already available in `depth_backbones.py`
    interface — needs a `predict_with_uncertainty()` method.
  - **Segmentation:** sigmoid logit margin per pixel; MC-dropout
    when the backbone supports it.
  - **Pose (SLAM):** inlier ratio + reprojection error from the
    matcher.
  - **TSDF mesh:** voxel weight (Open3D already tracks how many
    frames touched each voxel).
  - **Coverage / reveal:** number of frustum hits per vertex
    (already computed; just expose).
- **Calibration to make confidences honest:**
  - **Temperature scaling** on a held-out set (~5 LOC).
  - **Conformal prediction** for actual coverage guarantees
    ("90% of pixels are within ±N mm at this confidence").
  - **Deep ensembles / MC-dropout** where the backbone supports it.
- **How it surfaces on the dashboard:**
  1. **Translucency / hatching:** high-confidence regions opaque,
     low-confidence translucent or hatched, on every overlay
     (mask, mesh, coverage, reveal, predicted-from-prior regions).
  2. **Dedicated uncertainty panel:** grayscale heatmap, dark = trust,
     bright = doubt, sitting beside the depth panel.
  3. **HUD numbers:** "Mean depth uncertainty: 2.3 mm",
     "Polyp confidence: 0.78", "Pose drift: 1.2 mm/frame".
  4. **Alert thresholds:** popup when confidence drops below a
     threshold (obscured frame, occluded region, tracker lost).
- **Deliverable:** every overlay on the dashboard answers
  "how much should the surgeon believe this?" without requiring
  ML literacy. Same module is reused by phase 3's anatomical-prior
  uncertainty channel.
- **Effort:** ~250 LOC for the confidence interface + per-backbone
  hooks + dashboard rendering, ~6 hours of integration. Adding
  conformal prediction with a real labeled set: +1 week.

### Phase 7 — Clinical UI on top
- Bookmark / export findings (PDF report with thumbnails + 3D
  positions, exportable per-procedure).
- Withdrawal-quality scorecard (time, speed, coverage, missed
  regions, image-quality %).
- Missed-region alerts in real time ("you didn't look at the cecum
  for 90 s").
- Polyp size measurements via depth (click two points, report mm).
- Replay mode for case review with full overlays + uncertainties.
- **Effort:** 6+ weeks; minimum viable scorecard can land in 2.

### Suggested team / parallelism

| Phase | Can start immediately | Blocks on |
|---|---|---|
| 1 (SLAM)                  | yes | — |
| 2 (online 3DGS)           | yes | — |
| 3.1 (CT prior)            | yes | — |
| 3.2 (SSM)                 | yes (data work) | colon dataset |
| 3.3 (diffusion)           | yes (research) | — |
| 4 (4D)                    | partial | phase 2 |
| 5 (segmentation overlays) | yes | — |
| 6 (confidence / trust)    | yes | works best after 5 + 1 + 3 |
| 7 (clinical UI)           | yes | strongest after 5 + 6 |

A 3-person split would be: **SLAM + 4D lead** (phases 1 + 2 + 4),
**Anatomical-prior + segmentation lead** (phases 3 + 5),
**Confidence + clinical-UI lead** (phases 6 + 7). Phases 5 and 6
together are what turn the demo from "shows organ shape" into
"shows organ shape, what's in it, and how much to trust either" —
they're the highest clinical-value pair on this list.

---

## 6. Other dataset paths (not the focus, but supported)

- **EndoNeRF**: train via Endo-4DGS as documented in the original
  paper. Use `arguments/endonerf.py`.
- **StereoMIS**: run `prepare_stereomis.sh`, then standard Endo-4DGS
  training via `arguments/stereomis.py`.
- **C3VD**: prep with `prepare_c3vd.py` (defaults assume the wide-FOV
  Olympus colonoscope, hFOV ≈ 140°). Use the new
  `render_navigation_c3vd.py` instead of the Endo-4DGS dashboard.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Loss=0 / PSNR=inf` during Endo-4DGS training | All-white masks getting inverted to all-zero | `prepare_c3vd.py` now writes black masks; re-prep + retrain |
| Renders are noise / pixel-gradient depth | hFOV wrong | use `--hfov 140` in `prepare_c3vd.py`, retrain |
| `Revealed 0.0%` in dashboard | Trajectory not aligned to organ mesh | trajectory ICP runs by default in `render_navigation_c3vd.py`; check `Trajectory->organ ICP: fitness=...` line |
| EndoDAC dim error `not multiple of patch height 14` | Checkpoint metadata mismatched | wrapper now hardcodes 224×280 (matches `test_simple.py`) |
| `0 frames fused` with DAv2 | Metric variant predicts meters, gets clipped | use `--variant vitb` + per-frame calibration |
| ONNX `pthread_setaffinity_np` warnings | Cgroup affinity limits, harmless | ignore |

---

## 8. Citations

This work builds on:

```
@inproceedings{huang2024endo,
  title={Endo-4dgs: Endoscopic monocular scene reconstruction with 4d gaussian splatting},
  author={Huang, Yiming and Cui, Beilei and Bai, Long and Guo, Ziqi and Xu, Mengya and Islam, Mobarakol and Ren, Hongliang},
  booktitle={MICCAI},
  year={2024},
}

@inproceedings{cui2024endodac,
  title={EndoDAC: Efficient Adapting Foundation Model for Self-Supervised Endoscopic Depth Estimation},
  author={Cui, Beilei and Islam, Mobarakol and Bai, Long and Ren, Hongliang},
  booktitle={MICCAI},
  year={2024},
}

@inproceedings{depthanythingv2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and ...},
  booktitle={NeurIPS},
  year={2024},
}
```

Acknowledgements: [StereoMIS](https://arxiv.org/abs/2304.08023v1) ·
[diff-gaussian-rasterization-depth](https://github.com/leo-frank/diff-gaussian-rasterization-depth) ·
[EndoNeRF](https://github.com/med-air/EndoNeRF) ·
[4DGaussians](https://github.com/hustvl/4DGaussians) ·
[Depth-Anything-ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX) ·
[Open3D](http://www.open3d.org/) ·
[Hugging Face Transformers](https://huggingface.co/docs/transformers).
