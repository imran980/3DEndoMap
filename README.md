# 3DEndoMap — Surgical Navigation + Anomaly Detection for Endoscopy

A monocular endoscopic video → a live **navigation dashboard** plus a
new clinical signal: **how the observed inner surface deviates from a
"normal" organ**. Deviations correlate with widening (inflammation),
narrowing (obstruction), missing branches (structural anomaly), and
shape inconsistencies (lesions / tumors).

> **Primary target: bronchoscopy.** Rigid airway tree, branching
> topology (carinas = natural SLAM landmarks), patient-specific
> chest CT is standard of care, ATM'22 / LIDC datasets exist for the
> normal prior. Colonoscopy (C3VD) stays as a legacy reference path.

---

## 1. Core idea

The model does **not** generate organ geometry. It does three things:

1. **Reconstruct** the inner surface actually observed by the camera
   (depth + pose + TSDF — anchored to data, never hallucinated).
2. **Compare** that reconstruction to a learned **prior of a healthy
   organ** (PointNet++ / DeepSDF trained on a corpus of normal CTs).
3. **Surface deviations** as the clinical signal:
   *abnormal widening · narrowing · missing branches · shape outliers.*

**Why this framing is safer than "generate the unseen organ":** when
the prior is wrong, we flag *more* anomalies, not *false anatomy*.
False positives are dismissible; we never invent missing pathology.

---

## 2. What works today

| Capability | Module | Status |
|---|---|---|
| C3VD data prep (hFOV, masks, near/far) | `prepare_c3vd.py` | ✓ |
| Pluggable depth backbones (DAv2 / EndoDAC) | `depth_backbones.py`, `dav2_depth.py` | ✓ |
| TSDF fusion from GT depth + GT poses | `build_colon_from_c3vd.py` | ✓ ICP 0.64 vs phantom |
| TSDF fusion from learned depth + GT poses | `build_colon_from_dav2.py` | ✓ ICP 0.66 (EndoDAC) |
| Standalone GPS dashboard (3 modes: coverage / reveal / dynamic) | `render_navigation_c3vd.py` | ✓ |
| Trajectory ↔ organ-mesh ICP alignment | inside `render_navigation_c3vd.py` | ✓ |
| Coverage heatmap, reveal-as-you-go, live TSDF growth | dashboard | ✓ |
| Withdrawal-time / speed / coverage HUD | `dashboard_common.py` | ✓ |
| Monocular VO from RGB (ORB + RANSAC) | `pose_estimation.py` | ✓ module ready, not yet wired into a CLI |
| Endo-4DGS training & rendering (legacy) | `train.py`, `render.py` | works on EndoNeRF; finicky on C3VD |

Status today: end-to-end works on C3VD because C3VD ships GT poses +
exact CT mesh. Bronchoscopy port + anomaly detection are the next
modules.

---

## 3. Module-based plan

Each module is small, has a clear input/output contract, and can be
built independently. Modules slot into the existing pipeline
without touching unrelated code.

| # | Module | Input | Output | Status | Effort |
|---|---|---|---|---|---|
| **M1** | **Video → poses (SLAM v0)** | MP4 | per-frame c2w | partial — `pose_estimation.py` done, missing CLI + no-GT calib | 1–2 days |
| **M2** | **Bronchoscopy data prep** | EndoMapper-Bronchus / BREAD | C3VD-shaped folder (RGB, optional GT) | not started | 2–3 days |
| **M3** | **Airway-CT segmentation → mesh** | chest CT (DICOM) | airway tree `.ply` | not started | 1 day per CT (3D Slicer + Lung CT Analyzer) |
| **M4** | **Normal organ prior** | corpus of airway meshes (ATM'22) | trained PointNet++ / DeepSDF | not started | 3–6 weeks |
| **M5** | **Mesh comparator** | observed mesh + prior | per-vertex deviation score + branch-diameter ratios + topology graph | not started | 1–2 weeks |
| **M6** | **Deformation field (advanced)** | observed mesh | SE(3)-equivariant warp `prior → observed`; large warp magnitude = anomaly | research, optional | 2–3 months |
| **M7** | **Anomaly overlay on dashboard** | M5 / M6 scores | color the GPS organ by deviation, HUD anomaly counter | not started | ~1 week (reuses `render_gps_frame`) |
| **M8** | **Segmentation overlays** | endo frame | per-pixel masks (carina / lesion / tool / blood) | not started | ~1 week (SAM2 / MedSAM / Polyp-PVT) |
| **M9** | **Confidence / trust layer** | every ML output | per-pixel/voxel uncertainty, surfaced as alpha + HUD | not started | ~1 week + 1 wk calibration |
| **M10** | **Clinical UI** | dashboard + findings | exportable PDF report, missed-region alerts, polyp size from depth | not started | 6+ weeks (MVP in 2) |

### Recommended build order

1. **M1 → M2 → M3** (the "any video runs end-to-end" milestone — ~1 week).
2. **M5** with a placeholder prior (mean shape from a few CTs) so the comparison plumbing is exercised early.
3. **M4** in parallel — train the real prior on ATM'22.
4. **M7** to surface results.
5. **M8 + M9** in parallel for clinical safety.
6. **M6** (deformation field) is the most publishable angle and runs as a research track on top of the others.
7. **M10** last; it depends on M4–M9 to have something worth exporting.

### Module independence

| Edits live in | Built / changed by |
|---|---|
| `pose_estimation.py`, new `run_video_dashboard.py` | M1 |
| `prepare_bronchus.py` (new), `arguments/bronchus.py` (new) | M2 |
| docs only | M3 |
| `external/normal_prior/` (new submodule) | M4 |
| `mesh_compare.py` (new) | M5 |
| `external/deformation_net/` (new) | M6 |
| `dashboard_common.py:render_gps_frame` (extend with anomaly coloring) | M7 |
| `seg_backbones.py` (new), `dashboard_common.py` | M8 |
| `confidence.py` (new), `dashboard_common.py` | M9 |
| `clinical_ui/` (new) | M10 |

Modules sharing a file are explicitly listed; otherwise they don't
collide. Three engineers can split as: **(M1, M2, M5, M7)**,
**(M3, M4, M6)**, **(M8, M9, M10)**.

---

## 4. Quick start

```bash
# Install
conda create -n ED4DGS python=3.8 && conda activate ED4DGS
pip install -r requirements.txt
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -e submodules/diff-gaussian-rasterization-depth
pip install -e submodules/simple-knn

# (optional) EndoDAC for endoscopy-tuned depth
git clone https://github.com/BeileiCui/EndoDAC.git external/EndoDAC
pip install fvcore timm einops
# Drop checkpoints into external/EndoDAC/{pretrained_model,EndoDAC_fullmodel}/
# (links in section 5)

# Today: run the C3VD demo (legacy, but proves the pipeline)
python render_navigation_c3vd.py \
    --c3vd_dir dataset/trans_t1_b \
    --output_dir output/c3vd_dash_reveal \
    --backbone endodac \
    --endodac_repo external/EndoDAC \
    --endodac_weights external/EndoDAC/EndoDAC_fullmodel/depth_model.pth \
    --organ_mesh dataset/trans_model.obj \
    --mode reveal     # also: coverage  |  dynamic
```

Once **M1 + M2** land, the entry point becomes:

```bash
python run_video_dashboard.py \
    --video my_bronchoscopy.mp4 \
    --output_dir output/my_bronchus \
    --hfov 90 \
    --mode dynamic
```

Once **M4 + M5 + M7** land, anomaly overlay turns on automatically
when a normal prior is configured (via `--normal_prior path/to/prior.pt`).

---

## 5. Datasets & external models

**Bronchoscopy (primary):**
- **EndoMapper-Bronchus** — clinical bronchoscopy with intrinsics.
- **BREAD phantom** — silicone airway + tracker poses.
- **ATM'22**, **EXACT'09**, **LIDC-IDRI** — chest CTs with airway
  segmentation labels, used to train the normal prior (M4).

**Colonoscopy (legacy):**
- **C3VD** — phantom + GT poses + GT depth + organ mesh.
- **EndoNeRF**, **StereoMIS** — Endo-4DGS training datasets.

**External depth models:**
- **EndoDAC** — endoscopy-finetuned DepthAnything.
  Repo: https://github.com/BeileiCui/EndoDAC.
  [Backbone (390 MB)](https://drive.google.com/file/d/163ILZcnz_-IUoIgy1UF_r7PAQBqgDbll) →
  `external/EndoDAC/pretrained_model/depth_anything_vitb14.pth`
  [Adapter (folder)](https://drive.google.com/file/d/1qzAYBtwYJDN7hEi6pApqBOOz6pUhyY70) →
  `external/EndoDAC/EndoDAC_fullmodel/depth_model.pth`
- **Depth-Anything-V2** (HuggingFace) — generic fallback backbone.

---

## 6. Repo layout

```
# Data prep
prepare_c3vd.py             C3VD → EndoNeRF format (legacy)
prepare_bronchus.py         (M2) bronchoscopy format
scripts/pre_dam_dep.py      Depth-Anything pseudo-depth

# Reconstruction
build_colon_from_c3vd.py    TSDF from GT depth + poses (sanity)
build_colon_from_dav2.py    TSDF from learned depth + poses
dynamic_organ.py            Open3D ScalableTSDFVolume wrapper
check_alignment.py          ICP sweep against a GT mesh

# Depth backbones
depth_backbones.py          Pluggable interface (DAv2 / EndoDAC)
dav2_depth.py               HF Depth-Anything-V2 wrapper

# Pose estimation
pose_estimation.py          (M1) ORB + RANSAC monocular VO

# Dashboards
render_navigation_c3vd.py   Standalone dashboard (RECOMMENDED)
render_navigation.py        Endo-4DGS-based dashboard (needs trained model)
dashboard_common.py         GPS panel, depth viz, HUD helpers

# (later modules — placeholders)
mesh_compare.py             (M5) observed-vs-prior comparator
seg_backbones.py            (M8) segmentation overlay backbones
confidence.py               (M9) calibrated trust layer

# Endo-4DGS infrastructure (inherited)
arguments/  scene/  gaussian_renderer/  utils/  submodules/
```

---

## 7. Honest status

**End-to-end works on C3VD phantom data.** Bronchoscopy port reuses
~90% of the code (M2 is just data plumbing; the dashboard, depth, and
fusion are unchanged). Anomaly detection is the new research track
and is gated on M4 (the prior); M5 + M7 are engineering once M4 lands.

**Not yet clinical.** Real-OR readiness needs M1 (real video, no
tracker), M9 (calibrated confidence), and clinical validation that
the deviation metric correlates with clinician-graded findings on a
labeled cohort. Each is its own multi-month workstream.

---

## 8. Citations

```
@inproceedings{huang2024endo,
  title={Endo-4DGS: Endoscopic monocular scene reconstruction with 4D Gaussian splatting},
  author={Huang, Yiming and Cui, Beilei and Bai, Long and Guo, Ziqi and Xu, Mengya and Islam, Mobarakol and Ren, Hongliang},
  booktitle={MICCAI}, year={2024}
}
@inproceedings{cui2024endodac,
  title={EndoDAC: Efficient Adapting Foundation Model for Self-Supervised Endoscopic Depth Estimation},
  author={Cui, Beilei and Islam, Mobarakol and Bai, Long and Ren, Hongliang},
  booktitle={MICCAI}, year={2024}
}
@inproceedings{depthanythingv2,
  title={Depth Anything V2}, author={Yang, Lihe and others},
  booktitle={NeurIPS}, year={2024}
}
```

Acknowledgements:
[Endo-4DGS](https://github.com/lastbasket/Endo-4DGS) ·
[EndoDAC](https://github.com/BeileiCui/EndoDAC) ·
[diff-gaussian-rasterization-depth](https://github.com/leo-frank/diff-gaussian-rasterization-depth) ·
[EndoNeRF](https://github.com/med-air/EndoNeRF) ·
[4DGaussians](https://github.com/hustvl/4DGaussians) ·
[Open3D](http://www.open3d.org/) ·
[Hugging Face Transformers](https://huggingface.co/docs/transformers).
