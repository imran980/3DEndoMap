# 3DEndoMap — Bronchoscopy Navigation + Anomaly Detection

A monocular **bronchoscopic** video → a live navigation dashboard
plus a clinical signal: **how the observed airway differs from a
"normal" airway** (widening, narrowing, missing branches, lesions).

Bronchoscopy first because the airway tree is rigid (cartilage rings),
branching (carinas = built-in SLAM landmarks), short (5–15 min), and
chest CTs are standard of care so a patient-specific prior is always
available.

---

## 1. Core idea (and why it's clinically defensible)

1. **Reconstruct only what the camera actually sees** — depth + pose
   → TSDF / learned-Gaussian map. Anchored to data, never invented.
2. **Compare** that reconstruction to a **learned prior of a healthy
   airway** trained on a corpus of normal chest-CT segmentations
   (ATM'22).
3. **Surface deviations** as the clinical signal:
   *abnormal widening · narrowing · missing branches · shape outliers.*

When the prior is wrong, we flag *more* anomalies, not *false
anatomy*. False positives are dismissible by the clinician;
hallucinated geometry is not.

---

## 2. Model stack

| Job | Today | Target (after M1+M2) |
|---|---|---|
| **Camera tracking** | ORB+RANSAC VO (`pose_estimation.py`) — drifts, but zero deps; kept long-term as a fallback | **Endo-2DTAM** |
| **Per-frame depth** | **EndoDAC** via `depth_backbones.py` | **Endo-2DTAM internal depth** (EndoDAC retired) |
| **3D map** | Open3D TSDF (`dynamic_organ.py`) | **Endo-2DTAM 2D-Gaussian map** (TSDF retired) |
| **GPS canvas** | procedural bronchial atlas (`bronchus_atlas.py`) | same — atlas-based, approximate; HUD discloses it |
| **Normal-airway prior (M4)** | — | **PointNet++ / DeepSDF** trained on ATM'22 |
| **Comparison metric (M5)** | — | per-vertex Hausdorff + per-branch diameter + topology graph |
| **Segmentation (M7)** | — | **SAM2** + lesion / tool / blood heads |
| **Confidence (M8)** | — | TTA std-dev (depth) + temperature-scaled logits (seg) + voxel weights |

Why two columns: today's column is the **working pipeline you can
run right now**; the target column is what the codebase converges to
once Endo-2DTAM is integrated, at which point we drop the depth +
TSDF stack (see the simplification gate in §4).

Removed from the plan entirely: **NRGS-SLAM** (deformable — overkill
for rigid airways), legacy Endo-4DGS dashboard.

---

## 3. What works today

| Capability | Module | Status |
|---|---|---|
| MP4 → frames + VO + dashboard end-to-end | `run_video_dashboard.py` | ✓ but VO is the placeholder |
| Pluggable depth backbone (EndoDAC default, DAv2 fallback) | `depth_backbones.py` | ✓ |
| TSDF fusion (depth + pose → growing mesh), with edge / gradient / motion-gated filters and connected-component cleanup | `dynamic_organ.py` | ✓ |
| 3-mode dashboard: dynamic / coverage / reveal | `render_navigation_c3vd.py` | ✓ |
| Camera trajectory polyline on the GPS panel | `dashboard_common.py` | ✓ |
| Withdrawal / speed / coverage HUD | `dashboard_common.py` | ✓ |
| C3VD demo (legacy reference) | `prepare_c3vd.py` + `build_colon_from_*.py` | ✓ |

**Not yet wired in:** Endo-2DTAM (next), normal-prior model, mesh
comparator, segmentation, confidence layer.

---

## 4. Module plan

Each module is independent, has a clear input/output, and can be
built without touching unrelated code.

| # | Module | Input | Output | Status | Effort |
|---|---|---|---|---|---|
| **M1** | **Pose backend = Endo-2DTAM** (replaces ORB+RANSAC VO) | RGB frames | per-frame c2w + (later) GS map | next | ~1 wk |
| **M2** | **Map = Endo-2DTAM Gaussians** (replaces TSDF) | M1 output | live 2D-Gaussian airway surface | gated on M1 | 1–2 wks |
| **M3** | **Airway-CT segmentation** | chest CT (DICOM) | airway tree `.ply` | docs only (3D Slicer + Lung CT Analyzer) | 1 day per CT |
| **M4** | **Normal-airway prior** | corpus of M3 outputs | trained PointNet++ / DeepSDF | not started | 3–6 wks |
| **M5** | **Mesh comparator** | observed mesh + prior | per-vertex deviation, branch-diameter ratios, topology graph | not started | 1–2 wks |
| **M6** | **Anomaly overlay on dashboard** | M5 scores | airway colored by deviation, HUD anomaly counter | reuses `render_gps_frame` | 1 wk |
| **M7** | **Segmentation overlays (SAM2 + heads)** | endo frame | per-pixel masks + 3D markers | not started | ~1 wk |
| **M8** | **Confidence / trust layer** | every ML output | per-pixel/voxel uncertainty → alpha + HUD | not started | 1–2 wks |
| **M9** | **Clinical UI** | dashboard + findings | PDF report, missed-region alerts, polyp size from depth | not started | 6+ wks (MVP in 2) |

**Build order:** M1 → M2 → M5 (placeholder prior) → M4 (real prior) →
M6 → M7 + M8 in parallel → M9. M3 happens in the dataset workstream.

**File ownership** so two engineers don't collide:

| Owner | Files |
|---|---|
| M1 + M2 | `pose_estimation.py` + new `external/Endo-2DTAM/` integration |
| M3 + M4 | `external/normal_prior/` + `mesh_compare.py` |
| M5 + M6 | `mesh_compare.py` + `dashboard_common.py:render_gps_frame` |
| M7 | new `seg_backbones.py` + `dashboard_common.py` |
| M8 | new `confidence.py` + `dashboard_common.py` |
| M9 | new `clinical_ui/` |

### Simplification gate — files slated for removal once M1+M2 land

Endo-2DTAM produces tracking, a 3D Gaussian map, **and** internal
depth reasoning, so once it's working the depth+TSDF stack becomes
redundant. The cleanup commit happens **after** Endo-2DTAM is
verified end-to-end on a real bronchoscopy video; until then these
files stay as the working fallback path.

| File | Why it leaves | Replacement |
|---|---|---|
| `dav2_depth.py` | DAv2 wrapper — no longer needed for fusion or the depth panel | Endo-2DTAM internal depth |
| `depth_backbones.py` (EndoDAC + DAv2) | depth backbone abstraction is irrelevant once the SLAM provides depth | Endo-2DTAM |
| `dynamic_organ.py` (TSDF fusion + cleanup) | TSDF map replaced by 2D-Gaussian map | Endo-2DTAM `.ply` |
| `build_colon_from_c3vd.py` | C3VD-only TSDF utility | — (regression-test path drops to one script) |
| `build_colon_from_dav2.py` | learned-depth + GT-pose TSDF | — |
| `prepare_c3vd.py` | C3VD-specific dataset prep | kept only if we still want a regression test |
| Calibration plumbing inside `render_navigation_c3vd.py` (`a_avg`, `b_avg`, `assumed_median_depth_mm`, GT-depth pairing) | depth no longer needs per-frame mm anchoring | — |

**Kept as-is** (small core that stays valuable):

| File | Why it stays |
|---|---|
| `run_video_dashboard.py` | end-to-end entry point |
| `tracking_backends.py` | the abstraction, with Endo-2DTAM as the one backend (and ORB kept as the zero-dep fallback) |
| `pose_estimation.py` | the ORB fallback — runs in any env, no extra deps |
| `bronchus_atlas.py` | procedural GPS canvas |
| `dashboard_common.py` | panels / HUD / GPS rendering |
| `render_navigation_c3vd.py` | orchestrator — trimmed by ~40% after the cleanup |
| `check_alignment.py` | useful debug utility |

After the simplification: ~1500 LOC removed, single conda env
(no EndoDAC env needed), and no dual-pipeline maintenance burden.

---

## 5. Quick start

```bash
# Install
conda create -n ED4DGS python=3.8 && conda activate ED4DGS
pip install -r requirements.txt
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -e submodules/diff-gaussian-rasterization-depth
pip install -e submodules/simple-knn

# EndoDAC depth (one-time)
git clone https://github.com/BeileiCui/EndoDAC.git external/EndoDAC
pip install fvcore timm einops
# Backbone (390 MB) -> external/EndoDAC/pretrained_model/depth_anything_vitb14.pth
# Adapter (folder)  -> external/EndoDAC/EndoDAC_fullmodel/depth_model.pth

# Run the dashboard on any bronchoscopy video
python run_video_dashboard.py \
    --video my_bronchoscopy.mp4 \
    --output_dir output/bronchus_demo \
    --hfov 90 \
    --mode dynamic     # 'reveal' / 'coverage' if you have an airway CT mesh

# Optional: with a patient-specific airway tree
python run_video_dashboard.py \
    --video my_bronchoscopy.mp4 \
    --output_dir output/bronchus_demo \
    --hfov 90 \
    --organ_mesh airway_tree.ply \
    --mode reveal
```

Outputs: `navigation_dashboard.mp4` + per-keyframe PNGs +
`dynamic_organ_mesh.ply` (the live-fused airway surface).

---

## 6. Datasets

**Bronchoscopy:**
- **EndoMapper-Bronchus** — clinical bronchoscopy with intrinsics.
- **BREAD** — silicone airway phantom + tracker poses.
- **ATM'22 / EXACT'09 / LIDC-IDRI** — chest CTs with airway
  segmentation labels. Used to train the M4 normal prior; segment
  individual patients with **3D Slicer + Lung CT Analyzer** for
  M3.

**Legacy (kept for sanity demos):**
- **C3VD** — colon phantom + GT depth + GT poses + organ mesh. Use
  via `build_colon_from_*.py` and `render_navigation_c3vd.py` for
  pipeline regression tests.

---

## 7. Citations

```
@inproceedings{huang2024endo,
  title={Endo-4DGS: Endoscopic monocular scene reconstruction with 4D Gaussian splatting},
  booktitle={MICCAI}, year={2024}}
@inproceedings{cui2024endodac,
  title={EndoDAC: Efficient Adapting Foundation Model for Self-Supervised Endoscopic Depth Estimation},
  booktitle={MICCAI}, year={2024}}
@inproceedings{endo2dtam,
  title={Endo-2DTAM: 2D Gaussian Tracking and Mapping for Endoscopy}, year={2024}}
```

Acknowledgements:
[Endo-4DGS](https://github.com/lastbasket/Endo-4DGS) ·
[EndoDAC](https://github.com/BeileiCui/EndoDAC) ·
[Endo-2DTAM](https://github.com/lastbasket/Endo-2DTAM) ·
[Open3D](http://www.open3d.org/) ·
[Hugging Face Transformers](https://huggingface.co/docs/transformers).
