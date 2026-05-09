# 3DEndoMap (bronchoscopy-clean)

Minimal, single-env, bronchoscopy-only scaffold for surgical
navigation. Everything that was C3VD-specific, Endo-4DGS-specific, or
research-prototype-only has been removed; what's left is a small,
honest core that takes a video → produces a dashboard.

> Status: **proof-of-concept, not clinical.** ORB+RANSAC tracking
> drifts on long clips, depth comes from EndoDAC (relative + scale-
> calibrated), the GPS canvas is a procedural atlas (anatomically
> credible but not patient-specific), and the HUD says so. The
> roadmap below lists what each piece needs to become real.

---

## 1. The pipeline today

```
video.mp4
   │
   ├── frames extracted        (cv2.VideoCapture)
   ├── monocular VO             (pose_estimation.MonocularVO  — ORB+RANSAC)
   ├── per-frame depth          (depth_backbones / EndoDAC | DAv2)
   ├── TSDF fusion              (dynamic_organ.DynamicOrganBuilder)
   ├── procedural atlas         (bronchus_atlas.build_procedural_airway)
   └── dashboard composite      (render_navigation_c3vd + dashboard_common)
        ├── endoscopic panel
        ├── depth panel
        └── 3D GPS panel       (atlas + cyan trajectory polyline)
                            (HUD: withdrawal time / speed / coverage)
```

## 2. Files (the whole repo)

| File | Purpose |
|---|---|
| `run_video_dashboard.py` | one-shot entry point: video → dashboard |
| `pose_estimation.py` | ORB+RANSAC monocular VO + pose I/O |
| `tracking_backends.py` | pluggable interface (just ORB today) |
| `depth_backbones.py` + `dav2_depth.py` | DepthAnything-V2 / EndoDAC wrappers |
| `dynamic_organ.py` | Open3D TSDF fusion + cleanup filters |
| `bronchus_atlas.py` | procedural 5-generation bronchial-tree atlas |
| `render_navigation_c3vd.py` | per-frame orchestrator (panels, HUD, GPS) |
| `dashboard_common.py` | shared rendering helpers (panels, HUD, GPS frame) |

That's it. ~12 source files.

## 3. Install

One conda env, no separate Endo-2DTAM env. Python 3.8+.

```bash
conda create -n endomap python=3.8 -y && conda activate endomap
pip install -r requirements.txt
```

For EndoDAC depth (recommended over DAv2 for endoscopy), one-time:

```bash
git clone https://github.com/BeileiCui/EndoDAC.git external/EndoDAC
pip install fvcore timm einops
# Backbone (~390 MB):
#   https://drive.google.com/file/d/163ILZcnz_-IUoIgy1UF_r7PAQBqgDbll
#   → external/EndoDAC/pretrained_model/depth_anything_vitb14.pth
# Adapter (folder):
#   https://drive.google.com/file/d/1qzAYBtwYJDN7hEi6pApqBOOz6pUhyY70
#   → external/EndoDAC/EndoDAC_fullmodel/depth_model.pth
```

If you skip EndoDAC, pass `--backbone dav2` (downloads the weights from
HuggingFace at first run — ~400 MB).

## 4. Run

```bash
python run_video_dashboard.py \
    --video your_bronchoscopy.mp4 \
    --output_dir output/bronchus_demo \
    --hfov 90
```

That's the whole interface. Outputs:

- `output/bronchus_demo/navigation_dashboard.mp4` — composite video.
- `output/bronchus_demo/dashboard_frames/*.png` — keyframes.
- `output/bronchus_demo/atlas_airway.ply` — the procedural canvas.
- `output/bronchus_demo/dynamic_organ_mesh.ply` — TSDF mesh, if dynamic mode.

Useful flags:

| Flag | Purpose |
|---|---|
| `--max_frames 400` | smoke test on a long video |
| `--skip_every 2` | take every Nth frame |
| `--mode reveal` | paint the atlas in as the camera moves (default with atlas) |
| `--atlas none` | use the live TSDF mesh as the GPS canvas instead |
| `--organ_mesh path.ply` | use a real CT-segmented airway mesh (best, when available) |
| `--hfov 110` | match your bronchoscope (Olympus BF-H190 ≈ 110°, generic ≈ 90°) |

## 5. What's known to be wrong

- **GPS dot tracking is approximate.** ORB+RANSAC drifts; trajectory is
  scale-anchored only by an assumed median depth. Real fix is
  Endo-2DTAM or another endoscopic SLAM (see roadmap).
- **Atlas is generic.** Anatomically credible, but not patient-
  specific. HUD discloses this.
- **No segmentation, no anomaly detection, no confidence channel.**
  All on the roadmap.

## 6. Roadmap (what to add, in priority order)

| # | Module | What it does | Effort |
|---|---|---|---|
| **R1** | `--start_branch` flag | Manual anatomical anchor — user labels frame-0 ("video starts at right intermediate bronchus"). Trajectory snaps onto that branch's centerline. | ~1 day |
| **R2** | Endo-2DTAM / EndoGSLAM tracker | Real endoscopic SLAM. Replaces ORB+RANSAC. Use as a separate process; the dashboard reads its `pose.txt` + map. | 1–2 weeks |
| **R3** | CT airway segmentation tooling | Script that turns a chest CT into the `--organ_mesh` we already accept. 3D Slicer + lungmask. | ~1 week (per CT: 1 day) |
| **R4** | Normal-airway prior + comparator | Train PointNet++ / DeepSDF on ATM'22; compare observed mesh to prior; surface deviations as the clinical signal. | 4–8 weeks |
| **R5** | Segmentation overlays | Pluggable seg backbone (SAM2 / MedSAM / Polyp-PVT). | 1 week |
| **R6** | Confidence / trust layer | Per-output uncertainty (TTA std-dev for depth, logit margin for seg, voxel weights for mesh). | 1–2 weeks |
| **R7** | Clinical UI | PDF report, missed-region alerts, replay. | 6+ weeks |

## 7. Cleanup notes (what was removed in this branch)

- All Endo-4DGS infrastructure: `train.py`, `render.py`, `metrics.py`,
  `arguments/`, `scene/`, `gaussian_renderer/`, `submodules/`,
  `utils/` (TTF fonts), `scripts/` (dataset utilities).
- All C3VD-specific code: `prepare_c3vd.py`, `build_colon_from_*.py`,
  `check_alignment.py`, `create_synthetic_organ.py`,
  `extract_surface.py`, `register_to_organ.py`, `localize_camera.py`,
  `run_navigation_demo.py`, `visualize_navigation.py`,
  `render_navigation.py` (Endo-4DGS dashboard).
- The Endo-2DTAM subprocess wrapper: integration was env-fragile and
  was reverted. It's tracked as R2 to be redone properly later.
- All shell scripts (`*.sh`), the legacy `How to run`, and the
  `implementation plans/` design docs.

Net delete: ~80% of the previous branch's files; codebase is now
~12 source files, single conda env, one entry point.

## 8. Citations / acknowledgements

Built on:
- **EndoDAC** (Cui et al., MICCAI 2024) — endoscopy depth.
- **Depth-Anything-V2** (Yang et al., NeurIPS 2024) — generic depth fallback.
- **Open3D** for TSDF + mesh I/O.

Future work targets **Endo-2DTAM** (ICRA 2025) for tracking + Gaussian
mapping.
