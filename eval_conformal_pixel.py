"""
Phase 3 eval: per-pixel split-conformal calibration on C3VD.

Given a C3VD sequence dir (color/, depth/, pose.txt, mesh.obj):
  1. Run EndoDAC under N-augmentation TTA -> per-pixel (mean, sigma)
     in disparity-like units (relative backbones).
  2. Fit a per-sequence affine `depth = a / pred + b` against the
     5k%-trimmed median of GT depth, the same way the dashboard does.
     Propagate sigma through the first-order Taylor of the inverse.
  3. Sample K pixels per frame uniformly from valid GT.
  4. Split frames into cal / test (default 50/50, frame-level — note
     this is the MVP single-sequence trick; for the paper, split at
     the SEQUENCE level for exchangeability).
  5. Fit q on cal scores, report coverage + width + AUROC on test
     at alpha in {0.1, 0.05, 0.01}.

Usage:
    python eval_conformal_pixel.py \\
        --c3vd_dir dataset/trans_t1_b \\
        --output_dir output/conformal_phase3 \\
        --endodac_repo external/EndoDAC \\
        --endodac_weights external/EndoDAC/EndoDAC_fullmodel/depth_model.pth \\
        --tta_n 4 \\
        --frames 80 \\
        --pixels_per_frame 5000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from depth_backbones import make_backbone
from dav2_depth import fit_disparity_to_depth
from tta_depth import TTAWrapper
from conformal import evaluate, calibration_curve


def _read_c3vd_depth_mm(path, target_hw=None):
    """Match render_navigation_c3vd._read_c3vd_depth_mm exactly so the
    eval and the dashboard use the same GT convention (mm)."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 3:
        raw = raw[..., 0]
    if target_hw is not None and raw.shape[:2] != target_hw:
        raw = cv2.resize(raw, (target_hw[1], target_hw[0]),
                         interpolation=cv2.INTER_NEAREST)
    if raw.dtype in (np.float32, np.float64):
        return raw.astype(np.float32)
    return raw.astype(np.float32) * (100.0 / 65535.0)


def _pair_frames(c3vd_dir: str) -> List[Tuple[str, str]]:
    """Return [(color_path, depth_path), ...] paired by stem."""
    color_dir = os.path.join(c3vd_dir, "color")
    depth_dir = os.path.join(c3vd_dir, "depth")
    # The C3VD layout user described: a `color/` and `depth/` folder
    # plus loose .tiff. Try the folder layout first, fall back to root.
    colors = sorted(glob.glob(os.path.join(color_dir, "*.png")))
    depths = sorted(glob.glob(os.path.join(depth_dir, "*.tiff"))) \
        + sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    if not colors:
        colors = sorted(glob.glob(os.path.join(c3vd_dir, "*_color.png")))
        depths = sorted(glob.glob(os.path.join(c3vd_dir, "*_depth.tiff"))) \
            + sorted(glob.glob(os.path.join(c3vd_dir, "*_depth.png")))
    n = min(len(colors), len(depths))
    if n == 0:
        sys.exit(f"ERROR: no color/depth pairs found under {c3vd_dir}")
    return list(zip(colors[:n], depths[:n]))


def _propagate_sigma_through_inverse(sigma_pred: np.ndarray,
                                     pred: np.ndarray, a: float
                                     ) -> np.ndarray:
    """First-order Taylor of depth = a / pred + b around pred:
    sigma_depth ~ |a| / pred^2 * sigma_pred.

    Same b cancels (it's an additive constant). Clamps pred away from
    zero so degenerate pixels don't blow up sigma — those get masked
    out by valid_mask anyway.
    """
    safe_pred = np.clip(np.asarray(pred), 1e-3, None)
    return np.abs(a) / (safe_pred ** 2) * np.asarray(sigma_pred)


def _calibrate_scale(bb, frames_color, frames_depth, n_cal_frames=20):
    """Fit a global (a, b) from disparity to depth_mm using up to N
    calibration frames. Median over per-frame fits — same procedure
    as render_navigation_c3vd.run()."""
    if getattr(bb, "is_metric", False):
        return None, None
    n_pick = min(n_cal_frames, len(frames_color))
    pick = np.linspace(0, len(frames_color) - 1, n_pick, dtype=int)
    a_list, b_list = [], []
    for i in tqdm(pick, desc="Scale calib"):
        rgb = cv2.cvtColor(cv2.imread(frames_color[i]), cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        # Use the BARE backbone (no TTA) for scale fit — we just want
        # a quick a, b. TTA noise would just average out anyway.
        pred = bb.backbone.predict(rgb) if isinstance(bb, TTAWrapper) \
            else bb.predict(rgb)
        gt = _read_c3vd_depth_mm(frames_depth[i], target_hw=(H, W))
        if gt is None:
            continue
        a, b = fit_disparity_to_depth(pred, gt)
        if a is not None:
            a_list.append(a)
            b_list.append(b)
    if not a_list:
        sys.exit("ERROR: no successful scale-calibration fits.")
    return float(np.median(a_list)), float(np.median(b_list))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c3vd_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--backbone", default="endodac",
                   choices=["dav2", "endodac"])
    p.add_argument("--variant", default="vitb")
    p.add_argument("--endodac_repo", default="external/EndoDAC")
    p.add_argument("--endodac_weights",
                   default="external/EndoDAC/EndoDAC_fullmodel/depth_model.pth")
    p.add_argument("--tta_n", type=int, default=4,
                   help="Number of TTA augmentations (>=2 for non-zero sigma).")
    p.add_argument("--frames", type=int, default=80,
                   help="Total frames to evaluate (uniform stride across "
                        "the sequence).")
    p.add_argument("--pixels_per_frame", type=int, default=5000)
    p.add_argument("--cal_frac", type=float, default=0.5,
                   help="Fraction of frames used for cal split. Note: "
                        "frame-level split is the MVP shortcut; for the "
                        "paper, split at the sequence level.")
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.1, 0.05, 0.01])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--depth_trunc_mm", type=float, default=80.0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    pairs = _pair_frames(args.c3vd_dir)
    print(f"Found {len(pairs)} color/depth pairs under {args.c3vd_dir}.")
    n_eval = min(args.frames, len(pairs))
    idx = np.linspace(0, len(pairs) - 1, n_eval, dtype=int)
    frames_color = [pairs[i][0] for i in idx]
    frames_depth = [pairs[i][1] for i in idx]

    # ---- Backbone (no TTA) for scale calibration ----
    print(f"Loading backbone ({args.backbone})...")
    if args.backbone == "endodac":
        bb_bare = make_backbone(
            "endodac",
            repo_dir=args.endodac_repo,
            weights_path=args.endodac_weights,
        )
    else:
        bb_bare = make_backbone("dav2", variant=args.variant)
    a, b = _calibrate_scale(bb_bare, frames_color, frames_depth,
                            n_cal_frames=min(20, n_eval))
    if a is not None:
        print(f"Disparity->depth scale: a={a:.3f}, b={b:.3f}")
    else:
        print("Metric backbone; skipping scale calibration.")

    # ---- TTA wrapper around the same backbone ----
    tta = TTAWrapper(bb_bare, n=args.tta_n)
    print(f"TTA wraps {tta.variant} with N={tta.n} augmentations.")

    # ---- Per-frame inference + per-pixel sampling ----
    preds_d, sigmas_d, gts, frame_ids = [], [], [], []
    for fi, (cp, dp) in enumerate(tqdm(list(zip(frames_color, frames_depth)),
                                       desc="TTA")):
        rgb = cv2.cvtColor(cv2.imread(cp), cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        mean_disp, sigma_disp = tta.predict_with_uncertainty(rgb)

        if a is not None:
            depth_mm = a / np.clip(mean_disp, 1e-6, None) + b
            sigma_mm = _propagate_sigma_through_inverse(sigma_disp,
                                                       mean_disp, a)
        else:
            depth_mm = mean_disp * 1000.0
            sigma_mm = sigma_disp * 1000.0

        gt = _read_c3vd_depth_mm(dp, target_hw=(H, W))
        if gt is None:
            continue
        valid = (
            np.isfinite(depth_mm)
            & np.isfinite(sigma_mm)
            & np.isfinite(gt)
            & (gt > 0.5)
            & (gt < args.depth_trunc_mm)
            & (depth_mm > 0.5)
            & (depth_mm < args.depth_trunc_mm)
            & (sigma_mm > 0)
        )
        n_valid = int(valid.sum())
        if n_valid < 100:
            continue
        sample_n = min(args.pixels_per_frame, n_valid)
        ys, xs = np.where(valid)
        pick = rng.choice(n_valid, sample_n, replace=False)
        preds_d.append(depth_mm[ys[pick], xs[pick]])
        sigmas_d.append(sigma_mm[ys[pick], xs[pick]])
        gts.append(gt[ys[pick], xs[pick]])
        frame_ids.append(np.full(sample_n, fi, dtype=np.int32))

    pred = np.concatenate(preds_d)
    sig = np.concatenate(sigmas_d)
    gt = np.concatenate(gts)
    fid = np.concatenate(frame_ids)
    print(f"Pooled {pred.size} pixels from {len(preds_d)} frames "
          f"(median sigma_mm={np.median(sig):.2f}, median |err|="
          f"{np.median(np.abs(pred - gt)):.2f} mm).")

    # ---- Frame-level cal/test split (MVP) ----
    uniq = np.unique(fid)
    rng.shuffle(uniq)
    n_cal_frames = int(len(uniq) * args.cal_frac)
    cal_frames = set(uniq[:n_cal_frames].tolist())
    cal_mask = np.isin(fid, list(cal_frames))
    test_mask = ~cal_mask
    print(f"Frame-level split: {cal_mask.sum()} cal pixels "
          f"({n_cal_frames} frames) / {test_mask.sum()} test pixels "
          f"({len(uniq) - n_cal_frames} frames).")

    # ---- Evaluate at each alpha ----
    reports = []
    for alpha in args.alphas:
        rep = evaluate(
            pred[cal_mask], sig[cal_mask], gt[cal_mask],
            pred[test_mask], sig[test_mask], gt[test_mask],
            alpha=alpha,
        )
        print(rep)
        reports.append(dict(
            alpha=rep.alpha, q=rep.q, coverage=rep.coverage,
            mean_width=rep.mean_width, auroc=rep.auroc,
            n_cal=rep.n_cal, n_test=rep.n_test,
        ))

    # ---- Calibration curve ----
    cs = np.abs(pred[cal_mask] - gt[cal_mask]) \
        / np.clip(sig[cal_mask], 1e-9, None)
    curve = calibration_curve(cs)

    # ---- Dump artifacts ----
    out = dict(
        config=vars(args),
        scale=dict(a=a, b=b),
        n_pixels=int(pred.size),
        n_frames=int(len(uniq)),
        reports=reports,
        calibration_curve=dict(
            alpha=curve["alpha"].tolist(),
            predicted=curve["predicted"].tolist(),
            observed=curve["observed"].tolist(),
        ),
    )
    with open(os.path.join(args.output_dir, "conformal_report.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {os.path.join(args.output_dir, 'conformal_report.json')}.")

    # Calibration plot, if matplotlib is around.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.plot(curve["predicted"], curve["observed"], "o-", ms=4)
        ax.set_xlabel("predicted coverage (1 - alpha)")
        ax.set_ylabel("observed coverage")
        ax.set_title("Calibration curve (cal-set self-check)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        plt.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "calibration_curve.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Skipped plot ({e}).")


if __name__ == "__main__":
    main()
