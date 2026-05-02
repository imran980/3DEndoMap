"""
Dashboard rendering helpers shared by all `render_navigation*` scripts.

Pulled out of render_navigation.py so the C3VD-based pipeline
(render_navigation_c3vd.py) can use the same depth visualization,
frustum coverage, GPS-frame layout and HUD without dragging in any
Endo-4DGS dependencies.
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ---------- depth ----------

def depth_to_colormap(depth_np, valid_mask=None, vmin=None, vmax=None,
                      colormap=cv2.COLORMAP_TURBO):
    """Per-frame percentile-normalized depth visualization.

    Returns (colored_bgr, vmin, vmax). near = warm (red), far = cool (blue).
    """
    if valid_mask is None:
        valid_mask = depth_np > 0
    valid = depth_np[valid_mask]
    if valid.size < 16:
        return np.zeros((*depth_np.shape, 3), dtype=np.uint8), 0.0, 0.0
    if vmin is None:
        vmin = float(np.percentile(valid, 5))
    if vmax is None:
        vmax = float(np.percentile(valid, 95))
    if vmax - vmin < 1e-4:
        vmax = vmin + 1e-4
    depth_norm = np.clip((depth_np - vmin) / (vmax - vmin), 0, 1)
    depth_u8 = ((1.0 - depth_norm) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_u8, colormap)
    colored[~valid_mask] = 0
    return colored, vmin, vmax


# ---------- coverage ----------

def frustum_visibility(cam_center, cam_forward, organ_pts,
                       hfov_deg=140.0, near=1.0, far=120.0):
    """Boolean array (N,) marking organ points inside the camera's FOV cone."""
    d = organ_pts - cam_center[None, :]
    dist = np.linalg.norm(d, axis=1)
    safe = dist > 1e-6
    d_norm = np.zeros_like(d)
    d_norm[safe] = d[safe] / dist[safe, None]
    cos_sim = d_norm @ cam_forward
    cos_half = float(np.cos(np.radians(hfov_deg) / 2.0))
    return safe & (cos_sim > cos_half) & (dist >= near) & (dist <= far)


# ---------- formatting ----------

def format_time_mmss(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def compute_organ_centerline(organ_mesh, n_points=200):
    """PCA-based centerline through an organ point cloud / mesh."""
    pts = np.asarray(organ_mesh.vertices)
    center = pts.mean(axis=0)
    centered = pts - center
    cov = np.cov(centered.T)
    _, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]
    projections = centered @ principal
    pmin, pmax = projections.min(), projections.max()
    t_values = np.linspace(pmin, pmax, n_points)
    bin_width = (pmax - pmin) / n_points * 2
    centerline = []
    for t in t_values:
        mask = np.abs(projections - t) < bin_width
        if mask.sum() > 0:
            centerline.append(pts[mask].mean(axis=0))
        else:
            centerline.append(center + principal * t)
    return np.array(centerline)


# ---------- GPS panel ----------

def render_gps_frame(gps_data, current_idx, coverage_counts=None,
                     reveal_mode=False, cam_pos=None, cam_forward=None,
                     cam_trajectory=None,
                     width=640, height=480, dpi=100):
    """3D matplotlib GPS view — see render_navigation.py for full docs."""
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    organ_pts = gps_data['organ_pts']
    centerline = gps_data['centerline']
    center = gps_data['center']
    extent = gps_data['extent']
    n_frames = gps_data['n_frames']

    n_organ = len(organ_pts)
    target_pts = 6000
    sub_idx = gps_data.get('sub_idx')
    if sub_idx is None or len(sub_idx) > n_organ:
        if n_organ > target_pts:
            sub_idx = np.random.RandomState(42).choice(
                n_organ, target_pts, replace=False)
        else:
            sub_idx = np.arange(n_organ)
        gps_data['sub_idx'] = sub_idx
    organ_sub = organ_pts[sub_idx] if n_organ > 0 else organ_pts

    if n_organ > 0 and coverage_counts is not None:
        cov = coverage_counts[sub_idx].astype(np.float32)
        if reveal_mode:
            seen = cov > 0
            if seen.any():
                pts = organ_sub[seen]
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                           c='salmon', alpha=0.55, s=5, depthshade=True)
            unseen = ~seen
            if unseen.any() and gps_data.get('reveal_show_ghost', True):
                pts = organ_sub[unseen]
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                           c=(0.4, 0.4, 0.45), alpha=0.04, s=2,
                           depthshade=True)
        else:
            scale = max(float(np.percentile(cov[cov > 0], 75))
                        if (cov > 0).any() else 1.0, 1.0)
            t = np.clip(cov / scale, 0.0, 1.0)
            colors = np.stack([
                np.clip(1.0 - t, 0, 1),
                np.clip(t, 0, 1),
                np.zeros_like(t),
            ], axis=1)
            ax.scatter(organ_sub[:, 0], organ_sub[:, 1], organ_sub[:, 2],
                       c=colors, alpha=0.25, s=4, depthshade=True)
            miss = cov == 0
            if miss.any():
                ax.scatter(organ_sub[miss, 0], organ_sub[miss, 1],
                           organ_sub[miss, 2], c='red', alpha=0.35, s=5,
                           depthshade=True)
    elif n_organ > 0:
        ax.scatter(organ_sub[:, 0], organ_sub[:, 1], organ_sub[:, 2],
                   c='salmon', alpha=0.09, s=3, depthshade=True)

    n_cl = len(centerline)
    cur_pos = (np.asarray(cam_pos, dtype=np.float64)
               if cam_pos is not None else None)
    if cur_pos is not None and n_cl > 1:
        d2 = np.sum((centerline - cur_pos[None, :]) ** 2, axis=1)
        nearest_cl_idx = int(np.argmin(d2))
    elif n_cl > 1:
        nearest_cl_idx = min(int(current_idx / max(1, n_frames - 1)
                                 * (n_cl - 1)), n_cl - 1)
        cur_pos = centerline[nearest_cl_idx]
    else:
        nearest_cl_idx = 0
        cur_pos = cur_pos if cur_pos is not None else np.zeros(3)

    if n_cl > 1:
        segments = []
        colors = []
        for i in range(n_cl - 1):
            segments.append([centerline[i], centerline[i + 1]])
            if i <= nearest_cl_idx:
                colors.append([0.2, 0.9, 0.2, 0.6])
            else:
                colors.append([0.5, 0.5, 0.5, 0.2])
        ax.add_collection3d(Line3DCollection(segments, colors=colors,
                                             linewidths=2.0))

    # Actual camera trajectory up to the current frame, drawn as a
    # cyan polyline on top of the centerline. This is the observed
    # path (whether from GT poses, SLAM, or VO), distinct from the
    # PCA-derived centerline of the organ.
    if cam_trajectory is not None and len(cam_trajectory) > 1:
        traj = np.asarray(cam_trajectory)
        upto = min(int(current_idx) + 1, len(traj))
        if upto >= 2:
            traj_segments = [
                [traj[i], traj[i + 1]] for i in range(upto - 1)
            ]
            ax.add_collection3d(Line3DCollection(
                traj_segments,
                colors=[(0.30, 0.85, 0.95, 0.9)] * len(traj_segments),
                linewidths=1.6))
            # Faint dot at the trajectory start so it's always visible
            ax.scatter(*traj[0], c=(0.30, 0.85, 0.95), s=80, zorder=8,
                       depthshade=False)

    ax.scatter(*cur_pos, c='lime', s=360, zorder=10, edgecolors='white',
               linewidths=2.5, depthshade=False)
    if n_cl > 1:
        ax.scatter(*centerline[0], c='green', s=160, marker='^', zorder=9,
                   depthshade=False)
        ax.scatter(*centerline[-1], c='red', s=160, marker='s', zorder=9,
                   depthshade=False)

    ax.view_init(elev=gps_data['elev'], azim=gps_data['azim'])
    max_range = float(extent.max()) * 0.6
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('none')
    ax.set_axis_off()

    pct = (nearest_cl_idx / max(1, n_cl - 1)) * 100.0 if n_cl > 1 else 0.0
    ax.set_title(f"GPS  •  Frame {current_idx}/{n_frames-1}  •  "
                 f"{pct:.0f}% along organ",
                 color='#58a6ff', fontsize=16, fontweight='bold', pad=6)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.94, bottom=0.0)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8) \
        .reshape(h, w, 3)
    plt.close(fig)
    return buf


# ---------- HUD ----------

def draw_hud(panel, *, gps_w, elapsed_s, speed_mms, dist_mm, mode,
             cov_pct=0.0, n_mesh_verts=0, n_fused=0):
    """Composite the per-frame HUD strip onto an already-rendered GPS panel."""
    hud_h = 74
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (gps_w, hud_h), (12, 12, 18), -1)
    cv2.addWeighted(overlay, 0.72, panel, 0.28, 0, panel)

    good_speed = 1.0 <= speed_mms <= 6.0
    speed_color = (80, 230, 120) if good_speed else (60, 120, 240)
    time_color = (80, 230, 120) if elapsed_s >= 360 else (220, 220, 220)
    cv2.putText(panel, f"Withdrawal  {format_time_mmss(elapsed_s)}",
                (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78, time_color, 2)
    cv2.putText(panel, "(target >= 06:00)",
                (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (140, 140, 150), 1)
    cv2.putText(panel, f"Speed  {speed_mms:5.1f} mm/s",
                (int(gps_w * 0.32), 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, speed_color, 2)
    cv2.putText(panel, f"Path  {dist_mm:6.1f} mm",
                (int(gps_w * 0.32), 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 210), 1)

    if mode == 'dynamic':
        c = (80, 230, 120) if n_mesh_verts > 20000 else \
            ((60, 200, 230) if n_mesh_verts > 5000 else (60, 120, 240))
        cv2.putText(panel, f"Mesh  {n_mesh_verts:,} pts",
                    (int(gps_w * 0.62), 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, c, 2)
        cv2.putText(panel, f"(live TSDF, {n_fused} frames fused)",
                    (int(gps_w * 0.62), 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 170), 1)
    else:
        label = "Revealed" if mode == 'reveal' else "Coverage"
        c = (80, 230, 120) if cov_pct >= 80 else \
            ((60, 200, 230) if cov_pct >= 50 else (60, 120, 240))
        cv2.putText(panel, f"{label}  {cov_pct:5.1f}%",
                    (int(gps_w * 0.62), 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, c, 2)
        bar_x1, bar_x2 = int(gps_w * 0.62), int(gps_w * 0.95)
        bar_y = 52
        cv2.rectangle(panel, (bar_x1, bar_y),
                      (bar_x2, bar_y + 10), (50, 50, 60), -1)
        fill_x = int(bar_x1 + (bar_x2 - bar_x1) * cov_pct / 100.0)
        cv2.rectangle(panel, (bar_x1, bar_y),
                      (fill_x, bar_y + 10), c, -1)
