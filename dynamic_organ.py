"""
Incremental TSDF organ reconstruction.

Fuses per-frame depth + camera pose into a Truncated Signed Distance
Field volume, with three filters that meaningfully clean up the
output:

1. Per-frame **depth reliability mask** — drops near-edge / steep-
   gradient / out-of-range pixels before integration.
2. **Motion-gated integration** — skips frames where the camera has
   barely moved since the last integration, so a slow / hovering
   pass doesn't smear the same near-camera pixels into a center
   blob.
3. **Connected-component cleanup** on the extracted mesh — discards
   isolated outlier triangles (the noisy "tail" you sometimes get
   from a single bad-depth frame), keeping only the largest few
   components by triangle count.

Coordinate convention: world-to-camera matrix (OpenCV style) for
the extrinsic, depth in the same units as the extrinsic translation
(mm for C3VD / bronchoscopy).
"""

from typing import Optional
import numpy as np
import cv2
import open3d as o3d


def _reliability_mask(depth_mm, edge_margin_pct=4.0, grad_thresh=8.0):
    """Per-pixel boolean mask of *reliable* depth pixels.

    - Pixels within `edge_margin_pct` of any image border are dropped
      (lens vignette, distortion-edge errors).
    - Pixels with steep local depth gradient (|∇depth| > grad_thresh
      times the median depth) are dropped — those are usually depth
      discontinuities the rasterizer can't resolve, and they project
      to floating triangles after fusion.
    """
    H, W = depth_mm.shape
    mask = depth_mm > 0

    if edge_margin_pct > 0:
        m = max(1, int(min(H, W) * edge_margin_pct / 100.0))
        edge = np.zeros_like(mask)
        edge[m:H - m, m:W - m] = True
        mask &= edge

    if grad_thresh > 0 and mask.any():
        valid = depth_mm[mask]
        if valid.size > 0:
            # Sobel on a smoothed depth so we don't amplify noise into a
            # rejection signal.
            dsm = cv2.GaussianBlur(depth_mm, (5, 5), 0)
            gx = cv2.Sobel(dsm, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(dsm, cv2.CV_32F, 0, 1, ksize=3)
            gmag = np.sqrt(gx * gx + gy * gy)
            med = float(np.median(valid))
            mask &= gmag < (grad_thresh * max(med, 1e-3) * 0.1)

    return mask


class DynamicOrganBuilder:
    """Online TSDF fuser with the filters described in the module docstring."""

    def __init__(self, voxel_size=0.5, sdf_trunc=None, depth_trunc=120.0,
                 depth_min=0.5,
                 edge_margin_pct=4.0, grad_thresh=8.0,
                 min_camera_motion_mm=0.0,
                 mesh_cleanup_keep_top=3,
                 mesh_cleanup_min_triangles=200):
        self.voxel_size = float(voxel_size)
        self.sdf_trunc = float(sdf_trunc if sdf_trunc is not None
                               else 4.0 * voxel_size)
        self.depth_trunc = float(depth_trunc)
        self.depth_min = float(depth_min)
        self.edge_margin_pct = float(edge_margin_pct)
        self.grad_thresh = float(grad_thresh)
        self.min_camera_motion_mm = float(min_camera_motion_mm)
        self.mesh_cleanup_keep_top = int(mesh_cleanup_keep_top)
        self.mesh_cleanup_min_triangles = int(mesh_cleanup_min_triangles)
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        self.n_integrated = 0
        self.n_skipped_motion = 0
        self.n_skipped_empty = 0
        self._last_cam_center: Optional[np.ndarray] = None

    def integrate(self, rgb_u8, depth_f32, w2c_4x4, intrinsic):
        """Single-frame fusion. Skips if motion-gated or empty after filtering."""
        # Motion gate: skip if the camera barely moved.
        c2w = np.linalg.inv(w2c_4x4)
        cam_center = c2w[:3, 3]
        if (self.min_camera_motion_mm > 0.0
                and self._last_cam_center is not None):
            d = float(np.linalg.norm(cam_center - self._last_cam_center))
            if d < self.min_camera_motion_mm:
                self.n_skipped_motion += 1
                return

        depth = np.ascontiguousarray(depth_f32, dtype=np.float32).copy()
        depth[depth < self.depth_min] = 0.0
        depth[depth > self.depth_trunc] = 0.0
        depth[~_reliability_mask(depth,
                                 edge_margin_pct=self.edge_margin_pct,
                                 grad_thresh=self.grad_thresh)] = 0.0
        if not np.any(depth > 0):
            self.n_skipped_empty += 1
            return

        color = o3d.geometry.Image(np.ascontiguousarray(rgb_u8))
        depth_img = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth_img,
            depth_scale=1.0,
            depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=False,
        )
        self.volume.integrate(rgbd, intrinsic, w2c_4x4)
        self.n_integrated += 1
        self._last_cam_center = cam_center.copy()

    def _cleanup_mesh(self, mesh):
        """Drop tiny disconnected components from a marching-cubes mesh."""
        if (self.mesh_cleanup_keep_top <= 0
                or self.mesh_cleanup_min_triangles <= 0):
            return mesh
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels = np.asarray(labels)
        counts = np.asarray(counts)
        if labels.size == 0 or counts.size == 0:
            return mesh
        order = np.argsort(-counts)
        keep_labels = set()
        for li in order[:self.mesh_cleanup_keep_top]:
            if counts[li] >= self.mesh_cleanup_min_triangles:
                keep_labels.add(int(li))
        if not keep_labels:
            return mesh
        drop_mask = np.array([int(l) not in keep_labels for l in labels])
        out = o3d.geometry.TriangleMesh(mesh)
        out.remove_triangles_by_mask(drop_mask)
        out.remove_unreferenced_vertices()
        return out

    def extract_mesh(self, min_vertices=64):
        """Run marching cubes + connected-component cleanup. None until populated."""
        mesh = self.volume.extract_triangle_mesh()
        if len(mesh.vertices) < min_vertices:
            return None
        mesh = self._cleanup_mesh(mesh)
        if len(mesh.vertices) < min_vertices:
            return None
        mesh.compute_vertex_normals()
        return mesh
