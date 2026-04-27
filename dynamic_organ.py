"""
Incremental TSDF organ reconstruction.

Fuses per-frame (rendered) depth + camera pose into a Truncated Signed
Distance Field volume. At any point during navigation, an up-to-date
mesh of the *inspected* organ surface can be extracted via marching cubes.

Unlike a static pre-op organ model, the geometry grows as the camera
moves — the GPS map therefore only shows anatomy that has actually
been observed, which is the honest thing to display clinically.
"""

import numpy as np
import open3d as o3d


class DynamicOrganBuilder:
    """
    Thin wrapper around Open3D's ScalableTSDFVolume for online fusion.

    Coordinate convention: world-to-camera matrix (OpenCV style) for the
    extrinsic, depth in the same units as the extrinsic translation
    (mm for C3VD).
    """

    def __init__(self, voxel_size=0.5, sdf_trunc=None, depth_trunc=120.0,
                 depth_min=0.5):
        self.voxel_size = float(voxel_size)
        self.sdf_trunc = float(sdf_trunc if sdf_trunc is not None
                               else 4.0 * voxel_size)
        self.depth_trunc = float(depth_trunc)
        self.depth_min = float(depth_min)
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        self.n_integrated = 0

    def integrate(self, rgb_u8, depth_f32, w2c_4x4, intrinsic):
        """
        rgb_u8     : HxWx3 uint8 (RGB, not BGR)
        depth_f32  : HxW   float32 (mm in C3VD)
        w2c_4x4    : 4x4   float64 world-to-camera
        intrinsic  : o3d.camera.PinholeCameraIntrinsic
        """
        depth = np.ascontiguousarray(depth_f32, dtype=np.float32).copy()
        # Clean pixels outside trusted depth range (renderer returns 0 for
        # uncovered pixels and can emit tiny noise near the near plane).
        depth[depth < self.depth_min] = 0.0
        depth[depth > self.depth_trunc] = 0.0
        if not np.any(depth > 0):
            return  # nothing to fuse this frame

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

    def extract_mesh(self, min_vertices=64):
        """Run marching cubes. Returns None until the volume is populated."""
        mesh = self.volume.extract_triangle_mesh()
        if len(mesh.vertices) < min_vertices:
            return None
        mesh.compute_vertex_normals()
        return mesh
