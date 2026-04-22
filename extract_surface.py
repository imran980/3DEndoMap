"""
Extract the 3D surface and camera trajectory from a trained Endo-4DGS model.

Instead of TSDF fusion (which fails due to non-orthogonal LLFF camera matrices),
this script:
  1. Loads the trained Gaussian point cloud directly
  2. Extracts camera positions from view.camera_center
  3. Optionally creates a Poisson mesh from the point cloud
  4. Saves the camera trajectory for downstream GPS visualization

Usage:
    python extract_surface.py --model_path output/endonerf/c3vd_trans \
        --configs arguments/endonerf.py --iteration 3000
"""

import os
import sys
import json
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from argparse import ArgumentParser
from plyfile import PlyData

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from scene import Scene, GaussianModel
from utils.general_utils import safe_state


def load_gaussian_pointcloud(ply_path):
    """
    Load the trained Gaussian point cloud from a PLY file.
    Returns an Open3D point cloud with positions and colors.
    """
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    xyz = np.stack([
        np.asarray(vertex['x']),
        np.asarray(vertex['y']),
        np.asarray(vertex['z']),
    ], axis=-1)
    
    # Extract colors from SH DC component (f_dc_0, f_dc_1, f_dc_2)
    # SH DC to RGB: color = SH_C0 * f_dc + 0.5, where SH_C0 = 0.28209479
    SH_C0 = 0.28209479177387814
    try:
        r = np.asarray(vertex['f_dc_0']) * SH_C0 + 0.5
        g = np.asarray(vertex['f_dc_1']) * SH_C0 + 0.5
        b = np.asarray(vertex['f_dc_2']) * SH_C0 + 0.5
        colors = np.stack([r, g, b], axis=-1).clip(0, 1)
    except (ValueError, KeyError):
        colors = np.ones((len(xyz), 3)) * 0.5
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    
    return pcd


def extract_surface(dataset, hyperparam, iteration, pipeline):
    """
    Main extraction pipeline.
    """
    print("=" * 60)
    print("STEP 1: Loading trained model")
    print("=" * 60)
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
    out_dir = os.path.join(dataset.model_path, "surface_reconstruction",
                           "iteration_{}".format(scene.loaded_iter))
    os.makedirs(out_dir, exist_ok=True)
    
    views = scene.getTrainCameras()
    print(f"Loaded {len(views)} training views")
    
    # ---- Step 2: Load Gaussian point cloud ----
    print("\n" + "=" * 60)
    print("STEP 2: Loading Gaussian point cloud")
    print("=" * 60)
    
    ply_path = os.path.join(dataset.model_path, "point_cloud",
                            f"iteration_{scene.loaded_iter}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"ERROR: Gaussian point cloud not found at {ply_path}")
        return
    
    pcd = load_gaussian_pointcloud(ply_path)
    pts = np.asarray(pcd.points)
    
    print(f"Gaussian point cloud: {len(pts):,} points")
    print(f"  X range: [{pts[:, 0].min():.3f}, {pts[:, 0].max():.3f}]")
    print(f"  Y range: [{pts[:, 1].min():.3f}, {pts[:, 1].max():.3f}]")
    print(f"  Z range: [{pts[:, 2].min():.3f}, {pts[:, 2].max():.3f}]")
    
    # ---- Step 3: Extract camera trajectory ----
    print("\n" + "=" * 60)
    print("STEP 3: Extracting camera trajectory")
    print("=" * 60)
    
    cam_positions = []
    cam_forwards = []
    cam_times = []
    
    with torch.no_grad():
        for idx, view in enumerate(views):
            # camera_center is correctly computed from world_view_transform
            pos = view.camera_center.cpu().numpy()
            cam_positions.append(pos)
            cam_times.append(getattr(view, 'time', idx / len(views)))
            
            # Forward direction: third column of c2w rotation
            # world_view_transform is w2c^T, so its inverse is c2w^T
            # c2w^T row 2 = c2w column 2 = forward direction
            wvt = view.world_view_transform.cpu().numpy()  # w2c^T (4x4)
            c2w_T = np.linalg.inv(wvt)  # c2w^T
            forward = c2w_T[2, :3]  # row 2 of c2w^T = column 2 of c2w
            cam_forwards.append(forward)
    
    cam_positions = np.array(cam_positions)
    cam_forwards = np.array(cam_forwards)
    
    print(f"Camera positions: {len(cam_positions)} frames")
    print(f"  X range: [{cam_positions[:, 0].min():.4f}, {cam_positions[:, 0].max():.4f}]")
    print(f"  Y range: [{cam_positions[:, 1].min():.4f}, {cam_positions[:, 1].max():.4f}]")
    print(f"  Z range: [{cam_positions[:, 2].min():.4f}, {cam_positions[:, 2].max():.4f}]")
    
    # Check if camera moves enough
    pos_range = cam_positions.max(axis=0) - cam_positions.min(axis=0)
    print(f"  Total movement: dx={pos_range[0]:.4f}, dy={pos_range[1]:.4f}, dz={pos_range[2]:.4f}")
    
    # ---- Step 4: Clean and optionally reconstruct mesh ----
    print("\n" + "=" * 60)
    print("STEP 4: Processing point cloud")
    print("=" * 60)
    
    # Remove statistical outliers
    n_before = len(pcd.points)
    pcd_clean, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"Outlier removal: {n_before:,} → {len(pcd_clean.points):,} points")
    
    # Downsample for manageable size
    extent = pts.max(axis=0) - pts.min(axis=0)
    voxel_size = float(np.median(extent)) / 300.0
    pcd_down = pcd_clean.voxel_down_sample(voxel_size)
    print(f"Downsampled (voxel={voxel_size:.4f}): {len(pcd_down.points):,} points")
    
    # Try Poisson surface reconstruction
    mesh = None
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4, max_nn=30))
    
    if pcd_down.has_normals() and len(pcd_down.points) > 100:
        pcd_down.orient_normals_consistent_tangent_plane(k=15)
        
        try:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd_down, depth=9)
            densities = np.asarray(densities)
            # Remove low-density vertices (artifacts)
            density_threshold = np.quantile(densities, 0.1)
            vertices_to_remove = densities < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)
            mesh.compute_vertex_normals()
            print(f"Poisson mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
        except Exception as e:
            print(f"Poisson reconstruction failed: {e}")
            mesh = None
    
    if mesh is None:
        # Fallback: save as vertex-only mesh
        print("Saving as vertex-only mesh (no triangles)")
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = pcd_clean.points
        if pcd_clean.has_colors():
            mesh.vertex_colors = pcd_clean.colors
    
    # ---- Step 5: Save outputs ----
    print("\n" + "=" * 60)
    print("STEP 5: Saving outputs")
    print("=" * 60)
    
    # Save mesh
    mesh_path = os.path.join(out_dir, "fused_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"Mesh: {mesh_path} ({len(mesh.vertices):,} vertices)")
    
    # Save point cloud
    pcd_path = os.path.join(out_dir, "fused_pointcloud.ply")
    o3d.io.write_point_cloud(pcd_path, pcd_clean)
    print(f"Point cloud: {pcd_path} ({len(pcd_clean.points):,} points)")
    
    # Save camera trajectory
    view0 = views[0]
    traj_data = {
        "metadata": {
            "model_path": dataset.model_path,
            "iteration": scene.loaded_iter,
            "n_frames": len(views),
            "gaussian_points": len(pcd.points),
        },
        "intrinsics": {
            "fx": float(view0.image_width / (2.0 * np.tan(view0.FoVx / 2.0))),
            "fy": float(view0.image_height / (2.0 * np.tan(view0.FoVy / 2.0))),
            "cx": float(view0.image_width / 2.0),
            "cy": float(view0.image_height / 2.0),
            "width": int(view0.image_width),
            "height": int(view0.image_height),
        },
        "frames": []
    }
    
    for i in range(len(views)):
        frame = {
            "frame_id": i,
            "time": float(cam_times[i]),
            "position": cam_positions[i].tolist(),
            "forward": cam_forwards[i].tolist(),
        }
        traj_data["frames"].append(frame)
    
    traj_path = os.path.join(out_dir, "camera_trajectory.json")
    with open(traj_path, 'w') as f:
        json.dump(traj_data, f, indent=2)
    print(f"Trajectory: {traj_path} ({len(views)} frames)")
    
    # Summary
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"  Mesh:         {mesh_path}")
    print(f"  Point cloud:  {pcd_path}")
    print(f"  Trajectory:   {traj_path}")
    print(f"\nNext step: Register to organ model:")
    print(f"  python register_to_organ.py --model_path {dataset.model_path} \\")
    print(f"      --iteration {scene.loaded_iter} --organ_mesh dataset/trans_model.obj")


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract surface from trained Endo-4DGS model")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=3000, type=int)
    parser.add_argument("--configs", type=str, default="")
    
    args = get_combined_args(parser)
    
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    safe_state(False)
    
    extract_surface(
        model.extract(args),
        hyperparam.extract(args),
        args.iteration,
        pipeline.extract(args),
    )
