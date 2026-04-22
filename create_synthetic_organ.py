"""
Create a synthetic organ model that encloses the endoscopic reconstruction.

This simulates a pre-operative CT/MRI-derived organ surface for registration demo.
The synthetic organ is a larger ellipsoidal/tubular cavity surrounding the endo mesh.

Usage:
    python create_synthetic_organ.py --model_path output/endonerf/pulling
"""

import os
import json
import numpy as np
import open3d as o3d
from argparse import ArgumentParser


def create_organ_mesh(endo_mesh_path, traj_path, output_dir, scale_factor=1.5, wall_thickness=5.0):
    """
    Create a synthetic organ mesh that encloses the endoscopic reconstruction.
    
    Strategy:
    1. Load the endo mesh and camera trajectory
    2. Compute the bounding ellipsoid of the reconstruction
    3. Create a larger tubular mesh around it (simulating organ wall)
    4. Add some surface variation to look organic
    """
    print("=" * 60)
    print("Creating synthetic organ model")
    print("=" * 60)
    
    # Load endoscopic mesh
    endo_mesh = o3d.io.read_triangle_mesh(endo_mesh_path)
    endo_pts = np.asarray(endo_mesh.vertices)
    
    # Load camera trajectory for path-based organ shape
    with open(traj_path) as f:
        traj = json.load(f)
    cam_positions = np.array([f['position'] for f in traj['frames']])
    
    print(f"Endo mesh: {len(endo_pts)} vertices")
    print(f"Camera positions: {len(cam_positions)} frames")
    
    # Compute center and extent of the reconstruction
    center = endo_pts.mean(axis=0)
    extent = endo_pts.max(axis=0) - endo_pts.min(axis=0)
    
    print(f"Center: {center}")
    print(f"Extent: {extent}")
    
    # Create an ellipsoid that's larger than the reconstruction
    # This simulates the organ cavity the endoscope is inside
    radii = extent * scale_factor / 2.0
    
    # Create a UV sphere mesh and scale it to an ellipsoid
    organ = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=40)
    
    # Scale to ellipsoid
    vertices = np.asarray(organ.vertices)
    vertices[:, 0] *= radii[0]
    vertices[:, 1] *= radii[1]
    vertices[:, 2] *= radii[2]
    
    # Center on the reconstruction
    vertices += center
    
    # Add organic-looking surface perturbation (bumps and folds)
    np.random.seed(42)
    n_verts = len(vertices)
    
    # Low-frequency noise for large folds
    for freq in [0.02, 0.05, 0.1]:
        noise_amp = np.mean(radii) * 0.03 / freq
        for dim in range(3):
            phase = np.random.rand() * 2 * np.pi
            noise = noise_amp * np.sin(freq * vertices[:, dim] + phase)
            # Apply noise along vertex normals (radial direction)
            direction = vertices - center
            direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
            direction_norm[direction_norm == 0] = 1
            direction /= direction_norm
            vertices += direction * noise[:, np.newaxis] * 0.3
    
    organ.vertices = o3d.utility.Vector3dVector(vertices)
    organ.compute_vertex_normals()
    
    # Flip normals to point inward (we're looking at the inside of the organ)
    organ.triangle_normals = o3d.utility.Vector3dVector(
        -np.asarray(organ.triangle_normals))
    organ.vertex_normals = o3d.utility.Vector3dVector(
        -np.asarray(organ.vertex_normals))
    
    # Color the organ with a pinkish/reddish tissue color
    n_vertices = len(organ.vertices)
    base_color = np.array([0.85, 0.55, 0.50])  # pinkish tissue
    color_variation = np.random.rand(n_vertices, 3) * 0.08
    colors = np.clip(base_color + color_variation - 0.04, 0, 1)
    organ.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    organ_path = os.path.join(output_dir, "synthetic_organ.ply")
    o3d.io.write_triangle_mesh(organ_path, organ)
    
    print(f"\nOrgan mesh: {len(organ.vertices)} vertices, {len(organ.triangles)} triangles")
    print(f"Saved to: {organ_path}")
    
    # Also save metadata
    meta = {
        "endo_mesh_center": center.tolist(),
        "endo_mesh_extent": extent.tolist(),
        "organ_radii": radii.tolist(),
        "scale_factor": scale_factor,
    }
    meta_path = os.path.join(output_dir, "organ_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    return organ, organ_path


if __name__ == "__main__":
    parser = ArgumentParser(description="Create synthetic organ model")
    parser.add_argument("--model_path", required=True, help="Path to trained model output")
    parser.add_argument("--iteration", default=3000, type=int, help="Iteration to use")
    parser.add_argument("--scale", default=1.5, type=float, help="Scale factor for organ size relative to endo mesh")
    args = parser.parse_args()
    
    recon_dir = os.path.join(args.model_path, "surface_reconstruction", f"iteration_{args.iteration}")
    endo_mesh = os.path.join(recon_dir, "fused_mesh.ply")
    traj = os.path.join(recon_dir, "camera_trajectory.json")
    
    if not os.path.exists(endo_mesh):
        print(f"Error: {endo_mesh} not found. Run extract_surface.py first.")
        exit(1)
    
    create_organ_mesh(endo_mesh, traj, recon_dir, scale_factor=args.scale)
