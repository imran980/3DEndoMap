"""
Surgical Navigation GPS Visualization.

Displays the 3D organ model with the camera trajectory showing
where the endoscope traveled inside the organ.

The visualization shows:
  - The organ surface (from trans_model.obj) as a semi-transparent wireframe
  - The registered camera trajectory as a colored path (green → red)
  - Start/end markers and camera frustums
  - The registered endoscopic reconstruction overlaid

Usage:
    python visualize_navigation.py \\
        --model_path output/endonerf/c3vd_trans --iteration 3000
"""

import os
import sys
import json
import copy
import numpy as np
import open3d as o3d
from argparse import ArgumentParser


def create_camera_frustum(position, forward, up=None, size=2.0, color=[0.2, 0.6, 1.0]):
    """Create a small camera frustum wireframe at the given position."""
    if up is None:
        # Compute an up vector perpendicular to forward
        forward = np.array(forward, dtype=np.float64)
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            forward = np.array([0, 0, -1.0])
        else:
            forward = forward / norm
        
        # Choose an arbitrary up
        if abs(forward[1]) < 0.9:
            up = np.array([0, 1, 0], dtype=np.float64)
        else:
            up = np.array([1, 0, 0], dtype=np.float64)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
    
    position = np.array(position, dtype=np.float64)
    right = np.cross(np.array(forward), np.array(up))
    if np.linalg.norm(right) > 1e-6:
        right = right / np.linalg.norm(right)
    
    # Frustum corners (small pyramid)
    s = size
    corners = [
        position + forward * s * 2 + right * s + up * s,    # top-right
        position + forward * s * 2 - right * s + up * s,    # top-left
        position + forward * s * 2 - right * s - up * s,    # bottom-left
        position + forward * s * 2 + right * s - up * s,    # bottom-right
    ]
    
    # Lines from camera to corners + rectangle
    points = [position] + corners
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # camera to corners
        [1, 2], [2, 3], [3, 4], [4, 1],   # rectangle
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    line_set.colors = o3d.utility.Vector3dVector(np.array([color] * len(lines)))
    
    return line_set


def create_sphere(center, radius, color, resolution=10):
    """Create a colored sphere mesh."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    return sphere


def create_trajectory_line(positions, colors=None):
    """Create a line set showing the camera path."""
    n = len(positions)
    if n < 2:
        return None
    
    points = np.array(positions)
    lines = [[i, i + 1] for i in range(n - 1)]
    
    if colors is None:
        # Green → Red gradient
        colors = []
        for i in range(n - 1):
            t = i / max(1, n - 2)
            colors.append([t, 1.0 - t, 0.0])
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    line_set.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    return line_set


def visualize_gps(model_path, iteration):
    """Main GPS visualization."""
    recon_dir = os.path.join(model_path, "surface_reconstruction",
                             f"iteration_{iteration}")
    
    print("=" * 60)
    print("Surgical Navigation GPS Visualization")
    print("=" * 60)
    
    # ---- Load data ----
    # Organ model
    organ_candidates = [
        os.path.join(recon_dir, "organ_model.ply"),
        "dataset/trans_model.obj",
    ]
    organ_path = None
    for p in organ_candidates:
        if os.path.exists(p):
            organ_path = p
            break
    
    # Registered trajectory
    traj_path = os.path.join(recon_dir, "registered_trajectory.json")
    if not os.path.exists(traj_path):
        traj_path = os.path.join(recon_dir, "camera_trajectory.json")
        print("  Using unregistered trajectory (no registered_trajectory.json)")
    
    # Registered endo point cloud
    endo_pcd_path = os.path.join(recon_dir, "registered_endo_pcd.ply")
    if not os.path.exists(endo_pcd_path):
        endo_pcd_path = os.path.join(recon_dir, "fused_pointcloud.ply")
    
    geometries = []
    
    # 1. Load organ model
    if organ_path:
        organ = o3d.io.read_triangle_mesh(organ_path)
        organ.compute_vertex_normals()
        
        # Make semi-transparent pinkish color
        if len(organ.vertex_colors) == 0:
            organ.paint_uniform_color([0.90, 0.60, 0.55])
        
        # Create wireframe version for see-through effect
        organ_wire = o3d.geometry.LineSet.create_from_triangle_mesh(organ)
        # Color the wireframe
        n_lines = len(organ_wire.lines)
        wire_colors = np.ones((n_lines, 3)) * np.array([0.85, 0.55, 0.50])
        organ_wire.colors = o3d.utility.Vector3dVector(wire_colors)
        
        geometries.append(organ_wire)
        print(f"✓ Organ: {len(organ.vertices):,} vertices (wireframe)")
        
        organ_extent = np.asarray(organ.vertices).max(0) - np.asarray(organ.vertices).min(0)
        vis_scale = float(np.median(organ_extent)) / 40.0  # for sizing markers
    else:
        print("✗ Organ model not found")
        vis_scale = 1.0
    
    # 2. Load endo reconstruction
    if os.path.exists(endo_pcd_path):
        endo_pcd = o3d.io.read_point_cloud(endo_pcd_path)
        if len(endo_pcd.points) > 0:
            # Downsample for visualization
            ds_size = vis_scale * 0.3
            endo_vis = endo_pcd.voxel_down_sample(ds_size)
            if not endo_vis.has_colors():
                endo_vis.paint_uniform_color([0.3, 0.8, 0.3])
            geometries.append(endo_vis)
            print(f"✓ Endo reconstruction: {len(endo_vis.points):,} points (downsampled)")
    
    # 3. Load camera trajectory
    if os.path.exists(traj_path):
        with open(traj_path) as f:
            traj = json.load(f)
        
        # Use organ-space positions if available
        positions = []
        forwards = []
        for frame in traj['frames']:
            if 'position_organ' in frame:
                positions.append(frame['position_organ'])
            else:
                positions.append(frame['position'])
            if 'forward_organ' in frame:
                forwards.append(frame['forward_organ'])
            elif 'forward' in frame:
                forwards.append(frame['forward'])
            else:
                forwards.append([0, 0, -1])
        
        positions = np.array(positions)
        forwards = np.array(forwards)
        print(f"✓ Trajectory: {len(positions)} frames")
        print(f"  Position range: [{positions.min(0).round(2)}] → [{positions.max(0).round(2)}]")
        
        # Camera path line
        path_line = create_trajectory_line(positions)
        if path_line is not None:
            geometries.append(path_line)
        
        # Start and end markers
        start_sphere = create_sphere(positions[0], vis_scale * 0.8, [0.0, 0.9, 0.0])
        end_sphere = create_sphere(positions[-1], vis_scale * 0.8, [0.9, 0.0, 0.0])
        geometries.append(start_sphere)
        geometries.append(end_sphere)
        
        # Camera frustums (every N frames)
        n_frustums = min(20, len(positions))
        step = max(1, len(positions) // n_frustums)
        for i in range(0, len(positions), step):
            t = i / max(1, len(positions) - 1)
            color = [t, 1.0 - t, 0.3]
            frustum = create_camera_frustum(
                positions[i], forwards[i], size=vis_scale * 0.5, color=color
            )
            geometries.append(frustum)
        print(f"✓ {len(range(0, len(positions), step))} camera frustums")
        
        # Progress spheres along trajectory (every 10%)
        for pct in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            idx = int(pct * (len(positions) - 1))
            t = pct
            color = [t, 1.0 - t, 0.0]
            dot = create_sphere(positions[idx], vis_scale * 0.4, color)
            geometries.append(dot)
    else:
        print(f"✗ Trajectory not found: {traj_path}")
    
    # ---- Save combined scene as PLY ----
    # Merge all point-like geometries
    combined_pcd = o3d.geometry.PointCloud()
    
    for geo in geometries:
        if isinstance(geo, o3d.geometry.PointCloud):
            combined_pcd += geo
        elif isinstance(geo, o3d.geometry.TriangleMesh):
            # Sample points from mesh for PLY export
            if len(geo.vertices) > 0:
                pts = np.asarray(geo.vertices)
                colors = np.asarray(geo.vertex_colors) if len(geo.vertex_colors) > 0 else np.ones((len(pts), 3)) * 0.5
                tmp = o3d.geometry.PointCloud()
                tmp.points = o3d.utility.Vector3dVector(pts)
                tmp.colors = o3d.utility.Vector3dVector(colors)
                combined_pcd += tmp
        elif isinstance(geo, o3d.geometry.LineSet):
            # Sample points along lines for PLY export
            pts = np.asarray(geo.points)
            if len(pts) > 0:
                tmp = o3d.geometry.PointCloud()
                tmp.points = o3d.utility.Vector3dVector(pts)
                if len(geo.colors) > 0:
                    # Use first color for all line points
                    c = np.asarray(geo.colors)
                    pt_colors = np.ones((len(pts), 3)) * 0.7
                    tmp.colors = o3d.utility.Vector3dVector(pt_colors)
                combined_pcd += tmp
    
    scene_path = os.path.join(recon_dir, "navigation_scene.ply")
    o3d.io.write_point_cloud(scene_path, combined_pcd)
    print(f"\nCombined scene saved to: {scene_path}")
    
    # ---- Launch viewer ----
    print("\n" + "=" * 60)
    print("Launching 3D viewer...")
    print("=" * 60)
    print("Controls:")
    print("  Mouse left:   Rotate")
    print("  Mouse wheel:  Zoom")
    print("  Mouse right:  Pan")
    print("  Q/Esc:        Quit")
    print()
    print("  Green sphere = START position")
    print("  Red sphere   = END position")
    print("  Path color   = green→red (time progression)")
    print("  Wireframe    = organ surface")
    
    try:
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Surgical Navigation GPS",
            width=1280,
            height=720,
        )
    except Exception as e:
        print(f"\nViewer failed (likely no display): {e}")
        print(f"Open the scene in MeshLab: {scene_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Surgical Navigation GPS Visualization")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", default=3000, type=int)
    args = parser.parse_args()
    
    visualize_gps(args.model_path, args.iteration)
