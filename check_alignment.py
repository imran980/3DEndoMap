"""
Brute-force-find the coordinate convention that aligns our fused mesh
to C3VD's ground-truth phantom. Tries 16 sign/axis conventions
(identity + axis-flips + swaps), runs ICP against the GT mesh, and
reports the best-fitting transform.

Usage:
    python check_alignment.py \
        --ours output/colon_gt_trans/colon_mesh_gt.ply \
        --gt   dataset/trans_model.obj
"""

import numpy as np
import open3d as o3d
from argparse import ArgumentParser


def candidate_transforms():
    """16 reasonable conventions: combinations of axis flips (±1) on each axis."""
    mats = []
    names = []
    for sx in (+1, -1):
        for sy in (+1, -1):
            for sz in (+1, -1):
                T = np.diag([sx, sy, sz, 1]).astype(np.float64)
                mats.append(T)
                names.append(f"flip(x={sx},y={sy},z={sz})")
    # Y/Z swap (OpenGL <-> Unity is a Z-flip; some datasets also swap Y/Z)
    swap_yz = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]], dtype=np.float64)
    for base, nm in list(zip(mats[:], names[:])):
        mats.append(swap_yz @ base)
        names.append(nm + "+swap_yz")
    return list(zip(names, mats))


def main(ours_path, gt_path, voxel=1.0, max_corr=10.0):
    gt_mesh = o3d.io.read_triangle_mesh(gt_path)
    our_mesh = o3d.io.read_triangle_mesh(ours_path)

    if len(gt_mesh.triangles) > 0:
        gt_pc = gt_mesh.sample_points_uniformly(50000)
    else:
        gt_pc = o3d.geometry.PointCloud()
        gt_pc.points = gt_mesh.vertices
    our_pc_raw = our_mesh.sample_points_uniformly(50000)

    gt_center = np.asarray(gt_pc.points).mean(0)

    print(f"GT   bbox: {np.asarray(gt_mesh.vertices).min(0).round(2)} -> "
          f"{np.asarray(gt_mesh.vertices).max(0).round(2)}")
    print(f"OURS bbox: {np.asarray(our_mesh.vertices).min(0).round(2)} -> "
          f"{np.asarray(our_mesh.vertices).max(0).round(2)}")
    print()

    gt_down = gt_pc.voxel_down_sample(voxel)
    results = []
    for name, T in candidate_transforms():
        our_t = o3d.geometry.PointCloud(our_pc_raw)
        our_t.transform(T)
        # Center-align to GT first so ICP has a fighting chance
        delta = gt_center - np.asarray(our_t.points).mean(0)
        T_full = np.eye(4); T_full[:3, 3] = delta
        our_t.transform(T_full)
        our_down = our_t.voxel_down_sample(voxel)

        icp = o3d.pipelines.registration.registration_icp(
            our_down, gt_down, max_corr, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        results.append((icp.fitness, icp.inlier_rmse, name, T, T_full, icp.transformation))

    results.sort(key=lambda r: (-r[0], r[1]))
    print("Top 5 conventions (by ICP fitness):")
    for fit, rmse, name, _, _, _ in results[:5]:
        print(f"  fitness={fit:.3f}  rmse={rmse:.3f}  {name}")

    # Save the best-aligned mesh for visual check
    best = results[0]
    fit, rmse, name, T_axes, T_center, T_icp = best
    aligned = o3d.geometry.TriangleMesh(our_mesh)
    aligned.transform(T_axes)
    aligned.transform(T_center)
    aligned.transform(T_icp)
    aligned.paint_uniform_color([0.2, 0.8, 0.3])
    out_path = ours_path.replace(".ply", "_aligned_to_gt.ply")
    o3d.io.write_triangle_mesh(out_path, aligned)
    print(f"\nBest-aligned mesh saved to: {out_path}")
    print("Open it alongside the GT mesh in MeshLab to verify visually.")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--ours", required=True, help="Our fused mesh .ply")
    p.add_argument("--gt", required=True, help="GT organ mesh .obj/.ply")
    p.add_argument("--voxel", default=1.0, type=float)
    p.add_argument("--max_corr", default=10.0, type=float)
    args = p.parse_args()
    main(args.ours, args.gt, args.voxel, args.max_corr)
