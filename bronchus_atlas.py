"""
Procedural bronchial-tree atlas generator.

Produces an anatomically-credible bronchial tree mesh + centerline
without needing any patient-specific CT. Use as the GPS canvas when
no patient CT is available — the camera position then snaps onto the
atlas via the existing trajectory ICP, giving the surgeon
**approximate, atlas-based** scope localization (not metric tracking).

Anatomy modeled (5 generations, 23 segments, mm units):
    Trachea (1)
      └ Main bronchi: R + L (2)
          └ Lobar: R-upper, R-middle, R-lower; L-upper, L-lower (5)
              └ Segmental: ~10 (typical adult: 10R + 8L = 18 segmental)
                  └ Subsegmental: a few representative ones for shape

Asymmetric branching pattern + realistic angles / diameters / lengths
based on Weibel's bronchial dimensions (mean adult). Numbers are
rounded; this is a reference shape, NOT a patient-specific model.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import open3d as o3d


@dataclass
class AirwayBranch:
    name: str
    start: np.ndarray            # 3-vector, mm
    end: np.ndarray              # 3-vector, mm
    radius_start: float          # mm
    radius_end: float            # mm
    generation: int
    parent: Optional["AirwayBranch"] = None
    children: List["AirwayBranch"] = field(default_factory=list)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> np.ndarray:
        d = self.end - self.start
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


# Coordinate convention: +z = caudal (down), patient supine.
# Trachea descends along -z from the larynx; main bronchi split
# laterally (±x = right/left) and inferiorly (-z).

def _rotate_around(axis, angle_deg, vec):
    """Rotate `vec` by `angle_deg` around unit `axis`. Right-handed."""
    a = np.radians(angle_deg)
    k = axis / max(np.linalg.norm(axis), 1e-9)
    cos, sin = np.cos(a), np.sin(a)
    return (vec * cos
            + np.cross(k, vec) * sin
            + k * np.dot(k, vec) * (1 - cos))


def _add_child(parent: AirwayBranch, name: str,
               turn_axis: np.ndarray, turn_deg: float,
               length: float, radius_end: float) -> AirwayBranch:
    """Append a child branch turning `turn_deg` from the parent's direction."""
    new_dir = _rotate_around(turn_axis, turn_deg, parent.direction)
    end = parent.end + new_dir * length
    child = AirwayBranch(
        name=name,
        start=parent.end.copy(),
        end=end,
        radius_start=parent.radius_end,
        radius_end=radius_end,
        generation=parent.generation + 1,
        parent=parent,
    )
    parent.children.append(child)
    return child


def build_procedural_airway() -> List[AirwayBranch]:
    """Return a list of all branches in DFS order. branches[0] is the trachea."""
    branches: List[AirwayBranch] = []

    # Trachea: 120 mm long, 18 mm diameter, descending along -z
    trachea = AirwayBranch(
        name="trachea",
        start=np.array([0.0, 0.0, 0.0]),
        end=np.array([0.0, 0.0, -120.0]),
        radius_start=9.0, radius_end=9.0,
        generation=0,
    )
    branches.append(trachea)

    # Main carina: trachea splits into R + L main bronchi, ~70° apart
    # Right is more vertical (~25° off trachea axis); left turns ~45°
    # Frontal plane axis = +y (anterior); sagittal axis = +x (right)
    main_R = _add_child(trachea, "main_R",
                        turn_axis=np.array([0, 1, 0]), turn_deg=-25,
                        length=22.0, radius_end=7.5)  # right main is shorter
    main_L = _add_child(trachea, "main_L",
                        turn_axis=np.array([0, 1, 0]), turn_deg=+45,
                        length=50.0, radius_end=6.5)  # left main is longer
    branches.append(main_R); branches.append(main_L)

    # Right lobar bronchi: upper, middle, lower
    rul = _add_child(main_R, "R_upper_lobar",
                     turn_axis=np.array([0, 0, 1]), turn_deg=+30,
                     length=18.0, radius_end=5.0)
    bi  = _add_child(main_R, "R_intermediate",
                     turn_axis=np.array([0, 1, 0]), turn_deg=-10,
                     length=20.0, radius_end=6.0)
    rml = _add_child(bi, "R_middle_lobar",
                     turn_axis=np.array([0, 0, -1]), turn_deg=+30,
                     length=15.0, radius_end=4.0)
    rll = _add_child(bi, "R_lower_lobar",
                     turn_axis=np.array([0, 1, 0]), turn_deg=-15,
                     length=22.0, radius_end=5.0)
    branches += [rul, bi, rml, rll]

    # Left lobar bronchi: upper (with lingula), lower
    lul = _add_child(main_L, "L_upper_lobar",
                     turn_axis=np.array([0, 0, 1]), turn_deg=-25,
                     length=18.0, radius_end=4.5)
    lll = _add_child(main_L, "L_lower_lobar",
                     turn_axis=np.array([0, 1, 0]), turn_deg=-15,
                     length=22.0, radius_end=4.5)
    branches += [lul, lll]

    # Segmental — a representative subset (anatomically there are ~18)
    def _segments(parent, prefix, n, spread_axis, base_radius, length=14.0):
        out = []
        if n <= 0:
            return out
        for i in range(n):
            # Spread evenly across spread_axis, with slight forward fan
            angle = -25 + (50 * i / max(n - 1, 1))
            seg = _add_child(parent, f"{prefix}_seg{i+1}",
                             turn_axis=spread_axis, turn_deg=angle,
                             length=length, radius_end=base_radius * 0.8)
            out.append(seg)
        return out

    branches += _segments(rul, "RUL", 3, np.array([0, 1, 0]), 3.5)
    branches += _segments(rml, "RML", 2, np.array([0, 0, 1]), 3.0)
    branches += _segments(rll, "RLL", 4, np.array([1, 0, 0]), 3.5)
    branches += _segments(lul, "LUL", 3, np.array([0, 1, 0]), 3.2)
    branches += _segments(lll, "LLL", 4, np.array([1, 0, 0]), 3.2)
    return branches


def _frame_for(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors orthogonal to `direction`."""
    d = direction / max(np.linalg.norm(direction), 1e-9)
    helper = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(d, helper)
    u /= max(np.linalg.norm(u), 1e-9)
    v = np.cross(d, u)
    return u, v


def airway_to_mesh(branches: List[AirwayBranch],
                   sides: int = 16,
                   ring_density_per_mm: float = 0.5) -> o3d.geometry.TriangleMesh:
    """Tube each branch as a truncated cone with `sides` facets,
    sampled along the centerline. Returns one merged TriangleMesh."""
    all_verts = []
    all_tris = []
    base = 0
    for b in branches:
        n_rings = max(2, int(b.length * ring_density_per_mm))
        u, v = _frame_for(b.direction)
        rings = []
        for i in range(n_rings):
            t = i / (n_rings - 1)
            center = b.start * (1 - t) + b.end * t
            r = b.radius_start * (1 - t) + b.radius_end * t
            ring = []
            for k in range(sides):
                theta = 2 * np.pi * k / sides
                offset = (np.cos(theta) * u + np.sin(theta) * v) * r
                ring.append(center + offset)
            rings.append(ring)
        # Verts
        for ring in rings:
            for p in ring:
                all_verts.append(p)
        # Triangles connecting consecutive rings
        for ri in range(n_rings - 1):
            for k in range(sides):
                a = base + ri * sides + k
                b1 = base + ri * sides + (k + 1) % sides
                c = base + (ri + 1) * sides + k
                d = base + (ri + 1) * sides + (k + 1) % sides
                all_tris.append([a, c, b1])
                all_tris.append([b1, c, d])
        base += n_rings * sides

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(all_verts))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(all_tris))
    mesh.compute_vertex_normals()
    return mesh


def airway_centerline_dfs(branches: List[AirwayBranch],
                          samples_per_branch: int = 12) -> np.ndarray:
    """Single Nx3 centerline polyline, depth-first traversal of the tree.

    DFS gives a continuous curve that visits every branch — good enough
    for the existing trajectory ICP / centerline progress display.
    """
    pts = []
    visited = set()

    def _visit(b: AirwayBranch):
        if id(b) in visited:
            return
        visited.add(id(b))
        for i in range(samples_per_branch):
            t = i / max(samples_per_branch - 1, 1)
            pts.append(b.start * (1 - t) + b.end * t)
        for c in b.children:
            _visit(c)
            # backtrack: re-visit branch's end so next sibling starts from it
            pts.append(b.end.copy())

    if branches:
        _visit(branches[0])
    return np.asarray(pts, dtype=np.float64)


def write_atlas(output_dir: str,
                mesh_filename: str = "atlas_airway.ply",
                centerline_filename: str = "atlas_centerline.ply",
                summary_filename: str = "atlas_summary.json"):
    """Convenience: generate atlas, dump to disk, return (mesh_path, branches)."""
    import os, json
    os.makedirs(output_dir, exist_ok=True)

    branches = build_procedural_airway()
    mesh = airway_to_mesh(branches)
    centerline = airway_centerline_dfs(branches)

    mesh_path = os.path.join(output_dir, mesh_filename)
    o3d.io.write_triangle_mesh(mesh_path, mesh)

    cl_pcd = o3d.geometry.PointCloud()
    cl_pcd.points = o3d.utility.Vector3dVector(centerline)
    o3d.io.write_point_cloud(
        os.path.join(output_dir, centerline_filename), cl_pcd)

    summary = {
        "n_branches": len(branches),
        "branches": [
            {
                "name": b.name, "generation": b.generation,
                "length_mm": round(b.length, 2),
                "radius_start_mm": round(b.radius_start, 2),
                "radius_end_mm": round(b.radius_end, 2),
            } for b in branches
        ],
        "note": ("Procedural atlas — anatomically credible, NOT patient-"
                 "specific. Use only for approximate, atlas-based scope "
                 "localization."),
    }
    with open(os.path.join(output_dir, summary_filename), "w") as f:
        json.dump(summary, f, indent=2)

    return mesh_path, branches


if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser(description="Generate the procedural bronchial atlas")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    mesh_path, branches = write_atlas(args.output_dir)
    print(f"Wrote {len(branches)} branches -> {mesh_path}")
