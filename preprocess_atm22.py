"""Preprocess ATM22 airway segmentation labels.

For each labeled subject in ATM22's labelsTr:
  - load the binary airway mask (NIfTI)
  - extract a 3D centerline + bifurcation graph (kimimaro)
  - extract a triangle mesh of the airway surface (marching cubes)
  - apply the NIfTI affine so all outputs live in patient mm coords

Step 1 of the patient-specific airway navigation pipeline. Output is
the per-subject corpus that SSM fitting (PCA across subjects) consumes
later.

Outputs per subject under <output_dir>/<subject_id>/:
  airway_mesh.ply           decimated surface mesh (mm)
  centerline.npz            nodes (Nx3, mm), edges (Mx2),
                            radii (N, voxels), is_bifurcation (N, bool)
  bifurcation_graph.gml     networkx graph; node attrs: pos, radius, kind

Usage:
  python preprocess_atm22.py \\
      --labels_dir /path/to/ATM22/labelsTr \\
      --output_dir preprocessed/atm22 \\
      --workers 4

If the labelsTr is still a folder of zip archives, unzip first:
  cd /path/to/ATM22/labelsTr && for z in *.zip; do unzip -o "$z"; done
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import networkx as nx
import numpy as np
import trimesh
from skimage import measure
from tqdm import tqdm

try:
    import kimimaro
except ImportError:
    sys.exit("ERROR: kimimaro is required. pip install -r "
             "requirements-atm22.txt")


def _largest_component(mask):
    """Keep the largest 26-connected component of a binary mask."""
    lab = measure.label(mask, connectivity=3)
    if lab.max() == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))


def _vox_to_mm(points_vox, affine):
    """Apply a 4x4 NIfTI affine to (N,3) voxel-index points."""
    h = np.concatenate(
        [points_vox, np.ones((len(points_vox), 1))], axis=1)
    return (h @ affine.T)[:, :3]


def _skeleton_to_graph(skel, affine, voxel_size_mm):
    """Wrap a kimimaro Skeleton in a networkx graph in mm coords.

    Bifurcations = nodes of degree >= 3. Endpoints = degree 1.
    Radii are converted from voxel units to mm using the mean voxel
    spacing — exact for isotropic CT, approximate for anisotropic
    (acceptable for graph-matching since matching uses geometry, not
    radius magnitude).
    """
    verts_mm = _vox_to_mm(skel.vertices, affine)
    radii_mm = np.asarray(skel.radii, dtype=np.float64) * float(voxel_size_mm)
    G = nx.Graph()
    for i, v in enumerate(verts_mm):
        G.add_node(int(i), pos=v.tolist(), radius=float(radii_mm[i]))
    for a, b in skel.edges:
        G.add_edge(int(a), int(b))
    for n, d in G.degree():
        if d >= 3:
            kind = 'bifurcation'
        elif d == 1:
            kind = 'endpoint'
        else:
            kind = 'shaft'
        G.nodes[n]['kind'] = kind
    return G


def _mesh_from_mask(mask, affine, decimate_target):
    """Marching cubes + (optional) quadric decimation, in mm coords."""
    verts, faces, _, _ = measure.marching_cubes(
        mask.astype(np.uint8), level=0.5)
    verts_mm = _vox_to_mm(verts, affine)
    mesh = trimesh.Trimesh(vertices=verts_mm, faces=faces, process=True)
    if decimate_target and len(mesh.faces) > decimate_target:
        try:
            mesh = mesh.simplify_quadric_decimation(decimate_target)
        except Exception as e:
            # Newer trimesh routes through fast-simplification; if that
            # backend is missing, fall back to the original mesh rather
            # than failing the whole subject.
            print(f"  (decimation skipped: {e})")
    return mesh


def _subject_id(path):
    base = os.path.basename(path)
    for ext in ('.nii.gz', '.nii'):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base


def process_one(label_path, out_root, decimate_target, delete_raw_after=False):
    sid = _subject_id(label_path)
    out_dir = os.path.join(out_root, sid)
    os.makedirs(out_dir, exist_ok=True)
    mesh_path = os.path.join(out_dir, 'airway_mesh.ply')
    cl_path = os.path.join(out_dir, 'centerline.npz')
    g_path = os.path.join(out_dir, 'bifurcation_graph.gml')
    affine_path = os.path.join(out_dir, 'affine.npy')
    spacing_path = os.path.join(out_dir, 'voxel_spacing_mm.npy')
    required = (mesh_path, cl_path, g_path, affine_path, spacing_path)
    if all(os.path.exists(p) for p in required):
        # Even on skip, honor delete-after if the source still exists.
        if delete_raw_after and os.path.exists(label_path):
            os.remove(label_path)
        return sid, 'skipped (already exists)'

    nii = nib.load(label_path)
    mask = (np.asarray(nii.dataobj) > 0)
    if not mask.any():
        return sid, 'empty label'
    affine = nii.affine.astype(np.float64)
    # Voxel sizes from a possibly-rotated affine: take column norms of
    # the 3x3 linear block, not just the diagonal.
    voxel_sizes = np.linalg.norm(affine[:3, :3], axis=0).astype(np.float32)
    voxel_size_mm = float(voxel_sizes.mean())

    mask = _largest_component(mask)

    # Anisotropy=(1,1,1) -> kimimaro returns voxel-index vertices, which
    # we then map to mm via the full affine (handles rotated CTs, not
    # just axis-aligned scaling).
    skels = kimimaro.skeletonize(
        mask.astype(np.uint32),
        teasar_params={
            'scale': 1.5, 'const': 300,
            'pdrf_scale': 100000, 'pdrf_exponent': 4,
            'soma_acceptance_threshold': 3500,
            'soma_detection_threshold': 750,
            'soma_invalidation_const': 300,
            'soma_invalidation_scale': 2,
        },
        anisotropy=(1.0, 1.0, 1.0),
        dust_threshold=100,
        fix_branching=True, fix_borders=True,
        progress=False, parallel=1,
    )
    if not skels:
        return sid, 'no skeleton produced'
    # ATM22 labels are single-label (=1); kimimaro returns {1: Skeleton}.
    skel = next(iter(skels.values()))

    G = _skeleton_to_graph(skel, affine, voxel_size_mm)
    nodes = np.array([G.nodes[n]['pos'] for n in G.nodes], dtype=np.float64)
    radii = np.array([G.nodes[n]['radius'] for n in G.nodes],
                     dtype=np.float64)
    is_bif = np.array([G.nodes[n]['kind'] == 'bifurcation'
                       for n in G.nodes], dtype=bool)
    edges = np.array(list(G.edges), dtype=np.int64)
    np.savez(cl_path, nodes=nodes, edges=edges, radii=radii,
             is_bifurcation=is_bif)
    nx.write_gml(G, g_path, stringizer=str)

    mesh = _mesh_from_mask(mask, affine, decimate_target=decimate_target)
    mesh.export(mesh_path)

    np.save(affine_path, affine)
    np.save(spacing_path, voxel_sizes)

    # Only delete the source after every output is confirmed on disk.
    # Any failure above raises before we get here, so the raw file is
    # preserved on error.
    if delete_raw_after and all(os.path.exists(p) for p in required):
        try:
            os.remove(label_path)
        except OSError as e:
            print(f"  (could not delete {label_path}: {e})")

    return sid, (f'ok verts={len(mesh.vertices)} faces={len(mesh.faces)} '
                 f'bifurcations={int(is_bif.sum())} '
                 f'endpoints={int((np.bincount(edges.ravel()) == 1).sum())}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels_dir', required=True,
                    help='Directory of ATM_*.nii.gz airway masks.')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--decimate_target', type=int, default=30000,
                    help='Target face count after decimation; 0 disables.')
    ap.add_argument('--limit', type=int, default=None,
                    help='Process only the first N subjects (for testing).')
    ap.add_argument('--delete_raw_after', action='store_true',
                    help='Delete each source NIfTI immediately after its '
                         'outputs are written and verified. Lets you '
                         'process the whole dataset without doubling disk '
                         'usage. IRREVERSIBLE — confirm with --limit first.')
    args = ap.parse_args()

    paths = sorted(
        glob.glob(os.path.join(args.labels_dir, '*.nii.gz'))
        + glob.glob(os.path.join(args.labels_dir, '*.nii')))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        sys.exit(f'ERROR: no NIfTI files under {args.labels_dir}. '
                 f'If labelsTr contains zips, unzip them first.')
    print(f'Found {len(paths)} subjects -> {args.output_dir}')
    os.makedirs(args.output_dir, exist_ok=True)

    decim = args.decimate_target if args.decimate_target > 0 else 0
    delete_raw = bool(args.delete_raw_after)
    if delete_raw:
        print('WARNING: --delete_raw_after is on. Source NIfTI files will '
              'be deleted as each subject completes.')
    failures = []
    if args.workers <= 1:
        for p in tqdm(paths, desc='subjects'):
            sid, status = process_one(p, args.output_dir, decim, delete_raw)
            tqdm.write(f'{sid}: {status}')
            if not status.startswith('ok') and not status.startswith('skipped'):
                failures.append((sid, status))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_one, p, args.output_dir, decim,
                              delete_raw): p for p in paths}
            for f in tqdm(as_completed(futs), total=len(futs),
                          desc='subjects'):
                sid, status = f.result()
                tqdm.write(f'{sid}: {status}')
                if not status.startswith('ok') \
                        and not status.startswith('skipped'):
                    failures.append((sid, status))

    print(f'\nDone. Failures: {len(failures)}/{len(paths)}')
    for sid, s in failures:
        print(f'  {sid}: {s}')


if __name__ == '__main__':
    main()
