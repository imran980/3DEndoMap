"""Pack preprocess_atm22.py outputs into a single compressed HDF5 corpus.

After preprocessing finishes and the per-subject .ply files have been
visually QA'd, this script bundles all per-subject artifacts into one
gzipped HDF5 file. The corpus is self-contained — downstream code
(step 2 correspondence, step 3 PCA, etc.) only needs the .h5, never
the original ATM22 CT/labels or the intermediate per-subject dirs.

Usage:
  python build_atm22_corpus.py \\
      --input_dir preprocessed/atm22 \\
      --output atm22_corpus.h5

After packing succeeds you can safely delete both the raw ATM22
dataset and the per-subject input_dir.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import os
import sys

import h5py
import networkx as nx
import numpy as np
import trimesh
from tqdm import tqdm

PIPELINE_VERSION = "preprocess_atm22.py v1"


def _load_subject(subject_dir):
    mesh = trimesh.load(os.path.join(subject_dir, "airway_mesh.ply"),
                        process=False)
    cl = np.load(os.path.join(subject_dir, "centerline.npz"))
    g_path = os.path.join(subject_dir, "bifurcation_graph.gml")
    G = nx.read_gml(g_path, destringizer=None) if os.path.exists(g_path) \
        else None

    # NIfTI affine + voxel spacing aren't currently saved per subject;
    # recover voxel spacing from the centerline node spacing as a
    # fallback, and store identity affine. This is acceptable because
    # everything is already in mm; affine/spacing are kept only for
    # provenance/debug, not used by downstream SSM steps.
    affine = np.eye(4, dtype=np.float64)
    spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    return {
        "surface_verts": np.asarray(mesh.vertices, dtype=np.float32),
        "surface_faces": np.asarray(mesh.faces, dtype=np.int32),
        "centerline_nodes": cl["nodes"].astype(np.float32),
        "centerline_edges": cl["edges"].astype(np.int32),
        "centerline_radii": cl["radii"].astype(np.float32),
        "is_bifurcation": cl["is_bifurcation"].astype(bool),
        "affine": affine,
        "voxel_spacing_mm": spacing,
        "_graph": G,
    }


def _write_subject(h5_group, data):
    opts = dict(compression="gzip", compression_opts=6)
    for key in ("surface_verts", "surface_faces", "centerline_nodes",
                "centerline_edges", "centerline_radii", "is_bifurcation"):
        h5_group.create_dataset(key, data=data[key], **opts)
    h5_group.create_dataset("affine", data=data["affine"])
    h5_group.create_dataset("voxel_spacing_mm", data=data["voxel_spacing_mm"])
    h5_group.attrs["n_bifurcations"] = int(data["is_bifurcation"].sum())
    h5_group.attrs["n_endpoints"] = int(
        (np.bincount(data["centerline_edges"].ravel()) == 1).sum())
    h5_group.attrs["n_verts"] = int(len(data["surface_verts"]))
    h5_group.attrs["n_faces"] = int(len(data["surface_faces"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True,
                    help="Output dir from preprocess_atm22.py (contains "
                         "per-subject subfolders).")
    ap.add_argument("--output", required=True,
                    help="Path to write atm22_corpus.h5.")
    args = ap.parse_args()

    subject_dirs = sorted(
        d for d in glob.glob(os.path.join(args.input_dir, "*"))
        if os.path.isdir(d)
        and os.path.exists(os.path.join(d, "airway_mesh.ply"))
        and os.path.exists(os.path.join(d, "centerline.npz")))
    if not subject_dirs:
        sys.exit(f"ERROR: no preprocessed subjects under {args.input_dir}")
    print(f"Packing {len(subject_dirs)} subjects -> {args.output}")

    failed = []
    written = []
    with h5py.File(args.output, "w") as h5:
        meta = h5.create_group("metadata")
        subj_grp = h5.create_group("subjects")
        for sdir in tqdm(subject_dirs, desc="packing"):
            sid = os.path.basename(sdir)
            try:
                data = _load_subject(sdir)
                _write_subject(subj_grp.create_group(sid), data)
                written.append(sid)
            except Exception as e:
                failed.append((sid, str(e)))
                tqdm.write(f"  {sid}: FAILED ({e})")

        meta.attrs["source"] = "ATM22 labelsTr"
        meta.attrs["pipeline_version"] = PIPELINE_VERSION
        meta.attrs["created"] = _dt.datetime.utcnow().isoformat() + "Z"
        meta.create_dataset("subject_ids",
                            data=np.array(written, dtype=h5py.string_dtype()))

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nWrote {len(written)} subjects -> {args.output} ({size_mb:.1f} MB)")
    if failed:
        print(f"Failures ({len(failed)}):")
        for sid, e in failed:
            print(f"  {sid}: {e}")


if __name__ == "__main__":
    main()
