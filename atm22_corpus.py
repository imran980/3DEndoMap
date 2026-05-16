"""Read-only loader for the packed ATM22 corpus (atm22_corpus.h5).

All downstream pipeline steps (correspondence, PCA / SSM fitting,
graph matching, etc.) should depend only on this module, not on the
raw ATM22 dataset or the per-subject preprocessing intermediates.

Typical use:
    from atm22_corpus import Corpus
    corpus = Corpus("atm22_corpus.h5")
    print(corpus.subject_ids)
    s = corpus.load("ATM_001")
    s.surface_verts        # (V, 3) float32 in mm
    s.centerline_nodes     # (N, 3) float32 in mm
    s.is_bifurcation       # (N,) bool
    bif_pts = s.bifurcation_points()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import h5py
import numpy as np


@dataclass
class Subject:
    subject_id: str
    surface_verts: np.ndarray       # (V, 3) float32, mm
    surface_faces: np.ndarray       # (F, 3) int32
    centerline_nodes: np.ndarray    # (N, 3) float32, mm
    centerline_edges: np.ndarray    # (E, 2) int32
    centerline_radii: np.ndarray    # (N,)   float32, mm
    is_bifurcation: np.ndarray      # (N,)   bool
    affine: np.ndarray              # (4, 4) float64
    voxel_spacing_mm: np.ndarray    # (3,)   float32

    def bifurcation_points(self) -> np.ndarray:
        return self.centerline_nodes[self.is_bifurcation]

    def endpoint_indices(self) -> np.ndarray:
        deg = np.bincount(self.centerline_edges.ravel(),
                          minlength=len(self.centerline_nodes))
        return np.where(deg == 1)[0]


class Corpus:
    """Lazy, read-only access to atm22_corpus.h5."""

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as h5:
            ids = h5["metadata/subject_ids"][:]
            self.subject_ids: List[str] = [
                s.decode() if isinstance(s, bytes) else str(s) for s in ids]
            self.metadata = dict(h5["metadata"].attrs)

    def __len__(self):
        return len(self.subject_ids)

    def __contains__(self, sid):
        return sid in self.subject_ids

    def load(self, subject_id: str) -> Subject:
        with h5py.File(self.h5_path, "r") as h5:
            g = h5[f"subjects/{subject_id}"]
            return Subject(
                subject_id=subject_id,
                surface_verts=g["surface_verts"][:],
                surface_faces=g["surface_faces"][:],
                centerline_nodes=g["centerline_nodes"][:],
                centerline_edges=g["centerline_edges"][:],
                centerline_radii=g["centerline_radii"][:],
                is_bifurcation=g["is_bifurcation"][:],
                affine=g["affine"][:],
                voxel_spacing_mm=g["voxel_spacing_mm"][:],
            )

    def iter_subjects(self):
        for sid in self.subject_ids:
            yield self.load(sid)
