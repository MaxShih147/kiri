"""Triangle meshes as two arrays, and the topology queries kiri needs."""

from dataclasses import dataclass, field

import numpy as np
import trimesh


@dataclass
class Mesh:
    """`v`: (n, 3) float64 positions. `f`: (m, 3) int64 vertex indices,
    counter-clockwise seen from outside."""

    v: np.ndarray
    f: np.ndarray
    # Per face: True on the skins kiri made to close a cut.
    cap: np.ndarray = None
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if self.cap is None:
            self.cap = np.zeros(len(self.f), dtype=bool)

    @property
    def n_faces(self):
        return len(self.f)

    def _memo(self, key, make):
        got = self._cache.get(key)
        if got is None:
            got = self._cache[key] = make()
        return got

    # -- geometry ---------------------------------------------------------
    def face_normals(self):
        def make():
            a, b, c = self.corners()
            n = np.cross(b - a, c - a)
            length = np.linalg.norm(n, axis=1)
            return n / np.where(length > 0, length, 1.0)[:, None]
        return self._memo("fn", make)

    def face_areas(self):
        def make():
            a, b, c = self.corners()
            return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        return self._memo("fa", make)

    def centroids(self):
        return self._memo("fc", lambda: self.v[self.f].mean(axis=1))

    def corners(self):
        return self.v[self.f[:, 0]], self.v[self.f[:, 1]], self.v[self.f[:, 2]]

    def vertex_normals(self):
        def make():
            acc = np.zeros_like(self.v)
            weighted = self.face_normals() * self.face_areas()[:, None]
            for k in range(3):
                np.add.at(acc, self.f[:, k], weighted)
            length = np.linalg.norm(acc, axis=1)
            return acc / np.where(length > 0, length, 1.0)[:, None]
        return self._memo("vn", make)

    def size(self):
        return float(np.linalg.norm(np.ptp(self.v, axis=0))) or 1.0

    def volume(self):
        a, b, c = self.corners()
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    # -- topology ---------------------------------------------------------
    def edges(self):
        """(unique edges (e, 2) with lo < hi, edge of each face side (m, 3)).
        Side k of a face runs from corner k to corner k+1."""
        def make():
            side = np.stack([self.f[:, [0, 1]], self.f[:, [1, 2]],
                             self.f[:, [2, 0]]], axis=1).reshape(-1, 2)
            lo, hi = side.min(axis=1), side.max(axis=1)
            key = lo * np.int64(len(self.v)) + hi
            keys, which = np.unique(key, return_inverse=True)
            e = np.stack([keys // len(self.v), keys % len(self.v)], axis=1)
            return e, which.reshape(-1, 3)
        return self._memo("edges", make)

    def edge_faces(self):
        """For edges shared by exactly two faces: (edge ids, face a, face b)."""
        def make():
            e, side = self.edges()
            flat = side.ravel()
            owner = np.repeat(np.arange(self.n_faces), 3)
            count = np.bincount(flat, minlength=len(e))
            order = np.argsort(flat, kind="stable")
            start = np.zeros(len(e) + 1, dtype=np.int64)
            np.cumsum(count, out=start[1:])
            two = np.flatnonzero(count == 2)
            fa = owner[order[start[two]]]
            fb = owner[order[start[two] + 1]]
            return two, fa, fb
        return self._memo("ef", make)

    def edge_lengths(self):
        def make():
            e, _ = self.edges()
            return np.linalg.norm(self.v[e[:, 0]] - self.v[e[:, 1]], axis=1)
        return self._memo("el", make)

    def face_graph(self):
        """CSR adjacency of faces across shared edges: (offsets, neighbours,
        edge id of each link)."""
        def make():
            eid, fa, fb = self.edge_faces()
            src = np.concatenate([fa, fb])
            dst = np.concatenate([fb, fa])
            via = np.concatenate([eid, eid])
            order = np.argsort(src, kind="stable")
            off = np.zeros(self.n_faces + 1, dtype=np.int64)
            np.cumsum(np.bincount(src, minlength=self.n_faces), out=off[1:])
            return off, dst[order], via[order]
        return self._memo("fg", make)

    def open_edges(self):
        e, side = self.edges()
        return int((np.bincount(side.ravel(), minlength=len(e)) == 1).sum())

    def bad_edges(self):
        e, side = self.edges()
        return int((np.bincount(side.ravel(), minlength=len(e)) > 2).sum())


def load(source, file_type=None):
    """Any format trimesh reads, as one welded triangle mesh."""
    got = trimesh.load(source, file_type=file_type, force="mesh", process=True)
    if not isinstance(got, trimesh.Trimesh) or not len(got.faces):
        raise ValueError("no triangles in this file")
    return Mesh(np.asarray(got.vertices, dtype=np.float64),
                np.asarray(got.faces, dtype=np.int64))


def to_stl(mesh):
    return trimesh.Trimesh(mesh.v, mesh.f, process=False).export(
        file_type="stl")


def components(n, a, b):
    """Connected-component id per node of a graph with edges (a, b), ids
    0.. in order of each component's lowest node."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    _count, lab = connected_components(g, directed=False)
    # Renumber by first appearance so ids do not depend on scipy's order.
    first = np.full(lab.max() + 1 if len(lab) else 0, n, dtype=np.int64)
    np.minimum.at(first, lab, np.arange(n))
    rank = np.empty_like(first)
    rank[np.argsort(first)] = np.arange(len(first))
    return rank[lab]


def to_zip(pieces, stem="piece"):
    """The pieces as STL files in one zip archive (bytes)."""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, p in enumerate(pieces, 1):
            z.writestr("%s_%02d.stl" % (stem, i), to_stl(p))
    return buf.getvalue()
