"""Moving part borders onto the folds of the full-detail model.

Katz and Tal ("Hierarchical mesh decomposition using fuzzy clustering and
cuts", SIGGRAPH 2003) settle a border by a minimum cut through a fuzzy band
between two parts, with edge capacities 1 / (1 + AngDist / mean AngDist): the
cut prefers concave edges. Here the band is a few rings of faces either side
of the current border, the capacity is also scaled by edge length (so a
shorter cut is preferred among equally concave ones), and the flow is
scipy's."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

from . import features
from .meshio import components

RINGS = 3
SCALE = 10_000          # capacities are integers for scipy's max flow
MOST_MOVED = 0.4        # of the band: beyond this a cut is not trusted


def _rings(off, nb, seed, allowed, rings):
    """Faces within `rings` steps of `seed`, staying in `allowed`."""
    inside = np.zeros(len(allowed), dtype=bool)
    inside[seed] = True
    front = seed
    for _ in range(rings):
        if not len(front):
            break
        nxt = np.unique(np.concatenate([nb[off[f]:off[f + 1]] for f in front]))
        nxt = nxt[allowed[nxt] & ~inside[nxt]]
        inside[nxt] = True
        front = nxt
    return inside


def snap_to_folds(mesh, labels, rings=RINGS):
    labels = labels.copy()
    eid, fa, fb, ang = features.concavity(mesh)
    length = mesh.edge_lengths()[eid]
    cap = (length / length.mean()) / (1.0 + ang / max(ang.mean(), 1e-12))
    weight = np.maximum(1, np.round(cap * SCALE)).astype(np.int64)
    off, nb, _via = mesh.face_graph()

    pairs = np.unique(np.sort(np.stack([labels[fa], labels[fb]], 1)[
        labels[fa] != labels[fb]], axis=1), axis=0)
    for A, B in pairs.tolist():
        on = ((labels[fa] == A) & (labels[fb] == B)) | (
            (labels[fa] == B) & (labels[fb] == A))
        if not on.any():
            continue
        allowed = (labels == A) | (labels == B)
        seed = np.unique(np.concatenate([fa[on], fb[on]]))
        band = _rings(off, nb, seed, allowed, rings)
        ids = np.flatnonzero(band)
        if len(ids) < 4:
            continue
        local = np.full(mesh.n_faces, -1, dtype=np.int64)
        local[ids] = np.arange(len(ids))
        s_node, t_node = len(ids), len(ids) + 1

        # Band faces touching the part beyond the band are tied to it.
        out_a = np.zeros(len(ids), dtype=bool)
        out_b = np.zeros(len(ids), dtype=bool)
        for x, y in ((fa, fb), (fb, fa)):
            m = band[x] & ~band[y]
            out_a[local[x[m & (labels[y] == A)]]] = True
            out_b[local[x[m & (labels[y] == B)]]] = True
        if not out_a.any() or not out_b.any():
            continue                  # a part lies wholly in the band

        inner = band[fa] & band[fb]
        u, v, w = local[fa[inner]], local[fb[inner]], weight[inner]
        big = int(weight.sum()) + 1
        src = np.concatenate([u, v, np.full(out_a.sum(), s_node),
                              np.flatnonzero(out_b)])
        dst = np.concatenate([v, u, np.flatnonzero(out_a),
                              np.full(out_b.sum(), t_node)])
        cap_e = np.concatenate([w, w, np.full(out_a.sum(), big),
                                np.full(out_b.sum(), big)])
        n = len(ids) + 2
        graph = csr_matrix((cap_e.astype(np.int32 if big < 2**31 else np.int64),
                            (src, dst)), shape=(n, n))
        graph.sum_duplicates()
        flow = maximum_flow(graph, s_node, t_node).flow
        residual = graph - flow
        residual.data = np.where(residual.data > 0, 1, 0)
        residual.eliminate_zeros()
        reach = breadth_first_order(residual, s_node, directed=True,
                                    return_predecessors=False)
        side_a = np.zeros(n, dtype=bool)
        side_a[reach] = True
        new = np.where(side_a[:len(ids)], A, B)
        if (new != labels[ids]).mean() > MOST_MOVED:
            continue
        trial = labels.copy()
        trial[ids] = new
        # The cut must not break either part in two.
        ok = True
        for part in (A, B):
            mask = trial == part
            sel = mask[fa] & mask[fb]
            faces = np.flatnonzero(mask)
            if len(faces) and len(np.unique(components(
                    mesh.n_faces, fa[sel], fb[sel])[faces])) > 1:
                ok = False
        if ok:
            labels = trial
    return labels
