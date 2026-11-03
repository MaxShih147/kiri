"""From parts to pieces: peel parts off one at a time, smallest first, until
every piece holds one part."""

import numpy as np

from .cut import CutFailed, peel
from .meshio import Mesh, components


def _areas(mesh, labels):
    return np.bincount(labels, weights=mesh.face_areas())


def shells(mesh, labels=None):
    """`mesh` as one Mesh per connected shell (with each one's labels, when
    `labels` are given)."""
    _eid, fa, fb = mesh.edge_faces()
    shell = components(mesh.n_faces, fa, fb)
    out = []
    for s in range(shell.max() + 1):
        mask = shell == s
        used, inv = np.unique(mesh.f[mask], return_inverse=True)
        piece = Mesh(mesh.v[used], inv.reshape(-1, 3), mesh.cap[mask])
        out.append(piece if labels is None else (piece, labels[mask]))
    return out


def split_into_pieces(mesh, labels, progress=None):
    """Closed pieces, one per part where the cuts succeed. Returns (pieces,
    the parts that could not be cut off, as (piece index, part) pairs)."""
    todo = shells(mesh, np.asarray(labels))
    done, refused = [], []
    why = {}
    cuts = 0
    while todo:
        piece, lab = todo.pop()
        parts = np.unique(lab)
        if len(parts) < 2:
            done.append(piece)
            continue
        area = _areas(piece, lab)
        order = sorted(parts.tolist(), key=lambda p: area[p])
        for part in order:
            try:
                part_piece, rest, rest_lab = peel(piece, lab, part)
            except CutFailed as err:
                why.setdefault(part, str(err))
                continue
            done.append(part_piece)
            # What is left can fall apart: each shell goes on by itself.
            todo.extend(shells(rest, rest_lab))
            cuts += 1
            if progress is not None:
                progress(cuts)
            break
        else:
            done.append(piece)
            refused.append((len(done) - 1, len(parts),
                            sorted(set(why.values()))))
    return _gather_crumbs(done, mesh), refused


CRUMB = 0.001    # of the model's area: a shell this small is not a piece


def _gather_crumbs(pieces, model):
    """Each piece as its shells; a shell too small to print alone (an eye
    modelled apart, a sliver) rides with the nearest larger piece."""
    from .nearest import Surface
    floor = CRUMB * model.face_areas().sum()
    whole, crumbs = [], []
    for p in pieces:
        for s in shells(p):
            (whole if s.face_areas().sum() >= floor else crumbs).append(s)
    if not whole:
        return crumbs
    if crumbs:
        # Nearest by surface: the piece whose skin lies closest to the
        # crumb's centre.
        near = [Surface.of(w) for w in whole]
        joined = {i: [w] for i, w in enumerate(whole)}
        for c in crumbs:
            centre = c.v.mean(axis=0)
            gaps = [s.nearest(centre[None])[2][0] for s in near]
            joined[int(np.argmin(gaps))].append(c)
        whole = [_concat(parts) for parts in joined.values()]
    return whole


def _concat(meshes):
    if len(meshes) == 1:
        return meshes[0]
    v, f, cap, base = [], [], [], 0
    for m in meshes:
        v.append(m.v)
        f.append(m.f + base)
        cap.append(m.cap)
        base += len(m.v)
    return Mesh(np.vstack(v), np.vstack(f), np.concatenate(cap))
