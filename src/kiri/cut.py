"""Peeling one part off a piece: smooth seams round it, a signed distance to
them over the surface, every triangle split where that distance changes sign
(marching triangles: one new point per crossed edge, shared by both sides, so
the two halves meet exactly), and each opening closed with one skin seen from
both sides."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from . import seam
from .cap import skin
from .meshio import Mesh, components
from .nearest import Surface


class CutFailed(Exception):
    pass


def _median_edge(mesh):
    return float(np.median(mesh.edge_lengths()))


def _vertex_graph(mesh):
    e, _ = mesh.edges()
    w = mesh.edge_lengths()
    n = len(mesh.v)
    return coo_matrix((np.concatenate([w, w]),
                       (np.concatenate([e[:, 0], e[:, 1]]),
                        np.concatenate([e[:, 1], e[:, 0]]))),
                      shape=(n, n)).tocsr()


def signed_distance(mesh, inside, curves, step):
    """Per vertex: distance to the nearest seam, positive on the `inside`
    side. Near a seam (along the surface) the side is read off the seam's
    own direction; further away it is the side the vertex's faces are on."""
    pts = np.vstack([c for c, _n in curves])
    nrm = np.vstack([n for _c, n in curves])
    tangent = np.vstack([np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)
                         for c, _n in curves])
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True),
                          1e-300)
    left = np.cross(nrm, tangent)       # the inside lies to the left

    # Which side each vertex is on by its faces.
    n = len(mesh.v)
    in_count = np.zeros(n)
    all_count = np.zeros(n)
    for k in range(3):
        np.add.at(in_count, mesh.f[:, k], inside.astype(float))
        np.add.at(all_count, mesh.f[:, k], 1.0)
    by_faces = in_count * 2 > all_count
    border = (in_count > 0) & (in_count < all_count)

    tree = cKDTree(pts)
    dist, j = tree.query(mesh.v)
    # How far the smoothed seams stray from the painted border sets how
    # wide a band reads its side from them; measured along the surface.
    stray = float(dist[border].max()) if border.any() else step
    reach = 1.5 * stray + 3.0 * step
    along = dijkstra(_vertex_graph(mesh), indices=np.flatnonzero(border),
                     min_only=True, limit=reach)
    near = np.isfinite(along)
    side = ((mesh.v - pts[j]) * left[j]).sum(axis=1) > 0
    positive = _no_islands(mesh, np.where(near, side, by_faces), by_faces,
                           near)
    tiny = mesh.size() * 1e-9
    return np.where(positive, 1.0, -1.0) * np.maximum(dist, tiny)


def _no_islands(mesh, positive, by_faces, near):
    """Flip patches of either sign that float loose of their side: the part
    is one patch, so only the positive patch most on the part's faces
    stays; a negative patch stays when it reaches beyond the seam band onto
    the rest's own faces (the rest may fall in several). Without this a
    seam that grazes the surface leaves bubbles -- a speck of skin cut out
    and closed on its own."""
    e, _ = mesh.edges()
    for _ in range(4):
        same = positive[e[:, 0]] == positive[e[:, 1]]
        patch = components(len(mesh.v), e[same, 0], e[same, 1])
        sign = np.zeros(patch.max() + 1, dtype=bool)
        sign[patch] = positive
        agree = np.bincount(patch, weights=(positive == by_faces),
                            minlength=len(sign))
        anchored = np.bincount(patch, weights=(positive == by_faces) & ~near,
                               minlength=len(sign)) > 0
        keep = ~sign & anchored
        pos_ids = np.flatnonzero(sign)
        if not len(pos_ids):
            break
        keep[pos_ids[np.argmax(agree[pos_ids])]] = True
        if not (~sign).any() or not keep[~sign].any():
            neg_ids = np.flatnonzero(~sign)
            if len(neg_ids):
                keep[neg_ids[np.argmax(agree[neg_ids])]] = True
        if keep.all():
            break
        positive = np.where(keep[patch], positive, ~positive)
    return positive


def split(mesh, field):
    """Cut every triangle where `field` changes sign. Returns (points,
    positive-side triangles, negative-side triangles, source face of each
    triangle on each side)."""
    pos = field > 0
    e, side = mesh.edges()
    crossed = pos[e[:, 0]] != pos[e[:, 1]]
    ce = np.flatnonzero(crossed)
    fa, fb = field[e[ce, 0]], field[e[ce, 1]]
    t = (fa / (fa - fb))[:, None]
    new = mesh.v[e[ce, 0]] + t * (mesh.v[e[ce, 1]] - mesh.v[e[ce, 0]])
    point_of = np.full(len(e), -1, dtype=np.int64)
    point_of[ce] = len(mesh.v) + np.arange(len(ce))
    points = np.vstack([mesh.v, new])

    count = pos[mesh.f].sum(axis=1)
    whole_pos = np.flatnonzero(count == 3)
    whole_neg = np.flatnonzero(count == 0)
    cut = np.flatnonzero((count == 1) | (count == 2))
    f = mesh.f[cut]
    s = pos[f]
    # The corner whose side differs from the other two.
    lone_k = np.where(count[cut] == 1, np.argmax(s, axis=1),
                      np.argmin(s, axis=1))
    k1, k2 = (lone_k + 1) % 3, (lone_k + 2) % 3
    rows = np.arange(len(cut))
    L, A, B = f[rows, lone_k], f[rows, k1], f[rows, k2]
    x_la = point_of[side[cut, lone_k]]       # side lone -> A
    x_bl = point_of[side[cut, k2]]           # side B -> lone
    lone_tri = np.stack([L, x_la, x_bl], axis=1)
    # The quad (x_la, A, B, x_bl), split along its shorter diagonal.
    d1 = np.linalg.norm(points[x_la] - points[B], axis=1)
    d2 = np.linalg.norm(points[A] - points[x_bl], axis=1)
    short = (d1 <= d2)[:, None]
    q1 = np.where(short, np.stack([x_la, A, B], 1), np.stack([x_la, A, x_bl], 1))
    q2 = np.where(short, np.stack([x_la, B, x_bl], 1), np.stack([A, B, x_bl], 1))
    lone_pos = pos[L]

    tri_pos = [mesh.f[whole_pos], lone_tri[lone_pos], q1[~lone_pos],
               q2[~lone_pos]]
    src_pos = [whole_pos, cut[lone_pos], cut[~lone_pos], cut[~lone_pos]]
    tri_neg = [mesh.f[whole_neg], lone_tri[~lone_pos], q1[lone_pos],
               q2[lone_pos]]
    src_neg = [whole_neg, cut[~lone_pos], cut[lone_pos], cut[lone_pos]]
    return (points, np.vstack(tri_pos), np.vstack(tri_neg),
            np.concatenate(src_pos), np.concatenate(src_neg))


def open_loops(faces):
    """Closed loops of edges used by one face only, each walked the way it
    runs in its face."""
    side = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                           faces[:, [2, 0]]])
    key = np.sort(side, axis=1)
    _u, inv, count = np.unique(key, axis=0, return_inverse=True,
                               return_counts=True)
    single = side[count[inv.ravel()] == 1]
    nxt = {}
    for a, b in single.tolist():
        nxt.setdefault(a, []).append(b)
    loops, used = [], set()
    for a0 in list(nxt):
        for b0 in nxt[a0]:
            if (a0, b0) in used:
                continue
            loop, b = [a0], b0
            used.add((a0, b0))
            while b != a0:
                loop.append(b)
                options = [c for c in nxt.get(b, []) if (b, c) not in used]
                if not options:
                    raise CutFailed("an opening that does not close")
                used.add((b, options[0]))
                b = options[0]
            loops.append(loop)
    return loops


def _compact(points, faces, cap):
    used, inv = np.unique(faces, return_inverse=True)
    return Mesh(points[used], inv.reshape(-1, 3).astype(np.int64), cap)


SMOOTHING = (4, 1, 0)    # passes tried in turn until a cut holds up


def peel(mesh, labels, part):
    """Cut part `part` off `mesh`. Returns (the part as a closed piece, the
    rest as a closed piece, the rest's labels).

    The seams are smoothed as far as the result stays sound -- both sides
    closed and solid. A seam smoothed across a thin strip can leave a sliver
    whose skin bulges out of the model; less smoothing keeps the seam on the
    painted border, which always divides."""
    inside = labels == part
    if inside.all() or not inside.any():
        raise CutFailed("nothing to cut")
    loops = seam.border_loops(mesh, inside)
    if not loops:
        raise CutFailed("the part has no clean border")
    step = _median_edge(mesh)
    surface = Surface.of(mesh)
    why = "no cut held up"
    for passes in SMOOTHING:
        try:
            got = _peel_once(mesh, labels, part, inside, loops, surface, step,
                             passes)
        except CutFailed as err:
            why = str(err)
            continue
        piece, rest, _lab = got
        if (piece.open_edges() or rest.open_edges() or piece.bad_edges()
                or rest.bad_edges()):
            why = "the cut left an opening"
            continue
        if piece.volume() <= 0 or rest.volume() <= 0:
            why = "the cut leaves a sliver turned inside out"
            continue
        return got
    raise CutFailed(why)


def _peel_once(mesh, labels, part, inside, loops, surface, step, passes):
    if passes:
        curves = [seam.smooth(mesh.v[loop], surface, step, passes=passes)
                  for loop in loops]
    else:
        # The painted border itself, evenly spaced: no smoothing at all.
        curves = []
        for loop in loops:
            pts = seam.resample(mesh.v[loop], step)
            on, face, _d = surface.nearest(pts)
            curves.append((on, surface.normals[face]))
    field = signed_distance(mesh, inside, curves, step)
    points, tri_pos, tri_neg, src_pos, src_neg = split(mesh, field)
    if not len(tri_pos) or not len(tri_neg):
        raise CutFailed("the seam does not divide the piece")

    # One skin per opening, seen from both sides.
    cap_pos, cap_neg, extra = [], [], []
    base = len(points)
    for loop in open_loops(tri_pos):
        rim = list(reversed(loop))      # the skin runs each rim edge back
        new, tris = skin(points[rim])
        ids = np.concatenate([np.array(rim), base + np.arange(len(new))])
        base += len(new)
        extra.append(new)
        mapped = ids[tris]
        cap_pos.append(mapped)
        cap_neg.append(mapped[:, ::-1])
    if not cap_pos:
        raise CutFailed("no opening to close")
    points = np.vstack([points] + extra)
    cap_pos, cap_neg = np.vstack(cap_pos), np.vstack(cap_neg)

    piece = _compact(points, np.vstack([tri_pos, cap_pos]), np.concatenate(
        [mesh.cap[src_pos], np.ones(len(cap_pos), dtype=bool)]))
    rest = _compact(points, np.vstack([tri_neg, cap_neg]), np.concatenate(
        [mesh.cap[src_neg], np.ones(len(cap_neg), dtype=bool)]))

    # The rest keeps its parts; faces that came over to this side from the
    # part peeled, and the new skin, take the part of what they touch.
    rest_labels = np.concatenate([labels[src_neg],
                                  np.full(len(cap_neg), -1)])
    rest_labels[rest_labels == part] = -1
    rest_labels = _flood(rest, rest_labels)
    return piece, rest, rest_labels


def _flood(mesh, labels):
    """Fill every -1 from labelled neighbours, the most common first."""
    labels = labels.copy()
    _eid, fa, fb = mesh.edge_faces()
    for _ in range(10000):
        hole = labels < 0
        if not hole.any():
            return labels
        votes = {}
        for x, y in ((fa, fb), (fb, fa)):
            m = hole[x] & ~hole[y]
            for a, l in zip(x[m].tolist(), labels[y[m]].tolist()):
                votes.setdefault(a, []).append(l)
        if not votes:
            labels[hole] = 0
            return labels
        for a, vs in votes.items():
            vals, cnt = np.unique(vs, return_counts=True)
            labels[a] = vals[cnt.argmax()]
    return labels
