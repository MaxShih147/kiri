"""Seams: where one part meets the rest of a piece, as closed polylines on the
surface, smoothed so the cut does not follow the staircase of triangle edges
the parts were painted with."""

import numpy as np

from .nearest import Surface

PASSES = 4          # filter, then put back on the surface, this many times
SIGMA = 2.5         # Gaussian width of each filter, in samples


def border_loops(mesh, inside):
    """Closed loops of edges between faces in `inside` (a mask) and the
    rest, each a list of vertex ids walked so that `inside` lies to the left
    (the direction the edge runs in the inside face). Returns [] if the
    border branches anywhere (two loops touching at a vertex are walked
    apart by always taking the leftmost turn)."""
    e, side = mesh.edges()
    eid, fa, fb = mesh.edge_faces()
    cross = inside[fa] != inside[fb]
    if not cross.any():
        return []
    in_face = np.where(inside[fa[cross]], fa[cross], fb[cross])
    edge_ids = eid[cross]
    # The direction each border edge runs in its inside face.
    nxt = {}
    for f, ed in zip(in_face.tolist(), edge_ids.tolist()):
        k = int(np.flatnonzero(side[f] == ed)[0])
        a, b = int(mesh.f[f, k]), int(mesh.f[f, (k + 1) % 3])
        nxt.setdefault(a, []).append(b)
    loops, used = [], set()
    for a0, outs in nxt.items():
        for b0 in outs:
            if (a0, b0) in used:
                continue
            loop, a, b = [a0], a0, b0
            used.add((a, b))
            while b != a0:
                loop.append(b)
                options = [c for c in nxt.get(b, []) if (b, c) not in used]
                if not options:
                    loop = None
                    break
                c = options[0] if len(options) == 1 else _leftmost(
                    mesh, a, b, options)
                used.add((b, c))
                a, b = b, c
            if loop is not None and len(loop) >= 3:
                loops.append(loop)
    return loops


def _leftmost(mesh, a, b, options):
    """At a vertex where the border meets itself, the way on that turns
    most to the left, seen along the surface normal there."""
    n = mesh.vertex_normals()[b]
    come = mesh.v[b] - mesh.v[a]
    best, best_angle = options[0], -np.inf
    for c in options:
        go = mesh.v[c] - mesh.v[b]
        angle = np.arctan2(np.dot(np.cross(come, go), n), np.dot(come, go))
        if angle > best_angle:
            best, best_angle = c, angle
    return best


def resample(points, step):
    """A closed polyline at even spacing (about `step`)."""
    pts = np.array(points, dtype=float)
    seg = np.sqrt(((np.roll(pts, -1, axis=0) - pts) ** 2).sum(axis=1))
    total = seg.sum()
    count = max(16, int(round(total / max(step, 1e-12))))
    at = np.concatenate([[0.0], np.cumsum(seg)])
    want = np.linspace(0.0, total, count, endpoint=False)
    i = np.clip(np.searchsorted(at, want, side="right") - 1, 0, len(pts) - 1)
    t = (want - at[i]) / np.where(seg[i] > 0, seg[i], 1.0)
    j = (i + 1) % len(pts)
    return pts[i] + (pts[j] - pts[i]) * t[:, None]


def _low_pass(pts, sigma):
    """Circular Gaussian filter along a closed polyline."""
    radius = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    out = np.zeros_like(pts)
    for shift, w in zip(range(-radius, radius + 1), k):
        out += w * np.roll(pts, shift, axis=0)
    return out


def smooth(points, surface, step, passes=PASSES, sigma=SIGMA):
    """The loop low-passed and laid back on the surface, a few times over;
    returns the curve and the surface normal under each of its points."""
    pts = resample(points, step)
    normals = None
    for _ in range(passes):
        pts = _low_pass(pts, sigma)
        pts, face, _d = surface.nearest(pts)
        normals = surface.normals[face]
    return pts, normals
