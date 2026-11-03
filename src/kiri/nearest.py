"""Exact nearest point on a triangle mesh, for many points at once.

A bounding-volume hierarchy split at the midpoint of each node's longest
axis, searched depth first with the nearer child first, on every core."""

import numpy as np
from numba import njit, prange

_LEAF = 6


@njit(cache=True)
def _point_triangle(p, a, b, c):
    """Closest point to `p` on triangle abc (Voronoi regions of the
    triangle's features, as in Ericson, Real-Time Collision Detection)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = ab @ ap
    d2 = ac @ ap
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = p - b
    d3 = ab @ bp
    d4 = ac @ bp
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + ab * (d1 / (d1 - d3))
    cp = p - c
    d5 = ab @ cp
    d6 = ac @ cp
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + ac * (d2 / (d2 - d6))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))
    s = 1.0 / (va + vb + vc)
    return a + ab * (vb * s) + ac * (vc * s)


@njit(cache=True)
def _build(lo, hi, mid):
    """Nodes as (box low, box high, first child or -1, item range)."""
    n = lo.shape[0]
    items = np.arange(n)
    cap = 2 * n
    blo = np.empty((cap, 3))
    bhi = np.empty((cap, 3))
    child = np.full(cap, -1, np.int64)
    first = np.zeros(cap, np.int64)
    last = np.zeros(cap, np.int64)
    todo = np.empty((cap, 3), np.int64)
    todo[0] = (0, 0, n)
    top, made = 1, 1
    while top:
        top -= 1
        node, s, e = todo[top]
        box_lo = lo[items[s]].copy()
        box_hi = hi[items[s]].copy()
        for i in range(s + 1, e):
            box_lo = np.minimum(box_lo, lo[items[i]])
            box_hi = np.maximum(box_hi, hi[items[i]])
        blo[node], bhi[node] = box_lo, box_hi
        first[node], last[node] = s, e
        if e - s <= _LEAF:
            continue
        ext = box_hi - box_lo
        axis = np.argmax(ext)
        cut = 0.5 * (box_lo[axis] + box_hi[axis])
        i, j = s, e - 1
        while i <= j:
            if mid[items[i], axis] < cut:
                i += 1
            else:
                items[i], items[j] = items[j], items[i]
                j -= 1
        if i == s or i == e:          # all on one side: split by count
            i = (s + e) // 2
        child[node] = made
        todo[top] = (made, s, i)
        todo[top + 1] = (made + 1, i, e)
        top += 2
        made += 2
    return blo[:made], bhi[:made], child[:made], first[:made], last[:made], items


@njit(cache=True)
def _box_gap(p, lo, hi):
    g = 0.0
    for k in range(3):
        if p[k] < lo[k]:
            g += (lo[k] - p[k]) ** 2
        elif p[k] > hi[k]:
            g += (p[k] - hi[k]) ** 2
    return g


@njit(cache=True, parallel=True)
def _query(pts, a, b, c, blo, bhi, child, first, last, items):
    n = pts.shape[0]
    out_p = np.empty((n, 3))
    out_f = np.empty(n, np.int64)
    out_d = np.empty(n)
    for q in prange(n):
        p = pts[q]
        best = np.inf
        best_f = -1
        best_p = p.copy()
        stack = np.empty(96, np.int64)
        stack[0] = 0
        top = 1
        while top:
            top -= 1
            node = stack[top]
            if _box_gap(p, blo[node], bhi[node]) > best:
                continue
            k = child[node]
            if k < 0:
                for i in range(first[node], last[node]):
                    t = items[i]
                    x = _point_triangle(p, a[t], b[t], c[t])
                    d = ((x - p) ** 2).sum()
                    if d < best or (d == best and t < best_f):
                        best, best_f, best_p = d, t, x
                continue
            g0 = _box_gap(p, blo[k], bhi[k])
            g1 = _box_gap(p, blo[k + 1], bhi[k + 1])
            near, far = (k, k + 1) if g0 <= g1 else (k + 1, k)
            stack[top] = far
            stack[top + 1] = near
            top += 2
        out_p[q] = best_p
        out_f[q] = best_f
        out_d[q] = np.sqrt(best)
    return out_p, out_f, out_d


class Surface:
    """Nearest-point queries against a fixed set of triangles."""

    def __init__(self, v, f):
        v = np.asarray(v, dtype=np.float64)
        f = np.asarray(f, dtype=np.int64)
        self.a = np.ascontiguousarray(v[f[:, 0]])
        self.b = np.ascontiguousarray(v[f[:, 1]])
        self.c = np.ascontiguousarray(v[f[:, 2]])
        lo = np.minimum(np.minimum(self.a, self.b), self.c)
        hi = np.maximum(np.maximum(self.a, self.b), self.c)
        mid = (self.a + self.b + self.c) / 3.0
        n = np.cross(self.b - self.a, self.c - self.a)
        length = np.linalg.norm(n, axis=1)
        self.normals = n / np.where(length > 0, length, 1.0)[:, None]
        self._tree = _build(lo, hi, mid)

    @classmethod
    def of(cls, mesh):
        return cls(mesh.v, mesh.f)

    def nearest(self, points):
        """(closest points, face ids, distances)."""
        pts = np.ascontiguousarray(np.asarray(points, dtype=np.float64)
                                   .reshape(-1, 3))
        return _query(pts, self.a, self.b, self.c, *self._tree)
