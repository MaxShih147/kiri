"""Closing a cut: a skin stretched over each cut loop.

The loop is projected onto its best-fit plane and filled there with a
quality triangulation (Shewchuk's Triangle: no new points on the rim, so the
skin meets the cut exactly, interior points as needed for well-shaped
triangles). The interior is then lifted off the plane by a harmonic solve
with the rim held fixed: the skin settles as a soap film would on that wire,
following a curved cut instead of cutting across the piece."""

import numpy as np
import triangle
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

MAX_TRIANGLES = 3000


def _frame(points):
    centre = points.mean(axis=0)
    _u, _s, vt = np.linalg.svd(points - centre, full_matrices=False)
    return centre, vt[0], vt[1], vt[2]


def _simple(xy):
    """Whether a closed 2-D polyline crosses itself (a sweep would be
    faster; loops here are a few hundred points)."""
    n = len(xy)
    a = xy
    b = np.roll(xy, -1, axis=0)
    for i in range(n):
        p, q = a[i], b[i]
        # Segments that do not share a vertex with segment i.
        js = np.array([j for j in range(n) if j not in (i, (i + 1) % n,
                                                        (i - 1) % n)])
        if not len(js):
            continue
        r, s = a[js], b[js]

        def orient(u, v, w):
            return (v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1]) - \
                (v[..., 1] - u[..., 1]) * (w[..., 0] - u[..., 0])

        d1 = orient(p, q, r)
        d2 = orient(p, q, s)
        d3 = orient(r, s, p)
        d4 = orient(r, s, q)
        if np.any((d1 * d2 < 0) & (d3 * d4 < 0)):
            return False
    return True


def _harmonic(points, tris, fixed):
    """Positions of the free vertices solving the graph Laplace equation
    with the `fixed` ones held."""
    n = len(points)
    i = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2],
                        tris[:, 1], tris[:, 2], tris[:, 0]])
    j = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0],
                        tris[:, 0], tris[:, 1], tris[:, 2]])
    adj = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    adj.data[:] = 1.0
    deg = np.asarray(adj.sum(axis=1)).ravel()
    free = np.flatnonzero(~fixed)
    if not len(free):
        return points
    lap = (coo_matrix((deg, (np.arange(n), np.arange(n))), shape=(n, n))
           .tocsr() - adj)
    a = lap[free][:, free]
    b = -lap[free][:, np.flatnonzero(fixed)] @ points[fixed]
    out = points.copy()
    sol = spsolve(a.tocsc(), b)
    out[free] = sol.reshape(len(free), 3)
    return out


def skin(rim):
    """Triangles closing the loop `rim` (an (n, 3) array, in order).
    Returns (new points, triangles): triangles index rim points as 0..n-1
    and new points as n.., wound so that the rim runs i -> i+1 along each
    triangle that holds a rim edge.

    A loop that projects onto its best-fit plane without crossing itself is
    meshed in that plane. One that folds over itself is laid out on a circle
    instead, by arc length -- a disk parameterisation (Tutte 1963, Floater
    1997): a circle never crosses itself, so the mesh is always sound -- and
    in both cases the interior is then placed by the harmonic solve."""
    rim = np.asarray(rim, dtype=np.float64)
    n = len(rim)
    if n < 3:
        raise ValueError("a loop needs three points")
    centre, u, v, _w = _frame(rim)
    xy = np.stack([(rim - centre) @ u, (rim - centre) @ v], axis=1)
    if not _simple(xy):
        xy = _circle(rim)
    got = _mesh_disk(xy)
    if got is None and not np.allclose(xy, _circle(rim)):
        xy = _circle(rim)
        got = _mesh_disk(xy)
    if got is not None:
        inner, tris = got
        pts = np.vstack([rim, np.zeros((len(inner), 3))])
        fixed = np.zeros(len(pts), dtype=bool)
        fixed[:n] = True
        pts = _harmonic(pts, tris, fixed)
        return pts[n:], _wind(tris, n)
    # Last resort: a fan from the loop's centre.
    pts = np.vstack([rim, rim.mean(axis=0)])
    tris = np.stack([np.arange(n), (np.arange(n) + 1) % n,
                     np.full(n, n)], axis=1)
    return pts[n:], _wind(tris, n)


def _circle(rim):
    """The loop on a circle of its own length, spaced by arc length."""
    gap = np.linalg.norm(np.roll(rim, -1, axis=0) - rim, axis=1)
    total = gap.sum()
    if total <= 0:
        gap = np.ones(len(rim))
        total = float(len(rim))
    angle = 2 * np.pi * np.concatenate([[0.0], np.cumsum(gap)[:-1]]) / total
    radius = total / (2 * np.pi)
    return radius * np.stack([np.cos(angle), np.sin(angle)], axis=1)


def _mesh_disk(xy):
    """Quality triangles inside the simple polygon `xy`, no new points on
    its edges. Returns (count of interior points, triangles) or None."""
    n = len(xy)
    seg = np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1)
    area = 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], -1))
                     - np.dot(xy[:, 1], np.roll(xy[:, 0], -1)))
    step = np.median(np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1))
    max_area = max(area / MAX_TRIANGLES, 0.5 * step * step)
    try:
        out = triangle.triangulate({"vertices": xy, "segments": seg},
                                   "pq28YQa%.12g" % max_area)
    except Exception:
        return None
    pts2 = out.get("vertices")
    tris = out.get("triangles")
    if (pts2 is None or tris is None or len(pts2) < n
            or not np.allclose(pts2[:n], xy)):
        return None
    return pts2[n:], tris.astype(np.int64)


def _wind(tris, n):
    """Flip all triangles if needed so the rim edge i -> i+1 appears in that
    order in the triangle holding it."""
    for t in tris:
        for k in range(3):
            a, b = t[k], t[(k + 1) % 3]
            if a < n and b < n:
                if (b - a) % n == 1:
                    return tris
                if (a - b) % n == 1:
                    return tris[:, ::-1].copy()
    return tris
