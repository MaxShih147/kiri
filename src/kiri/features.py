"""What the segmentation measures: a coarse stand-in of the model, how thick
the solid is under each of its faces, and how sharply it folds inward across
each of its edges.

* Thickness is the shape diameter function of Shapira, Shamir and Cohen-Or,
  "Consistent mesh partitioning and skeletonisation using the shape diameter
  function" (The Visual Computer, 2008): rays in a cone opposite the normal,
  outliers dropped, a weighted mean, log-normalised.
* Concavity is the angular distance of Katz and Tal, "Hierarchical mesh
  decomposition using fuzzy clustering and cuts" (SIGGRAPH 2003):
  eta * (1 - cos angle), with a small eta where the fold is convex."""

import math

import fast_simplification
import numpy as np
import trimesh

from .meshio import Mesh

# Shapira et al. use a 120 degree cone and 30 rays, and alpha = 4.
CONE_DEGREES = 120.0
RAYS = 30
LOG_ALPHA = 4.0
# Katz and Tal: eta = 1 on concave edges, "a small positive" eta on convex.
CONVEX_ETA = 0.2


def coarse(mesh, faces=None):
    """The model reduced to about `faces` triangles (by default 1/150 of it,
    kept between 2,500 and 15,000): everything the segmentation measures is
    measured on this."""
    if faces is None:
        faces = int(np.clip(mesh.n_faces // 150, 2500, 15000))
    if mesh.n_faces <= faces:
        return Mesh(mesh.v.copy(), mesh.f.copy())
    v, f = fast_simplification.simplify(
        mesh.v.astype(np.float32), mesh.f.astype(np.int32),
        1.0 - faces / mesh.n_faces)
    v, f = np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)
    # Drop anything the reduction left unused or degenerate.
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[keep]
    used, f = np.unique(f, return_inverse=True)
    return Mesh(v[used], f.reshape(-1, 3))


def _cone(count, degrees):
    """`count` unit directions spread evenly over a cone round +z (a
    Fibonacci spiral on the cap), and each one's angle off the axis."""
    half = math.radians(degrees) / 2.0
    i = np.arange(count) + 0.5
    cos_t = 1.0 - (1.0 - math.cos(half)) * i / count
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t ** 2))
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    dirs = np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)
    return dirs, np.arccos(np.clip(cos_t, -1.0, 1.0))


def thickness(mesh):
    """The shape diameter under every face, normalised to [0, 1] in log
    space (so thin parts stay distinct from each other)."""
    centres = mesh.centroids()
    normals = mesh.face_normals()
    inward = -normals
    # A frame per face with z along the inward normal.
    helper = np.where(np.abs(inward[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]],
                      [[1.0, 0.0, 0.0]])
    x_axis = np.cross(helper, inward)
    x_axis /= np.linalg.norm(x_axis, axis=1, keepdims=True)
    y_axis = np.cross(inward, x_axis)
    cone, angle = _cone(RAYS, CONE_DEGREES)
    dirs = (cone[None, :, 0:1] * x_axis[:, None]
            + cone[None, :, 1:2] * y_axis[:, None]
            + cone[None, :, 2:3] * inward[:, None])
    weight = 1.0 / np.maximum(angle, 1e-3)

    start = centres + inward * (mesh.size() * 1e-6)
    origins = np.repeat(start, RAYS, axis=0)
    rays = dirs.reshape(-1, 3)
    tm = trimesh.Trimesh(mesh.v, mesh.f, process=False)
    hit = tm.ray.intersects_first(origins, rays)
    depth = np.full(len(rays), np.nan)
    ok = hit >= 0
    if ok.any():
        n_hit = normals[hit[ok]]
        # Only rays that leave through the far wall measure the solid.
        through = (n_hit * rays[ok]).sum(axis=1) > 0.0
        plane = (n_hit * (mesh.v[mesh.f[hit[ok], 0]] - origins[ok])).sum(1)
        cos = (n_hit * rays[ok]).sum(axis=1)
        t = np.where(through, plane / np.where(cos == 0, 1.0, cos), np.nan)
        depth[np.flatnonzero(ok)] = np.where(t > 0, t, np.nan)
    depth = depth.reshape(-1, RAYS)

    # Keep rays within one standard deviation of the median, then the
    # angle-weighted mean (Shapira et al.).
    med = np.nanmedian(np.where(np.isnan(depth).all(1, keepdims=True), 0.0,
                                depth), axis=1, keepdims=True)
    std = np.nanstd(np.where(np.isnan(depth).all(1, keepdims=True), 0.0,
                             depth), axis=1, keepdims=True)
    keep = ~np.isnan(depth) & (np.abs(depth - med) <= std + 1e-12)
    w = np.where(keep, weight[None, :], 0.0)
    total = w.sum(axis=1)
    sdf = np.where(total > 0,
                   (np.where(keep, depth, 0.0) * w).sum(1)
                   / np.where(total > 0, total, 1.0), np.nan)
    if np.isnan(sdf).all():
        return np.zeros(mesh.n_faces)
    sdf = np.where(np.isnan(sdf), np.nanmedian(sdf), sdf)
    lo, hi = sdf.min(), sdf.max()
    if hi - lo < 1e-12:
        return np.zeros(mesh.n_faces)
    unit = (sdf - lo) / (hi - lo)
    return np.log1p(unit * LOG_ALPHA) / math.log1p(LOG_ALPHA)


def smooth_on_faces(mesh, values, rounds=3, contrast=0.1):
    """Bilateral smoothing across shared edges: neighbours pull a face's
    value toward theirs, less the more they differ, so steps survive."""
    eid, fa, fb = mesh.edge_faces()
    length = mesh.edge_lengths()[eid]
    out = values.astype(np.float64).copy()
    n = mesh.n_faces
    for _ in range(rounds):
        w = length * np.exp(-np.abs(out[fa] - out[fb]) / contrast)
        acc = np.bincount(fa, w * out[fb], n) + np.bincount(fb, w * out[fa], n)
        tot = np.bincount(fa, w, n) + np.bincount(fb, w, n)
        # Each face also holds on to its own value, as strongly as an
        # average face is pulled by its neighbours.
        own = tot.mean()
        out = (acc + own * out) / (tot + own)
    return out


def concavity(mesh):
    """For every edge shared by two faces: (edge ids, face a, face b,
    angular distance). Concave folds count in full, convex ones by eta."""
    eid, fa, fb = mesh.edge_faces()
    n = mesh.face_normals()
    c = mesh.centroids()
    cos = np.clip((n[fa] * n[fb]).sum(axis=1), -1.0, 1.0)
    concave = ((c[fb] - c[fa]) * n[fa]).sum(axis=1) > 0.0
    eta = np.where(concave, 1.0, CONVEX_ETA)
    return eid, fa, fb, eta * (1.0 - cos)
