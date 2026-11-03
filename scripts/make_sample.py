"""A test figure for kiri: a torso, head and four limbs blended into one
closed surface (an implicit surface of round-ended cylinders, meshed by
marching cubes), the way a scan or a generated model fuses its parts."""

import sys

import numpy as np
import trimesh
from skimage.measure import marching_cubes

LIMBS = [  # (start, end, radius)
    ((0.0, 0.0, 1.00), (0.0, 0.0, 1.40), 0.27),
    ((0.0, 0.0, 1.62), (0.0, 0.0, 1.70), 0.14),
    ((0.0, 0.0, 1.45), (0.0, 0.0, 1.55), 0.07),
    ((0.28, 0.0, 1.42), (0.75, 0.05, 1.10), 0.07),
    ((-0.28, 0.0, 1.42), (-0.75, 0.05, 1.10), 0.07),
    ((0.12, 0.0, 0.90), (0.15, 0.02, 0.10), 0.09),
    ((-0.12, 0.0, 0.90), (-0.15, 0.02, 0.10), 0.09),
]


def segment_distance(p, a, b, r):
    a, b = np.asarray(a), np.asarray(b)
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1) - r


def blend(d1, d2, k=0.05):
    h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return d2 + (d1 - d2) * h - k * h * (1.0 - h)


def main(path, step=0.008):
    lo, hi = np.array([-1.0, -0.4, -0.05]), np.array([1.0, 0.4, 1.95])
    grid = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i], step)
                                  for i in range(3)], indexing="ij"), -1)
    field = segment_distance(grid, *LIMBS[0])
    for limb in LIMBS[1:]:
        field = blend(field, segment_distance(grid, *limb))
    v, f, _n, _ = marching_cubes(field, step * 1e-3, spacing=(step,) * 3)
    m = trimesh.Trimesh(v + lo, f)
    m.update_faces(m.nondegenerate_faces())
    m.merge_vertices()
    m = max(m.split(only_watertight=False), key=lambda s: len(s.faces))
    if m.volume < 0:
        m.invert()
    m.export(path)
    print("%s: %d faces, closed: %s" % (path, len(m.faces), m.is_watertight))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples/figure.stl")
