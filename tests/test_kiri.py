"""End-to-end checks on shapes made here, so no outside file is needed."""

import numpy as np
import pytest
import trimesh

from kiri.meshio import Mesh, components, to_zip
from kiri.nearest import Surface
from kiri.pipeline import split_into_pieces
from kiri.segment import scan


def _mesh(t):
    return Mesh(np.asarray(t.vertices, float), np.asarray(t.faces, np.int64))


@pytest.fixture(scope="module")
def dumbbell():
    """Two balls joined by a thin bar: two parts with a clear neck."""
    from skimage.measure import marching_cubes
    h = 0.05
    x, y, z = np.mgrid[-3.4:3.4:h, -1.2:1.2:h, -1.2:1.2:h]
    ball = lambda cx: np.sqrt((x - cx) ** 2 + y ** 2 + z ** 2) - 1.0
    bar = np.maximum(np.sqrt(y ** 2 + z ** 2) - 0.3, np.abs(x) - 2.0)
    d = np.minimum(np.minimum(ball(-2.2), ball(2.2)), bar)
    # Off zero, so no corner of the grid lies exactly on the surface.
    v, f, _n, _ = marching_cubes(d, h * 1e-3, spacing=(h, h, h))
    v += [-3.4, -1.2, -1.2]
    t = trimesh.Trimesh(v, f, process=True)
    assert t.is_watertight and t.volume > 0
    return _mesh(t)


def test_nearest_matches_brute_force():
    rng = np.random.default_rng(1)
    t = trimesh.creation.icosphere(subdivisions=3)
    m = _mesh(t)
    q = rng.normal(size=(500, 3)) * 1.5
    pts, face, dist = Surface.of(m).nearest(q)
    tri = m.v[m.f]
    best = np.full(len(q), np.inf)
    for i in range(len(tri)):
        p = trimesh.triangles.closest_point(np.repeat(tri[i:i + 1], len(q), 0), q)
        best = np.minimum(best, np.linalg.norm(p - q, axis=1))
    assert np.allclose(dist, best, atol=1e-12)
    assert np.allclose(np.linalg.norm(pts - q, axis=1), dist, atol=1e-12)


def test_components_numbered_by_first_node():
    lab = components(5, np.array([3, 1]), np.array([4, 2]))
    assert lab.tolist() == [0, 1, 1, 2, 2]


def test_scan_finds_the_neck(dumbbell):
    s = scan(dumbbell, log=lambda *_: None)
    lab = s.labels(3)
    assert len(np.unique(lab)) == 3
    # Each ball lies (almost) wholly in one part.
    x = dumbbell.centroids()[:, 0]
    for side in (x < -2.4, x > 2.4):
        share = np.bincount(lab[side]).max() / side.sum()
        assert share > 0.95


@pytest.mark.parametrize("k", [2, 3])
def test_pieces_are_closed_and_keep_the_volume(dumbbell, k):
    s = scan(dumbbell, log=lambda *_: None)
    pieces, refused = split_into_pieces(dumbbell, s.labels(k))
    assert not refused
    assert len(pieces) == k
    for p in pieces:
        assert p.open_edges() == 0
        assert p.bad_edges() == 0
        assert p.volume() > 0
        assert p.cap.any()
    total = sum(p.volume() for p in pieces)
    assert total == pytest.approx(dumbbell.volume(), rel=1e-9)


def test_zip_holds_one_stl_per_piece(dumbbell):
    import io
    import zipfile
    s = scan(dumbbell, log=lambda *_: None)
    pieces, _ = split_into_pieces(dumbbell, s.labels(2))
    names = zipfile.ZipFile(io.BytesIO(to_zip(pieces, "db"))).namelist()
    assert names == ["db_01.stl", "db_02.stl"]
