"""Parts by region merging: every face of the coarse model starts as its own
region, and the two neighbouring regions whose shared border least looks like
a place to cut are merged, over and over, until one region is left. The order
of merges is kept, so any number of parts is a replay of a prefix.

How much a border looks like a cut combines what the measures say:
  contrast  -- how concave the border runs, and how different the two sides'
               thickness is;
  size      -- Ward's variance increase, aA*aB/(aA+aB): small regions merge
               early, whatever their border;
  enclosure -- the share of the smaller region's own outline this border is:
               a region all but surrounded by another is a fragment of it."""

import heapq
import time
from dataclasses import dataclass, field

import numpy as np

from . import features
from .meshio import Mesh, components
from .nearest import Surface

THICKNESS_WEIGHT = 2.0
FRAGMENT_SHARE = 0.0025


@dataclass
class Scan:
    mesh: Mesh
    coarse: Mesh
    merges: np.ndarray          # (n_coarse - 1, 2) region ids, in order
    costs: np.ndarray
    to_coarse: np.ndarray       # coarse face under each face of `mesh`
    seconds: float
    _labels: dict = field(default_factory=dict, repr=False)

    def coarse_labels(self, k):
        """Coarse face -> part, for `k` parts (0..k-1)."""
        n = self.coarse.n_faces
        parent = np.arange(n)

        def root(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in self.merges[:max(0, n - int(k))].tolist():
            ra, rb = root(a), root(b)
            if ra != rb:
                parent[rb] = ra
        roots = np.array([root(i) for i in range(n)])
        return np.unique(roots, return_inverse=True)[1]

    def labels(self, k, refine=True):
        """Face -> part on the full model, for `k` parts."""
        key = (int(k), bool(refine))
        got = self._labels.get(key)
        if got is None:
            lab = self.coarse_labels(k)[self.to_coarse]
            lab = tidy(self.mesh, lab)
            if refine:
                from .refine import snap_to_folds
                lab = tidy(self.mesh, snap_to_folds(self.mesh, lab))
            got = self._labels[key] = np.unique(lab, return_inverse=True)[1]
        return got

    def suggested(self, lo=2, hi=24):
        """The part count just before the steepest rise in merge cost."""
        n = self.coarse.n_faces
        best, best_k = 1.0, None
        for k in range(lo, min(hi, n - 1) + 1):
            i = n - k          # the merge that would leave k - 1 parts
            if i < 1 or i >= len(self.costs):
                continue
            prev = self.costs[i - 1]
            if prev > 1e-12 and self.costs[i] / prev > best:
                best, best_k = self.costs[i] / prev, k
        return best_k


def _merge_order(coarse, sdf):
    """The merge sequence and its costs."""
    n = coarse.n_faces
    eid, fa, fb, ang = features.concavity(coarse)
    length = coarse.edge_lengths()[eid]
    area = coarse.face_areas().astype(np.float64).copy()
    total = area.sum()
    thick = sdf.astype(np.float64).copy()

    # Per region: neighbour -> [shared length, length-weighted concavity].
    link = [dict() for _ in range(n)]
    for a, b, L, c in zip(fa.tolist(), fb.tolist(), length.tolist(),
                          (ang * length).tolist()):
        for x, y in ((a, b), (b, a)):
            got = link[x].get(y)
            if got is None:
                link[x][y] = [L, c]
            else:
                got[0] += L
                got[1] += c
    # A shell too small to be a part (an eye, a bead) is tied to the nearest
    # face of a larger one, so it merges early as a fragment rather than
    # standing as a part of its own until the last merges.
    shell = components(n, fa, fb)
    shell_area = np.bincount(shell, weights=area)
    small = shell_area < FRAGMENT_SHARE * total
    if small.any() and (~small).any():
        from scipy.spatial import cKDTree
        cen = coarse.centroids()
        big = np.flatnonzero(np.logical_not(small[shell]))
        tree = cKDTree(cen[big])
        tie = float(np.median(length))
        for s_id in np.flatnonzero(small).tolist():
            members = np.flatnonzero(shell == s_id)
            d, j = tree.query(cen[members])
            a, b = int(members[np.argmin(d)]), int(big[j[np.argmin(d)]])
            shared = [tie, 0.0]
            link[a][b] = shared
            link[b][a] = shared
    outline = np.array([sum(v[0] for v in d.values()) for d in link])
    alive = np.ones(n, dtype=bool)
    stamp = np.zeros(n, dtype=np.int64)

    def cost(a, b):
        L, c = link[a][b]
        contrast = c / max(L, 1e-12) + THICKNESS_WEIGHT * abs(thick[a] - thick[b])
        size = area[a] * area[b] / (area[a] + area[b]) / total
        enclosure = min(1.0, L / max(min(outline[a], outline[b]), 1e-12))
        fragment = min(area[a], area[b]) < FRAGMENT_SHARE * total
        return (0 if fragment else 1), contrast * size * (1.0 - 0.5 * enclosure)

    heap = []
    for a in range(n):
        for b in link[a]:
            if a < b:
                tier, c = cost(a, b)
                heapq.heappush(heap, (tier, c, a, b, 0, 0))
    merges, costs = [], []
    while heap:
        tier, c, a, b, sa, sb = heapq.heappop(heap)
        if not (alive[a] and alive[b]) or stamp[a] != sa or stamp[b] != sb:
            continue
        # Merge b into a.
        w = area[a] + area[b]
        thick[a] = (thick[a] * area[a] + thick[b] * area[b]) / w
        area[a] = w
        shared = link[a].pop(b)[0]
        link[b].pop(a)
        outline[a] += outline[b] - 2.0 * shared
        for nb, (L, cc) in link[b].items():
            link[nb].pop(b)
            if nb in link[a]:
                link[a][nb][0] += L
                link[a][nb][1] += cc
            else:
                link[a][nb] = [L, cc]
            link[nb][a] = link[a][nb]
        link[b] = {}
        alive[b] = False
        stamp[a] += 1
        merges.append((a, b))
        costs.append(c if tier else 0.0)
        for nb in link[a]:
            t2, c2 = cost(a, nb)
            heapq.heappush(heap, (t2, c2, min(a, nb), max(a, nb),
                                  stamp[min(a, nb)], stamp[max(a, nb)]))

    # Separate shells: join them last, nearest first.
    rest = np.flatnonzero(alive)
    if len(rest) > 1:
        cen = coarse.centroids()
        top = max(costs) if costs else 1.0
        anchor = rest[0]
        for r in sorted(rest[1:], key=lambda r: np.linalg.norm(cen[r] - cen[anchor])):
            top *= 1.05
            merges.append((anchor, r))
            costs.append(top)
    return np.array(merges, dtype=np.int64).reshape(-1, 2), np.array(costs)


def tidy(mesh, labels):
    """Every part one connected patch: smaller patches of a part go to the
    part they border most."""
    labels = labels.copy()
    eid, fa, fb = mesh.edge_faces()
    for _ in range(6):
        same = labels[fa] == labels[fb]
        patch = components(mesh.n_faces, fa[same], fb[same])
        size = np.bincount(patch)
        # The largest patch of each label keeps it.
        owner = np.zeros(len(size), dtype=np.int64)
        np.maximum.at(owner, patch, labels)   # every face of a patch shares it
        keep = np.zeros(len(size), dtype=bool)
        order = np.lexsort((-size, owner))
        first = np.ones(len(order), dtype=bool)
        first[1:] = owner[order][1:] != owner[order][:-1]
        keep[order[first]] = True
        stray = ~keep[patch]
        if not stray.any():
            return labels
        cross = patch[fa] != patch[fb]
        moved = False
        votes = {}
        for a, b in zip(fa[cross].tolist(), fb[cross].tolist()):
            for x, y in ((a, b), (b, a)):
                if stray[x]:
                    votes.setdefault(int(patch[x]), []).append(int(labels[y]))
        for p, vs in votes.items():
            vals, cnt = np.unique(vs, return_counts=True)
            labels[patch == p] = vals[cnt.argmax()]
            moved = True
        if not moved:
            return labels
    return labels


def scan(mesh, coarse_faces=None, log=print):
    """Measure the model and work out the whole merge order."""
    t0 = time.time()
    small = features.coarse(mesh, coarse_faces)
    log("coarse model: %d faces (%.1fs)" % (small.n_faces, time.time() - t0))
    t = time.time()
    sdf = features.smooth_on_faces(small, features.thickness(small))
    log("thickness: %.1fs" % (time.time() - t))
    t = time.time()
    merges, costs = _merge_order(small, sdf)
    log("merge order: %.1fs" % (time.time() - t))
    t = time.time()
    to_coarse = Surface.of(small).nearest(mesh.centroids())[1]
    log("mapped back: %.1fs" % (time.time() - t))
    return Scan(mesh, small, merges, costs, to_coarse, time.time() - t0)
