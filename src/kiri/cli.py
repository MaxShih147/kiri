"""kiri on the command line:

  kiri scan  MODEL            how the model would part, and a suggested count
  kiri cut   MODEL [-k N]     cut it into closed pieces, written as STL files
  kiri web                    the same in a browser page
"""

import argparse
import sys
import time
from pathlib import Path


def _scan(path, log):
    from .meshio import load
    from .segment import scan
    mesh = load(path)
    log("%s: %d faces, %s" % (Path(path).name, mesh.n_faces,
                             "closed" if mesh.open_edges() == 0
                             else "%d open edges" % mesh.open_edges()))
    return mesh, scan(mesh, log=log)


def cmd_scan(args):
    _mesh, s = _scan(args.model, print)
    print("suggested part count: %s" % s.suggested())
    n = s.coarse.n_faces
    print("\n parts   cost of the next merge")
    for k in range(2, 17):
        i = n - k
        if 0 <= i < len(s.costs):
            print("  %3d    %.3g" % (k, s.costs[i]))


def cmd_cut(args):
    from .meshio import to_stl, to_zip
    from .pipeline import split_into_pieces
    mesh, s = _scan(args.model, print)
    k = args.parts or s.suggested() or 2
    t = time.time()
    labels = s.labels(k)
    print("parts: %d (%.1fs)" % (k, time.time() - t))
    t = time.time()
    pieces, refused = split_into_pieces(
        mesh, labels, progress=lambda n: print("  cut %d" % n, end="\r"))
    print("pieces: %d (%.1fs)" % (len(pieces), time.time() - t))
    for i, n, why in refused:
        print("  piece %d still holds %d parts: %s" % (i + 1, n, "; ".join(why)))
    out = Path(args.out or Path(args.model).stem + "_pieces")
    stem = Path(args.model).stem
    if args.zip:
        out.parent.mkdir(parents=True, exist_ok=True)
        target = out.with_suffix(".zip")
        target.write_bytes(to_zip(pieces, stem))
        print("wrote %s" % target)
    else:
        out.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(pieces, 1):
            (out / ("%s_%02d.stl" % (stem, i))).write_bytes(to_stl(p))
        print("wrote %d files to %s/" % (len(pieces), out))
    lost = abs(sum(p.volume() for p in pieces) - mesh.volume())
    print("volume kept to %.2g%%" % (100 * lost / max(abs(mesh.volume()), 1e-12)))


def cmd_web(args):
    from .server import serve
    serve(args.host, args.port, open_browser=not args.no_browser)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kiri",
                                 description="Cut a 3D model into closed parts.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan", help="measure a model and suggest a part count")
    p.add_argument("model")
    p.set_defaults(run=cmd_scan)
    p = sub.add_parser("cut", help="cut a model into closed STL pieces")
    p.add_argument("model")
    p.add_argument("-k", "--parts", type=int, help="number of parts "
                   "(default: the suggested count)")
    p.add_argument("-o", "--out", help="output folder (default: MODEL_pieces)")
    p.add_argument("--zip", action="store_true", help="write one zip instead")
    p.set_defaults(run=cmd_cut)
    p = sub.add_parser("web", help="open the browser page")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(run=cmd_web)
    args = ap.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    sys.exit(main())
