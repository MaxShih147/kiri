"""The browser page's back end: one model at a time, held in memory.

  POST /api/model        the model file as the body (?name=...): load and scan
  GET  /api/state        what is going on, as JSON
  GET  /api/mesh         the model: header JSON line, then float32 positions
                         and uint32 triangles
  GET  /api/parts?k=N    uint16 part per face
  POST /api/cut?k=N      cut into pieces in the background
  GET  /api/piece?i=N    one piece, laid out like /api/mesh plus a uint8
                         skin flag per face
  GET  /api/zip          every piece as STL in one archive

Scanning and cutting run on a worker thread; the page polls /api/state."""

import json
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np


class Session:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self, name=""):
        self.name = name
        self.stage = "empty"       # empty, scanning, ready, cutting, cut, failed
        self.message = ""
        self.mesh = None
        self.scan = None
        self.pieces = []
        self.cut_k = None
        self.refused = []
        self.cuts_done = 0
        self.started = time.time()
        self.job = object()        # identity of the current job

    def state(self):
        out = {"stage": self.stage, "message": self.message, "name": self.name,
               "seconds": round(time.time() - self.started, 1)}
        if self.mesh is not None:
            out["faces"] = int(self.mesh.n_faces)
            out["open_edges"] = int(self.mesh.open_edges())
        if self.scan is not None:
            out["suggested"] = self.scan.suggested()
            out["max_parts"] = int(min(40, self.scan.coarse.n_faces - 1))
        if self.stage in ("cutting", "cut"):
            out["k"] = self.cut_k
            out["cuts"] = self.cuts_done
        if self.stage == "cut":
            out["pieces"] = [{"faces": int(p.n_faces),
                              "volume": float(p.volume())}
                             for p in self.pieces]
            out["refused"] = [{"piece": i, "parts": n, "why": w}
                              for i, n, w in self.refused]
        return out


SESSION = Session()


def _load_and_scan(data, name, job):
    from .meshio import load
    from .segment import scan
    s = SESSION
    try:
        s.message = "reading the file"
        mesh = load(BytesIO(data), file_type=Path(name).suffix.lstrip(".")
                    or "stl")
        with s.lock:
            if s.job is not job:
                return
            s.mesh = mesh
            s.message = "measuring the shape"
        result = scan(mesh, log=lambda m: setattr(s, "message", m))
        with s.lock:
            if s.job is not job:
                return
            s.scan = result
            s.stage = "ready"
            s.message = "scanned in %.1fs" % result.seconds
    except Exception as err:
        traceback.print_exc()
        with s.lock:
            if s.job is job:
                s.stage, s.message = "failed", str(err)


def _cut(k, job):
    from .pipeline import split_into_pieces
    s = SESSION

    def progress(n):
        s.cuts_done = n
        s.message = "%d cuts made" % n
    try:
        s.message = "placing the parts"
        labels = s.scan.labels(k)
        s.message = "cutting"
        pieces, refused = split_into_pieces(s.mesh, labels, progress=progress)
        with s.lock:
            if s.job is not job:
                return
            s.pieces, s.refused = pieces, refused
            s.stage = "cut"
            s.message = "%d pieces" % len(pieces)
    except Exception as err:
        traceback.print_exc()
        with s.lock:
            if s.job is job:
                s.stage, s.message = "ready", "cutting failed: %s" % err


def _mesh_bytes(mesh, extra=None):
    head = {"vertices": len(mesh.v), "faces": mesh.n_faces,
            "lo": mesh.v.min(axis=0).tolist(), "hi": mesh.v.max(axis=0).tolist()}
    parts = [json.dumps(head).encode() + b"\n",
             mesh.v.astype("<f4").tobytes(), mesh.f.astype("<u4").tobytes()]
    if extra is not None:
        parts.append(extra.astype("u1").tobytes())
    return b"".join(parts)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, body, kind="application/octet-stream", code=200,
              extra_headers=()):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in extra_headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj).encode(), "application/json", code)

    def _query(self):
        u = urlparse(self.path)
        return u.path, {k: v[0] for k, v in parse_qs(u.query).items()}

    def do_GET(self):
        path, q = self._query()
        s = SESSION
        if path in ("/", "/index.html"):
            page = resources.files("kiri").joinpath("web/index.html")
            return self._send(page.read_bytes(), "text/html; charset=utf-8")
        if path == "/favicon.ico":
            return self._send(b"", "image/x-icon", 204)
        if path == "/api/state":
            return self._json(s.state())
        if path == "/api/mesh":
            if s.mesh is None:
                return self._json({"error": "no model"}, 404)
            return self._send(_mesh_bytes(s.mesh))
        if path == "/api/parts":
            if s.scan is None:
                return self._json({"error": "not scanned"}, 409)
            k = int(q.get("k", s.scan.suggested() or 2))
            return self._send(s.scan.labels(k).astype("<u2").tobytes())
        if path == "/api/piece":
            i = int(q.get("i", 0))
            if s.stage != "cut" or not 0 <= i < len(s.pieces):
                return self._json({"error": "no such piece"}, 404)
            p = s.pieces[i]
            return self._send(_mesh_bytes(p, p.cap))
        if path == "/api/zip":
            if s.stage != "cut":
                return self._json({"error": "not cut"}, 409)
            from .meshio import to_zip
            stem = Path(s.name).stem or "piece"
            return self._send(to_zip(s.pieces, stem), "application/zip",
                              extra_headers=[("Content-Disposition",
                                              'attachment; filename="%s_%d.zip"'
                                              % (stem, len(s.pieces)))])
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        path, q = self._query()
        s = SESSION
        if path == "/api/model":
            n = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(n)
            name = q.get("name", "model.stl")
            with s.lock:
                s.reset(name)
                s.stage, job = "scanning", s.job
            threading.Thread(target=_load_and_scan, args=(data, name, job),
                             daemon=True).start()
            return self._json(s.state())
        if path == "/api/cut":
            with s.lock:
                if s.scan is None or s.stage == "cutting":
                    return self._json({"error": "not ready"}, 409)
                k = int(q.get("k", s.scan.suggested() or 2))
                s.job = job = object()
                s.stage, s.cut_k, s.cuts_done, s.pieces = "cutting", k, 0, []
                s.started = time.time()
            threading.Thread(target=_cut, args=(k, job), daemon=True).start()
            return self._json(s.state())
        self._json({"error": "not found"}, 404)


def serve(host="127.0.0.1", port=8765, open_browser=True):
    # Another program on the port: take the next free one.
    for tries in range(20):
        try:
            httpd = ThreadingHTTPServer((host, port), Handler)
            break
        except OSError:
            if tries == 19:
                raise
            port += 1
    url = "http://%s:%d/" % ("localhost" if host in ("127.0.0.1", "0.0.0.0")
                             else host, port)
    print("kiri is at %s  (Ctrl+C to stop)" % url)
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
