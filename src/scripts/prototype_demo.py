"""Read-only local/Colab viewer for one prototype run; no GPU or React required."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit

from scripts.prototype_artifacts import read_state, viewer_html


def make_server(root: Path, host: str = "127.0.0.1", port: int = 8765):
    root = root.resolve()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            try:
                if path in ("/", "/demo.html"):
                    body = viewer_html(read_state(root)).encode()
                    content_type = "text/html; charset=utf-8"
                elif path == "/api/state":
                    body = json.dumps(read_state(root)).encode()
                    content_type = "application/json"
                elif path == "/favicon.ico":
                    self.send_response(204)
                    self.end_headers()
                    return
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (OSError, ValueError) as e:
                self.send_error(503, "Artifact temporarily unavailable")

        def log_message(self, *args):
            pass
    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    if not (args.run_dir / "manifest.json").is_file():
        ap.error("run directory must contain manifest.json")
    server = make_server(args.run_dir, args.host, args.port)
    print(f"Prototype demo: http://{args.host}:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
