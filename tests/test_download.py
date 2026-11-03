"""scripts/download.py: resumable download against a local Range-capable server, and zip extraction."""

import os
import subprocess
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from download import download, unzip  # noqa: E402

PAYLOAD = os.urandom(300_000)


class RangeHandler(BaseHTTPRequestHandler):
    ranges = True

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/missing":
            self.send_error(404)
            return
        start, rng = 0, self.headers.get("Range")
        if rng and self.ranges:
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(PAYLOAD):
                self.send_error(416)
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        else:
            self.send_response(200)
        body = PAYLOAD[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    RangeHandler.ranges = True
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_download_and_resume(server, tmp_path):
    out = tmp_path / "sub" / "f.bin"
    download(f"{server}/f", out, quiet=True)
    assert out.read_bytes() == PAYLOAD

    out.write_bytes(PAYLOAD[:123_456])                       # partial file -> resumes
    download(f"{server}/f", out, quiet=True)
    assert out.read_bytes() == PAYLOAD

    download(f"{server}/f", out, quiet=True)                 # already complete (416) -> untouched
    assert out.read_bytes() == PAYLOAD


def test_server_without_range_restarts(server, tmp_path):
    RangeHandler.ranges = False
    out = tmp_path / "f.bin"
    out.write_bytes(b"garbage prefix")
    download(f"{server}/f", out, quiet=True)
    assert out.read_bytes() == PAYLOAD


def test_cli_exit_codes(server, tmp_path):
    cmd = [sys.executable, str(ROOT / "scripts" / "download.py"), "-q"]
    ok = subprocess.run(cmd + [f"{server}/f", str(tmp_path / "a.bin")])
    bad = subprocess.run(cmd + [f"{server}/missing", str(tmp_path / "b.bin")], capture_output=True, text=True)
    assert ok.returncode == 0 and bad.returncode == 1 and "404" in bad.stderr


def test_unzip_overwrite_modes(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("d/one.txt", "new")
        zf.writestr("two.txt", "new")
    dest = tmp_path / "out"
    (dest / "d").mkdir(parents=True)
    (dest / "d" / "one.txt").write_text("old")

    assert unzip(z, dest, overwrite=False) == 1              # only two.txt is extracted
    assert (dest / "d" / "one.txt").read_text() == "old" and (dest / "two.txt").read_text() == "new"
    unzip(z, dest, overwrite=True)
    assert (dest / "d" / "one.txt").read_text() == "new"
