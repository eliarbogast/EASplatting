"""scripts/download.py: resumable download against a local Range-capable server, local unzip, and
range-extraction of parts of a zip (stored + deflated, zip64, resume, CRC)."""

import os
import subprocess
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from download import Source, central_directory, download, extract, unzip  # noqa: E402

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
    cmd = [sys.executable, str(ROOT / "scripts" / "download.py"), "get", "-q"]
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


def make_zip(path, force_zip64):
    files = {
        "bonsai/sparse/0/cameras.bin": os.urandom(1000),
        "bonsai/images_2/a.jpg": b"A" * 300_000,                    # compressible -> deflate
        "bonsai/images_2/b.jpg": os.urandom(50_000),
        "bonsai/images/a.jpg": os.urandom(10),
        "garden/sparse/0/cameras.bin": os.urandom(10),
    }
    with zipfile.ZipFile(path, "w", allowZip64=True) as z:
        for i, (name, data) in enumerate(files.items()):
            zi = zipfile.ZipInfo(name)
            zi.compress_type = zipfile.ZIP_DEFLATED if i % 2 else zipfile.ZIP_STORED
            with z.open(zi, "w", force_zip64=force_zip64) as f:
                f.write(data)
    return files


def test_extract_prefixes_zip64_and_resume(tmp_path, monkeypatch):
    for z64 in (False, True):
        zp = tmp_path / f"t{int(z64)}.zip"
        if z64:
            # Shrink the zip64 thresholds so zipfile writes real zip64 records (0xFFFFFFFF sizes/offsets
            # + zip64 end-of-central-directory), like the 12 GB 360_v2.zip.
            monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 1000)
            monkeypatch.setattr(zipfile, "ZIP_FILECOUNT_LIMIT", 2)
        files = make_zip(zp, z64)
        if z64:
            raw = zp.read_bytes()
            assert b"PK\x06\x06" in raw and b"PK\x06\x07" in raw         # zip64 EOCD + locator present
        src = Source(str(zp))
        members = central_directory(src)
        assert {m["name"] for m in members} == set(files)
        sel = [m for m in members if m["name"].startswith(("bonsai/sparse/", "bonsai/images_2/"))]
        out = tmp_path / f"out{int(z64)}"
        assert sum(extract(src, m, out) for m in sel) == sum(len(files[m["name"]]) for m in sel)
        for m in sel:
            assert (out / m["name"]).read_bytes() == files[m["name"]]
        assert not (out / "bonsai/images/a.jpg").exists() and not (out / "garden").exists()
        assert sum(extract(src, m, out) for m in sel) == 0            # resume: nothing re-downloaded


def test_cli_list(tmp_path):
    zp = tmp_path / "t.zip"
    make_zip(zp, False)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "download.py"), "zip-members", str(zp),
                        "--list", "--prefix", "bonsai/"], capture_output=True, text=True)
    assert r.returncode == 0 and "selected 4" in r.stdout
