"""scripts/fetch_zip_members.py on local zips (stored + deflated, zip64, resume, CRC)."""

import os
import subprocess
import sys
import zipfile

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from fetch_zip_members import Source, central_directory, extract  # noqa: E402


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
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_zip_members.py"), str(zp),
                        "--list", "--prefix", "bonsai/"], capture_output=True, text=True)
    assert r.returncode == 0 and "selected 4" in r.stdout
