"""
download.py — resumable download and zip extraction in plain Python (replaces wget / curl / unzip,
which minimal Linux images and cluster nodes often lack).

    python scripts/download.py URL OUT                  # resumes a partial OUT (HTTP Range), like wget -c
    python scripts/download.py -q URL OUT               # no progress lines
    python scripts/download.py --unzip ZIP DESTDIR      # extract everything (overwrites)
    python scripts/download.py --unzip ZIP DESTDIR --no-overwrite   # skip files that already exist

Exits non-zero on HTTP/network errors, so shell scripts can use `|| ...`.
"""

import argparse
import sys
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CHUNK = 1 << 20


def download(url, out, quiet=False, timeout=60):
    """Download url to out, resuming if out already holds a prefix of the file."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    have = out.stat().st_size if out.exists() else 0
    headers = {"User-Agent": "easplatting-download/1.0"}
    if have:
        headers["Range"] = f"bytes={have}-"
    try:
        resp = urlopen(Request(url, headers=headers), timeout=timeout)
    except HTTPError as e:
        if e.code == 416 and have:                       # asked past the end: already complete
            return out
        raise
    with resp:
        if have and resp.status != 206:                  # server ignored Range: start over
            have = 0
        total = None
        cr = resp.headers.get("Content-Range")           # "bytes 100-999/1000"
        if cr and "/" in cr and cr.rsplit("/", 1)[1].isdigit():
            total = int(cr.rsplit("/", 1)[1])
        elif resp.headers.get("Content-Length"):
            total = have + int(resp.headers["Content-Length"])
        done, last = have, time.time()
        with open(out, "ab" if have else "wb") as f:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if not quiet and time.time() - last > 5:
                    last = time.time()
                    pct = f" ({100 * done / total:.0f}%)" if total else ""
                    print(f"  {out.name}: {done / 1e6:.0f} MB{pct}", file=sys.stderr, flush=True)
    if total is not None and out.stat().st_size != total:
        raise IOError(f"incomplete download: {out.stat().st_size} of {total} bytes; re-run to resume")
    if not quiet:
        print(f"  {out.name}: {out.stat().st_size / 1e6:.0f} MB done", file=sys.stderr)
    return out


def unzip(zip_path, dest, overwrite=True):
    """Extract zip_path into dest. zipfile sanitizes '..' and absolute member names."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            target = dest / info.filename
            if not overwrite and target.exists() and not info.is_dir():
                continue
            z.extract(info, dest)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--unzip", action="store_true", help="treat the two arguments as ZIP DESTDIR")
    ap.add_argument("--no-overwrite", action="store_true", help="with --unzip: keep files that already exist")
    ap.add_argument("src", help="URL (download) or zip file (--unzip)")
    ap.add_argument("dst", help="output file (download) or directory (--unzip)")
    args = ap.parse_args()
    try:
        if args.unzip:
            n = unzip(args.src, args.dst, overwrite=not args.no_overwrite)
            if not args.quiet:
                print(f"  extracted {n} files into {args.dst}", file=sys.stderr)
        else:
            download(args.src, args.dst, quiet=args.quiet)
    except (HTTPError, URLError, OSError, zipfile.BadZipFile) as e:
        print(f"download.py: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
