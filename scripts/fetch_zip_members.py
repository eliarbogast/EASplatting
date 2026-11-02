"""
fetch_zip_members.py — extract selected folders from a huge remote .zip without downloading it.

Uses HTTP Range requests: reads the zip's central directory (zip64 supported),
then fetches only the members whose names start with the given prefixes.
Lets you pull one Mip-NeRF 360 scene (~0.3-1.5 GB) out of the 12 GB 360_v2.zip.

    python scripts/fetch_zip_members.py http://storage.googleapis.com/gresearch/refraw360/360_v2.zip \
        --prefix bonsai/sparse/ --prefix bonsai/images_2/ --out data/360_v2

    python scripts/fetch_zip_members.py URL --list --prefix bonsai/    # just list

The URL may also be a local path (used by the tests). Members are CRC-checked;
existing files with the right size are skipped, so re-running resumes.
"""

import argparse
import os
import struct
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen

CHUNK = 1 << 20


class Source:
    """Random access to a remote (HTTP Range) or local file."""

    def __init__(self, url):
        self.url = url
        self.local = not url.startswith(("http://", "https://"))
        if self.local:
            self.size = os.path.getsize(url)
        else:
            with urlopen(Request(url, method="HEAD"), timeout=60) as r:
                self.size = int(r.headers["Content-Length"])
                if r.headers.get("Accept-Ranges", "bytes") == "none":
                    sys.exit("Server does not support range requests.")

    def read(self, start, length):
        if length <= 0:
            return b""
        if self.local:
            with open(self.url, "rb") as f:
                f.seek(start)
                return f.read(length)
        req = Request(self.url, headers={"Range": f"bytes={start}-{start + length - 1}"})
        with urlopen(req, timeout=120) as r:
            if r.status != 206:
                raise IOError(f"expected 206 Partial Content, got {r.status}")
            return r.read()

    def stream(self, start, length):
        """Yield the byte range in chunks (keeps memory flat for big members)."""
        if self.local:
            with open(self.url, "rb") as f:
                f.seek(start)
                left = length
                while left > 0:
                    b = f.read(min(CHUNK, left))
                    if not b:
                        break
                    left -= len(b)
                    yield b
            return
        req = Request(self.url, headers={"Range": f"bytes={start}-{start + length - 1}"})
        with urlopen(req, timeout=300) as r:
            while True:
                b = r.read(CHUNK)
                if not b:
                    break
                yield b


def central_directory(src):
    """Return list of dicts: name, method, csize, usize, crc, offset (local header)."""
    tail_len = min(src.size, 65536 + 22)
    tail = src.read(src.size - tail_len, tail_len)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise ValueError("End of central directory not found — not a zip?")
    (_, _, _, _, n_entries, cd_size, cd_off, _) = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    # zip64: locator sits right before the EOCD record
    j = i - 20
    if j >= 0 and tail[j:j + 4] == b"PK\x06\x07":
        (_, _, z64_off, _) = struct.unpack("<IIQI", tail[j:j + 20])
        rec = src.read(z64_off, 56)
        if rec[:4] != b"PK\x06\x06":
            raise ValueError("bad zip64 end-of-central-directory record")
        (_, _, _, _, _, _, _, n_entries, cd_size, cd_off) = struct.unpack("<IQHHIIQQQQ", rec)

    cd = src.read(cd_off, cd_size)
    out, p = [], 0
    for _ in range(n_entries):
        if cd[p:p + 4] != b"PK\x01\x02":
            raise ValueError("bad central directory entry")
        (method, crc, csize, usize, nlen, xlen, clen, off) = (
            struct.unpack("<H", cd[p + 10:p + 12])[0],
            *struct.unpack("<III", cd[p + 16:p + 28]),
            *struct.unpack("<HHH", cd[p + 28:p + 34]),
            struct.unpack("<I", cd[p + 42:p + 46])[0],
        )
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", errors="replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        if 0xFFFFFFFF in (csize, usize, off):          # zip64 extended info (id 0x0001)
            q = 0
            while q + 4 <= len(extra):
                hid, hlen = struct.unpack("<HH", extra[q:q + 4])
                if hid == 0x0001:
                    vals = list(struct.unpack(f"<{hlen // 8}Q", extra[q + 4:q + 4 + (hlen // 8) * 8]))
                    if usize == 0xFFFFFFFF:
                        usize = vals.pop(0)
                    if csize == 0xFFFFFFFF:
                        csize = vals.pop(0)
                    if off == 0xFFFFFFFF:
                        off = vals.pop(0)
                    break
                q += 4 + hlen
        out.append(dict(name=name, method=method, crc=crc, csize=csize, usize=usize, offset=off))
        p += 46 + nlen + xlen + clen
    return out


def extract(src, m, out_dir):
    dest = os.path.join(out_dir, m["name"])
    if m["name"].endswith("/"):
        os.makedirs(dest, exist_ok=True)
        return 0
    if os.path.exists(dest) and os.path.getsize(dest) == m["usize"]:
        return 0                                         # already there (resume)
    hdr = src.read(m["offset"], 30)
    if hdr[:4] != b"PK\x03\x04":
        raise ValueError(f"bad local header for {m['name']}")
    nlen, xlen = struct.unpack("<HH", hdr[26:30])
    data_off = m["offset"] + 30 + nlen + xlen

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    crc = 0
    dec = zlib.decompressobj(-15) if m["method"] == 8 else None
    if m["method"] not in (0, 8):
        raise ValueError(f"unsupported compression method {m['method']} for {m['name']}")
    with open(tmp, "wb") as f:
        for chunk in src.stream(data_off, m["csize"]):
            b = dec.decompress(chunk) if dec else chunk
            crc = zlib.crc32(b, crc)
            f.write(b)
        if dec:
            b = dec.flush()
            crc = zlib.crc32(b, crc)
            f.write(b)
    if crc != m["crc"]:
        os.remove(tmp)
        raise IOError(f"CRC mismatch for {m['name']}")
    os.replace(tmp, dest)
    return m["usize"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--prefix", action="append", default=[], help="member-name prefix (repeatable)")
    ap.add_argument("--out", default=".")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    src = Source(args.url)
    members = central_directory(src)
    sel = [m for m in members if not args.prefix or any(m["name"].startswith(p) for p in args.prefix)]
    total = sum(m["usize"] for m in sel)
    print(f"{args.url}: {len(members)} members; selected {len(sel)} ({total / 1e9:.2f} GB)")
    if args.list:
        for m in sel[:200]:
            print(f"  {m['usize'] / 1e6:9.1f} MB  {m['name']}")
        if len(sel) > 200:
            print(f"  ... {len(sel) - 200} more")
        return
    if not sel:
        sys.exit("Nothing matched the given --prefix values.")

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract, src, m, args.out): m for m in sel}
        for k, fut in enumerate(as_completed(futs), 1):
            done += fut.result()
            if k % 25 == 0 or k == len(sel):
                print(f"  {k}/{len(sel)} files, {done / 1e9:.2f} GB downloaded", flush=True)
    print(f"Extracted to {args.out}")


if __name__ == "__main__":
    main()
