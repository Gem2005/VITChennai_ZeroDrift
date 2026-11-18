#!/usr/bin/env python3
"""
fetch_euroc.py - pull ONLY the IMU + ground-truth CSVs of EuRoC MAV sequences.

The public EuRoC archives are 1-12 GB each (they bundle the stereo images).
This tool opens the remote zip through HTTP range requests and extracts just

    mav0/imu0/data.csv                       (~2 MB)
    mav0/state_groundtruth_estimate0/data.csv (~4 MB)

so a standalone sequence zip costs a few MB of traffic instead of gigabytes.
The bundle archives (machine_hall.zip, ...) hold one nested zip per sequence.
A nested zip that is stored uncompressed is range-read in place; one that is
deflated (MH_01_easy: 1.57 GB) has to be streamed once to a temporary file,
which is deleted as soon as the two CSVs are extracted.

Development/benchmark tooling only (standard library; never imported by
src/main.py).

Usage
    python scripts/fetch_euroc.py                 # V1_01_easy + MH_01_easy
    python scripts/fetch_euroc.py MH_01_easy --list
"""

import argparse
import io
import os
import sys
import time
import urllib.request
import zipfile
import zlib

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

# sequence -> list of (archive url, optional nested member suffix)
SOURCES = {
    "V1_01_easy": [
        (HF.format(repo="pepijn223/euroc-mirror", path="V1_01_easy.zip"), None),
        (HF.format(repo="GlowBond/EuRoC_MAV_Dataset", path="vicon_room1.zip"), "V1_01_easy.zip"),
    ],
    "MH_01_easy": [
        (HF.format(repo="GlowBond/EuRoC_MAV_Dataset", path="machine_hall.zip"), "MH_01_easy.zip"),
    ],
}
BUNDLES = {"MH": "machine_hall.zip", "V1": "vicon_room1.zip", "V2": "vicon_room2.zip"}

WANTED = {
    "mav0/imu0/data.csv": "imu.csv",
    "mav0/state_groundtruth_estimate0/data.csv": "groundtruth.csv",
}


class HttpRangeFile(io.RawIOBase):
    """Read-only seekable file backed by HTTP range requests (block cached)."""

    def __init__(self, url, block=1 << 18):
        self.url, self.block, self.pos, self.cache, self.traffic = url, block, 0, {}, 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "fetch-euroc/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.final_url = r.geturl()
            rng = r.headers.get("Content-Range")
            if r.status != 206 or not rng:
                raise OSError("server does not support HTTP range requests: " + url)
            self.size = int(rng.split("/")[-1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _fetch(self, lo, hi):
        req = urllib.request.Request(
            self.final_url, headers={"Range": "bytes=%d-%d" % (lo, hi - 1), "User-Agent": "fetch-euroc/1.0"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                self.traffic += len(data)
                return data
            except OSError:
                if attempt == 3:
                    raise

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        b0, b1 = self.pos // self.block, (self.pos + n - 1) // self.block
        missing = [b for b in range(b0, b1 + 1) if b not in self.cache]
        if missing:
            lo, hi = missing[0] * self.block, min(self.size, (missing[-1] + 1) * self.block)
            data = self._fetch(lo, hi)
            for b in range(missing[0], missing[-1] + 1):
                self.cache[b] = data[(b * self.block - lo):((b + 1) * self.block - lo)]
            if len(self.cache) > 256:
                for b in list(self.cache)[:128]:
                    del self.cache[b]
        out = b"".join(self.cache[b] for b in range(b0, b1 + 1))
        start = self.pos - b0 * self.block
        self.pos += n
        return out[start:start + n]

    def readinto(self, buf):
        data = self.read(len(buf))
        buf[:len(data)] = data
        return len(data)


class Window(io.RawIOBase):
    """Seekable view of bytes [start, start+size) of another file object."""

    def __init__(self, base, start, size):
        self.base, self.start, self.size, self.pos = base, start, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        self.base.seek(self.start + self.pos)
        data = self.base.read(n)
        self.pos += len(data)
        return data


def stored_member_window(base, zf, info):
    """Window over an uncompressed (stored) zip member, e.g. a nested zip."""
    if info.compress_type != zipfile.ZIP_STORED:
        return None
    base.seek(info.header_offset)
    hdr = base.read(30)
    name_len = int.from_bytes(hdr[26:28], "little")
    extra_len = int.from_bytes(hdr[28:30], "little")
    return Window(base, info.header_offset + 30 + name_len + extra_len, info.file_size)


def inflate_member_to_file(remote, info, dst, chunk=8 << 20):
    """Sequentially download + inflate one deflated member of a remote zip."""
    remote.seek(info.header_offset)
    hdr = remote.read(30)
    start = info.header_offset + 30 + int.from_bytes(hdr[26:28], "little") + int.from_bytes(hdr[28:30], "little")
    end = start + info.compress_size
    inflater = zlib.decompressobj(-15)
    t0, done, last = time.time(), 0, 0.0
    with open(dst, "wb") as f:
        pos = start
        while pos < end:
            data = remote._fetch(pos, min(end, pos + chunk))
            pos += len(data)
            done += len(data)
            f.write(inflater.decompress(data))
            if time.time() - last > 15:
                last = time.time()
                print("    %5.1f%%  %.0f / %.0f MB  (%.1f MB/s)" % (
                    100.0 * done / info.compress_size, done / 1e6, info.compress_size / 1e6,
                    done / 1e6 / max(1e-3, time.time() - t0)), flush=True)
        f.write(inflater.flush())
    if os.path.getsize(dst) != info.file_size:
        raise OSError("inflated size mismatch for " + info.filename)


def open_sequence_zip(url, nested, tmp_dir, verbose=True):
    """Returns (remote, ZipFile of the sequence, temp path to delete or None)."""
    remote = HttpRangeFile(url)
    if verbose:
        print("  archive: %s (%.2f GB)" % (url.rsplit("/", 1)[-1], remote.size / 1e9))
    zf = zipfile.ZipFile(remote)
    if nested is None:
        return remote, zf, None
    for info in zf.infolist():
        if info.filename.endswith(nested):
            win = stored_member_window(remote, zf, info)
            if win is not None:
                return remote, zipfile.ZipFile(win), None
            if info.compress_type != zipfile.ZIP_DEFLATED:
                raise OSError("unsupported compression for nested " + nested)
            os.makedirs(tmp_dir, exist_ok=True)
            tmp = os.path.join(tmp_dir, "_" + nested + ".tmp")
            print("  nested %s is deflated inside the bundle: streaming %.2f GB to a temp file"
                  % (nested, info.compress_size / 1e9), flush=True)
            inflate_member_to_file(remote, info, tmp)
            return remote, zipfile.ZipFile(tmp), tmp
    raise OSError("%s not found inside %s" % (nested, url))


def fetch(seq, out_root, list_only=False):
    out_dir = os.path.join(out_root, seq)
    targets = {k: os.path.join(out_dir, v) for k, v in WANTED.items()}
    if not list_only and all(os.path.exists(p) and os.path.getsize(p) > 0 for p in targets.values()):
        print("[fetch] %s: already present in %s" % (seq, out_dir))
        return True
    sources = SOURCES.get(seq)
    if sources is None:
        bundle = BUNDLES.get(seq[:2])
        if bundle is None:
            print("[fetch] %s: unknown sequence" % seq)
            return False
        sources = [(HF.format(repo="GlowBond/EuRoC_MAV_Dataset", path=bundle), seq + ".zip")]
    for url, nested in sources:
        tmp = zf = None
        try:
            print("[fetch] %s" % seq, flush=True)
            if list_only and nested is not None:
                remote = HttpRangeFile(url)
                for info in zipfile.ZipFile(remote).infolist():
                    print("   ", info.filename, info.file_size)
                return True
            remote, zf, tmp = open_sequence_zip(url, nested, out_root)
            names = zf.namelist()
            if list_only:
                for n in names:
                    if not n.lower().endswith(".png"):
                        print("   ", n)
                return True
            os.makedirs(out_dir, exist_ok=True)
            for suffix, dst in targets.items():
                member = next(n for n in names if n.endswith(suffix) and not n.startswith("__MACOSX"))
                with zf.open(member) as src, open(dst + ".part", "wb") as f:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                os.replace(dst + ".part", dst)
                print("  wrote %s (%.2f MB)" % (dst, os.path.getsize(dst) / 1e6))
            print("  network traffic: %.1f MB" % (remote.traffic / 1e6))
            return True
        except (OSError, StopIteration, zipfile.BadZipFile, zlib.error) as ex:
            print("  source failed: %s" % ex)
        finally:
            if zf is not None:
                zf.close()
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
    return False


def main():
    ap = argparse.ArgumentParser(description="Fetch EuRoC IMU + ground truth CSVs via HTTP range requests.")
    ap.add_argument("sequences", nargs="*", default=["V1_01_easy", "MH_01_easy"])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "euroc"))
    ap.add_argument("--list", action="store_true", help="list non-image members instead of downloading")
    args = ap.parse_args()
    ok = all([fetch(s, os.path.normpath(args.out), args.list) for s in args.sequences])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
