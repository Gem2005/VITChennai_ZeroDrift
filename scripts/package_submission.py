#!/usr/bin/env python3
"""
package_submission.py - build dist/VITChennai_ZeroDrift.zip and prove it runs on its own.

Non-destructive: nothing in the repository is moved, edited or deleted. Only
dist/VITChennai_ZeroDrift.zip is (re)written.

The archive holds exactly the Slide 17 + Slide 25 file set:

    estimated_trajectory.txt     metadata.json     README.md
    requirements.txt             Dockerfile        src/*.py  (no __pycache__, no .pyc)

Steps:
  1. the root estimated_trajectory.txt / metadata.json must pass
     scripts/verify_submission_contract.py (skip with --no-verify),
  2. the zip is written and its file tree and sizes are printed,
  3. the zip is extracted to /tmp/test_eval (or <system temp>/test_eval where /tmp
     does not exist) and src/main.py is run there, from the extracted root and from
     inside src/, with flags, positionally and without arguments. The Python
     process sees only the extracted files: the run must produce one pose per input
     row and may import nothing beyond numpy and the standard library.

    uv run python scripts/package_submission.py
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ZIP_NAME = "VITChennai_ZeroDrift.zip"
ROOT_FILES = ("estimated_trajectory.txt", "metadata.json", "README.md", "requirements.txt", "Dockerfile")
SMOKE_ROWS = 2400            # 12 s at 200 Hz: protected start, estimation, 5 s blind window

IMPORT_AUDIT = r"""
import os, sys
before = set(sys.modules)                       # whatever the interpreter itself loaded
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import main
code = main.main(["--input", os.path.join("imu0", "data.csv"), "--output", os.path.join("out", "audit.txt"),
                  "--meta", os.path.join("out", "audit.json")])
here = os.path.realpath(os.getcwd())
own, third = set(), set()
for name, mod in list(sys.modules.items()):
    top = name.split(".")[0]
    path = getattr(mod, "__file__", None)
    if name in before or top in sys.stdlib_module_names or top.startswith("_"):
        continue
    if path and os.path.realpath(path).startswith(here + os.sep):
        own.add(top)
    else:
        third.add(top)
print("own modules      :", ", ".join(sorted(own)))
print("third-party      :", ", ".join(sorted(third)))
extra = third - {"numpy"}
print("beyond numpy     :", ", ".join(sorted(extra)) or "none")
sys.exit(1 if code or extra or own != {"main", "es_ekf", "inverse_degrade"} else 0)
"""


def source_files():
    """Everything under src/ except byte-code caches, as (absolute path, archive name)."""
    out = []
    src = os.path.join(ROOT, "src")
    for dirpath, dirnames, files in os.walk(src):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(files):
            if name.endswith((".pyc", ".pyo")):
                continue
            full = os.path.join(dirpath, name)
            out.append((full, os.path.relpath(full, ROOT).replace(os.sep, "/")))
    return out


def build_zip(zip_path):
    missing = [f for f in ROOT_FILES if not os.path.isfile(os.path.join(ROOT, f))]
    if missing:
        raise SystemExit("cannot package, missing at the repository root: " + ", ".join(missing))
    members = [(os.path.join(ROOT, f), f) for f in ROOT_FILES] + source_files()
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    tmp = zip_path + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for full, arc in members:
            z.write(full, arc)
    os.replace(tmp, zip_path)
    return [arc for _, arc in members]


def print_tree(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        infos = sorted(z.infolist(), key=lambda i: (i.filename.count("/"), i.filename))
        bad = z.testzip()
    print("\n%s" % os.path.relpath(zip_path, ROOT).replace(os.sep, "/"))
    print("  %-34s %14s %14s" % ("file", "uncompressed", "compressed"))
    top = [i for i in infos if "/" not in i.filename]
    sub = [i for i in infos if "/" in i.filename]
    for i in top:
        print("  %-34s %12d B %12d B" % (i.filename, i.file_size, i.compress_size))
    folder = None
    for i in sub:
        d, name = i.filename.rsplit("/", 1)
        if d != folder:
            folder = d
            print("  %s/" % d)
        print("  %-34s %12d B %12d B" % ("    " + name, i.file_size, i.compress_size))
    raw, packed = sum(i.file_size for i in infos), sum(i.compress_size for i in infos)
    print("  %-34s %12d B %12d B" % ("total (%d files)" % len(infos), raw, packed))
    print("  uncompressed %.2f MB, compressed %.2f MB, archive on disk %.2f MB (%d bytes), CRC check %s" % (
        raw / 1048576.0, packed / 1048576.0, os.path.getsize(zip_path) / 1048576.0, os.path.getsize(zip_path),
        "ok" if bad is None else "FAILED at " + bad))
    return bad is None


def write_smoke_csv(path):
    """
    A small degraded-style recording in the organisers' layout (nanosecond timestamps,
    accelerometer then gyroscope): a slow tilt oscillation under gravity, one dropout
    gap, two duplicated rows, and a zeroed last 5 seconds. Returns the row count.
    """
    import math
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0, dt_ns, rows = 1_700_000_000_000_000_000, 5_000_000, 0
    fail_from = SMOKE_ROWS - 1000
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("timestamp,ax,ay,az,gx,gy,gz\n")
        for i in range(SMOKE_ROWS):
            if 600 <= i < 640:                       # 0.2 s of physically missing rows
                continue
            t = i * 0.005
            if i >= fail_from:
                vals = (0.0,) * 6
            else:
                tilt = 0.05 * math.sin(2.0 * math.pi * 0.5 * t)
                rate = 0.05 * 2.0 * math.pi * 0.5 * math.cos(2.0 * math.pi * 0.5 * t)
                jitter = 0.01 * math.sin(37.0 * t)
                vals = (9.80665 * math.sin(tilt) + jitter, 0.02 + jitter, 9.80665 * math.cos(tilt), 0.001, rate, -0.002)
            line = "%d,%s\n" % (t0 + i * dt_ns, ",".join("%.6f" % v for v in vals))
            f.write(line)
            rows += 1
            if i in (300, 900):                      # duplicated rows (same timestamp)
                f.write(line)
                rows += 1
    return rows


def smoke_dir():
    base = "/tmp" if os.name != "nt" and os.path.isdir("/tmp") else tempfile.gettempdir()
    return os.path.join(base, "test_eval")


def pose_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip() and not line.lstrip().startswith("#"))


def smoke_test(zip_path, expected_members):
    work = smoke_dir()
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(work)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(work)
    extracted = sorted(os.path.relpath(os.path.join(d, f), work).replace(os.sep, "/")
                       for d, _, files in os.walk(work) for f in files)
    print("\nextraction smoke test in %s" % work)
    ok = extracted == sorted(expected_members)
    print("  [%s] extracted exactly the packaged files (%d)" % ("PASS" if ok else "FAIL", len(extracted)))

    n_rows = write_smoke_csv(os.path.join(work, "imu0", "data.csv"))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    src = os.path.join(work, "src")
    out = os.path.join(work, "out")
    os.makedirs(out)
    j = os.path.join
    runs = [
        ("python src/main.py --input imu0/data.csv --output out/a.txt --meta out/a.json", work,
         ["src/main.py", "--input", j("imu0", "data.csv"), "--output", j("out", "a.txt"), "--meta", j("out", "a.json")],
         j(out, "a.txt"), j(out, "a.json")),
        ("python src/main.py -i imu0/data.csv -o out/b.txt -m out/b.json", work,
         ["src/main.py", "-i", j("imu0", "data.csv"), "-o", j("out", "b.txt"), "-m", j("out", "b.json")],
         j(out, "b.txt"), j(out, "b.json")),
        ("python src/main.py imu0/data.csv out/pos/c.txt", work,
         ["src/main.py", j("imu0", "data.csv"), j("out", "pos", "c.txt")],
         j(out, "pos", "c.txt"), j(out, "pos", "metadata.json")),
        ("cd src && python main.py --input ../imu0/data.csv --output ../out/d.txt", src,
         ["main.py", "--input", j("..", "imu0", "data.csv"), "--output", j("..", "out", "d.txt"),
          "--meta", j("..", "out", "d.json")], j(out, "d.txt"), j(out, "d.json")),
        ("python src/main.py                      (no arguments)", work, ["src/main.py"],
         j(work, "estimated_trajectory.txt"), j(work, "metadata.json")),
        ("cd src && python main.py                (no arguments)", src, ["main.py"],
         j(work, "estimated_trajectory.txt"), j(work, "metadata.json")),
    ]
    reference = None
    for label, cwd, args, traj, meta in runs:
        for p in (traj, meta):
            if os.path.exists(p):
                os.remove(p)
        proc = subprocess.run([sys.executable] + args, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=600)
        good = proc.returncode == 0 and os.path.isfile(traj) and os.path.isfile(meta)
        detail = "exit %d" % proc.returncode
        if good:
            with open(meta, "r", encoding="utf-8") as f:
                m = json.load(f)
            n = pose_rows(traj)
            with open(traj, "rb") as f:
                data = f.read()
            reference = data if reference is None else reference
            good = n == n_rows == m["rows_in"] == m["rows_out"] and data == reference
            detail = "%d rows in -> %d poses, heap %.3f MB, %.2f ms per 1 s IMU%s" % (
                n_rows, n, m["peak_memory_mb"], m["ms_per_1s_imu"], "" if data == reference else ", output differs")
        print("  [%s] %s\n         %s" % ("PASS" if good else "FAIL", label, detail))
        if not good:
            print(proc.stdout[-800:] + proc.stderr[-2000:])
        ok = ok and good

    audit = subprocess.run([sys.executable, "-c", IMPORT_AUDIT], cwd=work, env=env, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=600)
    print("  [%s] import audit of the extracted src/ (numpy + standard library only)" % (
        "PASS" if audit.returncode == 0 else "FAIL"))
    for line in (audit.stdout + audit.stderr).strip().split("\n"):
        print("         " + line)
    ok = ok and audit.returncode == 0
    leftovers = [p for p in extracted if "__pycache__" in p or p.endswith(".pyc")]
    print("  [%s] no byte-code files in the archive" % ("PASS" if not leftovers else "FAIL"))
    return ok and not leftovers


def main():
    ap = argparse.ArgumentParser(description="Build and smoke-test dist/%s (non-destructive)" % ZIP_NAME)
    ap.add_argument("--out", default=os.path.join(ROOT, "dist", ZIP_NAME))
    ap.add_argument("--no-verify", action="store_true", help="skip scripts/verify_submission_contract.py")
    ap.add_argument("--no-smoke", action="store_true", help="skip the extraction smoke test")
    args = ap.parse_args()

    if not args.no_verify:
        print("contract check of the files to be packaged", flush=True)
        code = subprocess.run([sys.executable, os.path.join(HERE, "verify_submission_contract.py")],
                              cwd=ROOT, stdin=subprocess.DEVNULL).returncode
        if code != 0:
            print("\nNOT PACKAGED: the submission files fail the contract (see above)")
            return 1

    members = build_zip(args.out)
    ok = print_tree(args.out)
    if not args.no_smoke:
        ok = smoke_test(args.out, members) and ok
    print("\n%s  %s" % ("SUBMISSION BUNDLE READY:" if ok else "SUBMISSION BUNDLE FAILED:", args.out))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
