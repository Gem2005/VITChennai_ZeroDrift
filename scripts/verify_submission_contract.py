#!/usr/bin/env python3
"""
verify_submission_contract.py - Slide 16 / Track 3 FAQ checks on the submission files.

Checks estimated_trajectory.txt and metadata.json (repository root by default)
against the automated grading contract:

  1. estimated_trajectory.txt exists, is space-separated, and every line that is
     not a pose starts with '#'
  2. every pose row is exactly 8 floats: timestamp tx ty tz qx qy qz qw
  3. timestamps are the input timestamps in seconds (ns * 1e-9 for nanoseconds)
  4. one pose per IMU row of the input CSV, in input order, including the blind
     window and the duplicated timestamps; nothing added, nothing dropped, no NaN / Inf
  5. every quaternion norm is within 1e-4 of 1
  6. metadata.json: peak_memory_mb <= 8.0 and ms_per_1s_imu <= 20.0

The input CSV is needed for checks 3 and 4. Without --input it is the same file
src/main.py uses when run without arguments (imu0/data.csv, else data/imu0/data.csv).

    uv run python scripts/verify_submission_contract.py
    uv run python scripts/verify_submission_contract.py --input imu0/data.csv \
        --trajectory estimated_trajectory.txt --meta metadata.json

Exit status 0 only if every check passes. Needs numpy only.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MEMORY_LIMIT_MB = 8.0
SPEED_LIMIT_MS = 20.0
QUAT_TOL = 1e-4
TIME_TOL_S = 1e-6
FLOAT_CHARS = set("0123456789+-.eE")


def read_input_timestamps(path):
    """
    Timestamp of every IMU data row, in file order.

    A first line whose first cell is not a number is the header. Every other
    non-empty line is a data row and must be counted, whatever its sensor cells
    hold (the blind window is zeros, a broken cell may be 'nan').
    """
    stamps, bad = [], []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        first = True
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            cell = line.split(",")[0].strip()
            try:
                stamps.append(float(cell))
            except ValueError:
                if not first:
                    bad.append(lineno)
            first = False
    return np.asarray(stamps, dtype=np.float64), bad


def timestamp_scale(stamps):
    """Seconds per input tick, from the median sample period (the degrade_imu.py thresholds)."""
    d = np.diff(stamps)
    d = d[d > 0]
    dt = float(np.sort(d)[d.size // 2]) if d.size else 1.0
    if dt < 1.0:
        return 1.0, "s"
    if dt < 1e3:
        return 1e-3, "ms"
    if dt < 1e6:
        return 1e-6, "us"
    return 1e-9, "ns"


def read_trajectory(path):
    """(n x 8 array, format problems, number of '#' lines)."""
    rows, problems, comments = [], [], 0
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            text = line.rstrip("\r\n")
            if not text.strip():
                if text:
                    problems.append("line %d holds only whitespace" % lineno)
                continue
            if text.lstrip().startswith("#"):
                comments += 1
                continue
            if "," in text or "\t" in text or ";" in text:
                problems.append("line %d is not space-separated" % lineno)
                continue
            parts = text.split(" ")
            if "" in parts:
                parts = text.split()
            if len(parts) != 8:
                problems.append("line %d has %d columns, expected 8" % (lineno, len(parts)))
                continue
            if not all(set(p) <= FLOAT_CHARS for p in parts):
                problems.append("line %d holds a non-numeric value" % lineno)
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                problems.append("line %d holds a non-numeric value" % lineno)
    arr = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 8))
    return arr, problems, comments


def default_input():
    for base in (os.getcwd(), ROOT):
        for rel in (os.path.join("imu0", "data.csv"), os.path.join("data", "imu0", "data.csv")):
            path = os.path.join(base, rel)
            if os.path.isfile(path):
                return path
    return None


def verify(traj_path, meta_path, input_path):
    """List of (check name, passed, detail)."""
    out = []

    def add(name, ok, detail):
        out.append((name, bool(ok), detail))

    # 1 + 2: file, separator, columns
    exists = os.path.isfile(traj_path)
    add("1. estimated_trajectory.txt exists", exists and os.path.basename(traj_path) == "estimated_trajectory.txt",
        traj_path)
    if not exists:
        return out
    est, problems, comments = read_trajectory(traj_path)
    n = est.shape[0]
    add("1. space-separated, other lines start with '#'", not problems and n > 0,
        "%d pose rows, %d comment lines" % (n, comments) if not problems else "; ".join(problems[:3]))
    add("2. exactly 8 floats per pose row", not problems and n > 0,
        "timestamp tx ty tz qx qy qz qw" if not problems else "%d bad lines" % len(problems))
    finite = n > 0 and bool(np.isfinite(est).all())
    add("4. no NaN / Inf", finite, "all %d values finite" % est.size if finite else "non-finite values present")

    # 3 + 4: against the input CSV
    if input_path is None or not os.path.isfile(input_path):
        add("4. one pose per IMU row", False,
            "input CSV not found (%s): pass --input" % (input_path or "imu0/data.csv, data/imu0/data.csv"))
        add("3. timestamps in seconds", False, "needs the input CSV")
    else:
        stamps, bad = read_input_timestamps(input_path)
        same = stamps.shape[0] == n and not bad
        dup = int(np.count_nonzero(np.diff(stamps) == 0)) if stamps.size else 0
        add("4. one pose per IMU row", same,
            "%s: %d rows (%d repeat the previous timestamp) -> %d poses%s" % (
                os.path.relpath(input_path, os.getcwd()), stamps.shape[0], dup, n,
                "" if not bad else "; %d input lines without a numeric timestamp" % len(bad)))
        if stamps.shape[0] == n and n > 0:
            scale, unit = timestamp_scale(stamps)
            err = float(np.max(np.abs(est[:, 0] - stamps * scale)))
            order = bool(np.all(np.diff(est[:, 0]) >= 0)) == bool(np.all(np.diff(stamps) >= 0))
            add("3. timestamps in seconds", err <= TIME_TOL_S and order,
                "input in %s (x %g): max |t_out - t_in| = %.2e s, first %.9f" % (unit, scale, err, est[0, 0]))
            if dup:
                idx = np.flatnonzero(np.diff(stamps) == 0) + 1
                gap = float(np.max(np.abs(est[idx, 1:] - est[idx - 1, 1:])))
                add("4. duplicated timestamps keep their own row", gap == 0.0,
                    "%d duplicated rows, pose identical to the row they repeat (max diff %.1e)" % (dup, gap))
        else:
            add("3. timestamps in seconds", False, "row counts differ, cannot compare")

    # 5: quaternions
    if n > 0 and finite:
        qerr = float(np.max(np.abs(np.sqrt(np.sum(est[:, 4:8] ** 2, axis=1)) - 1.0)))
        add("5. unit quaternions (within %.0e)" % QUAT_TOL, qerr <= QUAT_TOL, "max | |q| - 1 | = %.2e" % qerr)
    else:
        add("5. unit quaternions (within %.0e)" % QUAT_TOL, False, "no finite pose rows")

    # 6: efficiency contract
    if not os.path.isfile(meta_path):
        add("6. metadata.json exists", False, meta_path)
        return out
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    def number(key):
        v = meta.get(key)
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None

    peak, speed = number("peak_memory_mb"), number("ms_per_1s_imu")
    add("6. peak_memory_mb <= %.1f" % MEMORY_LIMIT_MB, peak is not None and peak <= MEMORY_LIMIT_MB,
        "missing" if peak is None else "%.3f MB" % peak)
    add("6. ms_per_1s_imu <= %.1f" % SPEED_LIMIT_MS, speed is not None and speed <= SPEED_LIMIT_MS,
        "missing" if speed is None else "%.3f ms per 1 s of IMU" % speed)
    rows_meta = (meta.get("rows_in"), meta.get("rows_out"))
    add("6. metadata row counts match the trajectory", rows_meta == (n, n),
        "rows_in=%s rows_out=%s, trajectory %d" % (rows_meta[0], rows_meta[1], n))
    return out


def main():
    ap = argparse.ArgumentParser(description="Check estimated_trajectory.txt + metadata.json against the Track 3 contract")
    ap.add_argument("--trajectory", default=os.path.join(ROOT, "estimated_trajectory.txt"))
    ap.add_argument("--meta", default=os.path.join(ROOT, "metadata.json"))
    ap.add_argument("--input", default=None, help="IMU CSV the trajectory was estimated from")
    args = ap.parse_args()
    results = verify(args.trajectory, args.meta, args.input or default_input())
    for name, ok, detail in results:
        print("[%s] %-46s %s" % ("PASS" if ok else "FAIL", name, detail))
    passed = bool(results) and all(ok for _, ok, _ in results)
    print("\nSUBMISSION CONTRACT: %s" % ("PASSED" if passed else "FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
