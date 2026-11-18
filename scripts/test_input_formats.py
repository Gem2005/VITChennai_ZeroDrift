#!/usr/bin/env python3
"""
test_input_formats.py - the column-order / unit auto-detector and degenerate inputs.

Takes one officially degraded recording (default: results/SYN_room_walk/degraded.csv,
written by benchmark_all.py), rewrites it in every layout the hidden test file could
plausibly use and runs src/main.py on each:

    column order   accel-first / gyro-first, with header, with EuRoC header, without header
    units          accelerometer in m/s^2 or g, gyroscope in rad/s or deg/s
    timestamps     s, ms, us, ns
    file quirks    CRLF line ends + trailing blank lines, NaN / Inf cells
    degenerate     1 .. 2000 rows, all-zero sensors, constant sensors, empty file

Every variant must exit 0, keep the row count, be finite with unit quaternions, report
the layout that was written, and reproduce the reference trajectory (same physical data,
so any difference is detection error or text rounding).

    uv run python scripts/test_input_formats.py [--input results/<seq>/degraded.csv]
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from benchmark_all import G, load_degraded  # noqa: E402

HDR = "timestamp,ax,ay,az,gx,gy,gz"
EUROC_HDR = ("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
             "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]")
POS_TOL = 0.05      # m, against the reference run of the same data
MAIN = os.path.join(ROOT, "src", "main.py")


def write(path, arr, header, fmt, eol="\n", tail=""):
    with open(path, "w", newline="") as f:
        if header:
            f.write(header + eol)
        for r in arr:
            f.write(",".join(fmt[i] % v for i, v in enumerate(r)) + eol)
        f.write(tail)
    return path


def run(path):
    out, meta = path + ".tum", path + ".json"
    for p in (out, meta):
        if os.path.exists(p):
            os.remove(p)
    proc = subprocess.run([sys.executable, MAIN, "--input", path, "--output", out, "--meta", meta],
                          capture_output=True, text=True)
    m = None
    if proc.returncode == 0 and os.path.exists(meta):
        with open(meta) as f:
            m = json.load(f)
    est = np.loadtxt(out, ndmin=2) if os.path.exists(out) and os.path.getsize(out) else np.zeros((0, 8))
    return proc, m, est


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", default=os.path.join(ROOT, "results", "SYN_room_walk", "degraded.csv"))
    ap.add_argument("--work", default=os.path.join(ROOT, "results", "_formats"))
    args = ap.parse_args()
    stats_path = os.path.join(os.path.dirname(args.input), "stats.json")
    if not (os.path.isfile(args.input) and os.path.isfile(stats_path)):
        print("missing %s (+ stats.json): run scripts/benchmark_all.py first" % args.input)
        return 1
    os.makedirs(args.work, exist_ok=True)
    with open(stats_path) as f:
        stats = json.load(f)

    # canonical copy: integer nanoseconds, accel-first, m/s^2, rad/s
    deg = load_degraded(args.input)
    deg[:, 0] = np.rint(deg[:, 0] * {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0}[stats["timestamp_unit"]])
    if stats["accel_unit"] == "g":
        deg[:, 1:4] *= G
    ifmt = ["%d"] + ["%.6f"] * 6
    path = lambda name: os.path.join(args.work, name)  # noqa: E731

    proc, base_m, base = run(write(path("reference.csv"), deg, HDR, ifmt))
    if base_m is None:
        print(proc.stdout + proc.stderr)
        return 1
    failures = []

    def check(name, arr, header, fmt, expect=None, compare=True, **kw):
        proc, m, est = run(write(path(name), arr, header, fmt, **kw))
        problems = []
        if proc.returncode != 0 or m is None:
            problems.append("exit %d: %s" % (proc.returncode, (proc.stderr or proc.stdout).strip().split("\n")[-1][:120]))
        else:
            if not (est.shape[0] == arr.shape[0] == m["rows_in"] == m["rows_out"]):
                problems.append("rows in %d out %d" % (arr.shape[0], est.shape[0]))
            if not np.isfinite(est).all():
                problems.append("non-finite output")
            elif est.size and np.abs(np.linalg.norm(est[:, 4:8], axis=1) - 1.0).max() > 1e-6:
                problems.append("non-unit quaternion")
            pp = m["preprocessing"]
            for key, want in (expect or {}).items():
                if pp.get(key) != want:
                    problems.append("%s detected as %r, expected %r" % (key, pp.get(key), want))
            if compare and est.shape == base.shape:
                d = float(np.abs(est[:, 1:4] - base[:, 1:4]).max())
                dt = float(np.abs(np.diff(est[:, 0]) - np.diff(base[:, 0])).max())
                if d > POS_TOL or dt > 1e-6:
                    problems.append("differs from the reference: %.4f m, %.1e s" % (d, dt))
        print("  [%s] %-36s %s" % ("FAIL" if problems else "PASS", name, "; ".join(problems) if problems else
                                    "rows %d" % est.shape[0]))
        if problems:
            failures.append(name)

    canon = {"timestamp_unit": "ns", "column_order": "accel,gyro", "accel_unit": "m/s^2", "gyro_unit": "rad/s"}
    swap = [0, 4, 5, 6, 1, 2, 3]
    print("column order")
    check("no_header.csv", deg, None, ifmt, canon)
    check("gyro_first_named_header.csv", deg[:, swap], "timestamp,gx,gy,gz,ax,ay,az", ifmt, dict(canon, column_order="gyro,accel"))
    check("gyro_first_euroc_header.csv", deg[:, swap], EUROC_HDR, ifmt, dict(canon, column_order="gyro,accel"))
    check("gyro_first_no_header.csv", deg[:, swap], None, ifmt, dict(canon, column_order="gyro,accel"))

    print("units")
    in_g = deg.copy()
    in_g[:, 1:4] /= G
    check("accel_in_g.csv", in_g, HDR, ["%d"] + ["%.7f"] * 3 + ["%.6f"] * 3, dict(canon, accel_unit="g"))
    in_deg = deg.copy()
    in_deg[:, 4:7] *= 180.0 / np.pi
    check("gyro_in_deg_per_s.csv", in_deg, HDR, ["%d"] + ["%.6f"] * 3 + ["%.5f"] * 3, dict(canon, gyro_unit="deg/s"))
    both = in_g.copy()
    both[:, 4:7] *= 180.0 / np.pi
    check("g_and_deg_gyro_first_no_header.csv", both[:, swap], None, ["%d"] + ["%.5f"] * 3 + ["%.7f"] * 3,
          dict(canon, accel_unit="g", gyro_unit="deg/s", column_order="gyro,accel"))

    print("timestamps")
    for unit, scale, f in (("s", 1e-9, "%.9f"), ("ms", 1e-6, "%.6f"), ("us", 1e-3, "%.3f")):
        a = deg.copy()
        a[:, 0] *= scale
        check("timestamps_in_%s.csv" % unit, a, HDR, [f] + ["%.6f"] * 6, dict(canon, timestamp_unit=unit))

    print("file quirks")
    check("crlf_and_trailing_blank_lines.csv", deg, HDR, ifmt, canon, eol="\r\n", tail="\r\n\r\n")
    bad = deg.copy()
    bad[deg.shape[0] // 3, 2] = np.nan
    bad[deg.shape[0] // 2, 5] = np.inf
    check("nan_and_inf_cells.csv", bad, HDR, ["%d"] + ["%s"] * 6, canon, compare=False)

    print("degenerate inputs")
    for n in (1, 2, 3, 10, 300, 1200, 2000):
        check("first_%d_rows.csv" % n, deg[:n], HDR, ifmt, compare=False)
    zero = deg[:3000].copy()
    zero[:, 1:] = 0.0
    check("all_zero_sensors.csv", zero, HDR, ifmt, compare=False)
    const = deg[:3000].copy()
    const[:, 1:] = const[0, 1:]
    check("constant_sensors.csv", const, HDR, ifmt, compare=False)
    proc = subprocess.run([sys.executable, MAIN, "--input", write(path("empty.csv"), deg[:0], HDR, ifmt)],
                          capture_output=True, text=True)
    clean_exit = proc.returncode == 1 and "no IMU rows" in (proc.stderr + proc.stdout)
    print("  [%s] %-36s %s" % ("PASS" if clean_exit else "FAIL", "empty.csv",
                                "refused with a clear message (exit 1)" if clean_exit else "exit %d" % proc.returncode))
    if not clean_exit:
        failures.append("empty.csv")

    print("\n%s" % ("ALL INPUT-FORMAT CHECKS PASSED" if not failures else "FAILED: " + ", ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
