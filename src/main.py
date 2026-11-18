"""
main.py - Track 3 "Path Finder" entry point (IMU-only trajectory estimation).

    python src/main.py --input imu0/data.csv --output estimated_trajectory.txt --meta metadata.json
    python src/main.py -i imu0/data.csv -o estimated_trajectory.txt -m metadata.json
    python src/main.py imu0/data.csv estimated_trajectory.txt
    python src/main.py                  # imu0/data.csv -> <repo root>/estimated_trajectory.txt
    cd src && python main.py ...        # same options, from inside src/

Without --meta the statistics file is written as metadata.json next to the
trajectory. Without any argument the input is imu0/data.csv (working directory,
then repository root), else the degraded test CSV under data/, and both output
files go to the repository root.

Input : 7-column IMU CSV as produced by the organiser's degrade_imu.py
        (timestamp + accelerometer + gyroscope; column order, header and
        units are auto-detected).
Output: TUM trajectory, exactly one pose per input row, in input row order:
            timestamp[s] tx ty tz qx qy qz qw
        plus a JSON file with runtime / heap / preprocessing statistics.

Runs single-threaded and imports only numpy + the standard library. Heap use
is measured with tracemalloc over the whole pipeline (load -> estimate ->
write), which is how the 8 MB algorithm-level limit of the FAQ is defined;
the reported ms_per_1s_imu is measured in the same run and therefore
includes the tracemalloc overhead (--no-memtrace gives the plain timing).
The exit status is 0 whenever a trajectory was written; compliance with the
limits is reported in the metadata ("compliant") and on stdout.
"""

import os
import sys

# One CPU thread, also inside BLAS. Must happen before numpy is imported, and is
# assigned (not setdefault) so the caller's environment cannot raise the count.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# The helper modules live next to this file: importable from any working
# directory (python src/main.py from the repo root, python main.py from src/).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import math
import time
import tracemalloc

import numpy as np

from inverse_degrade import inverse_degrade, read_imu_csv
from es_ekf import estimate_trajectory

MEMORY_LIMIT_MB = 8.0
SPEED_LIMIT_MS_PER_S = 20.0
WRITE_CHUNK = 1024
ROW_FORMAT = "%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n"
# (decimals, integer digits reserved) per output column
FIELDS = ((9, 12), (6, 7), (6, 7), (6, 7), (9, 2), (9, 2), (9, 2), (9, 2))


def format_rows(row):
    """
    ASCII-encode an (n, 8) float64 block exactly like ROW_FORMAT, vectorised.

    Every column is split into its integer part and its rounded fractional
    digits, which are written into a fixed-width uint8 canvas; unused leading
    positions stay 0 and are squeezed out at the end. This avoids creating Python objects per
    row (which is what makes per-row "%" formatting slow under tracemalloc).
    Returns None if a value does not fit the reserved width.
    """
    n = row.shape[0]
    buf = np.zeros((n, sum(d + i + 3 for d, i in FIELDS)), dtype=np.uint8)
    col = 0
    for c, (dec, nint) in enumerate(FIELDS):
        v = row[:, c]
        mag = np.abs(v)
        if not float(mag.max()) < 10.0 ** nint - 1.0:
            return None
        whole = np.floor(mag)
        s = np.rint((mag - whole) * 10.0 ** dec).astype(np.int64)      # fractional digits
        carry = s >= 10 ** dec
        s -= carry * 10 ** dec
        whole = whole.astype(np.int64) + carry
        buf[:, col] = 45 * ((v < 0.0) & ((s > 0) | (whole > 0)))       # '-'
        dot = col + nint + 1
        for j in range(dec):
            s, r = np.divmod(s, 10)
            buf[:, dot + dec - j] = r + 48
        buf[:, dot] = 46                                               # '.'
        s, r = np.divmod(whole, 10)
        buf[:, dot - 1] = r + 48
        for j in range(2, nint + 1):
            if not s.any():
                break
            more = s > 0
            s, r = np.divmod(s, 10)
            buf[:, dot - j] = (r + 48) * more
        col = dot + dec + 1
        buf[:, col] = 10 if c == len(FIELDS) - 1 else 32     # newline / space
        col += 1
    return buf[buf != 0].tobytes()


def write_tum(path, pre, pos, quat):
    """
    Write one pose per ORIGINAL input row. Rows that degrade_imu.py duplicated
    (dt == 0) were dropped before estimation; here they receive the exact pose
    of the row they repeat, so the row count is preserved 1-to-1.
    """
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    dup, t, t0 = pre.dup, pre.t, pre.t0_abs
    n_rows, written, base = pre.n_rows, 0, 0
    last_p = np.zeros(3)
    last_q = np.array([0.0, 0.0, 0.0, 1.0])
    with open(path, "wb") as f:
        for a in range(0, n_rows, WRITE_CHUNK):
            b = min(n_rows, a + WRITE_CHUNK)
            uniq = ~dup[a:b]
            idx = base + np.cumsum(uniq) - 1          # compact index of each original row
            base += int(uniq.sum())
            row = np.empty((b - a, 8), dtype=np.float64)
            row[:, 0] = t[idx] + t0
            row[:, 1:4] = pos[idx]
            row[:, 4:8] = quat[idx]
            # hard guarantees: finite values and unit quaternions
            bad_p = ~np.isfinite(row[:, 1:4]).all(axis=1)
            qn = np.sqrt(np.sum(row[:, 4:8] ** 2, axis=1))
            bad_q = ~np.isfinite(qn) | (qn < 1e-6)
            if bad_p.any() or bad_q.any():
                for i in np.flatnonzero(bad_p | bad_q):
                    if bad_p[i]:
                        row[i, 1:4] = row[i - 1, 1:4] if i > 0 else last_p
                    if bad_q[i]:
                        row[i, 4:8] = row[i - 1, 4:8] if i > 0 else last_q
                qn = np.sqrt(np.sum(row[:, 4:8] ** 2, axis=1))
            row[:, 4:8] /= qn[:, None]
            last_p, last_q = row[-1, 1:4].copy(), row[-1, 4:8].copy()
            text = format_rows(row)
            if text is None:                           # out-of-range values: plain formatting
                text = "".join([ROW_FORMAT % tuple(r) for r in row.tolist()]).encode("ascii")
            f.write(text)
            written += b - a
    return written


def run_pipeline(input_path, output_path, denoise=True):
    """Load -> inverse-degrade -> estimate -> write. Returns a statistics dict."""
    header, ts, x = read_imu_csv(input_path)
    rows_in = int(ts.shape[0])
    if rows_in == 0:
        raise SystemExit("no IMU rows found in %s" % input_path)
    pre = inverse_degrade(header, ts, x, denoise=denoise)
    pos, quat, summary = estimate_trajectory(pre)
    rows_out = write_tum(output_path, pre, pos, quat)
    return {
        "rows_in": rows_in,
        "rows_out": int(rows_out),
        "imu_duration_s": float(pre.duration),
        "median_dt_s": float(pre.median_dt),
        "has_header": header is not None,
        "preprocessing": pre.info,
        "estimator": summary,
    }


def repo_root():
    """The directory that contains src/ (the submission root)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_input_csv():
    """
    Input for a run without an input argument, or None.

    Order: imu0/data.csv in the working directory, imu0/data.csv at the
    repository root, data/imu0/data.csv, then the first degraded.csv or
    .../imu0/data.csv found under data/ (sorted, so the choice is stable).
    """
    bases = [os.getcwd(), repo_root()]
    for rel in (os.path.join("imu0", "data.csv"), os.path.join("data", "imu0", "data.csv")):
        for base in bases:
            path = os.path.join(base, rel)
            if os.path.isfile(path):
                return path
    found = []
    for base in bases:
        for dirpath, dirnames, files in os.walk(os.path.join(base, "data")):
            dirnames.sort()
            if "degraded.csv" in files:
                found.append((0, os.path.join(dirpath, "degraded.csv")))
            if "data.csv" in files and os.path.basename(dirpath) == "imu0":
                found.append((1, os.path.join(dirpath, "data.csv")))
    return min(found)[1] if found else None


def resolve_paths(ap, args, no_arguments):
    """(input, output, metadata) paths for the three invocation styles."""
    if args.input and args.input_pos and args.output_pos is None:
        # "--input a.csv out.txt": the lone positional is the output
        args.output_pos, args.input_pos = args.input_pos, None
    elif args.input and args.input_pos:
        ap.error("the input CSV was given twice (%s and %s)" % (args.input, args.input_pos))
    if args.output and args.output_pos:
        ap.error("the output file was given twice (%s and %s)" % (args.output, args.output_pos))
    input_path = args.input or args.input_pos or default_input_csv()
    if not input_path:
        ap.error("no input CSV: pass --input <imu0/data.csv> (or a positional path), "
                 "or place the file at imu0/data.csv")
    if not os.path.isfile(input_path):
        ap.error("input file not found: %s" % input_path)
    output_path = args.output or args.output_pos
    if not output_path:
        # zero-argument run: the submission files belong at the repository root
        output_path = os.path.join(repo_root() if no_arguments else os.getcwd(), "estimated_trajectory.txt")
    meta_path = args.meta
    if meta_path is None:
        meta_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), "metadata.json")
    return input_path, output_path, meta_path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="VITChennai_ZeroDrift - IMU-only trajectory estimator (Track 3)",
        usage="main.py [--input CSV] [--output TXT] [--meta JSON]  |  main.py CSV [TXT]  |  main.py")
    ap.add_argument("input_pos", nargs="?", default=None, metavar="CSV",
                    help="degraded IMU CSV (positional alternative to --input)")
    ap.add_argument("output_pos", nargs="?", default=None, metavar="TXT",
                    help="TUM trajectory to write (positional alternative to --output)")
    ap.add_argument("--input", "-i", default=None, help="degraded IMU CSV (default: imu0/data.csv)")
    ap.add_argument("--output", "-o", default=None, help="TUM trajectory to write (default: estimated_trajectory.txt)")
    ap.add_argument("--meta", "-m", "--metadata", dest="meta", default=None,
                    help="runtime / memory / preprocessing statistics (JSON); default: metadata.json "
                         "next to the trajectory; '' disables")
    ap.add_argument("--no-denoise", action="store_true", help="skip the severity-adaptive smoother (ablation)")
    ap.add_argument("--no-memtrace", action="store_true",
                    help="do not trace the heap (timing without tracemalloc overhead)")
    argv = sys.argv[1:] if argv is None else list(argv)
    args = ap.parse_args(argv)
    input_path, output_path, meta_path = resolve_paths(ap, args, no_arguments=not argv)

    trace = not args.no_memtrace
    if trace:
        tracemalloc.start()
    t_start = time.perf_counter()
    stats = run_pipeline(input_path, output_path, denoise=not args.no_denoise)
    wall = time.perf_counter() - t_start
    peak_mb = None
    if trace:
        peak_mb = tracemalloc.get_traced_memory()[1] / (1024.0 * 1024.0)
        tracemalloc.stop()

    duration = stats["imu_duration_s"] if stats["imu_duration_s"] > 0 else float("nan")
    ms_per_s = 1000.0 * wall / duration
    ok_rows = stats["rows_in"] == stats["rows_out"]
    ok_speed = bool(ms_per_s <= SPEED_LIMIT_MS_PER_S)
    ok_mem = None if peak_mb is None else bool(peak_mb <= MEMORY_LIMIT_MB)
    meta = {
        "project": "VITChennai_ZeroDrift",
        "track": "Samsung XR Hackathon 2026 - Track 3: Path Finder",
        "input": os.path.basename(input_path),
        "output": os.path.basename(output_path),
        "rows_in": stats["rows_in"],
        "rows_out": stats["rows_out"],
        "imu_duration_s": round(stats["imu_duration_s"], 6),
        "median_dt_s": round(stats["median_dt_s"], 9),
        "wall_time_s": round(wall, 4),
        "ms_per_1s_imu": round(ms_per_s, 3) if math.isfinite(ms_per_s) else None,
        "peak_memory_mb": None if peak_mb is None else round(peak_mb, 3),
        "memory_traced_with": "tracemalloc (whole pipeline, timing includes its overhead)" if trace else None,
        "limits": {"peak_memory_mb": MEMORY_LIMIT_MB, "ms_per_1s_imu": SPEED_LIMIT_MS_PER_S},
        "compliant": bool(ok_rows and ok_speed and (ok_mem is not False)),
        "threads": 1,
        "preprocessing": stats["preprocessing"],
        "estimator": stats["estimator"],
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
    }
    if meta_path:
        meta_dir = os.path.dirname(os.path.abspath(meta_path))
        os.makedirs(meta_dir, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print("[zerodrift] %s -> %s" % (input_path, output_path))
    print("[zerodrift] rows_in=%d rows_out=%d duration=%.2fs" % (
        stats["rows_in"], stats["rows_out"], stats["imu_duration_s"]))
    print("[zerodrift] %.2f ms per 1 s of IMU (limit %.0f)%s" % (
        ms_per_s, SPEED_LIMIT_MS_PER_S,
        "" if peak_mb is None else ", peak heap %.3f MB (limit %.0f)" % (peak_mb, MEMORY_LIMIT_MB)))
    if not meta["compliant"]:
        print("[zerodrift] WARNING: outside the Track 3 limits (see %s)" % (meta_path or "the lines above"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
