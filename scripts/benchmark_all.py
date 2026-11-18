#!/usr/bin/env python3
"""
benchmark_all.py - end-to-end verification suite for Track 3 (Path Finder).

For every benchmark sequence (EuRoC V1_01_easy / MH_01_easy and two synthetic
recordings) this script

  1. writes the clean recording as a 7-column CSV in degrade_imu.py layout
     (timestamp, ax, ay, az, gx, gy, gz) plus a TUM ground-truth file,
  2. degrades it with the organiser's unmodified scripts/degrade_imu.py
     (--meta stats.json),
  3. runs   uv run python src/main.py --input degraded.csv
                 --output estimated_trajectory.txt --meta metadata.json,
  4. asserts the submission rules:
        rows_out == rows_in (one pose per row of degraded.csv, same order,
        same timestamps), 8 finite columns, |q| = 1,
        peak_memory_mb <= 8.0 (tracemalloc heap), ms_per_1s_imu <= 20.0,
  5. scores it with evo exactly as the FAQ describes (SE(3) Umeyama):
        evo_ape tum gt.tum estimated_trajectory.txt -va --align
        evo_rpe tum gt.tum estimated_trajectory.txt -va --align --delta <frames in 1 s> --delta_unit f
     (evo offers no seconds unit for --delta; 1 s is given as a frame count),
  6. scores two reference baselines with the same alignment (naive unfiltered
     strapdown integration of the degraded file, and a stationary trajectory),
  7. renders the comparison plots into plots/.

Run through uv:   uv run python scripts/benchmark_all.py [--fetch] [--sequences ...]
Exit status is non-zero if any compliance assertion fails.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import synthetic_sequences as syn  # noqa: E402

G = 9.80665
EUROC = ("V1_01_easy", "MH_01_easy")
DEFAULT_SEQUENCES = EUROC + tuple(syn.SEQUENCES)
MEMORY_LIMIT_MB = 8.0
SPEED_LIMIT_MS = 20.0
ASSOC_MAX_DIFF = 0.01


# ----------------------------------------------------------------------------
# 1. clean recordings in degrade_imu.py layout
# ----------------------------------------------------------------------------

def write_tum(path, t, pos, quat_xyzw):
    arr = np.column_stack((t, pos, quat_xyzw))
    np.savetxt(path, arr, fmt=["%.9f"] + ["%.6f"] * 3 + ["%.9f"] * 4)


def prepare_euroc(seq, out_dir, data_root):
    src = os.path.join(data_root, "euroc", seq)
    imu_path, gt_path = os.path.join(src, "imu.csv"), os.path.join(src, "groundtruth.csv")
    if not (os.path.isfile(imu_path) and os.path.isfile(gt_path)):
        return None
    imu = pd.read_csv(imu_path)
    # raw EuRoC order is [t, gx, gy, gz, ax, ay, az]; degrade_imu.py wants accel first
    clean = pd.DataFrame({
        "timestamp": imu.iloc[:, 0].to_numpy(),
        "ax": imu.iloc[:, 4].to_numpy(), "ay": imu.iloc[:, 5].to_numpy(), "az": imu.iloc[:, 6].to_numpy(),
        "gx": imu.iloc[:, 1].to_numpy(), "gy": imu.iloc[:, 2].to_numpy(), "gz": imu.iloc[:, 3].to_numpy(),
    })
    clean.to_csv(os.path.join(out_dir, "clean.csv"), index=False)
    gt = pd.read_csv(gt_path).to_numpy(dtype=np.float64)
    write_tum(os.path.join(out_dir, "gt.tum"), gt[:, 0] * 1e-9, gt[:, 1:4], gt[:, [5, 6, 7, 4]])
    return {"kind": "euroc", "rows_clean": int(len(clean))}


def prepare_synthetic(seq, out_dir):
    spec = syn.SEQUENCES[seq]
    d = spec["make"]()
    ts = d["t"] * {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}[spec["ts_unit"]]
    acc = d["acc"] / G if spec["accel_unit"] == "g" else d["acc"]
    clean = pd.DataFrame(np.column_stack((ts, acc, d["gyr"])),
                         columns=["timestamp", "ax", "ay", "az", "gx", "gy", "gz"])
    clean.to_csv(os.path.join(out_dir, "clean.csv"), index=False, header=spec["header"])
    write_tum(os.path.join(out_dir, "gt.tum"), d["t"], d["pos"], d["quat"])
    return {"kind": "synthetic", "rows_clean": int(len(clean))}


# ----------------------------------------------------------------------------
# 2-3. degrade + run
# ----------------------------------------------------------------------------

def run(cmd, log_path=None, cwd=ROOT):
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("$ " + " ".join(cmd) + "\n" + proc.stdout)
    return proc.returncode, proc.stdout


def python_cmd():
    """`uv run python` when uv is available (the documented way), else this interpreter."""
    if shutil.which("uv") and not os.environ.get("ZERODRIFT_NO_UV"):
        return ["uv", "run", "--quiet", "python"]
    return [sys.executable]


def evo_cmd(tool):
    exe = shutil.which(tool) or shutil.which(tool, path=os.path.dirname(sys.executable))
    if exe:
        return [exe]
    return [sys.executable, "-c", "from evo.entry_points import %s as m; m()" % tool.split("_")[1]]


# ----------------------------------------------------------------------------
# 4. compliance
# ----------------------------------------------------------------------------

def load_degraded(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        first = f.readline().split(",")[0]
    try:
        float(first)
        header = None
    except ValueError:
        header = 0
    return pd.read_csv(path, header=header).to_numpy(dtype=np.float64)


def check_compliance(deg, est_path, meta, stats):
    """Returns (list of (name, passed, detail)), est array."""
    checks = []

    def add(name, ok, detail):
        checks.append((name, bool(ok), detail))

    with open(est_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.read().split("\n") if ln.strip()]
    cols = {len(ln.split()) for ln in lines}
    add("8 columns per row", cols == {8}, "column counts seen: %s" % sorted(cols))
    est = np.array([[float(v) for v in ln.split()] for ln in lines]) if cols == {8} else np.zeros((0, 8))

    rows_in = deg.shape[0]
    add("rows_out == rows_in", est.shape[0] == rows_in and meta["rows_out"] == rows_in == meta["rows_in"]
        and stats["rows_out"] == rows_in,
        "degraded.csv=%d estimated=%d metadata=%d/%d degrade stats=%d" % (
            rows_in, est.shape[0], meta["rows_in"], meta["rows_out"], stats["rows_out"]))
    if est.shape[0] == rows_in:
        scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[stats["timestamp_unit"]]
        err = np.abs(est[:, 0] - deg[:, 0] * scale).max()
        add("timestamps = input rows in seconds", err < 1e-6, "max |dt| = %.2e s" % err)
        dup = np.flatnonzero(np.diff(deg[:, 0]) == 0) + 1
        same = np.abs(est[dup, 1:] - est[dup - 1, 1:]).max() if dup.size else 0.0
        add("duplicate rows repeat the original pose", same == 0.0, "%d duplicates, max diff %.1e" % (dup.size, same))
    finite = np.isfinite(est).all() if est.size else False
    add("no NaN / Inf", finite, "all finite" if finite else "non-finite values present")
    qerr = np.abs(np.linalg.norm(est[:, 4:8], axis=1) - 1.0).max() if est.size else 1.0
    add("unit quaternions", qerr < 1e-6, "max | |q| - 1 | = %.1e" % qerr)
    add("peak_memory_mb <= %.1f" % MEMORY_LIMIT_MB, meta["peak_memory_mb"] is not None
        and meta["peak_memory_mb"] <= MEMORY_LIMIT_MB, "%.3f MB (tracemalloc heap)" % meta["peak_memory_mb"])
    add("ms_per_1s_imu <= %.1f" % SPEED_LIMIT_MS, meta["ms_per_1s_imu"] <= SPEED_LIMIT_MS,
        "%.2f ms per 1 s of IMU, 1 thread, incl. tracemalloc overhead" % meta["ms_per_1s_imu"])
    return checks, est


# ----------------------------------------------------------------------------
# 5-6. scoring
# ----------------------------------------------------------------------------

def associate(t_est, t_gt, max_diff=ASSOC_MAX_DIFF):
    j = np.clip(np.searchsorted(t_gt, t_est), 1, len(t_gt) - 1)
    j = np.where(np.abs(t_gt[j - 1] - t_est) <= np.abs(t_gt[j] - t_est), j - 1, j)
    ok = np.abs(t_gt[j] - t_est) <= max_diff
    return np.flatnonzero(ok), j[ok]


def umeyama_se3(x, y):
    """Rigid (R, t) minimising ||y - (R x + t)||."""
    mx, my = x.mean(axis=0), y.mean(axis=0)
    cov = (y - my).T @ (x - mx) / x.shape[0]
    U, _, Vt = np.linalg.svd(cov)
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt)) or 1.0])
    R = U @ S @ Vt
    return R, my - R @ mx


def ate_se3(est_t, est_p, gt):
    """ATE after SE(3) Umeyama alignment. Returns (rmse, aligned positions, errors, gt positions, times)."""
    ie, ig = associate(est_t, gt[:, 0])
    x, y = est_p[ie], gt[ig, 1:4]
    R, t = umeyama_se3(x, y)
    xa = x @ R.T + t
    err = np.linalg.norm(xa - y, axis=1)
    return float(np.sqrt(np.mean(err ** 2))), xa, err, y, est_t[ie]


def naive_integration(deg, stats):
    """
    Reference baseline: plain strapdown integration of the degraded file with
    no preprocessing, no bias estimation and no regularisation. Attitude starts
    from the gravity direction of the first 2 s.
    """
    scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[stats["timestamp_unit"]]
    t = (deg[:, 0] - deg[0, 0]) * scale
    acc = deg[:, 1:4] * (G if stats["accel_unit"] == "g" else 1.0)
    gyr = deg[:, 4:7]
    f0 = acc[t < 2.0].mean(axis=0)
    g0 = np.linalg.norm(f0)
    u = f0 / g0
    axis = np.array([u[1], -u[0], 0.0])
    q = np.array([axis[0], axis[1], axis[2], 1.0 + u[2]])
    q /= np.linalg.norm(q)
    pos = np.zeros((len(t), 3))
    v = np.zeros(3)
    p = np.zeros(3)
    qx, qy, qz, qw = q
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        wx, wy, wz = 0.5 * dt * gyr[i]
        nx = qx + qw * wx + qy * wz - qz * wy
        ny = qy + qw * wy - qx * wz + qz * wx
        nz = qz + qw * wz + qx * wy - qy * wx
        nw = qw - qx * wx - qy * wy - qz * wz
        n = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz + nw * nw)
        qx, qy, qz, qw = nx * n, ny * n, nz * n, nw * n
        ax, ay, az = acc[i]
        tx, ty, tz = 2 * (qy * az - qz * ay), 2 * (qz * ax - qx * az), 2 * (qx * ay - qy * ax)
        a = np.array([ax + qw * tx + qy * tz - qz * ty, ay + qw * ty + qz * tx - qx * tz,
                      az + qw * tz + qx * ty - qy * tx - g0])
        p = p + v * dt + 0.5 * a * dt * dt
        v = v + a * dt
        pos[i] = p
    return deg[:, 0] * scale, pos


def parse_evo_zip(path):
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("stats.json"))


def run_evo(tool, gt_path, est_path, out_dir, extra):
    res_zip = os.path.join(out_dir, tool + ".zip")
    if os.path.exists(res_zip):
        os.remove(res_zip)
    cmd = evo_cmd(tool) + ["tum", gt_path, est_path, "-va", "--align"] + extra
    code, out = run(cmd + ["--save_results", res_zip, "--no_warnings"], os.path.join(out_dir, tool + ".log"))
    if code != 0 or not os.path.exists(res_zip):
        print(out)
        raise RuntimeError("%s failed" % tool)
    return parse_evo_zip(res_zip), " ".join([tool] + cmd[len(evo_cmd(tool)):])


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def benchmark_sequence(seq, args):
    out_dir = os.path.join(args.out, seq)
    os.makedirs(out_dir, exist_ok=True)
    if seq in syn.SEQUENCES:
        prep = prepare_synthetic(seq, out_dir)
    else:
        prep = prepare_euroc(seq, out_dir, args.data)
        if prep is None and args.fetch:
            run([sys.executable, os.path.join(HERE, "fetch_euroc.py"), seq, "--out",
                 os.path.join(args.data, "euroc")])
            prep = prepare_euroc(seq, out_dir, args.data)
        if prep is None:
            print("[skip] %s: EuRoC CSVs not found under %s (run with --fetch)" % (seq, args.data))
            return None

    paths = {k: os.path.join(out_dir, v) for k, v in {
        "clean": "clean.csv", "deg": "degraded.csv", "stats": "stats.json", "gt": "gt.tum",
        "est": "estimated_trajectory.txt", "meta": "metadata.json"}.items()}

    code, out = run([sys.executable, os.path.join(HERE, "degrade_imu.py"), paths["clean"], paths["deg"],
                     "--meta", paths["stats"]], os.path.join(out_dir, "degrade.log"))
    if code != 0:
        print(out)
        raise RuntimeError("degrade_imu.py failed for " + seq)
    with open(paths["stats"]) as f:
        stats = json.load(f)

    main_cmd = python_cmd() + [os.path.join("src", "main.py"), "--input", paths["deg"],
                               "--output", paths["est"], "--meta", paths["meta"]] + args.main_args
    t0 = time.perf_counter()
    code, out = run(main_cmd, os.path.join(out_dir, "main.log"))
    wall = time.perf_counter() - t0
    if code != 0:
        print(out)
        raise RuntimeError("src/main.py failed for " + seq)
    with open(paths["meta"]) as f:
        meta = json.load(f)
    # second, untraced run: timing without tracemalloc overhead (information only)
    untraced = os.path.join(out_dir, "metadata_untraced.json")
    run(python_cmd() + [os.path.join("src", "main.py"), "--input", paths["deg"], "--output",
                        os.path.join(out_dir, "_untraced.txt"), "--meta", untraced, "--no-memtrace"] + args.main_args)
    with open(untraced) as f:
        ms_untraced = json.load(f)["ms_per_1s_imu"]
    os.remove(os.path.join(out_dir, "_untraced.txt"))

    deg = load_degraded(paths["deg"])
    checks, est = check_compliance(deg, paths["est"], meta, stats)

    gt = np.loadtxt(paths["gt"])
    ape, ape_cmd = run_evo("evo_ape", paths["gt"], paths["est"], out_dir, [])
    # evo has no time-based delta unit (choices: f, d, r, m), so "1 second" is
    # expressed as the number of frames that span one second at this IMU rate.
    frames_1s = max(1, int(round(1.0 / meta["median_dt_s"])))
    rpe, rpe_cmd = run_evo("evo_rpe", paths["gt"], paths["est"], out_dir,
                           ["--delta", str(frames_1s), "--delta_unit", "f"])

    own_rmse, est_al, est_err, gt_al, t_al = ate_se3(est[:, 0], est[:, 1:4], gt)
    nt, npos = naive_integration(deg, stats)
    naive_rmse, naive_al, naive_err, _, naive_t = ate_se3(nt, npos, gt)
    ie, ig = associate(est[:, 0], gt[:, 0])
    gp = gt[ig, 1:4]
    # for drawing only: the naive trajectory in the estimate's aligned frame (both start at
    # the origin of the same gravity-aligned frame), so the plot shows where it departs
    R_est, t_est = umeyama_se3(est[ie, 1:4], gp)
    naive_draw = npos @ R_est.T + t_est
    stationary_rmse = float(np.sqrt(np.mean(np.sum((gp - gp.mean(axis=0)) ** 2, axis=1))))
    path_len = float(np.linalg.norm(np.diff(gt[:, 1:4], axis=0), axis=1).sum())

    result = {
        "sequence": seq, "kind": prep["kind"],
        "rows_clean": stats["rows_in"], "rows_degraded": stats["rows_out"], "rows_estimated": int(est.shape[0]),
        "duration_s": stats["duration_s"], "gt_path_length_m": round(path_len, 2),
        "degradation": {k: stats[k] for k in ("spikes", "stuck_samples", "dropped_single", "dropped_burst_gaps",
                                               "dropped_total", "duplicated_rows", "failure_frames")},
        "detected": meta["preprocessing"],
        "estimator": meta["estimator"],
        "peak_memory_mb": meta["peak_memory_mb"],
        "ms_per_1s_imu": meta["ms_per_1s_imu"],
        "ms_per_1s_imu_untraced": ms_untraced,
        "process_wall_s": round(wall, 3),
        "ate_rmse_m": ape["rmse"], "ate_mean_m": ape["mean"], "ate_median_m": ape["median"], "ate_max_m": ape["max"],
        "rpe_1s_rmse_m": rpe["rmse"], "rpe_1s_mean_m": rpe["mean"], "rpe_1s_max_m": rpe["max"],
        "ate_rmse_own_m": own_rmse,
        "naive_ate_rmse_m": naive_rmse,
        "stationary_ate_rmse_m": stationary_rmse,
        "evo_commands": [ape_cmd, rpe_cmd],
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
        "compliant": all(p for _, p, _ in checks),
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== %s  (%s, %.1f s, %d -> %d rows) ===" % (
        seq, prep["kind"], stats["duration_s"], stats["rows_in"], stats["rows_out"]))
    for n, p, d in checks:
        print("  [%s] %-42s %s" % ("PASS" if p else "FAIL", n, d))
    print("  evo ATE  (SE3 Umeyama)  rmse %.3f m   mean %.3f   max %.3f" % (ape["rmse"], ape["mean"], ape["max"]))
    print("  evo RPE  (delta 1 s)    rmse %.3f m   mean %.3f   max %.3f" % (rpe["rmse"], rpe["mean"], rpe["max"]))
    print("  baselines ATE rmse: naive integration %.1f m | stationary %.3f m" % (naive_rmse, stationary_rmse))

    if not args.no_plots:
        import make_plots
        make_plots.plot_preprocessing(seq, paths["deg"], stats, os.path.join(args.plots, seq + "_imu_preprocessing.png"))
        make_plots.plot_trajectory(seq, result, t_al, gt_al, est_al, est_err, naive_t, naive_err, nt, naive_draw,
                                   meta, os.path.join(args.plots, seq + "_trajectory.png"))
    return result


def main():
    ap = argparse.ArgumentParser(description="Track 3 verification suite (degrade -> estimate -> assert -> evo -> plots)")
    ap.add_argument("--sequences", nargs="*", default=list(DEFAULT_SEQUENCES))
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--plots", default=os.path.join(ROOT, "plots"))
    ap.add_argument("--fetch", action="store_true", help="download missing EuRoC IMU/ground-truth CSVs")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--publish", action="store_true",
                    help="copy the first sequence's estimated_trajectory.txt, metadata.json and ground truth "
                         "(ground_truth_trajectory.txt) to the repo root")
    ap.add_argument("--main-args", nargs=argparse.REMAINDER, default=[],
                    help="extra arguments forwarded to src/main.py (e.g. --no-denoise)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.plots, exist_ok=True)

    results = []
    for seq in args.sequences:
        r = benchmark_sequence(seq, args)
        if r is not None:
            results.append(r)
    if not results:
        print("no sequence could be benchmarked")
        return 1

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"limits": {"peak_memory_mb": MEMORY_LIMIT_MB, "ms_per_1s_imu": SPEED_LIMIT_MS},
                   "sequences": results}, f, indent=2)
    if not args.no_plots:
        import make_plots
        make_plots.plot_dashboard(results, MEMORY_LIMIT_MB, SPEED_LIMIT_MS,
                                  os.path.join(args.plots, "compliance_dashboard.png"))
    if args.publish:
        first = os.path.join(args.out, results[0]["sequence"])
        shutil.copyfile(os.path.join(first, "estimated_trajectory.txt"), os.path.join(ROOT, "estimated_trajectory.txt"))
        shutil.copyfile(os.path.join(first, "metadata.json"), os.path.join(ROOT, "metadata.json"))
        shutil.copyfile(os.path.join(first, "gt.tum"), os.path.join(ROOT, "ground_truth_trajectory.txt"))
        print("\npublished %s -> estimated_trajectory.txt, metadata.json, ground_truth_trajectory.txt" % results[0]["sequence"])

    print("\n%-16s %7s %7s %9s %9s %9s %8s %8s  %s" % (
        "sequence", "rows", "ATE[m]", "RPE1s[m]", "naive[m]", "static[m]", "heap[MB]", "ms/s", "rules"))
    for r in results:
        print("%-16s %7d %7.3f %9.3f %9.1f %9.3f %8.3f %8.2f  %s" % (
            r["sequence"], r["rows_estimated"], r["ate_rmse_m"], r["rpe_1s_rmse_m"], r["naive_ate_rmse_m"],
            r["stationary_ate_rmse_m"], r["peak_memory_mb"], r["ms_per_1s_imu"],
            "PASS" if r["compliant"] else "FAIL"))
    ok = all(r["compliant"] for r in results)
    print("\n%s" % ("ALL COMPLIANCE CHECKS PASSED" if ok else "COMPLIANCE FAILURES - see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
