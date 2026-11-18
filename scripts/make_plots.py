#!/usr/bin/env python3
"""
make_plots.py - figures for the Track 3 verification suite (called by benchmark_all.py).

  plot_preprocessing()  raw degraded IMU vs the output of src/inverse_degrade.py:
                        whole recording (|acc|, |gyr|, sample interval) plus zooms on
                        one removed spike, one interpolated stuck run and the 5 s
                        blind window.
  plot_trajectory()     ground truth vs estimate (after SE(3) Umeyama alignment) vs
                        naive unfiltered integration, in 3D and top view, with the
                        position error over time.
  plot_dashboard()      ATE / RPE against the baselines and heap / runtime against the
                        Track 3 limits for every benchmarked sequence.

Plotting needs matplotlib and is never imported by src/main.py.
"""

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from inverse_degrade import CONFIG, G, inverse_degrade, read_imu_csv, severity  # noqa: E402

C_GT = "#222222"
C_EST = "#1f6fb4"
C_RAW = "#d1495b"
C_CLEAN = "#1f6fb4"
C_NAIVE = "#e69f00"
C_BLIND = "#8e6bbf"
C_OK = "#2a9d8f"
C_BAD = "#d1495b"
C_REF = "#9aa0a6"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6, "figure.dpi": 100, "savefig.dpi": 200,
    "font.family": "DejaVu Sans",
})


def _save(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 1. raw degraded vs inverse-preprocessed IMU
# ----------------------------------------------------------------------------

def _load_stages(deg_path):
    """Raw (SI units, de-duplicated), repaired (spikes + stuck) and fully cleaned sensor arrays."""
    header, ts, x = read_imu_csv(deg_path)
    raw = x.copy()
    pre = inverse_degrade(header, ts, x)
    info = pre.info
    raw = raw[~pre.dup]
    if info["column_order"].startswith("gyro"):
        raw = raw[:, [3, 4, 5, 0, 1, 2]]
    raw = raw.astype(np.float64)
    if info["accel_unit"] == "g":
        raw[:, :3] *= G
    if info["gyro_unit"] == "deg/s":
        raw[:, 3:] *= np.pi / 180.0
    clean = np.column_stack((pre.acc, pre.gyr)).astype(np.float64)
    t = pre.t.copy()
    header, ts, x = read_imu_csv(deg_path)
    rep = inverse_degrade(header, ts, x, denoise=False)
    repaired = np.column_stack((rep.acc, rep.gyr)).astype(np.float64)
    n = pre.n_active
    stuck = np.zeros(t.shape[0], dtype=bool)
    stuck[1:n] = (np.abs(raw[1:n] - raw[:n - 1]) < 1e-7).all(axis=1)
    spike = np.zeros(t.shape[0], dtype=bool)
    spike[:n] = (np.abs(repaired[:n] - raw[:n]) > 1e-4).any(axis=1) & ~stuck[:n]
    return t, raw, repaired, clean, stuck, spike, pre


def _zoom(ax, t, raw, clean, lo, hi, ch, title, unit, mark=None, mark_label=None):
    sl = slice(max(lo, 0), min(hi, t.shape[0]))
    ax.plot(t[sl], raw[sl, ch], "-o", color=C_RAW, ms=3, lw=1.0, label="raw degraded")
    ax.plot(t[sl], clean[sl, ch], "-o", color=C_CLEAN, ms=2.5, lw=1.2, label="inverse-preprocessed")
    if mark is not None and mark.size:
        ax.plot(t[mark], raw[mark, ch], "o", ms=9, mfc="none", mec="k", mew=1.2, label=mark_label)
    ax.set_title(title)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(unit)
    ax.legend(loc="best", frameon=False)


def plot_preprocessing(seq, deg_path, stats, out_png):
    t, raw, repaired, clean, stuck, spike, pre = _load_stages(deg_path)
    info, n = pre.info, pre.n_active
    T = pre.duration
    t_fail = t[n] if n < t.shape[0] else None
    an_raw, an_cl = np.linalg.norm(raw[:, :3], axis=1), np.linalg.norm(clean[:, :3], axis=1)
    gn_raw, gn_cl = np.linalg.norm(raw[:, 3:], axis=1), np.linalg.norm(clean[:, 3:], axis=1)

    fig = plt.figure(figsize=(14, 11.5))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1.0, 1.0, 0.7, 1.25], hspace=0.42, wspace=0.25)

    def shade(ax):
        ax.axvspan(0.0, CONFIG["protect_seconds"], color=C_OK, alpha=0.10, lw=0)
        if t_fail is not None:
            ax.axvspan(t_fail, T, color=C_BLIND, alpha=0.15, lw=0)
        ax.set_xlim(0.0, T)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, an_raw, color=C_RAW, lw=0.5, label="raw degraded", rasterized=True)
    ax.plot(t[:n], an_cl[:n], color=C_CLEAN, lw=0.5, label="inverse-preprocessed", rasterized=True)
    if spike.any():
        ax.plot(t[spike], an_raw[spike], "v", color="k", ms=4, label="spike removed (%d)" % int(spike.sum()))
    shade(ax)
    ax.set_ylabel("|acc| [m/s²]")
    ax.set_title("%s — raw degraded IMU vs inverse-preprocessed IMU   (green: protected 2 s, purple: 5 s sensor failure)" % seq)
    ax2 = ax.twinx()
    ax2.plot(t, severity(t / max(T, 1e-9)), color=C_REF, lw=1.2, ls="--")
    ax2.set_ylabel("severity sev(t)", color=C_REF)
    ax2.set_ylim(0, 1.05)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.legend(loc="upper left", ncol=3, frameon=False)

    ax = fig.add_subplot(gs[1, :])
    ax.plot(t, gn_raw, color=C_RAW, lw=0.5, rasterized=True)
    ax.plot(t[:n], gn_cl[:n], color=C_CLEAN, lw=0.5, rasterized=True)
    if spike.any():
        ax.plot(t[spike], gn_raw[spike], "v", color="k", ms=4)
    shade(ax)
    ax.set_ylabel("|gyr| [rad/s]")

    ax = fig.add_subplot(gs[2, :])
    dt = np.diff(t)
    ax.semilogy(t[1:], np.maximum(dt, 1e-6) * 1e3, ".", color="#444444", ms=1.5, rasterized=True)
    ax.axhline(1.5 * pre.median_dt * 1e3, color=C_NAIVE, lw=1.0, ls="--", label="1.5 × median dt → RK4 sub-stepping")
    shade(ax)
    ax.set_ylabel("sample interval [ms]")
    ax.set_xlabel("time [s]")
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.995, 0.92, "dropout gaps: %d (max %.0f ms)   duplicated rows removed: %d   stuck samples: %d in %d runs" % (
        info["dropout_gaps"], info["max_gap_s"] * 1e3, info["duplicate_rows"], info["stuck_samples"], info["stuck_runs"]),
        transform=ax.transAxes, ha="right", va="top", fontsize=8)

    # zoom 1: the largest removed spike
    ax = fig.add_subplot(gs[3, 0])
    if spike.any():
        dev = np.abs(repaired - raw)[:, :3] * spike[:, None]
        i = int(np.argmax(dev.max(axis=1)))
        ch = int(np.argmax(dev[i]))
        m = np.flatnonzero(spike[max(i - 14, 0):i + 15]) + max(i - 14, 0)
        _zoom(ax, t, raw, clean, i - 14, i + 15, ch, "spike removal (Hampel, joint 6-channel test)",
              "acc %s [m/s²]" % "xyz"[ch], m, "flagged sample")
    else:
        ax.set_title("no spikes detected")

    # zoom 2: the longest stuck run
    ax = fig.add_subplot(gs[3, 1])
    edges = np.flatnonzero(np.diff(np.concatenate(([False], stuck, [False])).astype(np.int8)))
    if edges.size:
        runs = np.column_stack((edges[0::2], edges[1::2]))
        s, e = runs[int(np.argmax(runs[:, 1] - runs[:, 0]))]
        win = slice(max(s - 12, 0), min(e + 12, n))
        ch = 3 + int(np.argmax(np.ptp(clean[win, 3:], axis=0)))
        _zoom(ax, t, raw, clean, s - 12, e + 12, ch, "stuck-run repair (%d identical samples → linear interpolation)" % (e - s + 1),
              "gyr %s [rad/s]" % "xyz"[ch - 3], np.arange(s, e), "stuck sample")
    else:
        ax.set_title("no stuck runs detected")

    # zoom 3: blind window
    ax = fig.add_subplot(gs[3, 2])
    if t_fail is not None:
        lo = int(np.searchsorted(t, t_fail - 4.0))
        ax.plot(t[lo:], an_raw[lo:], color=C_RAW, lw=0.9, label="raw degraded |acc|")
        ax.plot(t[lo:n], an_cl[lo:n], color=C_CLEAN, lw=0.9, label="inverse-preprocessed |acc|")
        ax.axvspan(t_fail, T, color=C_BLIND, alpha=0.15, lw=0)
        ax.text(0.5 * (t_fail + T), 0.45 * float(np.max(an_raw[lo:n])),
                "blind window\n%d rows, acc = gyr = 0\nexcluded from the filter,\npose extrapolated" % info["failure_rows"],
                ha="center", va="center", fontsize=8.5, color="#4b3a75")
        ax.set_title("5 s failure window (detected at t = %.2f s)" % t_fail)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("|acc| [m/s²]")
        ax.legend(loc="upper left", frameon=False)
    else:
        ax.set_title("no failure window (recording ≤ 7 s)")

    fig.text(0.5, 0.005,
             "degrade_imu.py injected: %d spikes, %d stuck samples, %d dropped rows (%d burst gaps), %d duplicated rows, %d failure rows      "
             "detected: %d spikes, %d stuck, %d gaps, %d duplicates, %d failure rows      units: %s / %s / %s, order %s" % (
                 stats["spikes"], stats["stuck_samples"], stats["dropped_total"], stats["dropped_burst_gaps"],
                 stats["duplicated_rows"], stats["failure_frames"], info["spikes_removed"], info["stuck_samples"],
                 info["dropout_gaps"], info["duplicate_rows"], info["failure_rows"],
                 info["timestamp_unit"], info["accel_unit"], info["gyro_unit"], info["column_order"]),
             ha="center", va="bottom", fontsize=8, color="#333333")
    _save(fig, out_png)


# ----------------------------------------------------------------------------
# 2. trajectory: ground truth vs estimate vs naive integration
# ----------------------------------------------------------------------------

def _clip_to_box(p, lo, hi):
    """Samples of p that lie inside the (padded) box."""
    return ((p >= lo) & (p <= hi)).all(axis=1)


def plot_trajectory(seq, result, t_al, gt_al, est_al, est_err, naive_t, naive_err, draw_t, naive_draw, meta, out_png):
    """
    naive_draw is the naive integration mapped with the *estimate's* alignment (both start
    from the same gravity-aligned frame), so the picture shows where it departs; its own
    best SE(3) fit over a kilometre-scale divergence would not be near the plot at all.
    naive_err (at naive_t) is the error after the naive trajectory's own SE(3) Umeyama alignment.
    """
    naive_al = naive_draw
    t0 = t_al[0]
    tt = t_al - t0
    fail_s = meta["preprocessing"].get("failure_start_s")
    blind = tt >= fail_s if fail_s is not None else np.zeros(tt.shape[0], dtype=bool)
    both = np.vstack((gt_al, est_al))
    lo, hi = both.min(axis=0), both.max(axis=0)
    pad = 0.15 * max(float((hi - lo).max()), 1e-3)
    lo, hi = lo - pad, hi + pad
    inside = _clip_to_box(naive_al, lo, hi)
    first_out = int(np.argmax(~inside)) if (~inside).any() else naive_al.shape[0]
    nv = naive_al[:first_out]

    fig = plt.figure(figsize=(15, 9.5))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1.0], height_ratios=[1.0, 0.8], hspace=0.28, wspace=0.12)

    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax.plot(gt_al[:, 0], gt_al[:, 1], gt_al[:, 2], color=C_GT, lw=1.3, label="ground truth")
    ax.plot(est_al[~blind, 0], est_al[~blind, 1], est_al[~blind, 2], color=C_EST, lw=1.3,
            label="ZeroDrift estimate (SE(3) Umeyama aligned)")
    if blind.any():
        ax.plot(est_al[blind, 0], est_al[blind, 1], est_al[blind, 2], color=C_BLIND, lw=2.2, label="estimate, 5 s blind window")
    if nv.shape[0] > 1:
        ax.plot(nv[:, 0], nv[:, 1], nv[:, 2], color=C_NAIVE, lw=1.0, ls="--",
                label="naive integration, same frame (leaves the plot after %.1f s)" % (draw_t[first_out - 1] - draw_t[0]))
    ax.scatter(*gt_al[0], color=C_OK, s=30, depthshade=False, label="start")
    ax.scatter(*gt_al[-1], color=C_BAD, s=30, depthshade=False, label="end")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(np.maximum(hi - lo, 1e-6))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=28, azim=-58)
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(-0.08, 1.05))
    ax.set_title("%s — 3D trajectory" % seq)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(gt_al[:, 0], gt_al[:, 1], color=C_GT, lw=1.2, label="ground truth")
    ax.plot(est_al[~blind, 0], est_al[~blind, 1], color=C_EST, lw=1.2, label="estimate")
    if blind.any():
        ax.plot(est_al[blind, 0], est_al[blind, 1], color=C_BLIND, lw=2.2, label="blind window")
    if nv.shape[0] > 1:
        ax.plot(nv[:, 0], nv[:, 1], color=C_NAIVE, lw=1.0, ls="--", label="naive integration")
    ax.plot(*gt_al[0, :2], "o", color=C_OK, ms=6)
    ax.plot(*gt_al[-1, :2], "o", color=C_BAD, ms=6)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("top view")
    ax.legend(loc="best", frameon=False)

    ax = fig.add_subplot(gs[1, :])
    ax.semilogy(naive_t - t0, np.maximum(naive_err, 1e-3), color=C_NAIVE, lw=1.1, ls="--",
                label="naive unfiltered integration  (ATE rmse %.0f m)" % result["naive_ate_rmse_m"])
    ax.semilogy(tt, np.maximum(est_err, 1e-3), color=C_EST, lw=1.1,
                label="ZeroDrift estimate  (evo ATE rmse %.3f m)" % result["ate_rmse_m"])
    ax.axhline(result["stationary_ate_rmse_m"], color=C_REF, lw=1.2, ls=":",
               label="stationary trajectory  (ATE rmse %.3f m)" % result["stationary_ate_rmse_m"])
    if fail_s is not None:
        ax.axvspan(fail_s, tt[-1], color=C_BLIND, alpha=0.15, lw=0, label="5 s blind window")
    ax.set_xlim(0.0, tt[-1])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error after SE(3) alignment [m]")
    ax.set_title("absolute position error over time   (path length %.1f m, %d poses = %d input rows)" % (
        result["gt_path_length_m"], result["rows_estimated"], result["rows_degraded"]))
    ax.set_ylim(0.5 * max(float(est_err.min()), 1e-2), 3.0 * float(naive_err.max()))
    ax.legend(loc="center", ncol=2, frameon=False)
    _save(fig, out_png)


# ----------------------------------------------------------------------------
# 3. accuracy + compliance dashboard
# ----------------------------------------------------------------------------

def _bar_labels(ax, bars, fmt="%.2f", dy=0.01):
    top = ax.get_ylim()[1]
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + dy * top, fmt % b.get_height(),
                ha="center", va="bottom", fontsize=7.5)


def plot_dashboard(results, mem_limit, speed_limit, out_png):
    names = [r["sequence"] for r in results]
    x = np.arange(len(names))
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.28, height_ratios=[1.0, 0.95], width_ratios=[1.0, 1.0, 1.35])

    ax = fig.add_subplot(gs[0, 0:2])
    w = 0.38
    b1 = ax.bar(x - w / 2, [r["ate_rmse_m"] for r in results], w, color=C_EST, label="ZeroDrift (evo_ape, SE(3) Umeyama)")
    b2 = ax.bar(x + w / 2, [r["stationary_ate_rmse_m"] for r in results], w, color=C_REF, label="stationary trajectory baseline")
    ax.set_ylim(0, 1.25 * max(max(r["stationary_ate_rmse_m"], r["ate_rmse_m"]) for r in results))
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)
    for i, r in enumerate(results):
        ax.text(i, 0.985 * ax.get_ylim()[1], "naive integration: %s m" % ("{:,.0f}".format(r["naive_ate_rmse_m"])),
                ha="center", va="top", fontsize=7.5, color="#a36b00")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("ATE rmse [m]")
    ax.set_title("Absolute trajectory error on the degraded recordings (lower is better)")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.93))

    ax = fig.add_subplot(gs[0, 2])
    b = ax.bar(x, [r["rpe_1s_rmse_m"] for r in results], 0.55, color=C_EST)
    ax.set_ylim(0, 1.25 * max(r["rpe_1s_rmse_m"] for r in results))
    _bar_labels(ax, b, "%.3f")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("RPE rmse over 1 s [m]")
    ax.set_title("Relative pose error (evo_rpe, Δ = 1 s)")

    ax = fig.add_subplot(gs[1, 0])
    b = ax.bar(x, [r["peak_memory_mb"] for r in results], 0.55,
               color=[C_OK if r["peak_memory_mb"] <= mem_limit else C_BAD for r in results])
    ax.axhline(mem_limit, color=C_BAD, lw=1.4, ls="--", label="Track 3 limit %.0f MB" % mem_limit)
    ax.axhline(4.0, color=C_NAIVE, lw=1.0, ls=":", label="design target 4 MB")
    ax.set_ylim(0, 1.12 * mem_limit)
    _bar_labels(ax, b)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("peak heap [MB] (tracemalloc)")
    ax.set_title("Algorithm heap memory")
    ax.legend(loc="center right", frameon=False)

    ax = fig.add_subplot(gs[1, 1])
    w = 0.38
    b1 = ax.bar(x - w / 2, [r["ms_per_1s_imu"] for r in results], w,
                color=[C_OK if r["ms_per_1s_imu"] <= speed_limit else C_BAD for r in results], label="with tracemalloc (reported)")
    b2 = ax.bar(x + w / 2, [r["ms_per_1s_imu_untraced"] for r in results], w, color="#9ad1c9", label="without tracemalloc")
    ax.axhline(speed_limit, color=C_BAD, lw=1.4, ls="--", label="Track 3 limit %.0f ms" % speed_limit)
    ax.set_ylim(0, 1.12 * speed_limit)
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("ms per 1 s of IMU data (1 thread)")
    ax.set_title("Runtime")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.86))

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    checks = [c["name"] for c in results[0]["checks"]]
    cells = [["PASS" if c["passed"] else "FAIL" for c in r["checks"]] for r in results]
    wcol = 0.44 / max(len(names), 1)
    table = ax.table(cellText=[[checks[i]] + [row[i] for row in cells] for i in range(len(checks))],
                     colLabels=["assertion"] + [n.replace("_", "\n") for n in names],
                     colWidths=[0.56] + [wcol] * len(names), bbox=[-0.08, 0.0, 1.08, 0.93], cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if j == 0:
            cell.get_text().set_ha("left")
            cell.PAD = 0.03
        if i == 0:
            cell.set_height(cell.get_height() * 2.2)
            cell.get_text().set_fontweight("bold")
        if i > 0 and j >= 1:
            ok = cell.get_text().get_text() == "PASS"
            cell.set_facecolor("#d8f0ec" if ok else "#f6d5da")
            cell.get_text().set_color("#1c6b60" if ok else "#9b2335")
            cell.get_text().set_fontweight("bold")
    ax.set_title("Submission-rule assertions", pad=14)

    fig.suptitle("VITChennai_ZeroDrift — Track 3 Path Finder: accuracy and compliance on officially degraded recordings",
                 fontsize=12, y=0.955)
    _save(fig, out_png)


if __name__ == "__main__":
    # re-render the dashboard from an existing results/summary.json
    import json
    root = os.path.dirname(HERE)
    with open(os.path.join(root, "results", "summary.json")) as f:
        summary = json.load(f)
    plot_dashboard(summary["sequences"], summary["limits"]["peak_memory_mb"], summary["limits"]["ms_per_1s_imu"],
                   os.path.join(root, "plots", "compliance_dashboard.png"))
    print("wrote plots/compliance_dashboard.png")
