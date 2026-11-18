"""
inverse_degrade.py - white-box inverse of the organiser's degrade_imu.py.

The official degradation script (SEED = 42, G = 9.80665) is public, so every
artifact it injects has a known signature. This module undoes the reversible
ones, offline, in pure NumPy, before any trajectory integration:

    artifact (degrade_imu.py)          signature in the CSV           remedy
    ---------------------------------  -----------------------------  ---------------------------
    duplicated rows (np.repeat)        dt == 0, identical row         dropped, pose copied back
    failure window (last 5 s zeroed)   trailing ||acc|| = ||gyr|| = 0 excluded, dead-reckoned
    stuck sensor runs (2..10 samples)  identical consecutive rows     linear interpolation in t
    spikes (all 6 channels, 1 sample)  joint Hampel outlier           neighbour median
    white noise ramping with sev(t)    high-frequency energy          zero-phase adaptive smoother
    dropped rows (single + bursts)     dt > 1.5 * median dt           sub-stepped by the estimator

Bias and scale random walks are not reversible here; the estimator tracks
them (see es_ekf.py).

Memory model: the six sensor channels are held once as float32 and edited in
place; every rolling operation works on bounded chunks, so the heap cost is
~32 bytes per CSV row plus a fixed few hundred kB.

Only numpy + the standard library are imported (hackathon heap constraint).
"""

import math

import numpy as np

G = 9.80665

# Mirror of degrade_imu.py CONFIG (the white-box prior).
CONFIG = {
    "max_level": 1.0,
    "gamma": 1.7,
    "floor": 0.02,
    "protect_seconds": 2.0,
    "failure_seconds": 5.0,
    "accel_white_max": 0.20,
    "gyro_white_max": 0.015,
    "accel_bias_rw_max": 0.010,
    "gyro_bias_rw_max": 0.0006,
    "scale_rw_max": 0.0015,
    "spike_prob_max": 0.003,
    "accel_spike_mag": 6.0,
    "gyro_spike_mag": 1.0,
    "stuck_prob_max": 0.008,
    "stuck_len": (2, 10),
    "drop_prob_max": 0.06,
    "burst_prob_max": 0.0015,
    "burst_len": (5, 50),
    "dup_prob_max": 0.003,
}

GAP_FACTOR = 1.5          # dt > 1.5 * median dt  -> rows are physically missing
STUCK_TOL = 1e-7          # identical-sample tolerance (values are rounded to 1e-6)
FAIL_ACC_TOL = 1e-4       # m/s^2
FAIL_GYR_TOL = 1e-5       # rad/s
HAMPEL_K = 3.0            # deviation > 3 * MAD of the 5-sample window
JUMP_ACC = 2.5            # m/s^2 single-step jump-and-return
JUMP_GYR = 0.4            # rad/s single-step jump-and-return
SMOOTH_PASSES_MAX = 3.0   # binomial passes at the last timestamp (sev = 1)
CHUNK = 2048
HAMPEL_CHUNK = 512        # the Hampel test stacks 5-sample windows: keep its blocks small


def severity(t_norm, cfg=CONFIG):
    """sev(t) = floor + (max_level - floor) * (t / T) ** gamma  (degrade_imu.py)."""
    return cfg["floor"] + (cfg["max_level"] - cfg["floor"]) * np.power(np.clip(t_norm, 0.0, 1.0), cfg["gamma"])


def median(a, axis=None):
    """
    Median via np.partition. np.median / np.percentile are avoided on purpose:
    their first call lazily imports numpy.ma, which costs ~1 MB of heap.
    """
    a = np.asarray(a)
    if axis is None:
        a, axis = a.reshape(-1), 0
    n = a.shape[axis]
    if n == 0:
        return float("nan")
    lo, hi = (n - 1) // 2, n // 2
    part = np.partition(a, (lo, hi), axis=axis)
    return 0.5 * (np.take(part, lo, axis=axis) + np.take(part, hi, axis=axis))


def quantile(a, q):
    """Linear-interpolation quantile of a 1-D array (see median() for why not numpy's)."""
    srt = np.sort(np.asarray(a).reshape(-1))
    if srt.size == 0:
        return float("nan")
    pos = q * (srt.size - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, srt.size - 1)
    return float(srt[lo]) + (float(srt[hi]) - float(srt[lo])) * (pos - lo)


def row_norms(v):
    """Euclidean norm of every row of a float32 (n, 3) array, as float32, in bounded chunks."""
    out = np.empty(v.shape[0], dtype=np.float32)
    for a in range(0, v.shape[0], CHUNK):
        blk = v[a:a + CHUNK].astype(np.float64)
        out[a:a + CHUNK] = np.sqrt(np.sum(blk * blk, axis=1))
    return out


class Preprocessed(object):
    """Result of inverse_degrade(): cleaned, de-duplicated IMU stream."""

    __slots__ = ("t", "acc", "gyr", "dup", "n_rows", "n_active", "median_dt",
                 "duration", "t0_abs", "info")


# --------------------------------------------------------------------------
# CSV loader (two passes, bounded transient memory)
# --------------------------------------------------------------------------

def _parse_slow(text, out_t, out_x, w):
    """Tolerant per-line parser, used only when a block is not clean CSV."""
    for line in text.split("\n"):
        line = line.strip()
        if not line or line[0] == "#":
            continue
        tok = line.replace(";", ",").replace("\t", ",").split(",")
        if len(tok) < 7:
            tok = line.split()
        vals = []
        for s in tok[:7]:
            try:
                vals.append(float(s))
            except ValueError:
                vals.append(float("nan"))
        vals += [float("nan")] * (7 - len(vals))
        out_t[w] = vals[0]
        out_x[w] = vals[1:7]
        w += 1
    return w


def read_imu_csv(path, block_bytes=1 << 17):
    """
    Load a 7-column IMU CSV (optional header) with bounded transient memory.

    Returns (header_line or None, timestamps float64 (N,), sensors float32 (N, 6))
    with the six sensor columns still in file order and file units.
    """
    with open(path, "rb") as f:
        first = f.readline()
        if first[:3] == b"\xef\xbb\xbf":
            first = first[3:]
        head = first.decode("utf-8", "replace").strip()
        try:                                   # same rule as degrade_imu.detect_header
            float(head.split(",")[0])
            header, data_start = None, 0
        except ValueError:
            header, data_start = head, f.tell()
        f.seek(data_start)
        n_max, last = 0, b"\n"
        while True:
            buf = f.read(block_bytes)
            if not buf:
                break
            n_max += buf.count(b"\n")
            last = buf[-1:]
        if last != b"\n":
            n_max += 1

        ts = np.empty(n_max, dtype=np.float64)
        x = np.empty((n_max, 6), dtype=np.float32)
        f.seek(data_start)
        w, rest = 0, b""
        while True:
            buf = f.read(block_bytes)
            if not buf and not rest:
                break
            buf = rest + buf if buf else rest + b"\n"
            cut = buf.rfind(b"\n")
            if cut < 0:
                rest = buf
                continue
            block, rest = buf[:cut], buf[cut + 1:]
            if block[:3] == b"\xef\xbb\xbf":
                block = block[3:]
            text = block.replace(b"\r", b"").decode("ascii", "replace")
            n_lines = text.count("\n") + 1
            vals = None
            try:
                vals = np.fromstring(text.replace("\n", ","), dtype=np.float64, sep=",")
            except (ValueError, DeprecationWarning):
                vals = None
            if vals is not None and vals.size == 7 * n_lines:
                vals = vals.reshape(n_lines, 7)
                ts[w:w + n_lines] = vals[:, 0]
                x[w:w + n_lines] = vals[:, 1:]
                w += n_lines
            else:
                w = _parse_slow(text, ts, x, w)
            del vals, text, block
    return header, ts[:w], x[:w]


# --------------------------------------------------------------------------
# Layout / unit auto-detection
# --------------------------------------------------------------------------

def detect_ts_scale(ts):
    """Timestamp unit from the median sample period (same thresholds as degrade_imu.py)."""
    d = np.diff(ts)
    d = d[d > 0]
    dt = float(median(d)) if d.size else 1.0
    if dt < 1.0:
        return 1.0, "s"
    if dt < 1e3:
        return 1e-3, "ms"
    if dt < 1e6:
        return 1e-6, "us"
    return 1e-9, "ns"


def _column_role(name):
    n = name.strip().strip('"').lower()
    if not n:
        return "?"
    if any(k in n for k in ("w_rs", "gyr", "ang", "omega", "rot")):
        return "g"
    if any(k in n for k in ("a_rs", "acc", "lin")):
        return "a"
    if len(n) <= 3 and n[0] == "a":
        return "a"
    if len(n) <= 3 and n[0] in "gw":
        return "g"
    return "?"


def _median_norm(block):
    return float(median(row_norms(block)))


def _in_gravity_range(m):
    return (0.5 <= m <= 2.0) or (5.0 <= m <= 15.0)


def detect_accel_first(header, x_protect):
    """
    True if columns 1:4 are the accelerometer (degrade_imu.py order), False if
    they are the gyroscope (raw EuRoC order). Header names win; otherwise the
    gravity signature in the protected first seconds decides.
    """
    if header:
        roles = [_column_role(c) for c in header.split(",")[1:7]]
        if len(roles) == 6:
            if roles[:3] == ["a"] * 3 and roles[3:] == ["g"] * 3:
                return True, "header"
            if roles[:3] == ["g"] * 3 and roles[3:] == ["a"] * 3:
                return False, "header"
    m1, m2 = _median_norm(x_protect[:, 0:3]), _median_norm(x_protect[:, 3:6])
    r1, r2 = _in_gravity_range(m1), _in_gravity_range(m2)
    if r1 != r2:
        return r1, "gravity-norm"
    if r1 and r2:
        # both plausible (e.g. gyro in deg/s): gravity has the steadier norm
        def spread(b):
            nrm = row_norms(b)
            return (quantile(nrm, 0.75) - quantile(nrm, 0.25)) / max(quantile(nrm, 0.5), 1e-12)
        return spread(x_protect[:, 0:3]) <= spread(x_protect[:, 3:6]), "norm-steadiness"
    return m1 >= m2, "larger-norm"


def _hf_sigma(v):
    """Robust high-frequency noise level from second differences."""
    n = v.shape[0]
    if n < 8:
        return 0.0
    d2 = np.empty((n - 2, v.shape[1]), dtype=np.float32)
    for a in range(0, n - 2, CHUNK):
        b = min(n - 2, a + CHUNK)
        d2[a:b] = np.abs(np.diff(v[a:b + 2].astype(np.float64), n=2, axis=0))
    return float(1.4826 * median(d2) / math.sqrt(6.0))


def detect_gyro_deg(acc_ms2, gyr):
    """
    deg/s detection. degrade_imu.py injects white noise with a fixed
    gyro/accel ratio of 0.015 / 0.20 = 0.075 (rad/s per m/s^2); in deg/s the
    ratio is 57x larger. Implausibly large rates are the second cue.
    """
    n = gyr.shape[0]
    if n < 16:
        return False
    tail = slice(int(0.6 * n), n)
    p99 = quantile(row_norms(gyr), 0.99)
    sa, sg = _hf_sigma(acc_ms2[tail]), _hf_sigma(gyr[tail])
    ratio = sg / sa if sa > 1e-9 else 0.0
    return bool(p99 > 12.0 or ratio > 1.0)


# --------------------------------------------------------------------------
# Artifact detectors / repairs
# --------------------------------------------------------------------------

def _median5(w):
    """Median along axis 0 of a (5, m, c) stack."""
    w = np.sort(w, axis=0)
    return w[2]


def _hampel_chunk(x):
    """
    Joint 5-point Hampel test on a (m, 6) block [acc | gyr].

    A sample is a spike when its deviation from the 5-sample median exceeds
    HAMPEL_K * MAD (floored by the block's typical deviation so that a
    near-zero local MAD cannot trigger) on at least four of the six channels
    or on all three gyro channels - degrade_imu.py hits all six channels of a
    row at once - or when a channel jumps and immediately returns.

    Returns (mask (m,), replacement (m, 6)); the two edge rows on each side
    are never flagged.
    """
    m = x.shape[0]
    mask = np.zeros(m, dtype=bool)
    if m < 5:
        return mask, x
    c = x[2:-2]
    w = np.stack((x[:-4], x[1:-3], c, x[3:-1], x[4:]))
    med = _median5(w)
    dev = np.abs(c - med)
    mad = _median5(np.abs(w - med))
    floor = 1.4826 * median(dev, axis=0)
    exceed = dev > HAMPEL_K * np.maximum(mad, floor)
    hit = (exceed.sum(axis=1) >= 4) | exceed[:, 3:].all(axis=1)

    # single-step jump and immediate return
    up, dn = c - x[1:-3], c - x[3:-1]
    thr = np.empty(6, dtype=x.dtype)
    thr[:3] = np.maximum(JUMP_ACC, 8.0 * floor[:3])
    thr[3:] = np.maximum(JUMP_GYR, 8.0 * floor[3:])
    jump = (np.abs(up) > thr) & (np.abs(dn) > thr) & (up * dn > 0) \
        & (np.abs(x[3:-1] - x[1:-3]) < 0.5 * thr)
    hit |= jump.sum(axis=1) >= 2          # two channels: one alone is usually vibration

    # replacement: median of the four neighbours (sample itself excluded)
    nb = np.sort(np.stack((x[:-4], x[1:-3], x[3:-1], x[4:])), axis=0)
    rep = 0.5 * (nb[1] + nb[2])
    mask[2:-2] = hit
    out = x.copy()
    out[2:-2][hit] = rep[hit]
    return mask, out


def _smooth_chunk(x, passes, brk):
    """
    Zero-phase binomial smoothing of a (m, 6) block.

    passes : (m,) float, number of [1 2 1]/4 passes wanted at each sample
             (fractional passes are blended), brk[i] = True forbids averaging
             across the interval between sample i-1 and i (burst dropout).
    Each pass is symmetric, hence zero-phase, and conserves the sample sum
    away from breaks, so integrals of the signal are preserved.
    """
    n_pass = int(math.ceil(float(passes.max()))) if passes.size else 0
    if n_pass <= 0:
        return x
    y = x.astype(np.float32, copy=True)
    wl = np.ones(y.shape[0], dtype=np.float32)
    wl[0] = 0.0
    wl[brk] = 0.0
    wr = np.ones(y.shape[0], dtype=np.float32)
    wr[-1] = 0.0
    wr[:-1][brk[1:]] = 0.0
    den = (2.0 + wl + wr)[:, None]
    wl, wr = wl[:, None], wr[:, None]
    for j in range(n_pass):
        a = np.clip(passes - j, 0.0, 1.0).astype(np.float32)[:, None]
        s = 2.0 * y
        s[1:] += wl[1:] * y[:-1]
        s[:-1] += wr[:-1] * y[1:]
        s /= den
        y += a * (s - y)
    return y


def inverse_degrade(header, ts, x, cfg=CONFIG, denoise=True):
    """
    Undo the reversible degrade_imu.py artifacts in place.

    header : header line of the CSV or None
    ts     : (N,) float64 raw timestamps (overwritten with seconds from start)
    x      : (N, 6) float32 sensor columns in file order (edited in place)

    Returns a Preprocessed record whose t / acc / gyr are views of the inputs
    restricted to rows with unique timestamps (duplicates removed).
    """
    n_rows = ts.shape[0]
    info = {}

    # ---- 1. timestamps -> seconds from start --------------------------------
    bad_t = ~np.isfinite(ts)
    if bad_t.any():
        good = np.flatnonzero(~bad_t)
        if good.size >= 2:
            ts[bad_t] = np.interp(np.flatnonzero(bad_t), good, ts[good])
        else:
            ts[:] = np.arange(n_rows, dtype=np.float64) * 0.005
    scale, ts_unit = detect_ts_scale(ts)
    t0_abs = float(ts[0]) * scale if n_rows else 0.0
    if n_rows:
        ts -= ts[0]
        ts *= scale
    t = ts
    info["timestamp_unit"] = ts_unit

    # ---- 2. duplicates (dt == 0): flag, then compact in place -----------------
    dup = np.zeros(n_rows, dtype=bool)
    if n_rows > 1:
        dup[1:] = np.diff(t) <= 0.0
    n_dup = int(dup.sum())
    if n_dup:
        keep = np.flatnonzero(~dup)
        t[:keep.size] = t[keep]
        for a in range(0, keep.size, CHUNK):
            k = keep[a:a + CHUNK]
            x[a:a + k.size] = x[k]
        del keep
    m = n_rows - n_dup
    t, x = t[:m], x[:m]
    dt = np.diff(t)
    median_dt = float(median(dt)) if m > 1 else 0.005
    duration = float(t[-1]) if m else 0.0
    info["duplicate_rows"] = n_dup

    # ---- 3. column order + units ----------------------------------------------
    n_prot = int(np.searchsorted(t, cfg["protect_seconds"]))
    n_prot = max(min(m, 16), n_prot)
    accel_first, how = detect_accel_first(header, x[:n_prot])
    if not accel_first:
        for a in range(0, m, CHUNK):
            tmp = x[a:a + CHUNK, 0:3].copy()
            x[a:a + CHUNK, 0:3] = x[a:a + CHUNK, 3:6]
            x[a:a + CHUNK, 3:6] = tmp
    acc, gyr = x[:, 0:3], x[:, 3:6]
    info["column_order"] = "accel,gyro" if accel_first else "gyro,accel"
    info["column_order_source"] = how

    norm0 = _median_norm(acc[:n_prot]) if m else G
    accel_in_g = 0.5 < norm0 < 2.0                # same rule as degrade_imu.detect_accel_scale
    if accel_in_g:
        acc *= np.float32(G)
    info["accel_unit"] = "g" if accel_in_g else "m/s^2"

    # ---- 4. failure window: trailing block of dead samples --------------------
    n_active = m
    while n_active > 0:
        a = max(0, n_active - CHUNK)
        blk = x[a:n_active].astype(np.float64)
        an = np.sqrt(np.sum(blk[:, 0:3] ** 2, axis=1))
        gn = np.sqrt(np.sum(blk[:, 3:6] ** 2, axis=1))
        dead = ((an < FAIL_ACC_TOL) & (gn < FAIL_GYR_TOL)) | ~np.isfinite(an + gn)
        alive = np.flatnonzero(~dead)
        if alive.size:
            n_active = a + int(alive[-1]) + 1
            break
        n_active = a
    info["failure_rows"] = int(m - n_active)
    info["failure_start_s"] = float(t[n_active]) if n_active < m else None

    gyro_deg = detect_gyro_deg(acc[:n_active], gyr[:n_active])
    if gyro_deg:
        gyr *= np.float32(math.pi / 180.0)
    info["gyro_unit"] = "deg/s" if gyro_deg else "rad/s"

    # ---- 5. stuck runs: identical consecutive samples with dt > 0 -------------
    stuck = np.zeros(n_active, dtype=bool)
    for a in range(1, n_active, CHUNK):
        b = min(n_active, a + CHUNK)
        cur = x[a:b]
        same = (np.abs(cur - x[a - 1:b - 1]) < STUCK_TOL).all(axis=1)
        stuck[a:b] = same | ~np.isfinite(cur).all(axis=1)
    if n_active and not np.isfinite(x[0]).all():
        stuck[0] = True
    healthy = np.flatnonzero(~stuck)
    n_stuck = int(stuck.sum())
    info["stuck_samples"] = n_stuck
    info["stuck_runs"] = int(np.count_nonzero(stuck[1:] & ~stuck[:-1])) if n_active > 1 else 0

    # ---- 6. Hampel spike surgery on the healthy sequence -----------------------
    # (protected first seconds are spike-free by construction and left alone)
    n_spike = 0
    h0 = int(np.searchsorted(t[healthy], cfg["protect_seconds"])) if healthy.size else 0
    h0 = max(0, h0 - 2)
    for a in range(h0, healthy.size, HAMPEL_CHUNK):
        lo, hi = max(h0, a - 2), min(healthy.size, a + HAMPEL_CHUNK + 2)
        idx = healthy[lo:hi]
        mask, rep = _hampel_chunk(x[idx])
        own = np.zeros(idx.size, dtype=bool)          # rows this chunk is responsible for
        own[a - lo:a - lo + HAMPEL_CHUNK] = True
        mask &= own
        if mask.any():
            n_spike += int(mask.sum())
            x[idx[mask]] = rep[mask]
    info["spikes_removed"] = n_spike

    # ---- 7. stuck-run repair: linear interpolation in time ---------------------
    if n_stuck and healthy.size >= 2:
        bad = np.flatnonzero(stuck)
        th, tb = t[healthy], t[bad]
        for c in range(6):
            x[bad, c] = np.interp(tb, th, x[healthy, c])
        del bad, th, tb
    elif n_stuck:
        x[:n_active][stuck] = 0.0
    del healthy, stuck

    # ---- 8. severity-adaptive zero-phase smoother ------------------------------
    gap = np.zeros(n_active, dtype=bool)
    if n_active > 1:
        gap[1:] = dt[:n_active - 1] > GAP_FACTOR * median_dt
    info["dropout_gaps"] = int(gap.sum())
    info["max_gap_s"] = float(dt[:n_active - 1].max()) if n_active > 1 else 0.0
    info["smoothing_passes_max"] = SMOOTH_PASSES_MAX if denoise else 0.0
    if denoise and n_active > 4 and duration > 0:
        ov = int(math.ceil(SMOOTH_PASSES_MAX)) + 1
        brk_all = np.zeros(n_active, dtype=bool)
        brk_all[1:] = dt[:n_active - 1] > 2.5 * median_dt
        tail = None                                    # original rows preceding the chunk
        for a in range(0, n_active, CHUNK):
            b = min(n_active, a + CHUNK)
            e = min(n_active, b + ov)
            cur = x[a:e]
            lead = 0 if tail is None else tail.shape[0]
            blk = cur if tail is None else np.concatenate((tail, cur))
            passes = SMOOTH_PASSES_MAX * np.power(t[a - lead:e] / duration, cfg["gamma"])
            sm = _smooth_chunk(blk, passes, brk_all[a - lead:e])
            tail = x[max(a, b - ov):b].copy()
            x[a:b] = sm[lead:lead + (b - a)]
        del brk_all
    del gap, dt

    out = Preprocessed()
    out.t, out.acc, out.gyr, out.dup = t, acc, gyr, dup
    out.n_rows, out.n_active = n_rows, n_active
    out.median_dt, out.duration, out.t0_abs, out.info = median_dt, duration, t0_abs, info
    return out
