"""
es_ekf.py - IMU-only 6-DoF trajectory estimator for the degraded Track 3 data.

Pipeline (offline / non-causal, as allowed by the Track 3 FAQ):

  1. protected_init()    gravity magnitude g0, roll/pitch quaternion q0 and
                         gyro bias from the guaranteed-clean first 2 seconds.
  2. preintegrate()      RK4 quaternion + velocity + position kinematics for
                         every sample interval, vectorised over rows. Intervals
                         with dt > 1.5 * median_dt (dropped rows) are split into
                         round(dt / median_dt) sub-steps with linearly
                         interpolated (w, a). Increments are chained into a
                         gyro-only reference attitude and summed per 0.1 s block.
  3. run_filter()        13-state error-state Kalman filter on the blocks:
                         [dv, dtheta, d_bg, d_ba, d_scale]. Process noise
                         follows the white-box severity ramp of degrade_imu.py
                         (bias / scale random walks, white noise). Measurements:
                           - pseudo-drag: velocity ~ 0 with a Gauss-Markov-sized
                             covariance (bounds tilt, observes accel bias under
                             yaw motion),
                           - zero-velocity + direct gyro-bias updates on
                             detected standstill,
                           - gravity-norm consistency |f| = g0 in low-dynamics
                             blocks (accel scale / bias along gravity).
                         Two passes: the first learns the biases, the second
                         re-linearises the preintegration around them.
  4. spline_trend()      zero-phase drift removal: the dead-reckoned position is
                         split into drift + motion by a penalised least-squares
                         problem (smooth drift, bounded motion, zero displacement
                         at standstill). The cut-off shrinks as severity grows.
  4b. drag_aided_trend() only where flight_mask() finds the platform
                         thrust-borne (a multirotor in flight): the specific
                         force off the thrust axis is rotor drag, i.e. a body
                         velocity measurement. It enters the same least-squares
                         problem as displacement information and recovers the
                         slow motion that inertial integration cannot.
  5. blind_extrapolate() weighted least-squares fit of the last 3 s (terminal
                         velocity, acceleration trend, yaw rate), propagated as
                         a damped coordinated turn over the 5 s failure window.

Memory model: the float32 sensor arrays are overwritten in place by the
world-frame increments and finally by the positions; the only extra per-row
storage is one float32 quaternion. Everything else is per-block (10 Hz).

Only numpy + the standard library are imported (hackathon heap constraint).
Hot loops live in small functions and avoid creating Python objects: under
tracemalloc every allocation pays a line-number lookup that is linear in the
position inside the enclosing function.
"""

import math

import numpy as np

from inverse_degrade import CONFIG, GAP_FACTOR, median, severity

CHUNK_ROWS = 512        # rows per vectorised preintegration chunk
NSUB_MAX = 256          # cap on RK4 sub-steps inside one dropout gap
FILTER_CHUNK = 32       # blocks per relinearisation of the error-state filter
SOLVE_BLOCK = 32        # dense block size of the banded drift solver

PARAMS = {
    "block_dt": 0.10,           # s, filter / covariance step
    # sensor noise floors (independent of the injected degradation)
    "vrw_floor": 0.010,         # m/s/sqrt(s)
    "arw_floor": 5e-4,          # rad/sqrt(s)
    "bg_rw_floor": 2e-5,        # rad/s/sqrt(s)
    "ba_rw_floor": 3e-3,        # m/s^2/sqrt(s)
    "scale_rw_floor": 1e-5,     # 1/sqrt(s)
    "rate_rms": 0.3,            # rad/s, typical rate used for gyro scale error
    # pseudo-drag (velocity ~ 0 modelled as a Gauss-Markov process)
    "vel_sigma": 0.5,           # m/s
    "vel_tau": 2.0,             # s
    # standstill
    "zupt_sigma": 0.05,         # m/s
    "zupt_bias_sigma": 0.004,   # rad/s
    # gravity-norm consistency
    "gnorm_sigma": 0.4,         # m/s^2
    "gnorm_tau": 0.5,           # s
    # zero-phase drift removal: spline cut-off period [s] at sev=0 and sev=1
    # (longer when the gyro bias was calibrated at a standstill, see estimate_trajectory)
    "trend_period_clean": 30.0,
    "trend_period_degraded": 15.0,
    "trend_period_uncal_clean": 14.0,
    "trend_period_uncal_degraded": 7.0,
    "calibrated_min_zupt": 20,  # blocks of standstill that count as a calibration
    "zupt_trend_weight": 1.0e4,
    # rotor-drag velocity aiding (only where the platform is detected as thrust-borne)
    "drag_aid": True,
    "fly_window": 51,           # blocks (~5 s) for the thrust-borne test
    "fly_excitation": 0.10,     # m/s^2 rms gravity-tilt variation needed to decide
    "fly_beta_max": 0.35,       # |slope of specific force vs gravity tilt|
    "fly_ratio_max": 0.85,      # off-axis force variation / gravity-tilt variation
    "fly_min_blocks": 200,      # >= 20 s of flight before the aid is trusted
    "drag_period_clean": 120.0,  # s, drag velocity is trusted up to this period
    "drag_period_degraded": 60.0,
    "drag_mu": 0.25,            # 1/s, rotor-drag coefficient (platform prior for small multirotors)
    "drag_bias_order": 1,       # off-axis bias model: 1 constant, 2 linear in time
    # blind-window extrapolation
    "fit_seconds": 3.0,
    "fit_weight_tau": 1.0,
    "vel_damp_tau": 0.75,
    "acc_damp_tau": 0.5,
    "turn_damp_tau": 1.0,
    "vel_max": 3.0,
    "acc_max": 2.0,
}


# ----------------------------------------------------------------------------
# quaternion helpers  (Hamilton, stored [x, y, z, w], body -> world)
# ----------------------------------------------------------------------------

def qmul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    out = np.empty(np.broadcast(ax, bx).shape + (4,), dtype=np.float64)
    out[..., 0] = aw * bx + ax * bw + ay * bz - az * by
    out[..., 1] = aw * by - ax * bz + ay * bw + az * bx
    out[..., 2] = aw * bz + ax * by - ay * bx + az * bw
    out[..., 3] = aw * bw - ax * bx - ay * by - az * bz
    return out


def _qmul_pure(q, w):
    """q (x) [w, 0] for an (m, 3) rate array."""
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
    out = np.empty_like(q)
    out[:, 0] = qw * wx + qy * wz - qz * wy
    out[:, 1] = qw * wy - qx * wz + qz * wx
    out[:, 2] = qw * wz + qx * wy - qy * wx
    out[:, 3] = -(qx * wx + qy * wy + qz * wz)
    return out


def qrot(q, v):
    """Rotate (m, 3) vectors by (m, 4) quaternions (norm need not be exactly 1)."""
    ux, uy, uz, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    tx = 2.0 * (uy * vz - uz * vy)
    ty = 2.0 * (uz * vx - ux * vz)
    tz = 2.0 * (ux * vy - uy * vx)
    inv = 1.0 / (ux * ux + uy * uy + uz * uz + w * w)
    out = np.empty_like(v)
    out[:, 0] = vx + inv * (w * tx + uy * tz - uz * ty)
    out[:, 1] = vy + inv * (w * ty + uz * tx - ux * tz)
    out[:, 2] = vz + inv * (w * tz + ux * ty - uy * tx)
    return out


def rotmats(q):
    """(m, 4) unit quaternions -> (m, 3, 3) rotation matrices."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    r[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    r[:, 0, 1] = 2.0 * (x * y - w * z)
    r[:, 0, 2] = 2.0 * (x * z + w * y)
    r[:, 1, 0] = 2.0 * (x * y + w * z)
    r[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    r[:, 1, 2] = 2.0 * (y * z - w * x)
    r[:, 2, 0] = 2.0 * (x * z - w * y)
    r[:, 2, 1] = 2.0 * (y * z + w * x)
    r[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return r


def qexp(phi):
    """(m, 3) rotation vectors -> (m, 4) quaternions."""
    ang = np.sqrt(np.sum(phi * phi, axis=1))
    half = 0.5 * ang
    k = np.where(ang > 1e-8, np.sin(half) / np.maximum(ang, 1e-12), 0.5 - ang * ang / 48.0)
    out = np.empty((phi.shape[0], 4), dtype=np.float64)
    out[:, :3] = phi * k[:, None]
    out[:, 3] = np.cos(half)
    return out


def _qnorm_s(q):
    n = 1.0 / math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return (q[0] * n, q[1] * n, q[2] * n, q[3] * n)


def _rotmat_s(q):
    x, y, z, w = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)]])


def gravity_alignment(f):
    """Yaw-free quaternion that rotates the measured specific force onto +z."""
    n = math.sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2])
    if n < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    ux, uy, uz = f[0] / n, f[1] / n, f[2] / n
    if uz < -0.999999:
        return (1.0, 0.0, 0.0, 0.0)
    return _qnorm_s((uy, -ux, 0.0, 1.0 + uz))


# ----------------------------------------------------------------------------
# 1. protected initialisation
# ----------------------------------------------------------------------------

def protected_init(t, acc, gyr, n_act, protect_seconds):
    """
    Initial conditions from the degradation-free window 0 <= t < protect_seconds.

    Returns (mean specific force, mean gyro, is_static). Standstill is judged
    on the stability of sub-window means rather than on variance, so a platform
    that is at rest but vibrating (motors idling) still qualifies.
    """
    n2 = int(np.searchsorted(t[:n_act], protect_seconds))
    n2 = max(min(n_act, 8), min(n2, n_act))
    a = acc[:n2].astype(np.float64)
    w = gyr[:n2].astype(np.float64)
    fbar, wbar = a.mean(axis=0), w.mean(axis=0)
    static = False
    if n2 >= 40:
        parts = np.array_split(np.arange(n2), 4)
        wm = np.array([w[p].mean(axis=0) for p in parts])
        fm = np.array([a[p].mean(axis=0) for p in parts])
        static = bool(np.abs(wm - wbar).max() < 0.010 and np.abs(fm - fbar).max() < 0.15
                      and math.sqrt(float(wbar @ wbar)) < 0.2)
    return fbar, wbar, static


# ----------------------------------------------------------------------------
# 2. vectorised RK4 preintegration with dropout sub-stepping
# ----------------------------------------------------------------------------

class Blocks(object):
    """Per-block (filter step) preintegrated quantities, reference frame."""
    __slots__ = ("bnd", "tb", "dt", "U", "W", "Gam", "wbar", "fbar", "gap")


def block_boundaries(t, n_act, block_dt):
    """Row indices b_0 = 0 < b_1 < ... < b_K = n_act - 1 spaced ~block_dt apart."""
    if n_act < 2:
        return np.zeros(1, dtype=np.int64)
    edges = np.arange(t[0] + block_dt, t[n_act - 1], block_dt)
    idx = np.searchsorted(t[:n_act], edges, side="left")
    b = np.concatenate(([0], idx, [n_act - 1])).astype(np.int64)       # already sorted
    return b[np.concatenate(([True], np.diff(b) > 0))]                 # (np.unique would import numpy.ma)


def _rk4_step(q, v, p, wa, wb, fa, fb, hs):
    """
    One RK4 step of dq/dt = q(x)w/2, dv/dt = R(q) f, dp/dt = v with rate and
    specific force varying linearly from (wa, fa) to (wb, fb) over hs seconds.
    Vectorised over intervals; started from (identity, 0, 0) it returns the
    body-frame increments of each interval.
    """
    wm = 0.5 * (wa + wb)
    fm = 0.5 * (fa + fb)
    hf = hs[:, None]
    hh = 0.5 * hf
    k1 = 0.5 * _qmul_pure(q, wa)
    y2 = q + hh * k1
    k2 = 0.5 * _qmul_pure(y2, wm)
    y3 = q + hh * k2
    k3 = 0.5 * _qmul_pure(y3, wm)
    y4 = q + hf * k3
    k4 = 0.5 * _qmul_pure(y4, wb)
    a1, a2, a3, a4 = qrot(q, fa), qrot(y2, fm), qrot(y3, fm), qrot(y4, fb)
    qn = q + (hf / 6.0) * (k1 + 2.0 * (k2 + k3) + k4)
    qn /= np.sqrt(np.sum(qn * qn, axis=1))[:, None]
    pn = p + hf * v + (hf * hf / 6.0) * (a1 + a2 + a3)
    vn = v + (hf / 6.0) * (a1 + 2.0 * (a2 + a3) + a4)
    return qn, vn, pn


def _expand_gaps(w0, w1, f0, f1, h, gaps, median_dt):
    """
    Dropout sub-stepping: every interval with dt > 1.5 * median_dt is split
    into N = round(dt / median_dt) sub-intervals with linearly interpolated
    rate and specific force. Returns the expanded interval arrays plus, per
    original row, the index of its first / last sub-interval and the
    sub-interval -> row map.
    """
    m = h.shape[0]
    rep = np.ones(m, dtype=np.intp)
    rep[gaps] = np.clip(np.rint(h[gaps] / median_dt), 1, NSUB_MAX).astype(np.intp)
    first = np.cumsum(rep) - rep
    row = np.repeat(np.arange(m), rep)
    inv = 1.0 / rep[row]
    ja = ((np.arange(row.shape[0]) - first[row]) * inv)[:, None]
    jb = ja + inv[:, None]
    dw, df = (w1 - w0)[row], (f1 - f0)[row]
    w0, f0 = w0[row], f0[row]
    return w0 + dw * ja, w0 + dw * jb, f0 + df * ja, f0 + df * jb, h[row] * inv, first, row, first + rep - 1


def _chain(G_c, R_c, w0, w1, f0, f1, h):
    """
    RK4 increments of every interval, chained into the reference attitude.
    Returns G (attitude after each interval), its rotation matrices R, the
    interval-mean rotation R_mid and the reference-frame increments.
    """
    ms = h.shape[0]
    q_id = np.zeros((ms, 4))
    q_id[:, 3] = 1.0
    zero = np.zeros((ms, 3))
    dq, dv, dp = _rk4_step(q_id, zero, zero, w0, w1, f0, f1, h)
    G = qmul(G_c[None], _prefix_qprod(dq))
    G_prev = np.concatenate((G_c[None], G[:-1]))
    R = rotmats(G)
    R_mid = 0.5 * (np.concatenate((R_c[None], R[:-1])) + R)
    return G, R, R_mid, qrot(G_prev, dv), qrot(G_prev, dp)


def _prefix_qprod(dq):
    """P[i] = dq[0] (x) dq[1] (x) ... (x) dq[i]  (log-depth doubling scan)."""
    P = dq.copy()
    m, d = P.shape[0], 1
    while d < m:
        P[d:] = qmul(P[:-d], P[d:])
        d *= 2
    P /= np.sqrt(np.sum(P * P, axis=1))[:, None]
    return P


def preintegrate(t, acc, gyr, n_act, bnd, bg, ba, median_dt, G_out=None):
    """
    RK4-preintegrate the active rows and aggregate per block.

    The reference attitude G_i is the pure gyro integration (bias bg removed)
    starting from identity. Per row i the reference-frame increments are
        dv_ref_i = G_{i-1} * dv_i ,   dp_ref_i = G_{i-1} * dp_i .
    If G_out is given they are stored in place (acc <- dv_ref, gyr <- dp_ref,
    G_out <- G); otherwise only the block sums are produced.
    """
    K = bnd.shape[0] - 1
    blk = Blocks()
    blk.bnd = bnd
    blk.tb = t[bnd].astype(np.float64)
    blk.dt = np.diff(blk.tb)
    blk.U = np.zeros((K, 3))
    blk.W = np.zeros((K, 3))
    blk.Gam = np.zeros((K, 3, 3))
    blk.wbar = np.zeros((K, 3))
    blk.fbar = np.zeros((K, 3))
    blk.gap = np.zeros(K)
    if K <= 0:
        return blk

    store = G_out is not None
    w_c = gyr[0].astype(np.float64) - bg
    f_c = acc[0].astype(np.float64) - ba
    G_c = np.array([0.0, 0.0, 0.0, 1.0])
    R_c = np.eye(3)
    if store:
        G_out[0] = G_c
        acc[0] = 0.0
        gyr[0] = 0.0

    k0 = 0
    while k0 < K:
        k1 = k0 + 1
        while k1 < K and bnd[k1 + 1] - bnd[k0] <= CHUNK_ROWS:
            k1 += 1
        a, b = int(bnd[k0]) + 1, int(bnd[k1]) + 1          # rows [a, b)
        w1 = gyr[a:b].astype(np.float64) - bg
        f1 = acc[a:b].astype(np.float64) - ba
        w0 = np.concatenate((w_c[None], w1[:-1]))
        f0 = np.concatenate((f_c[None], f1[:-1]))
        h = t[a:b] - t[a - 1:b - 1]
        m = h.shape[0]
        gaps = h > GAP_FACTOR * median_dt
        starts_r = (bnd[k0:k1] + 1 - a).astype(np.intp)     # first interval of each block
        dtb = blk.dt[k0:k1]
        blk.gap[k0:k1] = np.add.reduceat(h * gaps, starts_r) / dtb
        w_n, f_n = w1[-1].copy(), f1[-1].copy()
        first = None
        if gaps.any():
            w0, w1, f0, f1, h, first, row, last = _expand_gaps(w0, w1, f0, f1, h, gaps, median_dt)
            starts = first[starts_r]
        else:
            starts = starts_r
        G, R, R_mid, dv_ref, dp_ref = _chain(G_c, R_c, w0, w1, f0, f1, h)
        ms = h.shape[0]

        hc = h[:, None]
        C_prev = np.cumsum(dv_ref, axis=0) - dv_ref
        counts = np.diff(np.concatenate((starts, [ms])))
        term = (C_prev - np.repeat(C_prev[starts], counts, axis=0)) * hc + dp_ref
        blk.U[k0:k1] = np.add.reduceat(dv_ref, starts, axis=0)
        blk.W[k0:k1] = np.add.reduceat(term, starts, axis=0)
        blk.Gam[k0:k1] = np.add.reduceat(R_mid * h[:, None, None], starts, axis=0)
        blk.wbar[k0:k1] = np.add.reduceat(0.5 * (w0 + w1) * hc, starts, axis=0) / dtb[:, None]
        blk.fbar[k0:k1] = np.add.reduceat(0.5 * (f0 + f1) * hc, starts, axis=0) / dtb[:, None]

        if store:
            if first is None:
                G_out[a:b] = G
                acc[a:b] = dv_ref
                gyr[a:b] = dp_ref
            else:
                term = (C_prev - C_prev[first][row]) * hc + dp_ref
                G_out[a:b] = G[last]
                acc[a:b] = np.add.reduceat(dv_ref, first, axis=0)
                gyr[a:b] = np.add.reduceat(term, first, axis=0)
        w_c, f_c, G_c, R_c = w_n, f_n, G[-1].copy(), R[-1].copy()
        k0 = k1
    return blk


# ----------------------------------------------------------------------------
# standstill detection (block level)
# ----------------------------------------------------------------------------

def _moving_mean(x, win):
    n = x.shape[0]
    c = np.cumsum(np.concatenate((np.zeros((1,) + x.shape[1:]), x)), axis=0)
    half = win // 2
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    cnt = (hi - lo).astype(np.float64)
    return (c[hi] - c[lo]) / (cnt if x.ndim == 1 else cnt[:, None])


def detect_static(blk, g0, bias_known, bias_drift_sigma, win=7, min_blocks=6):
    """
    Standstill flags per block from mean rate / mean specific force stability.

    Uses means (not variances) so that an idling, vibrating platform is still
    recognised, and so that the growing injected white noise does not break it.
    """
    K = blk.dt.shape[0]
    if K < min_blocks:
        return np.zeros(K, dtype=bool)
    wm = _moving_mean(blk.wbar, win)
    dev = blk.wbar - wm
    w_rms = np.sqrt(_moving_mean(np.sum(dev * dev, axis=1), win))
    fn = np.sqrt(np.sum(blk.fbar * blk.fbar, axis=1))
    fm = _moving_mean(fn, win)
    f_rms = np.sqrt(_moving_mean((fn - fm) ** 2, win))
    rate = np.sqrt(np.sum(wm * wm, axis=1))
    rate_max = (0.012 + 2.5 * bias_drift_sigma) if bias_known else 0.15
    st = (w_rms < 0.012) & (f_rms < 0.12) & (np.abs(fm - g0) < 0.30) & (rate < rate_max) & (blk.gap < 0.3)
    # drop runs shorter than min_blocks
    edges = np.flatnonzero(np.diff(np.concatenate(([False], st, [False])).astype(np.int8)))
    for s, e in zip(edges[0::2], edges[1::2]):
        if e - s < min_blocks:
            st[s:e] = False
    return st


# ----------------------------------------------------------------------------
# 3. block error-state Kalman filter
# ----------------------------------------------------------------------------

def process_noise(blk, sev, g0, median_dt, cfg, prm):
    """White-box per-block process noise (K, 13) from the degrade_imu.py model."""
    dt = blk.dt
    q = np.empty((dt.shape[0], 13))
    vrw2 = prm["vrw_floor"] ** 2 + (cfg["accel_white_max"] * sev) ** 2 * median_dt
    arw2 = prm["arw_floor"] ** 2 + (cfg["gyro_white_max"] * sev) ** 2 * median_dt
    bg2 = prm["bg_rw_floor"] ** 2 + (cfg["gyro_bias_rw_max"] * sev) ** 2 \
        + (cfg["scale_rw_max"] * sev * prm["rate_rms"]) ** 2
    ba2 = prm["ba_rw_floor"] ** 2 + (cfg["accel_bias_rw_max"] * sev) ** 2 \
        + (cfg["scale_rw_max"] * sev * g0) ** 2
    sc2 = prm["scale_rw_floor"] ** 2 + (cfg["scale_rw_max"] * sev) ** 2 / 3.0
    gap_t = blk.gap * dt                     # seconds of interpolated (unobserved) signal
    q[:, 0:3] = (vrw2 * dt + (0.3 * gap_t) ** 2)[:, None]
    q[:, 3:6] = (arw2 * dt + (0.05 * gap_t) ** 2)[:, None]
    q[:, 6:9] = (bg2 * dt)[:, None]
    q[:, 9:12] = (ba2 * dt)[:, None]
    q[:, 12] = sc2 * dt
    return q, np.sqrt(np.cumsum(bg2 * dt))


def _kf_steps(work, Fc, FcT, Qc, zv, zb, Hc, rvel, rg, zg, st, gn, xs, pvar):
    """
    Kalman recursion over the blocks of one chunk, in place on the work arrays.

    Kept as a small function of its own on purpose: tracemalloc looks up the
    current line for every allocation and that lookup is linear in the
    position inside the enclosing function.
    """
    (P, PT, Pd, Pd_bias, P00, PH0, PH0T, P66, PH6, PH6T, x, x03, x69, xt, dx, Ph, Kg1,
     T, S, Sd, Kg, inn, r_zb) = work
    matmul, inv = np.matmul, np.linalg.inv
    for j in range(len(st)):
        Fk = Fc[j]
        matmul(Fk, x, out=xt)
        x[:] = xt
        matmul(Fk, P, out=T)
        matmul(T, FcT[j], out=P)
        Pd += Qc[j]
        # velocity ~ 0: hard at standstill (ZUPT), soft Gauss-Markov otherwise
        S[:] = P00
        Sd += rvel[j]
        matmul(PH0, inv(S), out=Kg)
        np.subtract(zv[j], x03, out=inn)
        matmul(Kg, inn, out=dx)
        matmul(Kg, PH0T, out=T)
        P -= T
        x += dx
        if st[j]:
            # standstill: the mean rate is the gyro bias
            S[:] = P66
            Sd += r_zb
            matmul(PH6, inv(S), out=Kg)
            np.subtract(zb[j], x69, out=inn)
            matmul(Kg, inn, out=dx)
            matmul(Kg, PH6T, out=T)
            P -= T
            x += dx
        if gn[j]:
            # gravity-norm consistency (scalar)
            hj = Hc[j]
            matmul(P, hj, out=Ph)
            np.divide(Ph, float(hj @ Ph) + rg[j], out=Kg1)
            np.multiply(Kg1, zg[j] - float(hj @ x), out=dx)
            np.outer(Kg1, Ph, out=T)
            P -= T
            x += dx
        np.add(P, PT, out=T)
        np.multiply(T, 0.5, out=P)
        xs[j] = x
        pvar[j] = Pd_bias


def _kf_linearise(E, v, dbg, dba, kk, g0, U, Gam, dt, wbar, fbar):
    """
    Nominal propagation and linearisation of one chunk of blocks.

    Returns the mid-block attitude corrections Eh, the correction at the end
    of the chunk, the nominal velocity at the block ends, the 13 x 13
    transition matrices and the measurement rows / nominal residuals of the
    velocity, gyro-bias and gravity-norm measurements.
    """
    nc = dt.shape[0]
    # nominal attitude: reference rotated by the residual gyro bias
    half = qexp(-0.5 * (Gam @ dbg))
    cum = _prefix_qprod(qmul(half, half))
    Eh = qmul(qmul(E[None], np.concatenate(([[0.0, 0.0, 0.0, 1.0]], cum[:-1]))), half)
    Eh /= np.sqrt(np.sum(Eh * Eh, axis=1))[:, None]
    E_end = qmul(E[None], cum[-1:])[0]
    Rm = rotmats(Eh)
    one_k = 1.0 - kk
    Sw = np.einsum("kij,kj->ki", Rm, one_k * (U - Gam @ dba))
    dvn = Sw.copy()
    dvn[:, 2] -= g0 * dt
    vn = v + np.cumsum(dvn, axis=0)                 # nominal velocity at block ends
    RG = Rm @ Gam
    i13 = np.arange(13)
    Fc = np.zeros((nc, 13, 13))
    Fc[:, i13, i13] = 1.0
    Fc[:, 0, 4], Fc[:, 0, 5] = Sw[:, 2], -Sw[:, 1]
    Fc[:, 1, 3], Fc[:, 1, 5] = -Sw[:, 2], Sw[:, 0]
    Fc[:, 2, 3], Fc[:, 2, 4] = Sw[:, 1], -Sw[:, 0]
    Fc[:, 0:3, 9:12] = -one_k * RG
    Fc[:, 0:3, 12] = -Sw
    Fc[:, 3:6, 6:9] = -RG
    fb = fbar - dba                                 # gravity norm: |f - ba| (1 - k) = g0
    nf = np.sqrt(np.sum(fb * fb, axis=1))
    Hc = np.zeros((nc, 13))
    Hc[:, 9:12] = -one_k * fb / np.maximum(nf, 1e-9)[:, None]
    Hc[:, 12] = -nf
    return Eh, E_end, vn, Fc, -vn, wbar - dbg, Hc, (g0 - nf * one_k).tolist()


def _dead_reckon(xs, Eh, dba, kk, g0, U, W, Gam, dt, v0, p0):
    """
    Dead reckoning of one chunk with the filtered attitude / bias / scale but
    with the velocity left free (the drift it accumulates is removed later,
    zero-phase, by spline_trend). Returns attitude corrections, accel bias,
    scale error and the velocity / position at the block ends.
    """
    Ec = qmul(qexp(xs[:, 3:6]), Eh)
    Ec /= np.sqrt(np.sum(Ec * Ec, axis=1))[:, None]
    bac = dba + xs[:, 9:12]
    kc = np.clip(kk + xs[:, 12], -0.05, 0.05)
    okc = (1.0 - kc)[:, None]
    Rc = rotmats(Ec)
    Gb = np.einsum("kij,kj->ki", Gam, bac)
    dvr = np.einsum("kij,kj->ki", Rc, okc * (U - Gb))
    dvr[:, 2] -= g0 * dt
    vr = v0 + np.cumsum(dvr, axis=0)
    dpr = np.concatenate((v0[None], vr[:-1])) * dt[:, None] \
        + np.einsum("kij,kj->ki", Rc, okc * (W - (0.5 * dt)[:, None] * Gb))
    dpr[:, 2] -= 0.5 * g0 * dt * dt
    return Ec, bac, kc, vr, p0 + np.cumsum(dpr, axis=0)


def run_filter(blk, g0, q0, P0, Q, static, gnorm_ok, prm, record):
    """
    One forward pass of the block error-state Kalman filter.

    Nominal state: world correction E (attitude = E * G_ref), velocity v,
    residual gyro bias d_bg, residual accel bias d_ba, accel scale error k.
    Error state  : x = [dv(3), dphi(3, world frame), d_bg(3), d_ba(3), dk].

    The filter is linearised around the nominal trajectory FILTER_CHUNK blocks
    at a time: the nominal propagation, transition matrices and measurement
    rows of a chunk are computed vectorised, the per-block recursion
    (_kf_steps) then only does in-place 13 x 13 products, and at the end of
    the chunk the error state is folded back into the nominal state (closed
    loop).
    """
    K = blk.dt.shape[0]
    U, W, Gam, dtb, wbar, fbar = blk.U, blk.W, blk.Gam, blk.dt, blk.wbar, blk.fbar
    P = np.diag(np.asarray(P0, dtype=np.float64))
    Pd = P.reshape(-1)[::14]                       # diagonal (view)
    x = np.zeros(13)
    S = np.zeros((3, 3))
    work = (P, P.T, Pd, Pd[6:12], P[0:3, 0:3], P[:, 0:3], P[:, 0:3].T, P[6:9, 6:9], P[:, 6:9], P[:, 6:9].T,
            x, x[0:3], x[6:9], np.zeros(13), np.zeros(13), np.zeros(13), np.zeros(13),
            np.zeros((13, 13)), S, S.reshape(-1)[::4], np.zeros((13, 3)), np.zeros(3), prm["zupt_bias_sigma"] ** 2)

    E = np.asarray(q0, dtype=np.float64)
    v = np.zeros(3)
    dbg = np.zeros(3)
    dba = np.zeros(3)
    kk = 0.0
    r_soft = prm["vel_sigma"] ** 2 * 2.0 * prm["vel_tau"]
    r_zupt = prm["zupt_sigma"] ** 2
    r_gn = prm["gnorm_sigma"] ** 2 * 2.0 * prm["gnorm_tau"]
    st_l, gn_l = static.tolist(), gnorm_ok.tolist()

    if record:
        E_tr = np.empty((K, 4))
        ba_tr = np.empty((K, 3))
        k_tr = np.empty(K)
        vraw = np.zeros((K + 1, 3))
        praw = np.zeros((K + 1, 3))
    bias_est = np.empty((K, 6))
    bias_var = np.empty((K, 6))
    n_zupt = int(np.count_nonzero(static))

    for c0 in range(0, K, FILTER_CHUNK):
        c1 = min(K, c0 + FILTER_CHUNK)
        sl = slice(c0, c1)
        Eh, E_end, vn, Fc, zv, zb, Hc, zg = _kf_linearise(
            E, v, dbg, dba, kk, g0, U[sl], Gam[sl], dtb[sl], wbar[sl], fbar[sl])
        rvel = np.where(static[sl], r_zupt, r_soft / dtb[sl]).tolist()
        xs = np.empty((c1 - c0, 13))
        _kf_steps(work, Fc, Fc.transpose(0, 2, 1), Q[sl], zv, zb, Hc, rvel, (r_gn / dtb[sl]).tolist(), zg,
                  st_l[sl], gn_l[sl], xs, bias_var[sl])
        bias_est[sl, 0:3] = dbg + xs[:, 6:9]
        bias_est[sl, 3:6] = dba + xs[:, 9:12]
        if record:
            E_tr[sl], ba_tr[sl], k_tr[sl], vraw[c0 + 1:c1 + 1], praw[c0 + 1:c1 + 1] = _dead_reckon(
                xs, Eh, dba, kk, g0, U[sl], W[sl], Gam[sl], dtb[sl], vraw[c0], praw[c0])
        # closed loop: fold the error state into the nominal state
        v = vn[-1] + x[0:3]
        E = qmul(qexp(x[None, 3:6]), E_end[None])[0]
        E /= math.sqrt(float(E @ E))
        dbg = dbg + x[6:9]
        dba = dba + x[9:12]
        kk = min(0.05, max(-0.05, kk + float(x[12])))
        x[:] = 0.0

    kb = int(np.argmin(bias_var[:, 3:6].sum(axis=1)))
    kg = int(np.argmin(bias_var[:, 0:3].sum(axis=1)))
    out = {"E": E, "v": v, "dbg": dbg, "dba": dba, "scale": kk, "n_zupt": n_zupt,
           "best_ba": (float(bias_var[kb, 3:6].sum()), bias_est[kb, 3:6].copy(), bias_var[kb, 3:6].copy()),
           "best_bg": (float(bias_var[kg, 0:3].sum()), bias_est[kg, 0:3].copy(), bias_var[kg, 0:3].copy())}
    if record:
        out["E_tr"], out["ba_tr"], out["k_tr"], out["vraw"], out["praw"] = E_tr, ba_tr, k_tr, vraw, praw
    return out


# ----------------------------------------------------------------------------
# 4. zero-phase drift removal (smoothing spline on a non-uniform grid)
# ----------------------------------------------------------------------------

def spline_trend(tau, y, lam, mu, disp=None):
    """
    Drift c of the dead-reckoned track y (true position p = y - c), minimising
        sum_i hbar_i |y_i - c_i|^2                    position stays bounded
      + sum_i lam_i hbar_i |c''_i|^2                  inertial drift is smooth
      + sum_k mu_k |(p_{k+1} - p_k) - disp_k|^2       known displacement
    tau (n,) grid times, y (n, m), lam (n,) curvature penalty, mu (n-1,) weight
    of the displacement information for interval k and disp (n-1, m) that
    displacement (default 0: standstill).

    Without displacement terms y - c is y high-passed with the zero-phase
    response w^4 / (w^4 + 1/lam). A displacement term of weight mu = gamma/dt
    acts as a velocity measurement and takes over the band between
    sqrt(gamma/lam) and 1/sqrt(gamma) [rad/s]. Solved with a banded LDL^T
    factorisation in O(n); every column of y is one right-hand side.
    """
    n = tau.shape[0]
    if n < 3:
        return y.copy()
    h = np.diff(tau)
    h1, h2 = h[:-1], h[1:]
    hbar = np.empty(n)
    hbar[1:-1] = 0.5 * (h1 + h2)
    hbar[0], hbar[-1] = 0.5 * h[0], 0.5 * h[-1]
    al = 2.0 / (h1 * (h1 + h2))
    ga = 2.0 / (h2 * (h1 + h2))
    be = -(al + ga)
    wgt = lam[1:-1] * hbar[1:-1]

    d0 = hbar.copy()
    d1 = np.zeros(n - 1)
    d2 = np.zeros(n - 2)
    rhs = y * hbar[:, None]
    d0[:-2] += wgt * al * al
    d0[1:-1] += wgt * be * be
    d0[2:] += wgt * ga * ga
    d1[:-1] += wgt * al * be
    d1[1:] += wgt * be * ga
    d2 += wgt * al * ga
    d0[:-1] += mu
    d0[1:] += mu
    d1 -= mu
    dy = y[1:] - y[:-1]
    if disp is not None:
        dy = dy - disp
    dy *= mu[:, None]
    rhs[:-1] -= dy
    rhs[1:] += dy

    return _penta_solve(d0, d1, d2, rhs)


def _penta_solve(d0, d1, d2, rhs):
    """
    Solve the symmetric positive-definite pentadiagonal system with main
    diagonal d0 (n,), first off-diagonal d1 (n-1,) and second off-diagonal
    d2 (n-2,) for every column of rhs (n, m).

    Block-tridiagonal elimination with dense SOLVE_BLOCK x SOLVE_BLOCK blocks:
    neighbouring blocks are coupled only through a 2 x 2 corner, so each block
    costs one small dense solve. O(n) time, O(n) memory, no per-row Python.
    """
    n, nc = rhs.shape
    m = SOLVE_BLOCK
    nb = -(-n // m)
    N = nb * m
    D0 = np.ones(N)
    D0[:n] = d0
    D1 = np.zeros(N)
    D1[:n - 1] = d1
    D2 = np.zeros(N)
    D2[:n - 2] = d2
    X = np.zeros((N, nc))
    X[:n] = rhs
    Zc = np.zeros((nb, m, 2))
    Bc = np.zeros((nb, 2, 2))                    # coupling of block b (rows m-2, m-1) with b+1 (cols 0, 1)
    Bc[:, 0, 0] = D2[m - 2::m]
    Bc[:, 1, 0] = D1[m - 1::m]
    Bc[:, 1, 1] = D2[m - 1::m]
    S = np.zeros((m, m))
    rh = np.zeros((m, nc + 2))
    i = np.arange(m)
    corr = None
    for b in range(nb):
        o = b * m
        S[:] = 0.0
        S[i, i] = D0[o:o + m]
        S[i[:-1], i[1:]] = D1[o:o + m - 1]
        S[i[1:], i[:-1]] = D1[o:o + m - 1]
        S[i[:-2], i[2:]] = D2[o:o + m - 2]
        S[i[2:], i[:-2]] = D2[o:o + m - 2]
        if corr is not None:
            S[:2, :2] -= corr
        rh[:, :2] = 0.0
        rh[m - 2, 0] = 1.0
        rh[m - 1, 1] = 1.0
        rh[:, 2:] = X[o:o + m]
        Z = np.linalg.solve(S, rh)
        X[o:o + m] = Z[:, 2:]
        if b < nb - 1:
            Zc[b] = Z[:, :2]
            corr = Bc[b].T @ Z[m - 2:, :2] @ Bc[b]
            X[o + m:o + m + 2] -= Bc[b].T @ Z[m - 2:, 2:]
    for b in range(nb - 2, -1, -1):
        o = b * m
        X[o:o + m] -= Zc[b] @ (Bc[b] @ X[o + m:o + m + 2])
    return X[:n]


# ----------------------------------------------------------------------------
# 4b. rotor-drag velocity aiding (thrust-borne platforms only)
# ----------------------------------------------------------------------------

def _window_cov(a, b, win):
    """Moving covariance trace  E[a.b] - E[a].E[b]  over win blocks."""
    return _moving_mean(np.sum(a * b, axis=1), win) - np.sum(_moving_mean(a, win) * _moving_mean(b, win), axis=1)


def flight_mask(blk, Rm, g0, static, prm):
    """
    Blocks in which the platform is thrust-borne (multirotor in flight).

    Gravity-borne (resting, hand-held, walking, wheeled): the accelerometer
    sees the gravity reaction, so when the body tilts the specific force
    moves with the gravity direction in the body frame (regression slope ~1).
    Thrust-borne: specific force stays on the thrust axis whatever the tilt
    (slope ~0) and its off-axis part varies less than the gravity tilt does.
    The test needs tilt excitation; blocks without it inherit their neighbour's
    label, standstill is always gravity-borne.

    Returns (mask (K,), thrust axis in the body frame or None).
    """
    K = blk.dt.shape[0]
    win = int(prm["fly_window"])
    none = np.zeros(K, dtype=bool)
    if K < 4 * win:
        return none, None
    Rb = np.einsum("kij,kjl->kil", Rm, blk.Gam / blk.dt[:, None, None])
    gb = g0 * Rb[:, 2, :]                       # gravity reaction, body frame
    f = blk.fbar
    exc = prm["fly_excitation"] ** 2
    vg = _window_cov(gb, gb, win)
    beta = _window_cov(f, gb, win) / np.maximum(vg, 1e-12)
    cand = (np.abs(beta) < prm["fly_beta_max"]) & (vg > exc) & ~static
    if int(cand.sum()) < prm["fly_min_blocks"]:
        return none, None
    e_t = (f[cand] * blk.dt[cand, None]).sum(axis=0)
    e_t /= math.sqrt(float(e_t @ e_t))
    P = np.eye(3) - np.outer(e_t, e_t)
    fp, gp = f @ P, gb @ P
    vg = _window_cov(gp, gp, win)
    beta = _window_cov(fp, gp, win) / np.maximum(vg, 1e-12)
    ratio = _window_cov(fp, fp, win) / np.maximum(vg, 1e-12)
    known = (vg > exc) | static
    fly = (vg > exc) & ~static & (np.abs(beta) < prm["fly_beta_max"]) & (ratio < prm["fly_ratio_max"] ** 2)
    # blocks without tilt excitation inherit the previous (else next) known label
    idx = np.where(known, np.arange(K), -1)
    np.maximum.accumulate(idx, out=idx)
    first = int(np.argmax(known)) if known.any() else 0
    idx[idx < 0] = first
    fly = fly[idx] & ~static
    # drop short runs
    edges = np.flatnonzero(np.diff(np.concatenate(([False], fly, [False])).astype(np.int8)))
    for s, e in zip(edges[0::2], edges[1::2]):
        if e - s < win:
            fly[s:e] = False
    if int(fly.sum()) < prm["fly_min_blocks"]:
        return none, None
    return fly, e_t


def drag_aided_trend(blk, res, Rm, fly, e_t, static, lam, sev, prm):
    """
    Horizontal drift estimate with rotor-drag velocity aiding.

    On a multirotor the specific force perpendicular to the thrust axis is
    rotor drag, proportional to the body velocity:  f_perp = -mu_d v_perp + b.
    Over a flight block this gives the horizontal displacement
        disp_k = (u_k + B_k b(t_k)) / mu_d,
        u_k = -R_k int(P f) dt,     B_k = R_k int(R_ref) dt [e1 e2],
    which enters spline_trend() as displacement information. mu_d is a
    platform prior (PARAMS["drag_mu"]); the off-axis bias b (accelerometer
    bias + thrust-axis misalignment, a low-order polynomial in time because
    degrade_imu.py lets it random-walk) is unknown but linear, so it is
    eliminated exactly: one banded solve per basis column, then a small
    dense system.

    Returns (trend (K+1, 2), info dict).
    """
    K = blk.dt.shape[0]
    nb = int(prm["drag_bias_order"])
    inv_mu = 1.0 / prm["drag_mu"]
    par = blk.fbar @ e_t
    body = blk.U - (blk.Gam @ e_t) * par[:, None]
    u = -np.einsum("kij,kj->ki", Rm, body)[:, :2] * inv_mu
    u[~fly] = 0.0
    e1 = np.cross(e_t, [0.0, 0.0, 1.0] if abs(e_t[2]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= math.sqrt(float(e1 @ e1))
    e2 = np.cross(e_t, e1)
    Bm = np.einsum("kij,kjl->kil", Rm, blk.Gam) * inv_mu
    t_mid = 0.5 * (blk.tb[:-1] + blk.tb[1:])
    span = max(float(blk.tb[-1] - blk.tb[0]), 1e-9)
    tau = (t_mid - float(t_mid[fly].mean())) / span
    nt = 2 * nb
    D = np.zeros((K, 2, nt))
    for m in range(nb):
        ph = tau ** m
        D[:, :, 2 * m] = (Bm @ e1)[:, :2] * ph[:, None]
        D[:, :, 2 * m + 1] = (Bm @ e2)[:, :2] * ph[:, None]
    D[~fly] = 0.0

    sev_mid = 0.5 * (sev[:-1] + sev[1:])
    period = prm["drag_period_clean"] + (prm["drag_period_degraded"] - prm["drag_period_clean"]) * sev_mid
    gam = (period / (2.0 * math.pi)) ** 2
    mu = np.where(static, prm["zupt_trend_weight"], np.where(fly, gam / blk.dt, 0.0))
    praw = res["praw"]
    nc = nt + 1
    Y = np.zeros((K + 1, 2 * nc))
    disp = np.zeros((K, 2 * nc))
    for a in range(2):
        Y[:, a * nc] = praw[:, a]
        disp[:, a * nc] = u[:, a]
        disp[:, a * nc + 1:(a + 1) * nc] = D[:, a, :]
    C = spline_trend(blk.tb, Y, lam, mu, disp)
    dC = np.diff(C, axis=0)
    dy = np.diff(praw[:, :2], axis=0)
    A = np.zeros((nt, nt))
    g = np.zeros(nt)
    for a in range(2):
        Dw = D[:, a, :] * mu[:, None]
        A += Dw.T @ (D[:, a, :] + dC[:, a * nc + 1:(a + 1) * nc])
        g += Dw.T @ (dy[:, a] - dC[:, a * nc] - u[:, a])
    A.flat[::nt + 1] += 1e-9 * max(float(np.trace(A)), 1e-12)
    theta = np.linalg.solve(A, g)
    trend = np.empty((K + 1, 2))
    for a in range(2):
        trend[:, a] = C[:, a * nc] + C[:, a * nc + 1:(a + 1) * nc] @ theta
    info = {"flight_seconds": round(float(blk.dt[fly].sum()), 1),
            "thrust_axis_body": [round(float(x), 4) for x in e_t],
            "drag_coefficient_per_s": prm["drag_mu"],
            "drag_bias_ms2": [round(float(x), 4) for x in theta[:2]]}
    return trend, info


# ----------------------------------------------------------------------------
# per-row reconstruction
# ----------------------------------------------------------------------------

def reconstruct(t, pos, dp_ref, G, blk, res, trend, g0):
    """
    Per-row pose from block tracks (vectorised, chunked).

    On entry pos holds dv_ref and dp_ref holds dp_ref (float32, reference
    frame); on exit pos[i] is the drift-corrected world position and G[i] the
    world attitude quaternion of row i.
    """
    bnd, tb = blk.bnd, blk.tb
    K = blk.dt.shape[0]
    E_tr = res["E_tr"]
    flip = np.sum(E_tr[1:] * E_tr[:-1], axis=1) < 0.0
    if flip.any():
        E_tr[1:] *= np.cumprod(np.where(flip, -1.0, 1.0))[:, None]
    t_mid = 0.5 * (tb[:-1] + tb[1:])
    one_k = 1.0 - res["k_tr"]
    Rm_all_scale = one_k
    praw, vraw = res["praw"], res["vraw"]

    def attitude(rows):
        tt = t[rows]
        Ei = np.empty((tt.shape[0], 4))
        for c in range(4):
            Ei[:, c] = np.interp(tt, t_mid, E_tr[:, c])
        Ei /= np.sqrt(np.sum(Ei * Ei, axis=1))[:, None]
        q = qmul(Ei, G[rows].astype(np.float64))
        q /= np.sqrt(np.sum(q * q, axis=1))[:, None]
        G[rows] = q

    # row 0
    pos[0] = praw[0] - trend[0]
    attitude(slice(0, 1))

    k0 = 0
    while k0 < K:
        k1 = k0 + 1
        while k1 < K and bnd[k1 + 1] - bnd[k0] <= CHUNK_ROWS:
            k1 += 1
        a, b = int(bnd[k0]) + 1, int(bnd[k1]) + 1
        dv = pos[a:b].astype(np.float64)
        dp = dp_ref[a:b].astype(np.float64)
        tt = t[a:b]
        h = tt - t[a - 1:b - 1]
        starts = (bnd[k0:k1] + 1 - a).astype(np.intp)
        counts = np.diff(bnd[k0:k1 + 1])
        bid = np.repeat(np.arange(k1 - k0), counts)

        C = np.cumsum(dv, axis=0)
        C_prev = C - dv
        C_prev -= C_prev[starts][bid]
        term = C_prev * h[:, None] + dp
        D = np.cumsum(term, axis=0)
        D -= (D - term)[starts][bid]

        Rm = rotmats(E_tr[k0:k1]) * Rm_all_scale[k0:k1, None, None]
        ab = np.einsum("kij,kj->ki", Rm, np.einsum("kij,kj->ki", blk.Gam[k0:k1], res["ba_tr"][k0:k1]))
        ab /= blk.dt[k0:k1, None]
        gv = -ab
        gv[:, 2] -= g0
        tau = (tt - tb[k0:k1][bid])[:, None]
        p = praw[k0:k1][bid] + vraw[k0:k1][bid] * tau \
            + np.einsum("nij,nj->ni", Rm[bid], D) + 0.5 * gv[bid] * tau * tau
        for c in range(3):
            p[:, c] -= np.interp(tt, tb, trend[:, c])
        pos[a:b] = p
        attitude(slice(a, b))
        k0 = k1


# ----------------------------------------------------------------------------
# 5. blind-window extrapolation
# ----------------------------------------------------------------------------

def blind_extrapolate(t, pos, quat, n_act, w_term, prm):
    """
    Dead-reckon the failure window from a local kinematic model.

    Weighted least squares over the last fit_seconds of valid flight gives the
    terminal velocity and acceleration trend; w_term is the weighted mean body
    rate. The model is propagated as a damped coordinated turn across the
    timestamps that are present in the failure window (no rows added/removed).
    """
    m = t.shape[0]
    if n_act >= m:
        return {}
    if n_act < 1:
        pos[:] = 0.0
        quat[:] = (0.0, 0.0, 0.0, 1.0)
        return {}
    t_end = float(t[n_act - 1])
    i0 = int(np.searchsorted(t[:n_act], t_end - prm["fit_seconds"]))
    tau = t[i0:n_act] - t_end
    p_end = pos[n_act - 1].astype(np.float64)
    q_end = quat[n_act - 1].astype(np.float64)
    v_term = np.zeros(3)
    a_term = np.zeros(3)
    if tau.shape[0] >= 8 and tau[0] < -0.2:
        sw = np.sqrt(np.exp(tau / prm["fit_weight_tau"]))
        A = np.stack((np.ones_like(tau), tau, 0.5 * tau * tau), axis=1) * sw[:, None]
        coef = np.linalg.lstsq(A, pos[i0:n_act].astype(np.float64) * sw[:, None], rcond=None)[0]
        v_term, a_term = coef[1], coef[2]
        vn, an = math.sqrt(float(v_term @ v_term)), math.sqrt(float(a_term @ a_term))
        if vn > prm["vel_max"]:
            v_term = v_term * (prm["vel_max"] / vn)
        if an > prm["acc_max"]:
            a_term = a_term * (prm["acc_max"] / an)

    d = t[n_act:] - t_end
    R_end = _rotmat_s(tuple(q_end))
    yaw_rate = float((R_end @ w_term)[2])
    tw = prm["turn_damp_tau"]
    s_turn = tw * (1.0 - np.exp(-d / tw))
    psi = yaw_rate * s_turn
    vel = v_term[None, :] * np.exp(-d / prm["vel_damp_tau"])[:, None] \
        + a_term[None, :] * (d * np.exp(-d / prm["acc_damp_tau"]))[:, None]
    cp, sp = np.cos(psi), np.sin(psi)
    vx = cp * vel[:, 0] - sp * vel[:, 1]
    vy = sp * vel[:, 0] + cp * vel[:, 1]
    vel[:, 0], vel[:, 1] = vx, vy
    v_prev = np.concatenate((v_term[None, :], vel[:-1]))
    d_prev = np.concatenate(([0.0], d[:-1]))
    pos[n_act:] = p_end + np.cumsum(0.5 * (v_prev + vel) * (d - d_prev)[:, None], axis=0)
    qz = np.zeros((d.shape[0], 4))
    qz[:, 2], qz[:, 3] = np.sin(0.5 * psi), np.cos(0.5 * psi)
    q = qmul(qz, q_end[None, :])                 # turn about the world vertical: tilt is held
    q /= np.sqrt(np.sum(q * q, axis=1))[:, None]
    quat[n_act:] = q
    return {"v_term": [round(float(x), 4) for x in v_term],
            "a_trend": [round(float(x), 4) for x in a_term],
            "w_term": [round(float(x), 5) for x in w_term],
            "yaw_rate_term": round(yaw_rate, 5),
            "blind_seconds": round(float(d[-1]), 3)}


# ----------------------------------------------------------------------------
# top level
# ----------------------------------------------------------------------------

def estimate_trajectory(pre, cfg=CONFIG, prm=PARAMS):
    """
    Estimate one pose per (de-duplicated) row of a Preprocessed record.

    Returns (pos (M, 3) float32, quat (M, 4) float32 [x y z w], summary dict).
    pos aliases pre.acc and pre.gyr is used as scratch: both are consumed.
    """
    t, acc, gyr, n_act = pre.t, pre.acc, pre.gyr, pre.n_active
    m = t.shape[0]
    quat = np.zeros((m, 4), dtype=np.float32)
    quat[:, 3] = 1.0
    summary = {}
    if n_act < 3:
        acc[:] = 0.0
        return acc, quat, summary

    T = pre.duration if pre.duration > 0 else 1.0
    fbar0, wbar0, static0 = protected_init(t, acc, gyr, n_act, cfg["protect_seconds"])
    bnd = block_boundaries(t, n_act, prm["block_dt"])
    K = bnd.shape[0] - 1
    sev = severity(0.5 * (t[bnd][:-1] + t[bnd][1:]) / T, cfg)

    bg = wbar0.copy() if static0 else np.zeros(3)
    ba = np.zeros(3)

    def setup(blk, g0):
        Q, bg_drift = process_noise(blk, sev, g0, pre.median_dt, cfg, prm)
        static = detect_static(blk, g0, static0, bg_drift)
        wd = np.zeros_like(blk.wbar)
        if K > 2:
            wd[1:-1] = (blk.wbar[2:] - blk.wbar[:-2]) / (blk.dt[1:-1] + 0.5 * (blk.dt[2:] + blk.dt[:-2]))[:, None]
        fn = np.sqrt(np.sum(blk.fbar ** 2, axis=1))
        ok = (np.sqrt(np.sum(wd * wd, axis=1)) < 3.0) & (np.abs(fn - g0) < 1.5) & (blk.gap < 0.3)
        return Q, static, ok

    def gravity(blk):
        if static0:
            f = fbar0 - ba
            return math.sqrt(float(f @ f))
        return float(median(np.sqrt(np.sum(blk.fbar ** 2, axis=1))))

    # ---- pass 1: learn the biases (block sums only, sensors untouched) --------
    blk = preintegrate(t, acc, gyr, n_act, bnd, bg, ba, pre.median_dt)
    g0 = gravity(blk)
    if static0:
        P0 = [0.02 ** 2] * 3 + [0.08 ** 2, 0.08 ** 2, 1e-8] + [0.003 ** 2] * 3 + [0.3 ** 2] * 3 + [0.003 ** 2]
    else:
        P0 = [1.0] * 3 + [0.15 ** 2, 0.15 ** 2, 1e-8] + [0.05 ** 2] * 3 + [0.3 ** 2] * 3 + [0.003 ** 2]
    q0 = gravity_alignment(fbar0 - ba)
    Q, static, gn_ok = setup(blk, g0)
    res = run_filter(blk, g0, q0, P0, Q, static, gn_ok, prm, record=False)

    # ---- pass 2: re-linearise around the best-known biases ---------------------
    bg = bg + res["best_bg"][1]
    ba = ba + res["best_ba"][1]
    P0[6:9] = np.minimum(P0[6:9], 4.0 * res["best_bg"][2] + 1e-8).tolist()
    P0[9:12] = np.minimum(P0[9:12], 4.0 * res["best_ba"][2] + 1e-6).tolist()
    del blk, res, Q, static, gn_ok
    blk = preintegrate(t, acc, gyr, n_act, bnd, bg, ba, pre.median_dt, G_out=quat)
    g0 = gravity(blk)
    q0 = gravity_alignment(fbar0 - ba)
    Q, static, gn_ok = setup(blk, g0)
    res = run_filter(blk, g0, q0, P0, Q, static, gn_ok, prm, record=True)
    del Q, gn_ok

    # ---- zero-phase drift removal ---------------------------------------------
    sev_b = severity(blk.tb / T, cfg)
    calibrated = bool(static0 or res["n_zupt"] >= prm["calibrated_min_zupt"])
    key = "trend_period_" if calibrated else "trend_period_uncal_"
    period = prm[key + "clean"] + (prm[key + "degraded"] - prm[key + "clean"]) * sev_b
    lam = (period / (2.0 * math.pi)) ** 4
    mu = np.where(static, prm["zupt_trend_weight"], 0.0)
    trend = spline_trend(blk.tb, res["praw"], lam, mu)
    summary["drag_aid"] = {"used": False}
    if prm["drag_aid"]:
        Rm = rotmats(res["E_tr"])
        fly, e_t = flight_mask(blk, Rm, g0, static, prm)
        if e_t is not None:
            trend[:, :2], info = drag_aided_trend(blk, res, Rm, fly, e_t, static, lam, sev_b, prm)
            summary["drag_aid"].update(info)
            summary["drag_aid"]["used"] = True
    reconstruct(t, acc, gyr, quat, blk, res, trend, g0)

    # ---- blind window ---------------------------------------------------------
    t_end = blk.tb[-1]
    sel = blk.tb[1:] > t_end - prm["fit_seconds"]
    wts = np.exp((0.5 * (blk.tb[:-1] + blk.tb[1:])[sel] - t_end) / prm["fit_weight_tau"]) * blk.dt[sel]
    w_term = (blk.wbar[sel] * wts[:, None]).sum(axis=0) / max(float(wts.sum()), 1e-12) - res["dbg"]
    summary["blind"] = blind_extrapolate(t, acc, quat, n_act, w_term, prm)

    summary.update({
        "static_init": bool(static0),
        "gyro_bias_calibrated": calibrated,
        "g0": round(g0, 5),
        "gyro_bias": [round(float(x), 5) for x in (bg + res["dbg"])],
        "accel_bias": [round(float(x), 4) for x in (ba + res["dba"])],
        "accel_scale_error": round(float(res["scale"]), 5),
        "filter_blocks": int(K),
        "zupt_blocks": int(res["n_zupt"]),
    })
    return acc, quat, summary
