#!/usr/bin/env python3
"""
synthetic_sequences.py - clean synthetic IMU recordings with exact ground truth.

Each sequence is a smooth 6-DoF motion sampled analytically (8x oversampled
finite differences), turned into ideal body-frame specific force / angular
rate, and given a realistic constant sensor bias plus a little white noise so
that the "clean" file resembles a real IMU log. The official degrade_imu.py
is then applied to it by benchmark_all.py.

The two sequences deliberately differ in everything the estimator has to
auto-detect:

    SYN_room_walk    200 Hz, 90 s, timestamps in s,  accel in m/s^2, no header,
                     standstill at the start (headset picked up, room walk)
    SYN_fig8_moving  100 Hz, 70 s, timestamps in ms, accel in g, header row,
                     already moving at t = 0 (no standstill anywhere),
                     x-up sensor mounting as on the EuRoC MAV

Development/benchmark tooling only (never imported by src/main.py).
"""

import numpy as np

G = 9.80665
OVERSAMPLE = 8


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def _rot_zyx(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    R = np.empty(yaw.shape + (3, 3))
    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = cy * sp * sr - sy * cr
    R[..., 0, 2] = cy * sp * cr + sy * sr
    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = sy * sp * sr + cy * cr
    R[..., 1, 2] = sy * sp * cr - cy * sr
    R[..., 2, 0] = -sp
    R[..., 2, 1] = cp * sr
    R[..., 2, 2] = cp * cr
    return R


def _quat_from_R(R):
    """(n, 3, 3) -> (n, 4) [x, y, z, w], sign-continuous."""
    n = R.shape[0]
    q = np.empty((n, 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q[:, 3] = 0.5 * np.sqrt(np.maximum(0.0, 1.0 + tr))
    q[:, 0] = 0.5 * np.sqrt(np.maximum(0.0, 1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]))
    q[:, 1] = 0.5 * np.sqrt(np.maximum(0.0, 1.0 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2]))
    q[:, 2] = 0.5 * np.sqrt(np.maximum(0.0, 1.0 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2]))
    big = np.argmax(q, axis=1)
    for i in range(n):
        b = big[i]
        r = R[i]
        if b == 3:
            q[i, 0], q[i, 1], q[i, 2] = (r[2, 1] - r[1, 2]) / (4 * q[i, 3]), (r[0, 2] - r[2, 0]) / (4 * q[i, 3]), (r[1, 0] - r[0, 1]) / (4 * q[i, 3])
        elif b == 0:
            q[i, 3], q[i, 1], q[i, 2] = (r[2, 1] - r[1, 2]) / (4 * q[i, 0]), (r[0, 1] + r[1, 0]) / (4 * q[i, 0]), (r[0, 2] + r[2, 0]) / (4 * q[i, 0])
        elif b == 1:
            q[i, 3], q[i, 0], q[i, 2] = (r[0, 2] - r[2, 0]) / (4 * q[i, 1]), (r[0, 1] + r[1, 0]) / (4 * q[i, 1]), (r[1, 2] + r[2, 1]) / (4 * q[i, 1])
        else:
            q[i, 3], q[i, 0], q[i, 1] = (r[1, 0] - r[0, 1]) / (4 * q[i, 2]), (r[0, 2] + r[2, 0]) / (4 * q[i, 2]), (r[1, 2] + r[2, 1]) / (4 * q[i, 2])
        if i and np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _simulate(pose_fn, duration, rate, mount, gyro_bias, acc_bias, seed):
    """pose_fn(t) -> (position (m, 3), yaw, pitch, roll). Returns dict of arrays."""
    dt = 1.0 / rate
    dtf = dt / OVERSAMPLE
    n = int(round(duration * rate)) + 1
    tf = (np.arange((n - 1) * OVERSAMPLE + 1 + 4 * OVERSAMPLE) - 2 * OVERSAMPLE) * dtf
    p, yaw, pitch, roll = pose_fn(tf)
    v = np.gradient(p, dtf, axis=0)
    a = np.gradient(v, dtf, axis=0)
    R = _rot_zyx(yaw, pitch, roll) @ mount
    dR = np.einsum("nji,njk->nik", R[:-2], R[2:])            # R[j-1]^T R[j+1]
    w = np.stack((dR[:, 2, 1] - dR[:, 1, 2], dR[:, 0, 2] - dR[:, 2, 0], dR[:, 1, 0] - dR[:, 0, 1]), axis=1)
    sin_th = 0.5 * np.linalg.norm(w, axis=1)
    th = np.arcsin(np.clip(sin_th, 0.0, 1.0))
    w = 0.5 * w * np.where(sin_th > 1e-12, th / np.maximum(sin_th, 1e-12), 1.0)[:, None] / (2.0 * dtf)
    idx = 2 * OVERSAMPLE + np.arange(n) * OVERSAMPLE
    Rc = R[idx]
    f = np.einsum("nji,nj->ni", Rc, a[idx] + np.array([0.0, 0.0, G]))   # R^T (a - g_world)
    wc = w[idx - 1]

    rng = np.random.default_rng(seed)
    acc = f + np.asarray(acc_bias) + rng.normal(0.0, 0.02, f.shape)
    gyr = wc + np.asarray(gyro_bias) + rng.normal(0.0, 0.002, wc.shape)
    return {"t": np.arange(n) * dt, "acc": acc, "gyr": gyr, "pos": p[idx], "quat": _quat_from_R(Rc)}


def room_walk(duration=90.0, rate=200.0, seed=7):
    """Headset-like room-scale walk: 3.5 s standstill, looping walk with head motion."""
    t_start, t_stop = 3.5, duration - 1.0

    def pose(t):
        s = _smoothstep((t - t_start) / (t_stop - t_start))
        u = 2.0 * np.pi * 5.25 * s
        moving = _smoothstep((t - t_start) / 2.0) * _smoothstep((t_stop - t) / 2.0)
        x = 2.0 * np.cos(u) + 0.4 * np.cos(3.0 * u + 0.5)
        y = 1.4 * np.sin(u) + 0.5 * np.sin(2.0 * u)
        z = 1.6 + 0.12 * np.sin(0.5 * u) + 0.025 * moving * np.sin(2.0 * np.pi * 1.8 * t)
        du = 1e-4
        tx = (2.0 * np.cos(u + du) + 0.4 * np.cos(3.0 * (u + du) + 0.5)) - x
        ty = (1.4 * np.sin(u + du) + 0.5 * np.sin(2.0 * (u + du))) - y
        yaw = np.unwrap(np.arctan2(ty, tx)) + 0.35 * moving * np.sin(0.7 * t)
        pitch = 0.10 * moving * np.sin(0.5 * t + 0.3)
        roll = 0.05 * moving * np.sin(0.9 * t + 1.0)
        return np.stack((x, y, z), axis=1), yaw, pitch, roll

    mount = _rot_zyx(np.array(0.0), np.array(0.0), np.array(np.radians(8.0)))
    return _simulate(pose, duration, rate, mount, (0.012, -0.008, 0.020), (0.08, -0.05, 0.10), seed)


def fig8_moving(duration=70.0, rate=100.0, seed=11):
    """Figure-eight flight that is already under way at t = 0 (no standstill)."""
    om = 2.0 * np.pi / 12.0

    def pose(t):
        s, c = np.sin(om * t), np.cos(om * t)
        den = 1.0 + s * s
        x, y = 3.0 * c / den, 3.0 * s * c / den
        z = 1.2 + 0.4 * np.sin(0.5 * om * t)
        du = 1e-4
        s2, c2 = np.sin(om * t + du), np.cos(om * t + du)
        tx = 3.0 * c2 / (1.0 + s2 * s2) - x
        ty = 3.0 * s2 * c2 / (1.0 + s2 * s2) - y
        yaw = np.unwrap(np.arctan2(ty, tx))
        pitch = 0.12 * np.sin(om * t + 0.4)
        roll = 0.20 * np.sin(2.0 * om * t)
        return np.stack((x, y, z), axis=1), yaw, pitch, roll

    mount = _rot_zyx(np.array(0.0), np.array(np.radians(-90.0)), np.array(0.0))   # sensor x axis up
    return _simulate(pose, duration, rate, mount, (-0.004, 0.015, 0.030), (0.05, 0.12, -0.06), seed)


SEQUENCES = {
    "SYN_room_walk": {"make": room_walk, "ts_unit": "s", "accel_unit": "ms2", "header": False},
    "SYN_fig8_moving": {"make": fig8_moving, "ts_unit": "ms", "accel_unit": "g", "header": True},
}
