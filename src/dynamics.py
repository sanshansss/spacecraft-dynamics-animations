# -*- coding: utf-8 -*-
"""姿态动力学教学动画用的自写动力学仿真(无外部依赖, 仅 numpy)."""

import numpy as np


def skew(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_mult(a, b):
    """哈密顿乘积 a (x) b."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q, v):
    """用四元数 q 把向量 v 从本体系转到惯性系."""
    vv = np.array([0.0, v[0], v[1], v[2]])
    r = quat_mult(quat_mult(q, vv), quat_conj(q))
    return r[1:]


def quat_to_matrix(q):
    """单位四元数 -> 旋转矩阵 (本体系->惯性系, 列向量约定)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def body_to_euler_rates(w, angles, order):
    """
    本体角速度 -> 欧拉角速率. 课件 7.3.2 的两套姿态角运动学.
    '313' (第一套, 3-1-3):  铅垂轴 z -> x -> z
        phi̇ = (w_y sin φ + w_z cos φ)/cos θ   等标准关系, 见实现
    '312' (第二套, 3-1-2):  z -> x -> y
    返回 dψ/dt, dθ/dt, dφ/dt.
    """
    angles = np.asarray(angles, dtype=float)
    if order == '313':
        psi, theta, phi = angles
        wx, wy, wz = w
        st, ct = np.sin(theta), np.cos(theta)
        sf, cf = np.sin(phi), np.cos(phi)
        # 第一套 3-1-3 (R = Rz(phi) Rx(theta) Rz(psi)):
        # wx = psid*st*sf + thid*cf; wy = psid*st*cf - thid*sf; wz = psid*ct + phid
        theta_dot = wx * cf + wy * sf
        psi_dot = (wx * sf - wy * cf) / st
        phi_dot = wz - psi_dot * ct
        return np.array([psi_dot, theta_dot, phi_dot])
    if order == '312':
        psi, theta, phi = angles  # psi: 绕z, theta: 绕x, phi: 绕y
        wx, wy, wz = w
        st, ct = np.sin(theta), np.cos(theta)
        sf, cf = np.sin(phi), np.cos(phi)
        # 第二套 3-1-2: w = ψ̇ z3 + θ̇ x' + φ̇ y_b
        theta_dot = wx * cf - wz * sf
        psi_dot = (wx * sf + wz * cf) / ct
        phi_dot = wy + psi_dot * st
        return np.array([psi_dot, theta_dot, phi_dot])
    raise ValueError(order)


def tumbling(t, state, inertia, torque):
    """自由刚体(或常值外力矩)姿态运动: 状态 = [四元数, 角速度]."""
    q = quat_normalize(state[:4])
    w = state[4:]
    I = inertia
    Iw = I @ w
    w_cross_Iw = np.cross(w, Iw)
    w_dot = np.linalg.inv(I) @ (torque(t) - w_cross_Iw)
    # 本体角速度约定 (R: 本体系->惯性系): q_dot = 1/2 q (x) [0, w]
    q_dot = 0.5 * quat_mult(q, np.array([0.0, *w]))
    return np.concatenate([q_dot, w_dot])


def integrate_rk4(fun, t0, t1, y0, dt):
    """固定步长 RK4, 返回 (ts, ys)."""
    n = max(1, int(np.ceil((t1 - t0) / dt)))
    ts = np.linspace(t0, t1, n + 1)
    ys = np.zeros((n + 1, len(y0)))
    ys[0] = y0
    y = np.array(y0, dtype=float)
    for k in range(n):
        t = ts[k]
        h = ts[k + 1] - ts[k]
        k1 = fun(t, y)
        k2 = fun(t + h / 2, y + h / 2 * k1)
        k3 = fun(t + h / 2, y + h / 2 * k2)
        k4 = fun(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        y[:4] = quat_normalize(y[:4])
        ys[k + 1] = y
    return ts, ys


def run_tumbling(inertia, w0, q0=None, t_end=10.0, dt=2e-4, torque=None):
    torque = torque or (lambda t: np.zeros(3))
    if q0 is None:
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
    y0 = np.concatenate([quat_normalize(q0), np.asarray(w0, dtype=float)])
    return integrate_rk4(lambda t, y: tumbling(t, y, inertia, torque), 0.0, t_end, y0, dt)


def gravity_gradient_rod(t, state, params):
    """
    匀质杆在圆轨道上的平面引力梯度摆 (7.5.1).
    状态: [theta(偏离铅垂/本地垂线角), theta_dot]
    线性化方程 theta_ddot = -3 (B/A) n^2 * sin(theta)cos(theta) 近似 -> 稳定
    直接用非线性:  M = -3 n^2 (B) sin θ cos θ  归一化后
    """
    theta, theta_dot = state
    n2 = params['n'] ** 2
    ratio = params['inertia_ratio']  # B/A, 长杆绕横轴/绕纵轴 >1
    theta_ddot = -3.0 * n2 * ratio * np.sin(theta) * np.cos(theta)
    return np.array([theta_dot, theta_ddot])


def run_gravity_gradient_rod(theta0=0.3, theta_dot0=0.0, n_orbit=1.0,
                             inertia_ratio=2.0, periods=2.0, dt=1e-3):
    """
    n_orbit: 轨道角速率 rad/s. periods: 仿真多少个轨道周期.
    """
    t_end = periods * 2 * np.pi / n_orbit
    ts = np.arange(0.0, t_end + dt, dt)
    ys = np.zeros((len(ts), 2))
    y = np.array([theta0, theta_dot0])
    p = {'n': n_orbit, 'inertia_ratio': inertia_ratio}
    for k, t in enumerate(ts):
        ys[k] = y
        if k == len(ts) - 1:
            break
        h = ts[k + 1] - t
        f = lambda s, yy: gravity_gradient_rod(s, yy, p)
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ts, ys


def run_energy_dissipation(inertia, w0, k=0.08, t_end=60.0, dt=2e-3, q0=None):
    """
    能量耗散模型: 力矩 M = -k*(w - (w·e)e), e 为本体角动量方向.
    满足 d|H|/dt = 0 且 dE/dt = -k|w_perp|^2 <= 0,
    即课件 7.4.3 的两条分析依据的工程实现.
    """
    I = np.asarray(inertia, dtype=float)

    def fun(t, state):
        q = quat_normalize(state[:4])
        w = state[4:]
        H = I @ w
        e = H / np.linalg.norm(H)
        w_perp = w - (w @ e) * e
        M = -k * w_perp
        w_dot = np.linalg.inv(I) @ (M - np.cross(w, H))
        q_dot = 0.5 * quat_mult(q, np.array([0.0, *w]))
        return np.concatenate([q_dot, w_dot])

    if q0 is None:
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
    y0 = np.concatenate([quat_normalize(q0), np.asarray(w0, dtype=float)])
    ts, ys = integrate_rk4(fun, 0.0, t_end, y0, dt)
    energy = 0.5 * np.einsum('ij,jk,ik->i', ys[:, 4:], I, ys[:, 4:])
    return ts, ys, energy


def _gg_rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def _gg_rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def _gg_rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def _gg_kinematics_G(phi, theta):
    """3-2-1 姿态角速率 -> 本体角速度 (含轨道系 y 轴反向的符号约定)."""
    st, ct = np.sin(theta), np.cos(theta)
    sf, cf = np.sin(phi), np.cos(phi)
    tn = st / ct
    return np.array([
        [1.0, tn * sf, tn * cf],
        [0.0, cf, -sf],
        [0.0, sf / ct, cf / ct],
    ])


def run_gravity_gradient_satellite(I=(150.0, 200.0, 100.0),
                                   mu=3.986005e14, R_orbit=6580e3,
                                   phi0=3 * np.pi / 180,
                                   theta0=-5 * np.pi / 180,
                                   psi0=2 * np.pi / 180,
                                   n_orbits=10, dt=0.1):
    """
    逐行移植 references/10 重力梯度/test2/main.m 的线性化模型:
      引力梯度力矩: Mx=-3mu/R^3 (I2-I3) phi, My=-3mu/R^3 (I1-I3) theta,
                    Mz=-3mu/R^3 (I2-I1) theta*phi
      欧拉动力学 (本体绝对角速度 w), 运动学用相对角速度 w_r = w - w_orb_b,
      3-2-1 姿态角, 初始 w = [0, -w_orb, 0] (随轨道坐标系一起转动).
    返回 ts, angles(N,3), Lbos(N,3,3). Lbos[k] 为被动 DCM Lbo = X(phi)Y(theta)Z(psi),
    与 test2 的 x/y/z_rotation.m 约定一致 (分量变换矩阵).
    """
    I1, I2, I3 = I
    w_orb = np.sqrt(mu / R_orbit ** 3)
    t_end = n_orbits * 2 * np.pi / w_orb
    n_steps = int(round(t_end / dt))
    ts = np.arange(n_steps + 1) * dt
    angles = np.zeros((n_steps + 1, 3))
    Lbos = np.zeros((n_steps + 1, 3, 3))
    w = np.array([0.0, -w_orb, 0.0])
    ang = np.array([phi0, theta0, psi0])

    def step(a, wv):
        phi, theta, psi = a
        Mx = -3 * mu / R_orbit ** 3 * (I2 - I3) * phi
        My = -3 * mu / R_orbit ** 3 * (I1 - I3) * theta
        Mz = -3 * mu / R_orbit ** 3 * (I2 - I1) * theta * phi
        w_dot = np.array([
            (Mx + (I2 - I3) * wv[1] * wv[2]) / I1,
            (My + (I3 - I1) * wv[2] * wv[0]) / I2,
            (Mz + (I1 - I2) * wv[0] * wv[1]) / I3,
        ])
        Lbo = _gg_rot_x(phi) @ _gg_rot_y(theta) @ _gg_rot_z(psi)
        w_orb_b = Lbo @ np.array([0.0, -w_orb, 0.0])
        ang_dot = _gg_kinematics_G(phi, theta) @ (wv - w_orb_b)
        return ang_dot, w_dot

    for k in range(n_steps + 1):
        phi, theta, psi = ang
        angles[k] = ang
        Lbos[k] = _gg_rot_x(phi) @ _gg_rot_y(theta) @ _gg_rot_z(psi)
        if k == n_steps:
            break
        # RK4 与 test2 的显式 Euler 同一模型, 积分器换成 RK4 提高精度
        k1a, k1w = step(ang, w)
        k2a, k2w = step(ang + dt / 2 * k1a, w + dt / 2 * k1w)
        k3a, k3w = step(ang + dt / 2 * k2a, w + dt / 2 * k2w)
        k4a, k4w = step(ang + dt * k3a, w + dt * k3w)
        ang = ang + dt / 6 * (k1a + 2 * k2a + 2 * k3a + k4a)
        w = w + dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
    return ts, angles, Lbos
