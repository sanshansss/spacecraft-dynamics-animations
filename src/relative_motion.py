# -*- coding: utf-8 -*-
"""CW/Hill 相对运动与双脉冲交会 (自写, 仅依赖 numpy)."""

import numpy as np


def cw_stm(n, t):
    """Clohessy-Wiltshire 状态转移矩阵.

    状态顺序: [x, y, z, vx, vy, vz].
    x 为径向 (远离地心为正), y 为沿轨, z 为轨道面法向.
    """
    c = np.cos(n * t)
    s = np.sin(n * t)
    phi = np.zeros((6, 6))
    phi[0, 0] = 4.0 - 3.0 * c
    phi[0, 3] = s / n
    phi[0, 4] = 2.0 * (1.0 - c) / n
    phi[1, 0] = 6.0 * (s - n * t)
    phi[1, 1] = 1.0
    phi[1, 3] = 2.0 * (c - 1.0) / n
    phi[1, 4] = (4.0 * s - 3.0 * n * t) / n
    phi[2, 2] = c
    phi[2, 5] = s / n
    phi[3, 0] = 3.0 * n * s
    phi[3, 3] = c
    phi[3, 4] = 2.0 * s
    phi[4, 0] = -6.0 * n * (1.0 - c)
    phi[4, 3] = -2.0 * (1.0 - c)
    phi[4, 4] = 4.0 * c - 3.0
    phi[5, 2] = -n * s
    phi[5, 5] = c
    return phi


def propagate_cw(state0, n, times):
    """解析传播一组 CW 状态."""
    state0 = np.asarray(state0, dtype=float)
    return np.array([cw_stm(n, t) @ state0 for t in np.asarray(times)])


def cw_ode(state, n):
    """无推力 CW 方程的连续形式, 用于数值校验."""
    x, y, z, vx, vy, vz = state
    return np.array([
        vx,
        vy,
        vz,
        2.0 * n * vy + 3.0 * n ** 2 * x,
        -2.0 * n * vx,
        -n ** 2 * z,
    ])


def natural_ellipse_state(along_track_radius, n):
    """无推力闭合绕飞的初始状态.

    取 x0=rho/2, y0=0, vy0=-n*rho, 得到
      x = rho/2 cos(nt), y = -rho sin(nt).
    径向幅值是沿轨幅值的一半.
    """
    rho = float(along_track_radius)
    return np.array([rho / 2.0, 0.0, 0.0, 0.0, -n * rho, 0.0])


def two_impulse_rendezvous(r0, n, tof, v_initial=None):
    """给定初始位置和转移时间, 求双脉冲交会.

    第一次脉冲只保证在 tof 时刻到达目标位置; 到达时的剩余相对速度
    再由第二次脉冲抵消.
    """
    r0 = np.asarray(r0, dtype=float)
    v_initial = (np.zeros(3) if v_initial is None
                 else np.asarray(v_initial, dtype=float))
    phi = cw_stm(n, tof)
    phi_rr = phi[:3, :3]
    phi_rv = phi[:3, 3:]
    v0 = np.linalg.solve(phi_rv, -phi_rr @ r0)
    state0 = np.concatenate([r0, v0])
    rf = phi[:3] @ state0
    vf = phi[3:] @ state0
    dv1 = v0 - v_initial
    dv2 = -vf
    return {
        "state0": state0,
        "dv1": dv1,
        "dv2": dv2,
        "dv_total": float(np.linalg.norm(dv1) + np.linalg.norm(dv2)),
        "rf": rf,
        "vf_before_second_burn": vf,
        "tof": float(tof),
    }


def circular_orbit_rate(mu, radius):
    return np.sqrt(mu / radius ** 3)
