# -*- coding: utf-8 -*-
"""双曲线飞越/引力弹弓的二维 patched-conic 几何."""

import numpy as np


def hyperbolic_flyby(mu, v_inf, periapsis_radius, incoming_direction,
                     extent=0.72, samples=600):
    """生成行星中心系中的双曲线飞越轨迹.

    mu: 行星引力常数, m^3/s^2
    v_inf: 无穷远相对速度, m/s
    periapsis_radius: 近心点半径, m
    incoming_direction: 入射渐近线速度方向 (二维)
    extent: 显示到渐近线角度的比例 (0,1)

    返回的 positions 单位 m, velocities 单位 m/s.
    """
    v_inf = float(v_inf)
    rp = float(periapsis_radius)
    a = -mu / v_inf ** 2
    e = 1.0 - rp / a
    p = rp * (1.0 + e)
    nu_inf = np.arccos(-1.0 / e)
    h = np.sqrt(mu * p)

    nu = np.linspace(-extent * nu_inf, extent * nu_inf, samples)
    r = p / (1.0 + e * np.cos(nu))
    pos = np.column_stack([r * np.cos(nu), r * np.sin(nu)])
    vel = np.column_stack([
        -mu / h * np.sin(nu),
        mu / h * (e + np.cos(nu)),
    ])

    # 用渐近线方向把双曲线旋转到指定的入射方向.
    c_inf = -1.0 / e
    s_inf = np.sqrt(1.0 - c_inf ** 2)
    vin = np.array([-mu / h * (-s_inf), mu / h * (e + c_inf)])
    vout = np.array([-mu / h * s_inf, mu / h * (e + c_inf)])
    in_dir = np.asarray(incoming_direction, dtype=float)
    in_dir = in_dir / np.linalg.norm(in_dir)
    angle = np.arctan2(in_dir[1], in_dir[0]) - np.arctan2(vin[1], vin[0])
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])
    pos = pos @ rot.T
    vel = vel @ rot.T
    vin = rot @ vin
    vout = rot @ vout

    turn = np.arccos(np.clip(
        vin @ vout / (np.linalg.norm(vin) * np.linalg.norm(vout)),
        -1.0, 1.0))
    return {
        "positions": pos,
        "velocities": vel,
        "vin_inf": vin,
        "vout_inf": vout,
        "turn_angle": float(turn),
        "eccentricity": float(e),
        "nu_inf": float(nu_inf),
        "periapsis_radius": rp,
        "v_inf": v_inf,
    }


def heliocentric_velocity(planet_velocity, v_inf, direction):
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    return np.asarray(planet_velocity, dtype=float) + v_inf * direction
