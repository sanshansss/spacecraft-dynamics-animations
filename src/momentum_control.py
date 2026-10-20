# -*- coding: utf-8 -*-
"""反作用轮与单框架 CMG 的教学模型."""

import numpy as np


def reaction_wheel_spinup(J_body, J_wheel, motor_torque,
                          t_end=12.0, dt=1e-3):
    """恒值电机力矩驱动反作用轮; 本体角动量与轮子相反.

    返回 body_angle, body_rate, wheel_rate, wheel_momentum.
    """
    ts = np.arange(0.0, t_end + 0.5 * dt, dt)
    body_rate = -(motor_torque / J_body) * ts
    wheel_rate = (motor_torque / J_wheel) * ts
    body_angle = -0.5 * (motor_torque / J_body) * ts ** 2
    wheel_momentum = J_wheel * wheel_rate
    total_momentum = J_body * body_rate + wheel_momentum
    return ts, body_angle, body_rate, wheel_rate, wheel_momentum, total_momentum


def reaction_wheel_desaturation(J_body, J_wheel, wheel_momentum0,
                                thruster_torque, t_end=8.0, dt=1e-3):
    """外力矩卸载: 本体系施加 -tau_ext, 轮子角动量线性下降."""
    ts = np.arange(0.0, t_end + 0.5 * dt, dt)
    h_wheel = np.maximum(0.0, wheel_momentum0 - thruster_torque * ts)
    wheel_rate = h_wheel / J_wheel
    body_rate = np.zeros_like(ts)
    return ts, h_wheel, wheel_rate, body_rate


def cmg_h_vector(delta, H):
    """框架轴为 y, 转子动量在 x-z 平面内转动."""
    return H * np.array([np.cos(delta), 0.0, np.sin(delta)])


def cmg_torque(delta, delta_rate, H):
    """转子角动量变化率, 对航天器本体的反作用力矩为其负值."""
    h_dot = H * delta_rate * np.array(
        [-np.sin(delta), 0.0, np.cos(delta)])
    return -h_dot


def parallel_axis_cmg_torque_directions(delta1, delta2):
    t1 = np.array([np.cos(delta1), -np.sin(delta1)])
    t2 = np.array([np.cos(delta2), -np.sin(delta2)])
    return t1, t2


def parallel_axis_cmg_det(delta1, delta2):
    t1, t2 = parallel_axis_cmg_torque_directions(delta1, delta2)
    return float(t1[0] * t2[1] - t1[1] * t2[0])
