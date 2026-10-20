# -*- coding: utf-8 -*-
"""太阳同步轨道的基本计算 (圆轨道、J2 节点进动)."""

import numpy as np

MU = 3.986004418e14          # m^3/s^2
RE = 6378137.0               # m
J2 = 1.08262668e-3
DAY = 86400.0                # s
YEAR_DAYS = 365.2422

# 太阳在惯性系中的平均视运动, 单位: deg/day
SOLAR_RATE_DEG_PER_DAY = 360.0 / YEAR_DAYS


def orbit_radius(altitude_m):
    """圆轨道半径, m."""
    return RE + np.asarray(altitude_m, dtype=float)


def mean_motion(altitude_m):
    """圆轨道平均角速度, rad/s."""
    a = orbit_radius(altitude_m)
    return np.sqrt(MU / a ** 3)


def orbital_period_min(altitude_m):
    """圆轨道周期, min."""
    return 2.0 * np.pi / mean_motion(altitude_m) / 60.0


def nodal_precession_deg_per_day(altitude_m, inclination_deg):
    """J2 升交点赤经进动率, deg/day.

    使用近圆轨道形式:
      dOmega/dt = -(3/2) J2 (Re/a)^2 n cos(i)
    倾角大于 90 度时 cos(i)<0, 进动方向与地球公转方向一致 (向东).
    """
    a = orbit_radius(altitude_m)
    n = mean_motion(altitude_m)
    inc = np.radians(inclination_deg)
    rate_rad_s = -1.5 * J2 * (RE / a) ** 2 * n * np.cos(inc)
    return np.degrees(rate_rad_s) * DAY


def required_inclination_deg(altitude_m):
    """给定高度, 使节点进动率等于太阳平均视运动率的倾角, deg."""
    a = orbit_radius(altitude_m)
    n = mean_motion(altitude_m)
    target = np.radians(SOLAR_RATE_DEG_PER_DAY) / DAY
    cos_i = -target / (1.5 * J2 * (RE / a) ** 2 * n)
    return np.degrees(np.arccos(np.clip(cos_i, -1.0, 1.0)))


def local_time_drift_hours_per_year(altitude_m, inclination_deg):
    """节点地方时相对太阳的漂移量, h/year.

    对太阳同步轨道应接近 0; 对 i=90 度极轨约为 24 h/year.
    """
    drift_deg_per_day = (
        SOLAR_RATE_DEG_PER_DAY
        - nodal_precession_deg_per_day(altitude_m, inclination_deg)
    )
    return drift_deg_per_day / 15.0 * YEAR_DAYS
