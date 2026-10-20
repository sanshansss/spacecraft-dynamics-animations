# -*- coding: utf-8 -*-
"""
姿态动力学教学动画 (Manim Community 0.22).

公式与旋转约定对齐文件夹里的 MATLAB 代码:
  - references/02 Si2So / references/07 欧拉角转动: 分量变换 DCM,
    Lba = R3*R2*R1 (先转的在右边),
    x/y/z_rotation 均为"坐标架正转 +a"的被动矩阵.
  - references/10 重力梯度/test1: Explorer 1 模型 + 脚本化翻转
    (theta: 0->90°, 主轴角速度 50->0°/s, 横向 0->10°/s, 5 s 循环).
  - references/10 重力梯度/test2: 线性化引力梯度力矩 + 欧拉动力学 + 3-2-1 运动学.

渲染:
  manim -ql --disable_caching scenes.py EulerRotation312
  manim -qm --disable_caching scenes.py EnergyDissipationExplorer1
  manim -qm --disable_caching scenes.py GravityGradientSatellite
  manim -qm --disable_caching scenes.py SpinStability
  manim -qm --disable_caching scenes.py SunSynchronousOrbit
"""

import numpy as np
from manim import (Arrow, Arrow3D, Axes, BLUE, BLUE_E, Circle, Cube, Cylinder,
                   DashedLine, Dot, DL, DOWN, DR, Ellipse, FadeIn, FadeOut,
                   GREEN, GRAY, GRAY_B, Group, LEFT, Line, Line3D, ORANGE,
                   ORIGIN, OUT, ParametricFunction, Prism, RED, RIGHT,
                   Rectangle, Scene, Sphere, TEAL, Text, Create, ThreeDScene,
                   Transform, UL, UP, UR, ValueTracker, VGroup, VMobject,
                   WHITE, Write, YELLOW,
                   always_redraw, linear)

from dynamics import (quat_to_matrix, run_gravity_gradient_satellite,
                      run_tumbling)
from gravity_assist import heliocentric_velocity, hyperbolic_flyby
from momentum_control import (cmg_h_vector, cmg_torque,
                              parallel_axis_cmg_det,
                              parallel_axis_cmg_torque_directions,
                              reaction_wheel_spinup)
from relative_motion import (circular_orbit_rate, natural_ellipse_state,
                             propagate_cw, two_impulse_rendezvous)
from sun_sync import (SOLAR_RATE_DEG_PER_DAY, nodal_precession_deg_per_day,
                      orbital_period_min, required_inclination_deg)

CN_FONT = "Microsoft YaHei"


def cn(text, size=30, color=WHITE, weight="NORMAL"):
    return Text(text, font=CN_FONT, font_size=size, color=color, weight=weight)


# ---- 与 MATLAB x/y/z_rotation.m 完全一致的被动 DCM ----
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def active(dcm):
    """被动分量变换矩阵 -> Manim 用的主动旋转矩阵."""
    return np.asarray(dcm).T


def axis_angle_active(dA):
    """主动旋转增量矩阵 -> (角度, 轴)."""
    ang = np.arccos(np.clip((np.trace(dA) - 1) / 2, -1, 1))
    if ang < 1e-9:
        return 0.0, np.array([1.0, 0, 0])
    axis = np.array([dA[2, 1] - dA[1, 2],
                     dA[0, 2] - dA[2, 0],
                     dA[1, 0] - dA[0, 1]]) / (2 * np.sin(ang))
    return ang, axis


def make_triad(A, length=2.2, thickness=0.02, alpha=1.0):
    """按主动旋转矩阵 A 画三轴: x 红, y 绿, z 蓝."""
    arrows = []
    for e, c in zip(np.eye(3), [RED, GREEN, BLUE]):
        arrow = Arrow3D(ORIGIN, A @ (e * length), color=c, thickness=thickness,
                        height=0.32, base_radius=0.08)
        if alpha < 1.0:
            arrow.set_opacity(alpha)
        arrows.append(arrow)
    return Group(*arrows)


def orbit_basis(inc_deg):
    """返回轨道面内的两个正交单位向量 (惯性系, z 为地球自转轴)."""
    inc = np.radians(inc_deg)
    return (np.array([np.cos(inc), 0.0, -np.sin(inc)]),
            np.array([0.0, 1.0, 0.0]))


def orbit_point(inc_deg, theta, radius):
    u, v = orbit_basis(inc_deg)
    return radius * (np.cos(theta) * u + np.sin(theta) * v)


def make_orbit_plane(inc_deg, radius, color, fill_opacity=0.08):
    """画一个倾斜轨道面 (圆盘 + 外圈)."""
    plane = Circle(radius=radius, color=color, stroke_width=2.5)
    plane.set_fill(color, opacity=fill_opacity)
    plane.rotate(np.radians(inc_deg), axis=UP)
    return plane


def plane_line_2d(center, angle, length, color, width=5):
    d = np.array([np.cos(angle), np.sin(angle), 0.0])
    return Line(center - length * d, center + length * d,
                color=color, stroke_width=width)


def sun_arrow_2d(center, angle, length=2.0):
    d = np.array([np.cos(angle), np.sin(angle), 0.0])
    return Arrow(center, center + length * d, buff=0,
                 color=YELLOW, stroke_width=4)


class EulerRotation312(ThreeDScene):
    """第二套姿态角 3-1-2: 先绕 z 偏航 ψ, 再绕新 x 滚转 θ, 最后绕新 y 俯仰 φ."""

    def construct(self):
        self.set_camera_orientation(
            phi=70 * np.pi / 180, theta=-45 * np.pi / 180, zoom=1.3)
        title = cn("第二套姿态角: 3-1-2 旋转顺序", size=30,
                   weight="BOLD").to_corner(UL)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title), run_time=0.8)

        # 固定的参考坐标架 (灰)
        ref = Group(*[
            Arrow3D(ORIGIN, 2.6 * e, color=GRAY, thickness=0.006)
            for e in np.eye(3)
        ])
        labels = Group()
        for name, e, off in [("x", np.eye(3)[0], [0.35, 0.15, 0]),
                             ("y", np.eye(3)[1], [0, 0.35, 0.15]),
                             ("z", np.eye(3)[2], [0.35, 0, 0])]:
            t = Text(name, font="Times New Roman", font_size=30, color=GRAY)
            t.move_to(e * 2.6 + np.array(off))
            labels.add(t)
        self.add(ref)
        self.add_fixed_orientation_mobjects(*labels)

        body = make_triad(np.eye(3))
        self.play(FadeIn(body), run_time=0.6)

        psi, theta, phi = np.radians([50.0, 35.0, 25.0])
        A = np.eye(3)
        steps = [
            ("第一步: 绕 z 轴转 \u03c8 = 50°", np.array([0.0, 0, 1]), psi),
            ("第二步: 绕新 x 轴转 \u03b8 = 35°", np.array([1.0, 0, 0]), theta),
            ("第三步: 绕新 y 轴转 \u03c6 = 25°", np.array([0.0, 1, 0]), phi),
        ]
        for txt, axis_body, ang in steps:
            banner = cn(txt, size=26, color=YELLOW).to_edge(DOWN)
            self.add_fixed_in_frame_mobjects(banner)
            self.play(FadeIn(banner), run_time=0.4)
            axis_world = A @ axis_body
            self.play(body.animate.rotate(ang, axis=axis_world, about_point=ORIGIN),
                      run_time=2.0, rate_func=lambda t: t)
            if np.allclose(axis_body, [0, 0, 1]):
                A = active(rot_z(ang)) @ A
            elif np.allclose(axis_body, [1, 0, 0]):
                A = active(rot_x(ang)) @ A
            else:
                A = active(rot_y(ang)) @ A
            self.play(FadeOut(banner), run_time=0.3)

        note = cn("同样三个角, 若转序不同, 最终姿态不同!", size=28, color=ORANGE)
        note.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(note)
        self.play(FadeIn(note), run_time=0.6)
        self.wait(1.0)

        # 反序 2-1-3 作为对照
        self.play(FadeOut(body), run_time=0.5)
        body2 = make_triad(np.eye(3))
        self.add(body2)
        banner = cn("对照: 2-1-3 反序 (先 \u03c6 绕 y, 再 \u03b8 绕 x, 最后 \u03c8 绕 z)",
                    size=24, color=YELLOW).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(banner)
        self.play(FadeIn(banner), FadeOut(note), run_time=0.5)

        A2 = np.eye(3)
        for axis_body, ang, fun in [
            (np.array([0.0, 1, 0]), phi, rot_y),
            (np.array([1.0, 0, 0]), theta, rot_x),
            (np.array([0.0, 0, 1]), psi, rot_z),
        ]:
            axis_world = A2 @ axis_body
            self.play(body2.animate.rotate(ang, axis=axis_world, about_point=ORIGIN),
                      run_time=1.6, rate_func=lambda t: t)
            A2 = active(fun(ang)) @ A2

        dA = A @ A2.T
        diff = np.degrees(np.arccos(np.clip((np.trace(dA) - 1) / 2, -1, 1)))
        final = cn(f"两序列末态相差约 {diff:.0f}° \u2192 旋转顺序不可交换",
                   size=26, color=ORANGE).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(final)
        self.play(FadeOut(banner), FadeIn(final), run_time=0.6)
        self.wait(2)


def explorer1_model(scale=1.0):
    """Explorer 1 风格: 细长圆柱 + 球头 + 四根鞭天线 (对应 test1/drawTanxianzhe_test)."""
    parts = Group()
    for y0 in [0.5, -0.5]:
        cyl = Cylinder(radius=0.12 * scale, height=1.0 * scale,
                       direction=OUT, color=BLUE_E, resolution=(64, 4))
        cyl.set_opacity(1.0).set_stroke(width=0.4, color=BLUE_E)
        cyl.set_shade_in_3d(True)
        cyl.shift(OUT * y0 * scale)
        parts.add(cyl)
    ball = Sphere(radius=0.10 * scale, color=RED, resolution=(16, 16))
    ball.set_opacity(0.95)
    ball.shift(OUT * 1.10 * scale)
    parts.add(ball)
    # 自旋标记: 圆柱侧面的小色块, 让绕纵轴的自旋可见
    marker = Sphere(radius=0.06 * scale, color=ORANGE, resolution=(12, 12))
    marker.shift(RIGHT * 0.13 * scale + OUT * 0.25 * scale)
    parts.add(marker)
    # 四根鞭天线
    for k in range(4):
        a = k * np.pi / 2
        tip = np.array([0.55 * np.cos(a), 0.55 * np.sin(a), -1.5]) * scale
        parts.add(Line3D(ORIGIN, tip, color=WHITE, thickness=0.006)
                  .shift(OUT * -1.0 * scale))
    return parts


class EnergyDissipationExplorer1(ThreeDScene):
    """能量耗散翻转 (脚本化, 对应 references/10 重力梯度/test1/main.m 的角度脚本).

    test1 的复合: R = Y(theta) * Z(angle_omega) * Z(angle_Omega)  (被动),
    即主动旋转 A = Z(omega+Omega) * Y(theta). theta 0->90° 过程里,
    卫星从绕长轴 (最小惯量) 自旋过渡到绕横轴 (最大惯量) 自旋.
    """

    def construct(self):
        self.set_camera_orientation(
            phi=72 * np.pi / 180, theta=-60 * np.pi / 180, zoom=1.35)
        title = cn("能量耗散: 绕最小惯量轴自旋失稳 (Explorer 1)", size=26,
                   weight="BOLD").to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title), run_time=0.8)

        tb, tf = 1.0, 4.0

        def theta_of(t):
            if t < tb:
                return 0.0
            if t <= tf:
                return 90.0 * (t - tb) / (tf - tb)
            return 90.0

        def int_omega_of(t):
            """∫ω dt, ω: 50 -> 0 deg/s 线性下降 (t=1..4 s)."""
            if t <= tb:
                return 50.0 * t
            if t <= tf:
                return 50.0 + 50.0 * (t - tb) - 50.0 * (t - tb) ** 2 / 6.0
            return 50.0 + 150.0 - 75.0

        def int_big_of(t):
            """∫Ω dt, Ω: 0 -> 10 deg/s 线性上升 (t=1..4 s)."""
            if t <= tb:
                return 0.0
            if t <= tf:
                return 10.0 * (t - tb) ** 2 / 6.0
            return 15.0

        sat = explorer1_model(scale=1.0)
        t_tracker = ValueTracker(0.0)
        prev = [np.eye(3)]

        def upd(m, dt):
            t = t_tracker.get_value()
            th = np.radians(theta_of(t))
            # 0.35 倍视觉降速, 避免自旋把翻转方向转到视线方向
            spin = np.radians((int_omega_of(t) + int_big_of(t)) * 0.35)
            A_now = active(rot_z(spin)) @ active(rot_y(th))
            dA = A_now @ prev[0].T
            ang, ax = axis_angle_active(dA)
            if ang > 1e-9:
                m.rotate(ang, axis=ax, about_point=ORIGIN)
            prev[0] = A_now

        sat.add_updater(upd)
        self.add(sat)

        note1 = cn("0~1s: 绕长轴 (最小惯量轴) 自旋", size=24, color=GREEN).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(note1)
        self.play(t_tracker.animate.set_value(tb), run_time=1.0,
                  rate_func=lambda t: t)

        note2 = cn("结构阻尼耗能 \u2192 1~4s 翻转 90°, 改绕横轴 (最大惯量轴) 自旋",
                   size=24, color=ORANGE).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(note2)
        self.play(FadeOut(note1), FadeIn(note2), run_time=0.4)
        self.play(t_tracker.animate.set_value(tf), run_time=2.6,
                  rate_func=lambda t: t)

        note3 = cn("角动量守恒, 动能降到极小 \u2192 只有绕最大惯量轴自旋稳定",
                   size=24, color=YELLOW).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(note3)
        self.play(FadeOut(note2), FadeIn(note3), run_time=0.4)
        self.play(t_tracker.animate.set_value(6.0), run_time=1.6,
                  rate_func=lambda t: t)
        self.wait(2)


class GravityGradientSatellite(ThreeDScene):
    """引力梯度稳定/不稳定对比 (移植自 references/10 重力梯度/test2/main.m).

    稳定: A=150, B=200, C=100, 初始 3°/-5°/2° \u2192 天平动;
    不稳定: A=200, B=150, C=100 \u2192 姿态发散.
    """

    def construct(self):
        self.set_camera_orientation(
            phi=65 * np.pi / 180, theta=-45 * np.pi / 180, zoom=1.7)
        title = cn("引力梯度姿态稳定: 天平动 vs 发散", size=28,
                   weight="BOLD").to_corner(UL)
        title.set_z_index(50)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title), run_time=0.8)

        # 固定参考: 轨道坐标系 (灰)
        ref = Group(*[
            Arrow3D(ORIGIN, 2.4 * e, color=GRAY, thickness=0.006)
            for e in np.eye(3)
        ])
        self.add(ref)

        cases = [
            ((150.0, 200.0, 100.0), "稳定: A=150, B=200, C=100 \u2192 天平动", GREEN,
             1.5, 8.0),
            ((200.0, 150.0, 100.0), "不稳定: A=200, B=150, C=100 \u2192 发散", RED,
             2.5, 8.0),
        ]
        for I, label, color, n_orbits, t_show in cases:
            ts, angles, Lbos = run_gravity_gradient_satellite(
                I=I, n_orbits=n_orbits, dt=0.2)
            k = np.linspace(0, len(ts) - 1, int(t_show * 24)).astype(int)
            sub_L = Lbos[k]
            sub = angles[k]

            body = make_triad(np.eye(3), length=1.6)
            box = Cube(side_length=1.0, color=BLUE_E).set_opacity(0.9)
            panel1 = Prism(dimensions=[1.9, 0.06, 0.6], color=YELLOW)
            panel1.set_opacity(0.9).shift(RIGHT * 1.45)
            panel2 = Prism(dimensions=[1.9, 0.06, 0.6], color=YELLOW)
            panel2.set_opacity(0.9).shift(LEFT * 1.45)
            grp = Group(body, box, panel1, panel2)
            grp.set_shade_in_3d(True)

            t_tracker = ValueTracker(0.0)
            prev = [np.eye(3)]

            def upd(m, dt, sub_L=sub_L, prev=prev, t_tracker=t_tracker,
                    t_show=t_show):
                kk = min(int(t_tracker.get_value() / t_show * (len(sub_L) - 1)),
                         len(sub_L) - 1)
                A_now = sub_L[kk].T
                dA = A_now @ prev[0].T
                ang, ax = axis_angle_active(dA)
                if ang > 1e-9:
                    m.rotate(ang, axis=ax, about_point=ORIGIN)
                prev[0] = A_now

            grp.add_updater(upd)
            tag = cn(label, size=24, color=color).to_edge(DOWN)
            tag.set_z_index(50)
            self.add_fixed_in_frame_mobjects(tag)

            self.add(grp)
            self.play(FadeIn(tag), run_time=0.4)
            self.play(t_tracker.animate.set_value(t_show), run_time=t_show,
                      rate_func=lambda t: t)
            grp.clear_updaters()
            self.wait(1.0)
            self.play(FadeOut(grp), FadeOut(tag), run_time=0.5)

        conclusion = cn("满足稳定条件时姿态振荡 (天平动); 不满足时发散",
                        size=26, color=YELLOW).to_edge(DOWN)
        conclusion.set_z_index(50)
        self.add_fixed_in_frame_mobjects(conclusion)
        self.play(FadeIn(conclusion), run_time=0.6)
        self.wait(2)


class GravityGradientCurves(Scene):
    """同一天平动模型的两组姿态角曲线 (10 个轨道周期), 与 test2 数值算例对应."""

    def construct(self):
        mu, R_orb = 3.986005e14, 6580e3
        T_orb = 2 * np.pi / np.sqrt(mu / R_orb ** 3)
        title = cn("姿态角随时间变化 (10 个轨道周期)", size=30, weight="BOLD").to_edge(UP)
        self.add(title)

        cases = [
            ((150.0, 200.0, 100.0), [-6, 6, 3],
             "稳定: A=150, B=200, C=100 \u2192 有界振荡 (天平动)", GREEN),
            ((200.0, 150.0, 100.0), [-40, 160, 40],
             "不稳定: A=200, B=150, C=100 \u2192 姿态角发散", RED),
        ]
        for idx, (I, ycfg, cap, cap_color) in enumerate(cases):
            ts, ang, _ = run_gravity_gradient_satellite(I=I, n_orbits=10, dt=0.2)
            t_orb = ts / T_orb
            step = max(1, len(t_orb) // 2500)
            tt = t_orb[::step]

            axes = Axes(
                x_range=[0, 10, 2], y_range=ycfg,
                x_length=4.9, y_length=2.9, tips=False,
                axis_config={"include_numbers": True, "font_size": 20,
                             "stroke_width": 2},
            )
            axes.move_to((LEFT if idx == 0 else RIGHT) * 3.35 + DOWN * 0.35)
            self.play(Create(axes), run_time=0.6)

            legend = Group()
            for k, (name, c) in enumerate(zip(["\u03c6", "\u03b8", "\u03c8"],
                                              [RED, GREEN, BLUE])):
                t = Text(name, font="Times New Roman", font_size=22, color=c)
                t.next_to(axes, UP, buff=0.12)
                t.shift(RIGHT * (0.9 * k - 0.9) + LEFT * 0.4)
                legend.add(t)
            self.add(legend)

            for i, c in enumerate([RED, GREEN, BLUE]):
                vals = np.clip(np.degrees(ang[::step, i]),
                               ycfg[0] * 0.98, ycfg[1] * 0.98)
                graph = axes.plot_line_graph(
                    tt, vals, add_vertex_dots=False,
                    line_color=c)["line_graph"]
                graph.set_stroke(width=1.8)
                self.play(Create(graph), run_time=1.2, rate_func=linear)

            caption = cn(cap, size=20, color=cap_color)
            caption.next_to(axes, DOWN, buff=0.35)
            self.add(caption)

        note = cn("纵轴: 姿态角 (deg)", size=18, color=GRAY).to_corner(DL)
        self.add(note)
        self.wait(2)


class SpinStability(ThreeDScene):
    """刚体绕最大/中间惯量轴自旋 (PPT 7.4.2 数值算例)."""

    def construct(self):
        self.set_camera_orientation(
            phi=68 * np.pi / 180, theta=-45 * np.pi / 180, zoom=2.0)
        I = np.diag([1.0, 0.8, 0.6])

        for case, (w0, label, color, t_end, t_show) in enumerate([
            (np.array([2.3, 0.18, 0.12]), "绕最大惯量轴 (x) 自旋", GREEN, 40.0, 7.0),
            (np.array([0.15, 2.3, 0.15]), "绕中间惯量轴 (y) 自旋", RED, 40.0, 7.0),
        ]):
            ts, ys = run_tumbling(I, w0, t_end=t_end, dt=1e-3)
            k = np.linspace(0, len(ts) - 1, int(t_show * 24)).astype(int)
            sub = ys[k]

            H0 = quat_to_matrix(sub[0, :4]) @ (I @ sub[0, 4:])
            h_dir = H0 / np.linalg.norm(H0)
            h_arrow = Arrow3D(ORIGIN, 2.6 * h_dir, color=YELLOW, thickness=0.02,
                              height=0.3, base_radius=0.07)
            h_label = Text("H", font="Times New Roman", font_size=28, color=YELLOW)
            h_label.move_to(2.8 * h_dir)

            body = make_triad(np.eye(3), length=1.5)
            box = Cube(side_length=1.0).set_opacity(0.12)
            grp = Group(body, box)

            t_tracker = ValueTracker(0.0)
            prev = [np.eye(3)]

            def upd(m, dt, sub=sub, prev=prev, t_tracker=t_tracker,
                    t_show=t_show):
                kk = min(int(t_tracker.get_value() / t_show * (len(sub) - 1)),
                         len(sub) - 1)
                A_now = quat_to_matrix(sub[kk, :4])
                dA = A_now @ prev[0].T
                ang, ax = axis_angle_active(dA)
                if ang > 1e-9:
                    m.rotate(ang, axis=ax, about_point=ORIGIN)
                prev[0] = A_now

            def w_dir():
                kk = min(int(t_tracker.get_value() / t_show * (len(sub) - 1)),
                         len(sub) - 1)
                w_in = quat_to_matrix(sub[kk, :4]) @ sub[kk, 4:]
                n = np.linalg.norm(w_in)
                if n < 1e-9:
                    return np.array([1.0, 0, 0])
                return 1.7 * w_in / n

            w_arrow = always_redraw(
                lambda: Arrow3D(ORIGIN, w_dir(), color=ORANGE, thickness=0.012,
                                height=0.2, base_radius=0.05))

            grp.add_updater(upd)
            intro = cn(label, size=26, weight="BOLD").to_edge(DOWN)
            self.add_fixed_in_frame_mobjects(intro)
            self.add(grp, h_arrow, w_arrow)
            self.add_fixed_orientation_mobjects(h_label)
            self.play(FadeIn(intro), run_time=0.4)
            self.play(t_tracker.animate.set_value(t_show), run_time=t_show,
                      rate_func=lambda t: t)
            grp.clear_updaters()
            verdict = cn("角速度有界 \u2192 稳定" if case == 0 else
                         "角速度翻转 \u2192 不稳定 (网球拍效应)",
                         size=24, color=color).to_edge(DOWN)
            self.add_fixed_in_frame_mobjects(verdict)
            self.play(FadeOut(intro), FadeIn(verdict), run_time=0.5)
            self.wait(1.0)
            self.play(FadeOut(grp), FadeOut(h_arrow), FadeOut(h_label),
                      FadeOut(w_arrow), FadeOut(intro), FadeOut(verdict),
                      run_time=0.5)

        conclusion = cn("刚体: 绕最大/最小惯量轴自旋稳定, 绕中间惯量轴自旋不稳定",
                        size=24, color=YELLOW).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(conclusion)
        self.play(FadeIn(conclusion), run_time=0.6)
        self.wait(1.5)


class SunSynchronousOrbit(ThreeDScene):
    """太阳同步轨道: J2 节点进动与地球公转同步."""

    def switch_title(self, old_title, text, size=30):
        new_title = cn(text, size=size, weight="BOLD").to_corner(UL)
        self.add_fixed_in_frame_mobjects(new_title)
        self.play(FadeOut(old_title), FadeIn(new_title), run_time=0.5)
        return new_title

    def construct(self):
        self.set_camera_orientation(
            phi=72 * np.pi / 180, theta=-55 * np.pi / 180, zoom=1.0)

        inc700 = required_inclination_deg(700e3)
        rate700 = nodal_precession_deg_per_day(700e3, inc700)
        period700 = orbital_period_min(700e3)

        title = cn("太阳同步轨道：关键不是极轨", size=30,
                   weight="BOLD").to_corner(UL)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title), run_time=0.8)

        # Act 1: 普通极轨的轨道面在惯性空间固定, 地方时随太阳漂移.
        earth = Sphere(radius=1.0, resolution=(32, 32),
                       color=BLUE_E).set_opacity(0.9)
        equator = Circle(radius=1.02, color=GRAY_B, stroke_width=2)
        equator.set_stroke(opacity=0.5)
        axis = Arrow3D(ORIGIN, 1.9 * UP, color=GRAY_B, thickness=0.012,
                       height=0.22, base_radius=0.05)
        self.play(FadeIn(earth), FadeIn(equator), FadeIn(axis), run_time=1.0)

        polar = make_orbit_plane(90.0, 2.4, GRAY_B, fill_opacity=0.035)
        self.play(Create(polar), run_time=1.0)

        theta = ValueTracker(0.0)
        sat = Sphere(radius=0.075, resolution=(10, 10), color=WHITE)
        sat.add_updater(
            lambda m: m.move_to(orbit_point(90.0, theta.get_value(), 2.4)))
        self.add(sat)
        self.play(theta.animate.set_value(2 * np.pi), run_time=3.2,
                  rate_func=linear)
        sat.clear_updaters()

        legend = cn("灰色：i=90° 极轨    黄色：太阳方向",
                    size=19, color=GRAY_B).to_corner(DL).shift(UP * 0.8)
        clock = cn("地方时 10:30", size=22,
                   color=YELLOW).to_corner(DR).shift(UP * 0.5)
        note = cn("极轨的轨道面在惯性空间固定", size=22,
                  color=RED).to_edge(DOWN)
        for m in (legend, clock, note):
            self.add_fixed_in_frame_mobjects(m)
        self.play(FadeIn(legend), FadeIn(clock), FadeIn(note), run_time=0.6)

        sun = Arrow3D(ORIGIN, 4.2 * RIGHT, color=YELLOW, thickness=0.018,
                      height=0.28, base_radius=0.06)
        self.play(FadeIn(sun), run_time=0.4)
        self.play(
            sun.animate.rotate(np.pi / 2, axis=OUT, about_point=ORIGIN),
            run_time=2.0,
        )
        self.play(FadeOut(clock), FadeOut(note), run_time=0.4)
        clock = cn("地方时 16:30", size=22,
                   color=YELLOW).to_corner(DR).shift(UP * 0.5)
        note = cn("三个月后：极轨地方时漂移约 6 h", size=22,
                  color=RED).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(clock, note)
        self.play(FadeIn(clock), FadeIn(note), run_time=0.5)
        self.wait(1.0)

        # Act 2: J2 让逆行轨道面向东进动.
        title = self.switch_title(
            title, "J2 进动：让轨道面跟着太阳转")
        self.play(
            FadeOut(polar), FadeOut(sat), FadeOut(sun), FadeOut(clock),
            FadeOut(note), FadeOut(legend),
            run_time=0.6,
        )

        bulge = Circle(radius=1.12, color=ORANGE, stroke_width=8)
        bulge.set_stroke(opacity=0.55)
        bulge_label = cn("地球赤道隆起（J2）", size=20,
                         color=ORANGE).to_corner(DL).shift(UP * 0.8)
        self.add_fixed_in_frame_mobjects(bulge_label)
        self.play(FadeIn(bulge), FadeIn(bulge_label), run_time=0.8)

        sso = make_orbit_plane(inc700, 2.35, TEAL, fill_opacity=0.08)
        sso_label = cn("SSO: i≈98.19°（方向示意）", size=22,
                       color=TEAL).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(sso_label)
        self.play(Create(sso), FadeIn(sso_label), run_time=1.0)

        formula = VGroup(
            cn("dΩ/dt = -(3/2) J2 (R/a)^2 n cos i", size=20),
            cn("i > 90° → cos i < 0", size=20, color=YELLOW),
            cn("→ 轨道面向东进动", size=20, color=GREEN),
            cn(f"目标: {SOLAR_RATE_DEG_PER_DAY:.4f}°/天",
               size=20, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        formula.to_corner(UR).shift(DOWN * 0.8 + LEFT * 0.2)
        self.add_fixed_in_frame_mobjects(formula)
        self.play(FadeIn(formula), run_time=0.8)
        self.play(
            sso.animate.rotate(np.radians(55), axis=OUT, about_point=ORIGIN),
            run_time=4.0, rate_func=linear,
        )
        self.wait(1.0)

        # Act 2b: 不同高度的太阳同步倾角.
        title = self.switch_title(
            title, "不同高度需要多大的倾角？")
        self.play(
            FadeOut(bulge), FadeOut(bulge_label), FadeOut(sso),
            FadeOut(sso_label), FadeOut(formula), FadeOut(earth),
            FadeOut(equator), FadeOut(axis),
            run_time=0.6,
        )

        # 手动绘制坐标轴, 确保中文环境下刻度和右端 900 标签完整显示.
        plot_left, plot_right = -3.4, 3.5
        plot_bottom, plot_top = -1.70, 1.25
        h_lo, h_hi = 400.0, 900.0
        i_lo, i_hi = 96.5, 99.2

        def h_to_x(h):
            return plot_left + (h - h_lo) / (h_hi - h_lo) * (
                plot_right - plot_left)

        def inc_to_y(inc):
            return plot_bottom + (inc - i_lo) / (i_hi - i_lo) * (
                plot_top - plot_bottom)

        x_axis = Line(
            [plot_left, plot_bottom, 0], [plot_right, plot_bottom, 0],
            color=WHITE, stroke_width=2,
        )
        y_axis = Line(
            [plot_left, plot_bottom, 0], [plot_left, plot_top, 0],
            color=WHITE, stroke_width=2,
        )
        xticks = VGroup()
        for h in range(400, 901, 100):
            xpos = h_to_x(h)
            xticks.add(
                Line([xpos, plot_bottom - 0.06, 0],
                     [xpos, plot_bottom + 0.06, 0],
                     color=WHITE, stroke_width=1.5),
                cn(str(h), size=18).move_to(
                    [xpos, plot_bottom - 0.24, 0]),
            )
        yticks = VGroup()
        for inc in range(97, 100):
            ypos = inc_to_y(inc)
            yticks.add(
                Line([plot_left - 0.06, ypos, 0],
                     [plot_left + 0.06, ypos, 0],
                     color=WHITE, stroke_width=1.5),
                cn(str(inc), size=18).next_to(
                    [plot_left, ypos, 0], LEFT, buff=0.12),
            )

        xlab = cn("高度 (km)", size=20).move_to(
            [(plot_left + plot_right) / 2, plot_bottom - 0.60, 0])
        ylab = cn("所需倾角 (°)", size=20).rotate(np.pi / 2)
        ylab.move_to([plot_left - 0.72, (plot_bottom + plot_top) / 2, 0])

        graph = ParametricFunction(
            lambda t: np.array([
                h_to_x(h_lo + (h_hi - h_lo) * t),
                inc_to_y(required_inclination_deg(
                    (h_lo + (h_hi - h_lo) * t) * 1000.0)),
                0.0,
            ]),
            t_range=[0, 1], color=TEAL, stroke_width=4,
        )
        dot700 = Dot(
            [h_to_x(700), inc_to_y(inc700), 0],
            radius=0.09, color=YELLOW)
        vline = DashedLine(
            [h_to_x(700), plot_bottom, 0],
            [h_to_x(700), inc_to_y(inc700), 0],
            color=YELLOW, stroke_width=2, dash_length=0.08,
        ).set_stroke(opacity=0.65)
        mark_label = cn("700 km → i=98.19°", size=22,
                        color=YELLOW).next_to(dot700, DR, buff=0.18)
        curve_note = cn("低轨太阳同步轨道通常是 97°–99° 的逆行轨道",
                        size=20, color=GRAY_B).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(
            x_axis, y_axis, xticks, yticks, xlab, ylab,
            graph, dot700, vline, mark_label, curve_note)
        self.play(
            Create(x_axis), Create(y_axis), FadeIn(xticks), FadeIn(yticks),
            FadeIn(xlab), FadeIn(ylab), run_time=1.0)
        self.play(Create(graph), run_time=2.0)
        self.play(FadeIn(vline), FadeIn(dot700), Write(mark_label),
                  run_time=0.8)
        self.play(FadeIn(curve_note), run_time=0.5)
        self.wait(2.0)

        # Act 3: 一年时间尺度下, SSO 与普通极轨的对比.
        title = self.switch_title(
            title, "一年延时：SSO vs 普通极轨")
        self.play(
            FadeOut(x_axis), FadeOut(y_axis), FadeOut(xticks),
            FadeOut(yticks), FadeOut(xlab), FadeOut(ylab), FadeOut(graph),
            FadeOut(dot700), FadeOut(vline), FadeOut(mark_label),
            FadeOut(curve_note),
            run_time=0.6,
        )

        lc = LEFT * 3.35 + DOWN * 0.25
        rc = RIGHT * 3.35 + DOWN * 0.25
        panel_group = VGroup()
        for center, label in [
            (lc, "惯性系（一年延时）"),
            (rc, "随太阳方向看"),
        ]:
            bg = Rectangle(
                width=6.15, height=5.3, stroke_color=GRAY_B,
                stroke_width=2, fill_color=BLUE_E, fill_opacity=0.04,
            ).move_to(center)
            ptitle = cn(label, size=21).next_to(
                bg.get_top(), DOWN, buff=0.2)
            earth2 = Circle(
                radius=0.72, color=BLUE_E, stroke_width=2,
                fill_color=BLUE_E, fill_opacity=0.35,
            ).move_to(center)
            panel_group.add(bg, ptitle, earth2)
        self.add_fixed_in_frame_mobjects(panel_group)

        legend2 = VGroup(
            cn("青色：SSO (i≈98.19°)", size=18, color=TEAL),
            cn("灰色：极轨 (i=90°)", size=18, color=GRAY_B),
            cn("黄色：太阳方向", size=18, color=YELLOW),
        ).arrange(RIGHT, buff=0.55).move_to(UP * 2.65)
        left_sub = cn("SSO 地方时：10:30（全年不变）",
                      size=17, color=TEAL).move_to(lc + DOWN * 2.20)
        right_sub = cn("极轨：一年累计漂移 24 h",
                       size=17, color=GRAY_B).move_to(rc + DOWN * 2.20)
        self.add_fixed_in_frame_mobjects(legend2, left_sub, right_sub)
        self.play(FadeIn(legend2), FadeIn(left_sub), FadeIn(right_sub),
                  run_time=0.6)

        year = ValueTracker(0.0)
        offset = np.radians(30.0)
        left_sun = always_redraw(
            lambda: sun_arrow_2d(lc, 2 * np.pi * year.get_value(), 1.75))
        left_sso = always_redraw(
            lambda: plane_line_2d(
                lc, 2 * np.pi * year.get_value() + offset, 1.85, TEAL, 6))
        left_polar = plane_line_2d(lc, 0.0, 1.85, GRAY_B, 5)
        right_sun = sun_arrow_2d(rc, 0.0, 1.75)
        right_sso = plane_line_2d(rc, offset, 1.85, TEAL, 6)
        right_polar = always_redraw(
            lambda: plane_line_2d(
                rc, -2 * np.pi * year.get_value(), 1.85, GRAY_B, 5))
        for m in (left_sun, left_sso, left_polar,
                  right_sun, right_sso, right_polar):
            self.add_fixed_in_frame_mobjects(m)

        self.play(year.animate.set_value(1.0), run_time=9.0,
                  rate_func=linear)
        conclusion = cn(
            "SSO 的轨道面跟着太阳转；普通极轨相对太阳漂移",
            size=21, color=YELLOW).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(conclusion)
        self.play(FadeIn(conclusion), run_time=0.6)
        self.wait(1.5)

        # Act 4: 工程意义和边界.
        title = self.switch_title(
            title, "太阳同步轨道的工程意义")
        self.play(
            FadeOut(panel_group), FadeOut(legend2), FadeOut(left_sub),
            FadeOut(right_sub), FadeOut(left_sun), FadeOut(left_sso),
            FadeOut(left_polar), FadeOut(right_sun), FadeOut(right_sso),
            FadeOut(right_polar), FadeOut(conclusion),
            run_time=0.8,
        )

        earth4_center = LEFT * 3.55 + DOWN * 0.15
        earth4 = Circle(
            radius=1.15, color=BLUE_E, stroke_width=2,
            fill_color=BLUE_E, fill_opacity=0.28,
        ).move_to(earth4_center)
        track = ParametricFunction(
            lambda t: earth4_center + np.array(
                [0.55 * np.sin(2.5 * t), -1.0 + 2.0 * t / (2 * np.pi), 0.0]),
            t_range=[0, 2 * np.pi], color=YELLOW, stroke_width=3,
        )
        track_label = cn("星下点轨迹仍移动", size=18, color=YELLOW)
        track_label.next_to(earth4, DOWN, buff=0.2)

        bullets = VGroup(
            cn("✓ 同一地方时过境", size=22, color=GREEN),
            cn("→ 光照稳定，适合遥感/光学观测", size=21, color=GREEN),
            cn("× 仍会进入地影", size=21, color=RED),
            cn("× 星下点仍移动，不是静止轨道", size=21, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.24)
        bullets.move_to(RIGHT * 3.2 + DOWN * 0.2)

        params = cn(
            f"700 km | i={inc700:.2f}° | T≈{period700:.1f} min | "
            f"dΩ/dt≈{rate700:.4f}°/天",
            size=18, color=YELLOW).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(
            earth4, track, track_label, bullets, params)
        self.play(FadeIn(earth4), Create(track), FadeIn(track_label),
                  run_time=1.2)
        self.play(FadeIn(bullets), run_time=1.0)
        self.play(FadeIn(params), run_time=0.5)
        self.wait(2.5)


class CWRelativeRendezvous(Scene):
    """CW/Hill 相对运动、自然绕飞和 V-bar 双脉冲交会."""

    def switch_title(self, old_title, text, size=30):
        new_title = cn(text, size=size, weight="BOLD").to_corner(UL)
        self.play(FadeOut(old_title), FadeIn(new_title), run_time=0.5)
        return new_title

    def construct(self):
        mu = 3.986004418e14
        orbit_radius = 6378137.0 + 500e3
        n = circular_orbit_rate(mu, orbit_radius)
        period = 2.0 * np.pi / n

        title = cn("CW 相对运动：不点火也能绕飞", size=30,
                   weight="BOLD").to_corner(UL)
        self.play(Write(title), run_time=0.8)

        # 第一幕: LVLH 相对坐标系中的自然闭合椭圆.
        center = LEFT * 2.7 + DOWN * 0.55
        y_axis = Arrow(center, center + RIGHT * 3.7,
                       buff=0, color=GRAY_B, stroke_width=3)
        x_axis = Arrow(center, center + UP * 2.4,
                       buff=0, color=GRAY_B, stroke_width=3)
        axis_labels = VGroup(
            cn("沿轨 y", size=20, color=GRAY_B).next_to(
                y_axis.get_end(), RIGHT, buff=0.10),
            cn("径向 x", size=20, color=GRAY_B).next_to(
                x_axis.get_end(), UP, buff=0.10),
        )
        target = VGroup(
            Circle(radius=0.12, color=YELLOW, stroke_width=3),
            Dot(radius=0.045, color=YELLOW),
        ).move_to(center)
        target_label = cn("目标", size=19, color=YELLOW)
        target_label.next_to(target, DOWN, buff=0.12)

        ellipse = ParametricFunction(
            lambda t: center + RIGHT * (-1.75 * np.sin(t))
            + UP * (0.875 * np.cos(t)),
            t_range=[0, 2 * np.pi], color=TEAL, stroke_width=4,
        )
        phase = ValueTracker(0.0)
        chaser = always_redraw(
            lambda: Dot(
                center + RIGHT * (-1.75 * np.sin(phase.get_value()))
                + UP * (0.875 * np.cos(phase.get_value())),
                radius=0.075, color=WHITE,
            )
        )
        info = VGroup(
            cn("追踪器不点火", size=22, color=GRAY_B),
            cn("初始相对速度匹配", size=22, color=TEAL),
            cn("一个轨道周期后回到原处", size=22, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.24)
        info.move_to(RIGHT * 3.25 + UP * 0.95)

        self.play(
            Create(y_axis), Create(x_axis), FadeIn(axis_labels),
            FadeIn(target), FadeIn(target_label), run_time=0.9)
        self.play(Create(ellipse), FadeIn(chaser), FadeIn(info),
                  run_time=0.9)
        self.play(phase.animate.set_value(2 * np.pi), run_time=8.0,
                  rate_func=linear)

        # 第二幕: 方程与闭合条件.
        title = self.switch_title(title, "闭合绕飞从哪里来？")
        self.play(FadeOut(info), FadeOut(axis_labels), run_time=0.4)

        eq = VGroup(
            cn("无推力 CW 方程", size=22, color=YELLOW),
            cn("x'' - 2n y' - 3n² x = a_x", size=21),
            cn("y'' + 2n x' = a_y", size=21),
            cn("z'' + n² z = a_z", size=21),
            cn("x 径向 / y 沿轨 / z 法向", size=18, color=GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        eq.move_to(RIGHT * 3.15 + UP * 0.55)

        condition = cn("闭合条件:  v_y0 = -2 n x_0", size=24,
                       color=TEAL).to_edge(DOWN).shift(UP * 0.55)
        condition_note = cn("沿轨速度少一点，轨迹就不再闭合", size=20,
                            color=ORANGE).to_edge(DOWN)
        self.play(FadeIn(eq), run_time=0.8)
        self.play(FadeIn(condition), FadeIn(condition_note), run_time=0.6)
        self.wait(2.0)

        # 第三幕: V-bar 双脉冲交会.
        title = self.switch_title(title, "V-bar 双脉冲交会")
        self.play(
            FadeOut(eq), FadeOut(condition), FadeOut(condition_note),
            FadeOut(y_axis), FadeOut(x_axis), FadeOut(target),
            FadeOut(target_label), FadeOut(ellipse), FadeOut(chaser),
            run_time=0.6,
        )

        org = RIGHT * 2.1 + DOWN * 0.30
        y_ax = Arrow(org + LEFT * 5.4, org + RIGHT * 1.0,
                     buff=0, color=GRAY_B, stroke_width=3)
        x_ax = Arrow(org, org + UP * 2.0,
                     buff=0, color=GRAY_B, stroke_width=3)
        axes3 = VGroup(
            y_ax, x_ax,
            cn("沿轨 y", size=19, color=GRAY_B).next_to(
                y_ax.get_end(), RIGHT, buff=0.10),
            cn("径向 x", size=19, color=GRAY_B).next_to(
                x_ax.get_end(), RIGHT, buff=0.08),
        )

        target3 = VGroup(
            Circle(radius=0.11, color=YELLOW, stroke_width=3),
            Dot(radius=0.04, color=YELLOW),
        ).move_to(org)
        target3_label = cn("目标", size=18, color=YELLOW).next_to(
            target3, UR, buff=0.10)
        self.play(Create(axes3), FadeIn(target3), FadeIn(target3_label),
                  run_time=0.8)

        r0 = np.array([0.0, -2000.0, 0.0])
        tof = 0.5 * period
        result = two_impulse_rendezvous(r0, n, tof)
        ts = np.linspace(0.0, tof, 401)
        positions = propagate_cw(result["state0"], n, ts)

        def cw_screen(pos):
            return org + RIGHT * (pos[1] / 1000.0 * 2.25) \
                + UP * (pos[0] / 1000.0 * 3.20)

        pts = [cw_screen(p) for p in positions]
        transfer = VMobject(color=TEAL, stroke_width=4)
        transfer.set_points_smoothly(pts)
        start = Dot(pts[0], radius=0.085, color=WHITE)
        start_label = cn("起始：沿轨后方 2 km", size=18,
                         color=GRAY_B).next_to(start, LEFT, buff=0.14)

        progress = ValueTracker(0.0)

        def sample_position():
            s = np.clip(progress.get_value(), 0, 1) * (len(pts) - 1)
            i0 = int(np.floor(s))
            i1 = min(i0 + 1, len(pts) - 1)
            a = s - i0
            return pts[i0] * (1 - a) + pts[i1] * a

        mover = always_redraw(
            lambda: Dot(sample_position(), radius=0.08, color=WHITE))

        self.play(Create(transfer), FadeIn(start), FadeIn(start_label),
                  FadeIn(mover), run_time=1.0)

        # 第一次点火: 主要是径向加速.
        dv1 = result["dv1"]
        burn1 = Arrow(
            pts[0], pts[0] + UP * (dv1[0] / 0.55 * 0.55),
            buff=0, color=YELLOW, stroke_width=5,
        )
        burn1_label = cn("Δv₁", size=20, color=YELLOW).next_to(
            burn1, LEFT, buff=0.10)
        self.play(FadeIn(burn1), Write(burn1_label), run_time=0.5)
        self.play(progress.animate.set_value(1.0), run_time=7.0,
                  rate_func=linear)

        # 到达时仍有相对速度, 第二次点火把它抵消.
        vf = result["vf_before_second_burn"]
        v_before = Arrow(
            org, org + RIGHT * (vf[1] / 2.3 * 0.60)
            + UP * (vf[0] / 0.56 * 0.18),
            buff=0, color=ORANGE, stroke_width=4,
        )
        v_label = cn("剩余相对速度", size=18, color=ORANGE).next_to(
            v_before, UR, buff=0.08)
        dv2 = result["dv2"]
        burn2 = Arrow(
            org, org + RIGHT * (dv2[1] / 2.3 * 0.60)
            + UP * (dv2[0] / 0.56 * 0.18),
            buff=0, color=YELLOW, stroke_width=5,
        )
        burn2_label = cn("Δv₂ 制动", size=19, color=YELLOW).next_to(
            burn2, DOWN, buff=0.10)
        self.play(FadeIn(v_before), FadeIn(v_label), run_time=0.4)
        self.play(FadeIn(burn2), Write(burn2_label), run_time=0.5)
        self.play(FadeOut(v_before), FadeOut(v_label), run_time=0.4)

        metrics = cn(
            f"转移时间 {tof / 60:.1f} min  |  "
            f"Δv₁={np.linalg.norm(result['dv1']):.2f}  "
            f"Δv₂={np.linalg.norm(result['dv2']):.2f}  "
            f"总 Δv={result['dv_total']:.2f} m/s",
            size=20, color=TEAL).to_edge(DOWN).shift(UP * 0.45)
        self.play(FadeIn(metrics), run_time=0.5)
        self.wait(1.5)

        # 第四幕: 交会判据和工程边界.
        title = self.switch_title(title, "交会：位置和速度要同时匹配")
        self.play(
            FadeOut(axes3), FadeOut(target3), FadeOut(target3_label),
            FadeOut(transfer), FadeOut(start), FadeOut(start_label),
            FadeOut(mover), FadeOut(burn1), FadeOut(burn1_label),
            FadeOut(burn2), FadeOut(burn2_label), FadeOut(metrics),
            run_time=0.7,
        )

        summary = VGroup(
            cn("只把位置画到原点 → 可能高速掠过", size=23, color=RED),
            cn("位置 + 相对速度同时归零 → 才叫交会", size=23, color=GREEN),
            cn("双脉冲：一次转移，一次制动", size=22, color=YELLOW),
            cn("燃料、时间、安全走廊需要折中", size=22, color=GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.30)
        summary.move_to(UP * 0.15)
        note = cn(
            "CW 是线性化模型：适合近距离、短时间相对运动；"
            "远距离或大偏心率需用完整轨道动力学。",
            size=19, color=GRAY_B).to_edge(DOWN).shift(UP * 0.35)
        self.play(FadeIn(summary), run_time=0.8)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(2.5)


class GravityAssistSlingshot(Scene):
    """双曲线飞越与引力弹弓的二维教学动画."""

    def switch_title(self, old_title, text, size=30):
        new_title = cn(text, size=size, weight="BOLD").to_corner(UL)
        self.play(FadeOut(old_title), FadeIn(new_title), run_time=0.5)
        return new_title

    def construct(self):
        mu_e = 3.986004418e14
        r_earth = 6.378137e6
        v_inf = 5.0e3
        r_peri = 7.0e6
        planet_v = 30.0e3

        title = cn("引力弹弓：相对行星速率不变", size=30,
                   weight="BOLD").to_corner(UL)
        self.play(Write(title), run_time=0.8)

        ga = hyperbolic_flyby(
            mu_e, v_inf, r_peri, np.array([-1.0, 0.0]),
            extent=0.72, samples=501)
        center = LEFT * 2.55 + DOWN * 0.10
        screen_scale = 0.82 / r_earth
        pts = [center + np.array([
            p[0] * screen_scale, p[1] * screen_scale, 0.0])
            for p in ga["positions"]]

        planet = Circle(
            radius=0.80, color=BLUE_E, stroke_width=3,
            fill_color=BLUE_E, fill_opacity=0.78,
        ).move_to(center)
        planet_band = Circle(
            radius=0.80, color=TEAL, stroke_width=2,
        ).move_to(center).stretch(0.35, 1)
        planet_label = cn("行星", size=19, color=BLUE).next_to(
            planet, DOWN, buff=0.12)
        path = VMobject(color=WHITE, stroke_width=3)
        path.set_points_smoothly(pts)
        progress = ValueTracker(0.0)

        def point_at(u):
            s = np.clip(u, 0, 1) * (len(pts) - 1)
            i0 = int(np.floor(s))
            i1 = min(i0 + 1, len(pts) - 1)
            a = s - i0
            return pts[i0] * (1 - a) + pts[i1] * a

        craft = always_redraw(
            lambda: Dot(point_at(progress.get_value()),
                        radius=0.075, color=YELLOW))
        vhat_in = ga["vin_inf"] / np.linalg.norm(ga["vin_inf"])
        vhat_out = ga["vout_inf"] / np.linalg.norm(ga["vout_inf"])
        vin_arrow = Arrow(
            pts[0], pts[0] + 0.70 * np.array([vhat_in[0], vhat_in[1], 0.0]),
            buff=0, color=YELLOW, stroke_width=4,
        )
        vout_arrow = Arrow(
            pts[-1], pts[-1] + 0.70 * np.array([vhat_out[0], vhat_out[1], 0.0]),
            buff=0, color=YELLOW, stroke_width=4,
        )
        info = VGroup(
            cn("行星参考系", size=23, color=YELLOW),
            cn("|v∞| 不变", size=24, color=GREEN),
            cn(f"方向改变 {np.degrees(ga['turn_angle']):.1f}°",
               size=23, color=TEAL),
            cn(f"e = {ga['eccentricity']:.2f}", size=19, color=GRAY_B),
            cn(f"r_p = {r_peri / 1e3:.0f} km", size=19, color=GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        info.move_to(RIGHT * 3.0 + UP * 0.60)

        self.play(FadeIn(planet), FadeIn(planet_band),
                  FadeIn(planet_label), run_time=0.7)
        self.play(Create(path), FadeIn(craft), FadeIn(info),
                  run_time=0.8)
        self.play(FadeIn(vin_arrow), FadeIn(vout_arrow), run_time=0.4)
        self.play(progress.animate.set_value(1.0), run_time=8.0,
                  rate_func=linear)
        self.wait(1.0)

        # 第二幕: 把行星速度加到相对速度上.
        title = self.switch_title(title, "换到太阳系：速度矢量才改变")
        self.play(
            FadeOut(planet), FadeOut(planet_band), FadeOut(planet_label),
            FadeOut(path), FadeOut(craft), FadeOut(info),
            FadeOut(vin_arrow), FadeOut(vout_arrow),
            run_time=0.6,
        )

        vp = np.array([planet_v, 0.0])
        vin_h = heliocentric_velocity(vp, v_inf, vhat_in)
        vout_h = heliocentric_velocity(vp, v_inf, vhat_out)
        u = 0.082  # 屏幕单位 / (km/s)
        o1 = LEFT * 4.10 + UP * 0.15
        o2 = RIGHT * 1.55 + UP * 0.15

        def vec_arrow(origin, vec_km_s, color, width=5):
            return Arrow(
                origin, origin + RIGHT * vec_km_s[0] * u
                + UP * vec_km_s[1] * u,
                buff=0, color=color, stroke_width=width,
            )

        vp1 = vec_arrow(o1, [planet_v / 1e3, 0], YELLOW, 5)
        vinf1 = Arrow(
            o1 + RIGHT * planet_v / 1e3 * u,
            o1 + RIGHT * (planet_v - v_inf) / 1e3 * u,
            buff=0, color=TEAL, stroke_width=4,
        )
        vh1 = vec_arrow(o1, vin_h / 1e3, ORANGE, 7)

        vp2 = vec_arrow(o2, [planet_v / 1e3, 0], YELLOW, 5)
        tail2 = o2 + RIGHT * planet_v / 1e3 * u
        vinf2 = Arrow(
            tail2,
            tail2 + RIGHT * vhat_out[0] * v_inf / 1e3 * u
            + UP * vhat_out[1] * v_inf / 1e3 * u,
            buff=0, color=TEAL, stroke_width=4,
        )
        vh2 = vec_arrow(o2, vout_h / 1e3, ORANGE, 7)

        h1 = cn("入射：V_p + v∞", size=21, color=GRAY_B)
        h1.move_to(o1 + RIGHT * 1.10 + UP * 1.65)
        h2 = cn("出射：V_p + v∞′", size=21, color=GRAY_B)
        h2.move_to(o2 + RIGHT * 1.10 + UP * 1.65)
        s1 = cn(f"V_in = {np.linalg.norm(vin_h) / 1e3:.1f} km/s",
                size=20, color=ORANGE).next_to(vh1, DOWN, buff=0.30)
        s2 = cn(f"V_out = {np.linalg.norm(vout_h) / 1e3:.1f} km/s",
                size=20, color=ORANGE).next_to(vh2, DOWN, buff=0.30)
        vp1_label = cn("V_p = 30.0 km/s", size=18, color=YELLOW)
        vp1_label.next_to(vp1, UP, buff=0.08)
        vp2_label = cn("V_p = 30.0 km/s", size=18, color=YELLOW)
        vp2_label.next_to(vp2, UP, buff=0.08)
        vinf1_label = cn("v∞ = 5.0", size=18, color=TEAL)
        vinf1_label.next_to(vinf1, DOWN, buff=0.08)
        vinf2_label = cn("v∞ = 5.0", size=18, color=TEAL)
        vinf2_label.next_to(vinf2, RIGHT, buff=0.08)
        delta = cn(
            f"速度增量 ≈ {(np.linalg.norm(vout_h) - np.linalg.norm(vin_h)) / 1e3:+.1f} km/s",
            size=22, color=YELLOW).to_edge(DOWN).shift(UP * 0.55)

        self.play(FadeIn(h1), FadeIn(vp1), FadeIn(vinf1), FadeIn(vh1),
                  FadeIn(s1), FadeIn(vp1_label), FadeIn(vinf1_label),
                  run_time=0.9)
        self.play(FadeIn(h2), FadeIn(vp2), FadeIn(vinf2), FadeIn(vh2),
                  FadeIn(s2), FadeIn(vp2_label), FadeIn(vinf2_label),
                  run_time=0.9)
        self.play(FadeIn(delta), run_time=0.5)
        self.wait(2.0)

        # 第三幕: 同一套几何也能减速.
        title = self.switch_title(title, "弹弓不是只加速：也能减速")
        self.play(
            FadeOut(vp1), FadeOut(vinf1), FadeOut(vh1),
            FadeOut(vp2), FadeOut(vinf2), FadeOut(vh2),
            FadeOut(h1), FadeOut(h2), FadeOut(s1), FadeOut(s2),
            FadeOut(vp1_label), FadeOut(vp2_label),
            FadeOut(vinf1_label), FadeOut(vinf2_label), FadeOut(delta),
            run_time=0.6,
        )

        ga_dec = hyperbolic_flyby(
            mu_e, v_inf, r_peri, np.array([1.0, 0.0]),
            extent=0.72, samples=501)
        vin_dec = heliocentric_velocity(
            vp, v_inf, ga_dec["vin_inf"] / np.linalg.norm(ga_dec["vin_inf"]))
        vout_dec = heliocentric_velocity(
            vp, v_inf, ga_dec["vout_inf"] / np.linalg.norm(ga_dec["vout_inf"]))

        bar_left = LEFT * 4.8
        bar_scale = 0.078
        rows = [
            ("相对速度逆着公转 → 加速",
             np.linalg.norm(vin_h) / 1e3,
             np.linalg.norm(vout_h) / 1e3, GREEN),
            ("相对速度顺着公转 → 减速",
             np.linalg.norm(vin_dec) / 1e3,
             np.linalg.norm(vout_dec) / 1e3, RED),
        ]
        bars = Group()
        for k, (label, v0, v1, color) in enumerate(rows):
            y = UP * (1.55 - 2.15 * k)
            name = cn(label, size=22, color=color)
            name.move_to(LEFT * 1.55 + y + UP * 0.40)
            b0 = Rectangle(
                width=v0 * bar_scale, height=0.27,
                fill_color=GRAY_B, fill_opacity=0.50, stroke_width=0,
            ).move_to(bar_left + RIGHT * v0 * bar_scale / 2 + y)
            b1 = Rectangle(
                width=v1 * bar_scale, height=0.27,
                fill_color=color, fill_opacity=0.80, stroke_width=0,
            ).move_to(bar_left + RIGHT * v1 * bar_scale / 2 + y + DOWN * 0.42)
            n0 = cn(f"{v0:.1f}", size=18, color=GRAY_B).next_to(
                b0, RIGHT, buff=0.08)
            n1 = cn(f"{v1:.1f} km/s", size=18, color=color).next_to(
                b1, RIGHT, buff=0.08)
            bars.add(name, b0, b1, n0, n1)
        self.play(FadeIn(bars), run_time=0.9)
        note = cn(
            "改变的是航天器相对太阳的速度；行星自身也损失/获得很少的轨道能量",
            size=20, color=YELLOW).to_edge(DOWN).shift(UP * 0.35)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(2.0)

        # 第四幕: 设计参数和适用边界.
        title = self.switch_title(title, "弹弓设计：转角决定能量交换")
        self.play(FadeOut(bars), FadeOut(note), run_time=0.5)
        formula = VGroup(
            cn("双曲线飞越", size=24, color=YELLOW),
            cn("e = 1 + r_p v∞² / μ", size=22),
            cn("δ = 2 arcsin(1/e)", size=22),
            cn("r_p 越小、v∞ 越小 → 转角越大", size=21, color=TEAL),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        formula.move_to(LEFT * 2.4 + UP * 0.55)

        limits = VGroup(
            cn("工程含义", size=23, color=GREEN),
            cn("借行星公转能量改变日心轨道", size=20),
            cn("近心点高度受大气和碰撞约束", size=20),
            cn("精确任务需星历与多体数值积分", size=20, color=GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.20)
        limits.move_to(RIGHT * 3.2 + DOWN * 0.10)
        self.play(FadeIn(formula), FadeIn(limits), run_time=0.9)
        self.wait(2.5)


class MomentumWheelCMG(Scene):
    """反作用轮角动量交换、饱和卸载与 CMG 奇异几何."""

    def switch_title(self, old_title, text, size=30):
        new_title = cn(text, size=size, weight="BOLD").to_corner(UL)
        self.play(FadeOut(old_title), FadeIn(new_title), run_time=0.5)
        return new_title

    def construct(self):
        center = LEFT * 3.10 + DOWN * 0.15
        Jb = 12.0
        Jw = 0.05
        tau_m = 0.15
        t_end = 6.0
        ts, theta_b, wb, ww, hw, Htot = reaction_wheel_spinup(
            Jb, Jw, tau_m, t_end=t_end)
        dt = ts[1] - ts[0]
        wheel_angle = np.cumsum(ww) * dt

        def atm(arr, t):
            return float(np.interp(t, ts, arr))

        def spacecraft(theta_body, theta_wheel):
            bus = Rectangle(
                width=1.10, height=1.10, color=TEAL, stroke_width=3,
                fill_color=BLUE_E, fill_opacity=0.24,
            ).move_to(center)
            panel_l = Rectangle(
                width=1.55, height=0.48, color=BLUE_E, stroke_width=2,
                fill_color=BLUE_E, fill_opacity=0.45,
            ).move_to(center + LEFT * 1.48)
            panel_r = panel_l.copy().move_to(center + RIGHT * 1.48)
            body_axis = Line(
                center, center + UP * 0.78, color=WHITE, stroke_width=4)
            body = VGroup(bus, panel_l, panel_r, body_axis)
            body.rotate(theta_body, about_point=center)

            ring = Circle(radius=0.46, color=ORANGE, stroke_width=4)
            ring.move_to(center)
            spoke = Line(center, center + RIGHT * 0.40,
                         color=ORANGE, stroke_width=4)
            hub = Dot(center, radius=0.055, color=ORANGE)
            rotor = VGroup(ring, spoke, hub)
            rotor.rotate(theta_wheel, about_point=center)
            return VGroup(body, rotor)

        title = cn("反作用轮：内部交换角动量", size=30,
                   weight="BOLD").to_corner(UL)
        self.play(Write(title), run_time=0.8)

        time = ValueTracker(0.0)
        craft = always_redraw(
            lambda: spacecraft(
                atm(theta_b, time.get_value()),
                0.75 * atm(wheel_angle, time.get_value()),
            )
        )
        part_labels = VGroup(
            cn("航天器本体", size=18, color=TEAL).move_to(
                LEFT * 3.1 + UP * 1.45),
            cn("反作用轮", size=18, color=ORANGE).move_to(
                LEFT * 3.1 + DOWN * 1.15),
        )
        self.play(FadeIn(craft), FadeIn(part_labels), run_time=0.7)

        bar_center = RIGHT * 3.05 + DOWN * 1.25
        bar_left = bar_center + LEFT * 1.55
        bar_bg = Rectangle(
            width=3.10, height=0.18, color=GRAY_B, stroke_width=2,
            fill_color=GRAY, fill_opacity=0.25,
        ).move_to(bar_center)
        h_max = 0.95
        h_fill = always_redraw(
            lambda: Rectangle(
                width=max(0.001, 3.10 * atm(hw, time.get_value()) / h_max),
                height=0.18, stroke_width=0,
                fill_color=ORANGE, fill_opacity=0.85,
            ).move_to(
                bar_left + RIGHT * max(
                    0.001, 3.10 * atm(hw, time.get_value()) / h_max) / 2)
        )
        status = always_redraw(
            lambda: VGroup(
                cn(f"h_w = {atm(hw, time.get_value()):.2f} N·m·s",
                   size=20, color=ORANGE),
                cn(f"ω_b = {atm(wb, time.get_value()):+.3f} rad/s",
                   size=20, color=TEAL),
                cn("H_total = I_b ω_b + h_w ≈ 0",
                   size=19, color=YELLOW),
                cn("轮子加速 → 本体反向转动", size=20, color=WHITE),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to(
                RIGHT * 3.05 + UP * 0.70)
        )
        self.play(FadeIn(bar_bg), FadeIn(h_fill), FadeIn(status),
                  run_time=0.6)
        self.play(time.animate.set_value(t_end), run_time=6.0,
                  rate_func=linear)
        self.wait(1.0)

        # 第二幕: 饱和与用外力矩卸载.
        title = self.switch_title(title, "轮子饱和后：必须卸载")
        self.play(FadeOut(craft), FadeOut(part_labels), FadeOut(status),
                  run_time=0.5)

        wheel_icon = VGroup(
            Circle(radius=0.62, color=ORANGE, stroke_width=4).move_to(
                LEFT * 3.0 + DOWN * 0.25),
            Line(LEFT * 3.0 + DOWN * 0.25,
                 LEFT * 3.0 + DOWN * 0.25 + RIGHT * 0.54,
                 color=ORANGE, stroke_width=4),
            Dot(LEFT * 3.0 + DOWN * 0.25, radius=0.06, color=ORANGE),
            cn("反作用轮", size=19, color=ORANGE).move_to(
                LEFT * 3.0 + DOWN * 1.12),
            cn("角动量存储", size=18, color=GRAY_B).move_to(
                LEFT * 3.0 + DOWN * 1.48),
        )
        self.play(FadeIn(wheel_icon), run_time=0.4)

        h_limit = 0.75
        limit_x = bar_left + RIGHT * 3.10 * h_limit / h_max
        limit_line = DashedLine(
            limit_x + UP * 0.36, limit_x + DOWN * 0.36,
            color=RED, stroke_width=3, dash_length=0.06,
        )
        limit_label = cn("饱和线", size=18, color=RED).next_to(
            limit_line, UP, buff=0.08)
        zone = Rectangle(
            width=3.10 * (h_max - h_limit) / h_max, height=0.18,
            stroke_width=0,
            fill_color=RED, fill_opacity=0.18,
        ).move_to(
            limit_x + RIGHT * 3.10 * (h_max - h_limit) / h_max / 2)
        self.play(FadeIn(limit_line), FadeIn(limit_label),
                  FadeIn(zone), run_time=0.5)

        h_state = ValueTracker(hw[-1])
        h_fill2 = always_redraw(
            lambda: Rectangle(
                width=max(0.001, 3.10 * h_state.get_value() / h_max),
                height=0.18, stroke_width=0,
                fill_color=ORANGE, fill_opacity=0.85,
            ).move_to(
                bar_left + RIGHT * max(
                    0.001, 3.10 * h_state.get_value() / h_max) / 2)
        )
        load_readout = always_redraw(
            lambda: cn(
                f"轮子角动量 h_w = {h_state.get_value():.2f} N·m·s",
                size=22, color=ORANGE).move_to(RIGHT * 3.0 + UP * 0.55)
        )
        thruster = VGroup(
            cn("喷气卸载", size=20, color=RED).move_to(
                RIGHT * 3.0 + UP * 1.35),
            Arrow(RIGHT * 3.45 + DOWN * 0.15,
                  RIGHT * 2.55 + DOWN * 0.15,
                  buff=0, color=RED, stroke_width=4),
            Arrow(RIGHT * 3.45 + DOWN * 0.55,
                  RIGHT * 2.55 + DOWN * 0.55,
                  buff=0, color=RED, stroke_width=4),
        )
        note = cn("反作用轮不能凭空消掉角动量；外力矩才能卸载",
                  size=20, color=YELLOW).to_edge(DOWN).shift(UP * 0.55)
        self.play(FadeOut(h_fill), FadeIn(h_fill2),
                  FadeIn(load_readout), FadeIn(thruster), run_time=0.6)
        self.play(h_state.animate.set_value(0.0), run_time=6.0,
                  rate_func=linear)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(1.0)

        # 第三幕: CMG 改变角动量方向.
        title = self.switch_title(title, "CMG：改变角动量方向来出力矩")
        self.play(
            FadeOut(bar_bg), FadeOut(h_fill2), FadeOut(load_readout),
            FadeOut(limit_line), FadeOut(limit_label), FadeOut(zone),
            FadeOut(thruster), FadeOut(note), FadeOut(wheel_icon),
            run_time=0.6,
        )

        cmg_center = LEFT * 2.75 + DOWN * 0.10
        axes = VGroup(
            Arrow(cmg_center, cmg_center + RIGHT * 2.35,
                  buff=0, color=GRAY_B, stroke_width=2),
            Arrow(cmg_center, cmg_center + UP * 2.0,
                  buff=0, color=GRAY_B, stroke_width=2),
            cn("x", size=19, color=GRAY_B).next_to(
                cmg_center + RIGHT * 2.35, RIGHT, buff=0.08),
            cn("z", size=19, color=GRAY_B).next_to(
                cmg_center + UP * 2.0, UP, buff=0.08),
        )
        gimbal = VGroup(
            Dot(cmg_center, radius=0.05, color=WHITE),
            cn("框架轴 y（出纸面）", size=17, color=GRAY_B).move_to(
                cmg_center + LEFT * 0.85 + DOWN * 1.02),
        )
        delta = ValueTracker(0.0)

        def h_dir():
            a = delta.get_value()
            return np.array([np.cos(a), np.sin(a), 0.0])

        def tau_dir():
            a = delta.get_value()
            return np.array([np.sin(a), -np.cos(a), 0.0])

        disk = always_redraw(
            lambda: Ellipse(
                width=1.55, height=0.38, color=ORANGE, stroke_width=3,
                fill_color=ORANGE, fill_opacity=0.22,
            ).move_to(cmg_center).rotate(
                delta.get_value() + np.pi / 2, about_point=cmg_center)
        )
        h_arrow = always_redraw(
            lambda: Arrow(cmg_center, cmg_center + 1.55 * h_dir(),
                          buff=0, color=TEAL, stroke_width=5))
        tau_arrow = always_redraw(
            lambda: Arrow(cmg_center, cmg_center + 0.95 * tau_dir(),
                          buff=0, color=YELLOW, stroke_width=5))
        legend = VGroup(
            cn("青色：转子角动量 h", size=18, color=TEAL),
            cn("黄色：本体反作用力矩 τ", size=18, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        legend.move_to(RIGHT * 3.15 + DOWN * 1.35)
        cmg_status = always_redraw(
            lambda: VGroup(
                cn(f"框架角 δ = {np.degrees(delta.get_value()):.0f}°",
                   size=21, color=WHITE),
                cn("|h| = H 保持不变", size=21, color=TEAL),
                cn("τ = -ω_g × h", size=22, color=YELLOW),
                cn("|τ| = H δdot = 5 N·m", size=20, color=GRAY_B),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to(
                RIGHT * 3.15 + UP * 0.85)
        )
        self.play(FadeIn(axes), FadeIn(gimbal), FadeIn(disk),
                  FadeIn(h_arrow), FadeIn(tau_arrow),
                  FadeIn(legend), FadeIn(cmg_status), run_time=1.0)
        self.play(delta.animate.set_value(np.pi / 2), run_time=7.0,
                  rate_func=linear)
        self.wait(1.0)

        # 第四幕: 两个 CMG 的力矩方向共线时出现奇异.
        title = self.switch_title(title, "CMG 阵列：奇异方向从哪里来？")
        self.play(
            FadeOut(axes), FadeOut(gimbal), FadeOut(disk),
            FadeOut(h_arrow), FadeOut(tau_arrow), FadeOut(legend),
            FadeOut(cmg_status),
            run_time=0.6,
        )
        org = LEFT * 2.75 + DOWN * 0.15
        sep = ValueTracker(np.radians(60.0))

        def draw_torque(direction, color):
            return Arrow(org, org + 1.90 * np.array(
                [direction[0], direction[1], 0.0]),
                buff=0, color=color, stroke_width=6)

        t1_arrow = always_redraw(
            lambda: draw_torque(
                parallel_axis_cmg_torque_directions(
                    -sep.get_value() / 2, sep.get_value() / 2)[0],
                TEAL))
        t2_arrow = always_redraw(
            lambda: draw_torque(
                parallel_axis_cmg_torque_directions(
                    -sep.get_value() / 2, sep.get_value() / 2)[1],
                ORANGE))
        cmg_labels = VGroup(
            cn("CMG 1 力矩方向", size=18, color=TEAL).next_to(
                org + RIGHT * 1.25 + UP * 0.45, RIGHT, buff=0.10),
            cn("CMG 2 力矩方向", size=18, color=ORANGE).next_to(
                org + RIGHT * 1.25 + DOWN * 0.25, RIGHT, buff=0.10),
        )
        det_readout = always_redraw(
            lambda: cn(
                f"det A ∝ sin(δ₂-δ₁) = "
                f"{np.sin(sep.get_value()):.2f}",
                size=21, color=YELLOW).move_to(
                    RIGHT * 3.2 + UP * 0.95))
        jac_text = VGroup(
            cn("A δdot = -τ", size=21, color=WHITE),
            cn("rank(A) < 2 → 奇异", size=19, color=GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        jac_text.move_to(RIGHT * 3.2 + UP * 0.20)
        singular = cn(
            "两 CMG 力矩方向共线：某些方向不可控，框架角速度需求发散",
            size=19, color=RED).to_edge(DOWN).shift(UP * 0.45)
        self.play(FadeIn(t1_arrow), FadeIn(t2_arrow),
                  FadeIn(cmg_labels), FadeIn(det_readout), FadeIn(jac_text),
                  run_time=0.8)
        self.play(sep.animate.set_value(0.0), run_time=6.0,
                  rate_func=linear)
        self.play(FadeIn(singular), run_time=0.5)
        self.wait(2.0)
