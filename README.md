# 空间飞行器动力学教学动画

一组面向空间飞行器动力学、姿态动力学与轨道力学教学的中文动画，共 9 支。
作品以 Python 自写动力学模型为基础，通过 Manim 制作，覆盖欧拉角、自旋稳定性、
能量耗散、引力梯度、太阳同步轨道、CW 相对运动、引力弹弓和姿态执行机构等主题。

所有成片均为 `1280 x 720`、`30 fps`，采用中文标注，并配有与公式对应的动态过程、
曲线或三维姿态演示。

## 动画一览

| 动画 | 内容 | 时长 | 视频 |
| --- | --- | ---: | --- |
| `EulerRotation312` | 第二套姿态角 3-1-2 转序，并与 2-1-3 反序对比，展示转序不可交换 | 19.5 s | [观看](videos/EulerRotation312.mp4) |
| `EnergyDissipationExplorer1` | Explorer 1 绕最小惯量轴自旋时，在能量耗散作用下发生 90° 翻转 | 8.8 s | [观看](videos/EnergyDissipationExplorer1.mp4) |
| `SpinStability` | 刚体绕最大惯量轴与中间惯量轴自旋的稳定性对比，展示网球拍效应 | 20.9 s | [观看](videos/SpinStability.mp4) |
| `GravityGradientSatellite` | 引力梯度稳定构型的天平动与不稳定构型的发散过程三维对比 | 23.2 s | [观看](videos/GravityGradientSatellite.mp4) |
| `GravityGradientCurves` | 稳定与不稳定构型在 10 个轨道周期内的姿态角曲线 | 10.4 s | [观看](videos/GravityGradientCurves.mp4) |
| `SunSynchronousOrbit` | 太阳同步轨道的极轨地方时漂移、J2 节点进动、倾角—高度关系和一年尺度对比 | 46.3 s | [观看](videos/SunSynchronousOrbit.mp4) |
| `CWRelativeRendezvous` | LVLH 系中的自然闭合绕飞，以及 V-bar 半周期双脉冲交会和末端速度匹配 | 33.6 s | [观看](videos/CWRelativeRendezvous.mp4) |
| `GravityAssistSlingshot` | 双曲线飞越、行星系与日心系速度合成，以及引力弹弓的加速和减速 | 26.0 s | [观看](videos/GravityAssistSlingshot.mp4) |
| `MomentumWheelCMG` | 反作用轮角动量交换、饱和卸载、单框架 CMG 力矩和双 CMG 奇异几何 | 39.6 s | [观看](videos/MomentumWheelCMG.mp4) |

## 作品特点

- 以课程中的飞行器姿态动力学内容为基础，补充轨道相对运动、深空借力和姿态执行机构。
- 旋转矩阵、欧拉角、引力梯度、太阳同步轨道和 CW 交会均使用明确给出的动力学关系。
- 动画中的关键数值采用解析结果或独立数值检查进行核对。
- 每支动画不仅展示现象，也给出对应的公式、物理量、判据或工程含义。
- 明确区分教学模型与高保真仿真，避免把示意结果当作任务级设计结论。

## 仓库内容

- `videos/`：9 支最终成片。
- `src/scenes.py`：全部 Manim 动画场景。
- `src/dynamics.py`：刚体姿态动力学、欧拉角运动学和引力梯度模型。
- `src/sun_sync.py`：太阳同步轨道的一阶 J2 模型。
- `src/relative_motion.py`：CW 相对运动状态转移与双脉冲交会。
- `src/gravity_assist.py`：二维双曲线飞越与日心速度合成。
- `src/momentum_control.py`：反作用轮和单框架 CMG 教学模型。

## 模型范围

- 能量耗散翻转按照课件中的运动过程复现，用于说明最小惯量轴自旋失稳现象。
- 太阳同步轨道采用近圆轨道一阶 J2 长期项，不包含大气阻力、高阶重力场和轨道保持。
- CW 交会采用近距离线性化相对运动模型，不包含避碰、安全走廊和推力幅值约束。
- 引力弹弓采用二维 patched-conic 教学模型，不是星历、多体或实际任务设计。
- 反作用轮与 CMG 采用简化执行机构模型，不是完整构型、操纵律或奇异规避设计。

## 作者

**Yue Zixin**

- 个人主页：<https://sanshansss.github.io/astro-theme-retypeset/>
- GitHub：<https://github.com/sanshansss>
- 项目仓库：<https://github.com/sanshansss/spacecraft-dynamics-animations>
