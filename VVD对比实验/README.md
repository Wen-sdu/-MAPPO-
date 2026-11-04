# VVD 与 MAPPO 公平 HIL 对比说明

## 对比原则

两种控制器必须使用同一个 Typhoon 被控对象、同一组负载事件、同一套
DSP 接线、ADC 标定、PWM、一次下垂、电压外环和电流内环。唯一变化是
二次控制器：MAPPO 或论文式 (24) 的 VVD 控制器。

## DSP 模式

- `mode = 0`：固定占空比，仅作安全调试。
- `mode = 1`：一次下垂与电压/电流双闭环，不含二次控制。
- `mode = 2`：VVD 二次控制与相同的一次控制。

MAPPO 工程中的 `mode = 1` 同样作为“无二次控制”基线，MAPPO 的
`mode = 2` 作为学习控制方法。正式比较不使用固定占空比模式作为基线。

## Typhoon 到 DSP 接线

- AO1~AO4：`Vdc1~Vdc4 / 50`。
- AO5~AO8：`IL1~IL4 / 10`。
- AO9~AO12：`Vo1~Vo4 / 50`。
- DI1~DI4：DSP 的 ePWM1A~ePWM4A。

VVD 算法使用 `IL` 计算虚拟电压降，使用 `Vo` 完成一次电压闭环。
`Vdc` 只用于评价母线平均电压，不参与 VVD 控制。

## 实时调度

- PWM、ADC 与电流内环：10 kHz。
- 电压外环：1 kHz。
- VVD 离散控制器：1 kHz。
- Vdc 使用与 MAPPO 相同的五点中值和 IIR 滤波。
- IL、Vo 在每个 1 ms 电压环窗口内同步平均。

VVD 使用 1 kHz 是因为论文连续控制律采用前向欧拉离散后，在
`alpha=50`、`beta=100` 和四节点环形图下，10 ms 步长不满足离散稳定性。
正式报告应明确两种二次控制器的实际更新频率；其余条件保持一致。

## 运行顺序

1. 使用 `mode=1` 启动并确认一次双闭环稳定。
2. 在相同初始负载下切换到 `mode=2`，检查
   `secondarySaturated`、`dutySaturated` 和 `adcOverflowCount` 均不持续增加。
3. 施加与 MAPPO 完全相同的电阻/CPL、负载位置、阶跃幅值和事件时刻。
4. 保存四路 Vdc、平均 Vdc、四路 IL、占空比和二次控制量。
5. 至少比较平均电压最低点、超调、恢复时间、稳态误差、比例均流误差、
   峰值电流和占空比饱和时间。

## 离线检查

在仓库根目录运行：

```powershell
python ".\VVD对比实验\tools\test_vvd_discrete_controller.py"
```

该检查会验证一次控制关键参数与 MAPPO 一致，并检查环形 DAC 的离散稳定性、
均值保持和 VVD 控制不使用 Vdc 反馈。

