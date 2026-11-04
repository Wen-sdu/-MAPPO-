# DSP 与 Typhoon HIL IO 连接对照表

适用对象：

- DSP开发板：TI LAUNCHXL-F28P55X
- DSP芯片：TMS320F28P550SJ9PZ
- Typhoon设备：HIL404，Configuration 1
- CCS控制程序：`ccs_deployment/source/f28p55_mappo_double_loop.c`
- CCS接口配置：`ccs_deployment/source/mappo_double_loop_config.h`
- Typhoon模型：本包 `typhoon_hil` 目录中的电阻负载与CPL模型

## 1. DSP PWM输出到Typhoon数字输入

| 控制通道 | DSP外设输出 | DSP GPIO | LAUNCHXL-F28P55X排针 | Typhoon输入 | Typhoon受控对象 | 逻辑/频率 | 备注 |
|---|---|---:|---|---|---|---|---|
| Buck 1 | ePWM1A | GPIO0 | J4-40（BP Pin 40） | DI1 | Buck1 / S1 | Active high，10 kHz | `HIL_PWM1_GPIO_MUX = GPIO_0_EPWM1_A` |
| Buck 2 | ePWM2A | GPIO2 | J4-38（BP Pin 38） | DI2 | Buck2 / S1 | Active high，10 kHz | `HIL_PWM2_GPIO_MUX = GPIO_2_EPWM2_A` |
| Buck 3 | ePWM3A | GPIO4 | J4-32（BP Pin 32） | DI3 | Buck3 / S1 | Active high，10 kHz | S4必须置于UP/OFF，使GPIO4连接BoosterPack排针而不是CAN收发器 |
| Buck 4 | ePWM4A | GPIO6 | J8-78（BP Pin 78） | DI4 | Buck4 / S1 | Active high，10 kHz | `HIL_PWM4_GPIO_MUX = GPIO_6_EPWM4_A` |

连接方向：`DSP PWM输出 -> Typhoon DI输入`。

两套Typhoon `.tse` 模型都将Buck1～Buck4的数字输入编号设置为1～4，因此电阻负载和CPL工况采用相同PWM接线。

## 2. Typhoon模拟输出到DSP ADC输入

| Typhoon输出 | 反馈信号 | Typhoon缩放 | 典型AO电压 | DSP ADC模块/通道 | LAUNCHXL-F28P55X排针 | DSP程序索引 | 接线备注 |
|---|---|---|---:|---|---|---:|---|
| AO1 | Vdc1 | `Vdc1 / 50` | 48 V -> 0.96 V | ADCA / ADCINA0 | J7-70 | SOC0 / `g_hil.vdc[0]` | 接入母线1电压反馈 |
| AO2 | Vdc2 | `Vdc2 / 50` | 48 V -> 0.96 V | ADCA / ADCINA1 | J3-30 | SOC1 / `g_hil.vdc[1]` | 接入母线2电压反馈 |
| AO3 | Vdc3 | `Vdc3 / 50` | 48 V -> 0.96 V | ADCA / ADCINA14 | J3-26 | SOC2 / `g_hil.vdc[2]` | 不要改接ADCINA2/B6；项目记录该通道读数异常 |
| AO4 | Vdc4 | `Vdc4 / 50` | 48 V -> 0.96 V | ADCA / ADCINA3（PGA1_IN_N） | J10A-4 | SOC3 / `g_hil.vdc[3]` | 使用J10时将S6置于DOWN/J10位置；上电前确认实板标识 |
| AO5 | IL1 | `IL1 / 10` | 2 A -> 0.20 V | ADCA / ADCINA4 | J7-66 | SOC4 / `g_hil.il[0]` | 电感电流1反馈 |
| AO6 | IL2 | `IL2 / 10` | 2 A -> 0.20 V | ADCA / ADCINA5 | J7-65 | SOC5 / `g_hil.il[1]` | 电感电流2反馈 |
| AO7 | IL3 | `IL3 / 10` | 2 A -> 0.20 V | ADCA / ADCINA11 | J3-25 | SOC6 / `g_hil.il[2]` | 不要改接ADCINA6；项目记录该通道存在低读数/掉线 |
| AO8 | IL4 | `IL4 / 10` | 2 A -> 0.20 V | ADCA / ADCINA7 | J1-6 | SOC7 / `g_hil.il[3]` | 电感电流4反馈 |
| AO9 | Vo1 | `Vo1 / 50` | 48 V -> 0.96 V | ADCA / ADCINA9 | J5-46 | SOC8 / `g_hil.vo[0]` | 共享GPIO227，程序已启用模拟模式 |
| AO10 | Vo2 | `Vo2 / 50` | 48 V -> 0.96 V | ADCB / ADCINB0 | J7-63 | SOC9 / `g_hil.vo[1]` | 共享GPIO253，程序已启用模拟模式 |
| AO11 | Vo3 | `Vo3 / 50` | 48 V -> 0.96 V | ADCA / ADCINA10 | J7-64 | SOC10 / `g_hil.vo[2]` | 共享GPIO230，程序已启用模拟模式 |
| AO12 | Vo4 | `Vo4 / 50` | 48 V -> 0.96 V | ADCA / ADCINA12 | J7-69 | SOC11 / `g_hil.vo[3]` | 该引脚不需要额外GPIO模拟复用设置 |

连接方向：`Typhoon AO输出 -> DSP ADC输入`。

缩放含义为Typhoon物理输出电压等于被测工程量除以缩放系数：

```text
AO1～AO4  = Vdc / 50
AO5～AO8  = IL  / 10
AO9～AO12 = Vo  / 50
```

DSP代码按照各通道实测ADC零点和增益进行校准。当前ADC参考电压固定为2.5 V，不应向任一ADC引脚输入负电压或高于2.5 V的电压。

## 3. 公共地连接

| Typhoon端 | DSP端 | 要求 |
|---|---|---|
| AO GND / Analog COM | LAUNCHXL GND，例如J3-22或J7-62 | 所有AO反馈必须与DSP ADC共地 |
| DI GND / Digital COM | LAUNCHXL GND，例如J2-20或J6-60 | PWM数字输入必须与DSP GPIO共地 |

模拟地和数字地最终必须具有同一参考电位。不要把Typhoon AO端接到LaunchPad的3.3 V或5 V电源脚，也不要用AO端口给LaunchPad供电。

## 4. 上电前检查顺序

1. DSP和Typhoon均停止运行，先连接公共地。
2. 按AO1～AO12表连接全部模拟反馈线。
3. 设置S4为UP/OFF；若AO4使用J10A-4，设置S6为DOWN/J10。
4. 按DI1～DI4表连接4路PWM线。
5. 在Typhoon I/O设置中确认AO1～AO12的信号和缩放与表格一致。
6. 先用DSP `mode=0` 固定占空比检查DI1～DI4是否均有约10 kHz PWM，再进入闭环和MAPPO模式。
7. 在CCS中确认12路ADC原始值均非零且没有饱和，然后再接入二次控制。

## 5. 文件依据

- 项目接口定义：`ccs_deployment/source/mappo_double_loop_config.h`
- 项目ADC/PWM初始化：`ccs_deployment/source/f28p55_mappo_double_loop.c`
- Typhoon AO映射：`typhoon_hil/Fourbuck_MultiBus_Resitive.runx`
- Typhoon DI映射：两套 `.tse` 模型中的Buck和PWM Analyzer输入1～4
- TI LAUNCHXL-F28P55X Pinout Map：<https://www.ti.com/lit/ml/spaz056/spaz056.pdf>
- TI LAUNCHXL-F28P55X Schematic：<https://www.ti.com/lit/df/sprr499a/sprr499a.pdf>

> 注意：排针编号以LAUNCHXL-F28P55X官方Rev. A原理图/Pinout Map和当前项目配置为准。若换用不同板卡版本、修改CCS中的ADC/PWM宏、改变S4/S6位置或重新配置Typhoon I/O，必须同步更新本表。
