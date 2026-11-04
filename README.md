# DCMG MAPPO 可迁移项目说明

## 1. 项目用途

本目录是四母线直流微电网（DCMG）的可迁移完整项目包，用于：

- 训练能同时处理电阻负载、恒功率负载（CPL）和两者组合工况的前馈 MAPPO 模型。
- 对训练生成的 `actor.pt` 进行确定性评估，生成 CSV 和 JSON 结果。
- 将 PyTorch Actor 导出为 TI CCS 工程使用的 C 头文件。
- 验证 PyTorch 推理与导出头文件推理的数值一致性。
- 将控制算法部署到 LAUNCHXL-F28P55X（F28P550SJ9PZ），并与 Typhoon HIL404 进行 HIL 验证。
- 使用 MATLAB 绘制 Python 评估结果和 Typhoon HIL 导出的电压、电流数据。

该包的路径逻辑基于当前文件位置，不应依赖原始 `on-policy-main` 目录或原电脑的用户目录。

## 2. 对 Codex 的操作约束

在新电脑上使用 Codex 维护本项目时，应遵守：

1. 将本 `README.md` 所在目录视为唯一项目根目录。
2. 所有新增和修改只在本可迁移包内完成，不搜索、引用或修改其他备份项目。
3. 代码中不得新增固定盘符、用户名或原电脑绝对路径；使用 `Path(__file__)` 和包内相对路径。
4. 不要将 `.venv_mappo` 作为可迁移内容；每台电脑都应重新建立虚拟环境。
5. 训练必须优先通过 `mappo_training/start_mappo_train.py` 启动，不建议直接运行底层训练入口。
6. 每次更换 Actor 并重新导出头文件后，必须先运行导出一致性验证，通过后再烧录 DSP。
7. 修改 ADC、PWM、采样周期、缩放系数或 PI 参数时，必须同时核对 CCS 代码、Typhoon 原理图和实际接线。

## 3. 目录结构

```text
DCMG_MAPPO_Portable/
├─ onpolicy/                  MAPPO算法、Runner、经验缓存和公共电路环境
├─ mappo_training/
│  ├─ start_mappo_train.py   正式训练启动器
│  ├─ configs/               最终Mixed训练参数
│  ├─ checkpoints/           CPL热启动基础模型
│  ├─ training/              CPL/Mixed电路模型、环境和底层训练入口
│  ├─ scripts/               评估、Actor导出和导出一致性验证
│  ├─ results/               正式模型和新训练结果
│  └─ reference_evaluation/  打包时正式模型的参考评估数据
├─ ccs_deployment/
│  ├─ source/                DSP控制源码、配置头文件和Actor权重
│  └─ F28P55_MAPPO_Project/ 完整CCS工程的预留位置
├─ typhoon_hil/              Typhoon HIL404的TSE、CUS和RUNX文件
├─ matlab_tools/             Python评估CSV和HIL CSV绘图程序
├─ outputs/
│  ├─ training/              手动归档的训练日志、曲线或模型副本
│  ├─ evaluation/            Python评估输出
│  └─ hil_csv/               Typhoon Capture/Scope导出数据
├─ DSP_Typhoon_HIL_IO连接对照表.md
├─ README_SETUP.md           面向使用者的环境和操作说明
└─ README.md                 项目总览和Codex交接说明
```

## 4. 当前正式配置与模型

最终 Mixed 训练配置：

```text
mappo_training/configs/final_mixed_train_config.json
```

关键设定：

- 算法：前馈 MAPPO（非 RNN）。
- Agent 数量：4。
- 隐藏层宽度：32，实验名中记为 `ff32`。
- 训练步数：500,000。
- 容量比：`1,2,1,1`。
- 负载类型：电阻、CPL 及二者组合，负载可作用于任意母线。
- CPL 训练范围：当前配置为 100-500 W，混合工况总负载功率上限为 900 W。
- 电压目标：四母线平均电压 48 V，同时追踪按容量比的电流分配。

CPL 热启动检查点：

```text
mappo_training/checkpoints/warm_start_cpl/actor.pt
mappo_training/checkpoints/warm_start_cpl/critic.pt
```

`warm_start_cpl` 是先前在 CPL 工况下训练的基础模型。Mixed 训练从该 Actor/Critic 参数开始继续训练，最终输出才是 Mixed 模型。

打包时的正式 Mixed 模型：

```text
mappo_training/results/mappo_mixed_ff32_500000_20261015_145758/run1/models/actor.pt
mappo_training/results/mappo_mixed_ff32_500000_20261015_145758/run1/models/critic.pt
```

该 `run1` 目录下的 `config.json` 必须与 `models/` 一起保留，评估程序会从中恢复模型和环境参数。

## 5. Python 环境

建议环境：

- Windows。
- Python 3.10。
- 依赖版本以 `requirements_dc_microgrid.txt` 为准。

在包根目录手动建立环境：

```powershell
py -3.10 -m venv .venv_mappo
.\.venv_mappo\Scripts\python.exe -m pip install --upgrade pip
.\.venv_mappo\Scripts\python.exe -m pip install -r requirements_dc_microgrid.txt
```

也可运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_environment.ps1
```

不应将一台电脑的 `.venv_mappo` 压缩到迁移包中。

## 6. 标准工作流

### 6.1 训练

```powershell
.\.venv_mappo\Scripts\python.exe .\mappo_training\start_mappo_train.py --no-pause
```

新训练结果保存到：

```text
mappo_training/results/mappo_mixed_ff32_500000_新时间/run1/
```

### 6.2 确定性评估

```powershell
.\.venv_mappo\Scripts\python.exe .\mappo_training\scripts\eval_mappo_deterministic.py
```

程序会让用户选择 `actor.pt` 和输出目录，并生成 `normal` 和 `noise0` 的 CSV/JSON。建议将结果放在 `outputs/evaluation/<实验名>/`。

### 6.3 导出 CCS 头文件

```powershell
.\.venv_mappo\Scripts\python.exe .\mappo_training\scripts\export_mappo_actor_to_header.py
```

默认部署头文件为：

```text
ccs_deployment/source/mappo_actor_weights.h
```

首次导出新模型时建议先使用新文件名，一致性验证通过后再替换正式头文件。

### 6.4 验证 Actor 与头文件

```powershell
.\.venv_mappo\Scripts\python.exe `
  .\mappo_training\scripts\verify_mappo_export.py `
  "<actor.pt路径>" `
  "<mappo_actor_weights.h路径>"
```

只有显示以下结果才表示所选 `.h` 与所选 `actor.pt` 对应：

```text
PASSED: exported inference matches PyTorch
```

这项检查只验证模型导出和数值推理，不验证 ADC/PWM 接线、DSP 时序或 HIL 闭环性能。

## 7. CCS 部署

目标板卡和芯片：

```text
LAUNCHXL-F28P55X
F28P550SJ9PZ
```

CCS 工程应包含：

```text
ccs_deployment/source/f28p55_mappo_double_loop.c
ccs_deployment/source/mappo_double_loop_config.h
ccs_deployment/source/mappo_actor_weights.h
```

另需配置：

- 支持 F28P55x 的 CCS 和 C2000Ware。
- DriverLib、Device 支持和正确的链接器文件。
- TI 编译优化 `--opt_level=2`。
- 实际使用的 ADC 基准、PWM 引脚、ADC 引脚和控制周期。

如果 `ccs_deployment/F28P55_MAPPO_Project/` 中没有 `.project`、`.cproject` 等完整工程文件，则该目录仍是预留目录，新电脑需要新建或导入配置好的 CCS 工程。

## 8. Typhoon HIL

目标设备是 HIL404，原理图由 Typhoon HIL Control Center 2024.1 保存。新电脑应使用 2024.1 或兼容版本重新编译 `.tse`，然后加载对应 `.cus`。

DSP 与 Typhoon 的 PWM、模拟反馈和共地连接见：

```text
DSP_Typhoon_HIL_IO连接对照表.md
```

实验前至少检查：

- Typhoon AO 与 DSP ADC 的通道顺序。
- Typhoon DI1-DI4 与 DSP ePWM1A-ePWM4A 的顺序。
- Typhoon 与 DSP 信号地已可靠共地。
- ADC 输入电压不超出板卡允许范围。
- `mappo_double_loop_config.h` 中的量程、标定和实际接线一致。
- 重载/CPL 上电前先用固定 PWM 或轻载工况验证四路 PWM 和 12 路 ADC。

## 9. MATLAB 绘图

Python 评估 CSV：

```text
matlab_tools/evaluation/plot_eval_csv.m
```

Typhoon HIL CSV：

```text
matlab_tools/hil/Plot_Typhoon_HIL_RGBK_Width2.m
```

HIL CSV 建议前 10 列为：

```text
Time,Vdc1,Vdc2,Vdc3,Vdc4,Vaverage,IL1,IL2,IL3,IL4
```

## 10. 迁移和压缩规则

压缩前建议排除：

```text
.venv_mappo/
__pycache__/
*.pyc
.idea/        # 可选，不影响项目功能
```

新电脑建议解压到纯英文、无空格的短路径，例如：

```text
D:\DCMG_MAPPO_Portable
```

解压后应按照以下顺序验证：

```text
创建新虚拟环境
→ 安装requirements
→ 验证随包Actor与CCS头文件
→ 评估随包Actor
→ 需要时重新训练
→ 评估新Actor
→ 导出并验证新头文件
→ CCS编译和烧录
→ Typhoon编译
→ HIL闭环实验
```

## 11. 已知注意事项

- 四母线电压不必完全相等；主要电压目标是平均母线电压 48 V。
- 同时实现平均电压恢复和按容量比均流在理论上可行，但动态性能受电感、电容、CPL负阻抗、公共电压PI、电流前馈和限幅影响。
- 电感值对开关电流纹波有显著影响；电容增大可改善瞬态电压，但可能降低动态速度。
- CPL 阶跃下的大幅电压下冲不只是 Actor 问题，还受输出电容储能、内环带宽、功率/电流前馈和限幅约束影响。
- 模型训练范围不能突破物理电路、ADC/PWM、电流限制和 HIL 信号量程的可实现范围。

---

更详细的使用者操作步骤见 `README_SETUP.md`。
