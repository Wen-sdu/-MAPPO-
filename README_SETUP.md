# DCMG MAPPO 可迁移运行包

本包包含最终混合负载MAPPO训练、确定性评估、CSV绘图、Actor头文件导出、
导出一致性验证以及F28P55 CCS部署源码。所有运行路径均从本包根目录推导，
不依赖原电脑的盘符或工程位置。

## 目录职责

- `onpolicy/`：MAPPO基础算法、Runner和公共环境代码。
- `mappo_training/`：最终混合负载训练、评估、模型导出和检查点。
- `ccs_deployment/source/`：需要加入CCS工程的C/H文件。
- `matlab_tools/evaluation/`：Python评估CSV绘图。
- `matlab_tools/hil/`：Typhoon导出CSV绘图。
- `typhoon_hil/`：预留给Typhoon原理图、SCADA和编译文件。
- `outputs/`：用户实验输出目录。

建议将本包解压到纯英文、无空格的短路径，例如：

```text
D:\DCMG_MAPPO_Portable
```

## 1. Python环境

验证环境为Windows、Python 3.10.11、PyTorch 2.8.0 CPU版。不要复制其他电脑的
虚拟环境。在PowerShell中进入本包根目录后运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_environment.ps1
```

也可手动执行：

```powershell
py -3.10 -m venv .venv_mappo
.\.venv_mappo\Scripts\python.exe -m pip install --upgrade pip
.\.venv_mappo\Scripts\python.exe -m pip install -r requirements_dc_microgrid.txt
```

## 2. 训练

运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_training.ps1
```

训练入口读取：

```text
mappo_training/configs/final_mixed_train_config.json
```

并从以下CPL Actor/Critic热启动：

```text
mappo_training/checkpoints/warm_start_cpl
```

新训练结果自动写入`mappo_training/results/`。重新训练采用固定配置和种子，
但不同CPU/GPU、PyTorch版本或并行调度可能使权重不完全逐字节一致。

当前正式模型已经随包保存在：

```text
mappo_training/results/mappo_mixed_ff32_500000_20261015_145758/run1/models
```

## 3. 评估

运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_evaluation.ps1
```

在窗口中选择`actor.pt`和输出目录。程序分别生成`normal`和`noise0`的CSV及
摘要JSON。Actor所在`run1`目录必须保留对应的`config.json`。

MATLAB中运行：

```text
matlab_tools/evaluation/plot_eval_csv.m
```

然后选择评估CSV。

## 4. 导出CCS头文件

运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_export_actor.ps1
```

选择训练生成的`actor.pt`。默认输出位置为：

```text
ccs_deployment/source/mappo_actor_weights.h
```

该目录中的以下三个文件应一起加入CCS工程：

```text
f28p55_mappo_double_loop.c
mappo_double_loop_config.h
mappo_actor_weights.h
```

## 5. 验证导出结果

每次重新导出头文件后运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\verify_export.ps1
```

脚本默认比较随包最终Actor和CCS头文件。只有显示`PASSED`后才应烧录DSP。
当前正式Actor的SHA256为：

```text
66c2210d4733fa90efd29ef012dfceea371b72c3ed1b1820e5b307178e1c8a15
```

## 6. CCS工程

`ccs_deployment/F28P55_MAPPO_Project/`为工程预留位置。目标硬件为
LAUNCHXL-F28P55X、F28P550SJ9PZ。工程需配置DriverLib、Device支持、链接器文件
和TI编译优化`--opt_level=2`。ADC标定、PWM引脚、控制周期和PI参数位于
`mappo_double_loop_config.h`。

## 7. Typhoon与HIL CSV

Typhoon文件由用户放入`typhoon_hil/cpl/`和`typhoon_hil/resistive/`。
建议在目标电脑用相同或兼容的Typhoon HIL Control Center重新编译`.tse`。

Typhoon CSV应按以下前10列导出：

```text
Time,Vdc1,Vdc2,Vdc3,Vdc4,Vaverage,IL1,IL2,IL3,IL4
```

MATLAB中运行`matlab_tools/hil/Plot_Typhoon_HIL_RGBK_Width2.m`即可绘制四路母线
电压、平均母线电压和四路电流。脚本会按CSV实际采集时长设置横坐标。

