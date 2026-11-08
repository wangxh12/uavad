# ALFA 同架次切分实验：CrossAD 与跨变量关系版

## 协议与配置

- 数据：`data/processed/alfa_within_flight_v1`，18 特征、训练点拟合的 z-score。46 个 CSV 的各段分别拼接存储，加载时恢复为架次独立分段，窗口不跨文件。训练 47,184 个正常点，验证 15,734 个正常点，测试 31,180 点（正常 15,753、异常 15,427）。
- 每个源文件按故障前正常段的 60%/20%/20% 分成 train/val/test，完整故障段接入 test。真值故障起点参与**切分**，标签不作为模型输入、验证损失或阈值拟合的输入。此协议只衡量已见架次的后续片段，不能与旧的跨架次留出协议直接比较。
- 模型：原版 `crossad` 与现有跨变量 `crossad_relation`。两者独立从头训练，不热启动；相同种子 2025、96 点窗口、batch size 128、最多 20 epoch、早停 patience 3。训练步长 1；验证和测试非重叠窗口补末窗。单种子。
- 模型选择：最小正常验证损失；阈值：正常验证集逐点均值异常分数的 99% 分位数。测试无 point adjustment；指标按测试**逐点**计算，故障架次的 AUROC 单独在其正常+异常测试段内计算。
- 使用远端 `h100` 的独立 `ltsm` 环境，分别在空闲 GPU 1 和 2 运行。原版训练至第 20 epoch；关系版第 11 epoch 早停，最佳为第 8 epoch。两次小测分别使用 2 epoch、每轮 2 batch，另存于 `within_flight_*smoke_20261102`，不参与正式指标。

## 结果

| 模型 | AUROC | AUPRC | Precision | Recall | F1 | 故障架次检出 | AUROC >= 0.8 的故障架次 | F1 >= 0.5 的故障架次 | 正常点误报 | 全正常架次发生误报 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原版 CrossAD | 0.6248 | 0.5991 | 0.7609 | 0.0576 | 0.1070 | 9/36 | 13/36 | 4/36 | 279/15,753 | 1/10 |
| CrossAD relation | 0.6253 | 0.5999 | 0.7588 | 0.0575 | 0.1069 | 8/36 | 13/36 | 3/36 | 282/15,753 | 1/10 |

原版阈值 2.758905；关系版阈值 3.141939。两模型的分数标度不同，阈值不可交叉使用。关系版在 19/36 个故障架次上 AUROC 高于原版，但总体差异很小，固定阈值检测没有提升。异常点约占测试点的 49.5%，所以 AUPRC 0.60 的解释应结合此基线比例。AUROC 与固定阈值无关：例如原版 `2018-10-05-15-52-12_3_engine_failure_with_emr_traj` AUROC 为 0.993，F1 为 0，图上故障分数上升但未超过全局阈值；不能把高 AUROC 当成实际已报警。

## 输出与复核

- 原版：`results/crossad_alfa/within_flight_crossad_w96_seed2025/`。
- 关系版：`results/crossad_alfa/within_flight_relation_w96_seed2025/`。
- 对比：`results/crossad_alfa/within_flight_w96_comparison_seed2025/paired_metrics.png` 和 `flights.csv`。
- 每个运行目录含 `config.json`、`best.pt`、`latest.pt`、`metrics.json`、`flights.csv`、`scores/*.npz`、`figures/separation.png`、`figures/flight-timelines.png`、分故障类汇总图和 `figures/per-flight/*.png`（每模型 46 张）。图中粉色阴影为测试段的真值故障区间，虚线为各自验证阈值，橙线为超过阈值的点。
- 两个模型共 92 个逐架次分数文件、92 个逐架次图。复核每个测试点的标签与原始时间、每特征和平均分数、阈值判决、有限性以及报告中的 AUROC。相关 27 项 pytest 测试通过。

复现命令（在远端 `/workspace/code/uavad`，`ltsm`）：

```bash
CUDA_VISIBLE_DEVICES=1 python -m scripts.crossad_alfa train --processed-dir data/processed/alfa_within_flight_v1 --run-dir results/crossad_alfa/within_flight_crossad_w96_seed2025 --model crossad --window-size 96 --device cuda:0 --seed 2025 --batch-size 128 --epochs 20 --evaluate-after
CUDA_VISIBLE_DEVICES=2 python -m scripts.crossad_alfa train --processed-dir data/processed/alfa_within_flight_v1 --run-dir results/crossad_alfa/within_flight_relation_w96_seed2025 --model crossad_relation --window-size 96 --device cuda:0 --seed 2025 --batch-size 128 --epochs 20 --evaluate-after
python -m scripts.visualize_crossad --run-dir results/crossad_alfa/within_flight_crossad_w96_seed2025
python -m scripts.visualize_crossad --run-dir results/crossad_alfa/within_flight_relation_w96_seed2025
python -m scripts.visualize_within_flight_comparison --baseline results/crossad_alfa/within_flight_crossad_w96_seed2025 --candidate results/crossad_alfa/within_flight_relation_w96_seed2025 --output results/crossad_alfa/within_flight_w96_comparison_seed2025/paired_metrics.png
```

当前 Git HEAD 为 `49c2b7d13d826e348f81c2b7d6995f810266f7b4`，相关项目文件仍有未提交修改，**仅凭 HEAD 不足以复现**。本次使用的 SHA-256：数据 `manifest.json` 为 `60887fe674af4fb7efc968b3e27757d14ec8b098f0284e5e7da14767ef4c28f6`；`models/crossad.py` 为 `04a168328440cb379ec3da31272ca15c72b8bf7449f2af3e96ff0c609865a660`；`models/crossad_relation.py` 为 `d04bd5c1d1c452b36ca91bcc8bc2ceb517f6d8ab48ef8955a0266cec8236ec27`；`scripts/crossad_alfa.py` 为 `8662401f9acdafdaa06ea25da71f390c00fd605ad2fc20c5e29b78e37af76ba4`。
