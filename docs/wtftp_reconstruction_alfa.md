# WTFTP 主干重构版：ALFA 同架次实验

## 来源与改造边界

- 论文：*Flight trajectory prediction enabled by time-frequency wavelet transform*，Nature Communications (2023)，https://www.nature.com/articles/s41467-023-40903-9 。参考实现：https://github.com/MusDev7/wtftp-model ，核对提交 `d6293e23ad83b29861b781f84737aa7a59cd20c1`；未把原仓库代码复制到项目中。
- 原任务是用历史航迹预测未来位置；本实验是**WTFTP 主干的重构式异常检测改造**，不是原论文预测指标的复现，也没有使用其航迹数据。模型见 `models/wtftp_reconstruction.py`，运行入口见 `scripts/wtftp_alfa.py`，均不覆盖 CrossAD。
- 输入 `[B, 96, 18]`，Haar 一级分解；18→32→64 嵌入、4 层 LSTM 编码、两个时间增强注意力/步长 2 卷积分支、各 1 层 LSTM 解码并输出 18 维低频/高频系数。逆 Haar 得到同窗口重构。训练损失为低频 MSE + 高频 MSE；每特征异常分数为原输入与重构的平方误差，逐点分数为 18 维均值。为使长度 96 的输入对应 48 个 Haar 系数，注意力卷积 padding 从参考预测配置的 1 改为 0。输入投影与 LSTM 在嵌入空间混合通道，不是通道独立模型。

## 数据与评估协议

- 使用 `data/processed/alfa_within_flight_v1`：46 个有真值架次，每个源文件的故障前正常段依时间切为 60% 训练、20% 验证、20% 测试正常前缀，再把完整故障尾段加入测试。训练/验证只含正常点。虽然训练、验证、测试各自存为拼接文件，加载后按 `segment_offsets` 还原为每架次独立序列，滑窗不能跨架次。
- 原始 18 特征的 z-score 仅拟合训练正常点；窗口 96 点（20 Hz，4.8 秒），训练步长 1，验证/测试步长 96，补末窗，重叠处逐点分数取均值。最佳模型按正常验证重构损失选择，阈值为正常验证逐点分数的 99% 分位数；测试标签只用于指标、可视化及评价，不参与阈值计算。无 point adjustment。
- 这是**已见架次后续片段的离线窗口检测**。真值异常起点用于划分测试尾段，因此不能据此宣称未知架次泛化或在线报警；完整窗口会利用窗口内未来时刻的信号。指标是测试段逐点汇总（31,180 点，其中异常 15,427 点，占 49.5%），单种子、无置信区间。

## 训练与结果

H100 `ltsm`，seed 2025，Adam 0.001，batch size 128，10 轮学习率减半，最多 150 轮，patience 3。先完成 20 轮快照并保存为 `results/wtftp_alfa/within_flight_w96_seed2025_epoch20/`，再恢复训练至第 70 轮早停，正常验证损失最低的第 67 轮为最终模型。最终训练/验证系数损失分别约为 0.0020 / 0.00669。原版 CrossAD 是同一数据、96 点、种子 2025 的 20 轮结果；**最终 WTFTP 的训练预算更大，不是等轮数比较**。

| 模型 | AUROC | AUPRC | Precision | Recall | F1 | 故障架次报警 | 全正常架次误报 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CrossAD 原版，20 轮 | 0.6248 | 0.5991 | 0.7609 | 0.0576 | 0.1070 | 9/36 | 1/10 |
| WTFTP 重构，20 轮 | 0.7691 | 0.7810 | 0.9127 | 0.2284 | 0.3654 | 29/36 | 4/10 |
| WTFTP 重构，最佳第 67 轮 | **0.7816** | **0.7891** | **0.9029** | **0.3030** | **0.4538** | **31/36** | **5/10** |

最终阈值 `0.01918169`。36 个故障架次中，28 个 AUROC≥0.8，23 个固定阈值 F1≥0.5；29 个逐架次 AUROC 高于原版 CrossAD。发动机类 AUROC 0.910、故障点召回 0.449；舵类 AUROC 0.996、召回 0.996；副翼类 AUROC 0.611、召回仅 0.064。`2018-09-11-17-55-30_2_left_aileron_failure` 的 AUROC 只有 0.075，故障分数反而偏低；全正常 `2018-10-05-15-52-12_1_no_failure` 有 127 个误报点。正常架次中 5/10 发生过误报，改善故障检出的同时牺牲了误报控制。AUPRC 应结合测试异常比例 49.5% 阅读，AUROC 是阈值无关的排序指标，不代表实际已报警。

## 结果与复核

- 最终运行：`results/wtftp_alfa/within_flight_w96_seed2025/`，包含 `best.pt`、`latest.pt`、`config.json`（源码和数据 manifest SHA-256）、`training.csv`、`metrics.json`、`flights.csv`、46 个 `scores/*.npz` 与 46 张 `figures/per-flight/*.png`。
- [测试分数分布与故障类型](../results/wtftp_alfa/within_flight_w96_seed2025/figures/separation.png)，[CrossAD 逐架次对照](../results/wtftp_alfa/within_flight_w96_seed2025/comparison_vs_crossad.png)，[全部逐架次图目录](../results/wtftp_alfa/within_flight_w96_seed2025/figures/per-flight/)。例如[发动机故障检出](../results/wtftp_alfa/within_flight_w96_seed2025/figures/per-flight/2018-10-18-11-04-00_engine_failure_with_emr_traj.png)、[副翼漏报](../results/wtftp_alfa/within_flight_w96_seed2025/figures/per-flight/2018-09-11-17-55-30_2_left_aileron_failure.png)及[正常架次误报](../results/wtftp_alfa/within_flight_w96_seed2025/figures/per-flight/2018-10-05-15-52-12_1_no_failure.png)。图中粉色为真值故障段，橙色为超过阈值的点。
- `tests/test_wtftp_reconstruction.py` 审计 46 个文件的逐点覆盖、真实标签和时间、18 维分数均值、阈值判决、各架次与汇总指标，并在 CUDA 上从最佳模型复算正常验证阈值。CPU 与 CUDA 的 LSTM 推理有微小数值差异，严格相等复核使用原 CUDA 后端。

在远端 `/workspace/code/uavad` 运行：

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/ltsm/bin/python -m scripts.wtftp_alfa train --processed-dir data/processed/alfa_within_flight_v1 --run-dir results/wtftp_alfa/within_flight_w96_seed2025 --window-size 96 --seed 2025 --batch-size 128 --epochs 20 --device cuda:0 --evaluate-after
# 保存整个 20 轮目录快照后，在原目录接训：
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/ltsm/bin/python -m scripts.wtftp_alfa train --processed-dir data/processed/alfa_within_flight_v1 --run-dir results/wtftp_alfa/within_flight_w96_seed2025 --window-size 96 --seed 2025 --batch-size 128 --epochs 150 --device cuda:0 --resume --evaluate-after
/opt/conda/envs/ltsm/bin/python -m scripts.visualize_crossad --run-dir results/wtftp_alfa/within_flight_w96_seed2025
/opt/conda/envs/ltsm/bin/python -m scripts.visualize_within_flight_comparison --baseline results/crossad_alfa/within_flight_crossad_w96_seed2025 --candidate results/wtftp_alfa/within_flight_w96_seed2025 --baseline-label 'Original CrossAD' --candidate-label 'WTFTP reconstruction' --output results/wtftp_alfa/within_flight_w96_seed2025/comparison_vs_crossad.png
/opt/conda/envs/ltsm/bin/python -m pytest -q tests/test_wtftp_reconstruction.py tests/test_alfa_within_flight.py
```
