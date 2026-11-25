# ALFA 预测式异常注入（按飞行架次划分）

从仓库根目录运行：

```bash
python -m data_pipeline.injection.inject_alfa \
  --source-dir datasets/ALFA/intermediate/alfa_20hz_trim10s_18features \
  --output-dir datasets/ALFA/forecast/roll_bias_v1 \
  --seed 2024 --bias 8
```

筛选 `*_no_failure.csv`（预期 10 个），固定随机种子按 **6/2/2 架次**划分 Train/Val/Test。Train、Val 不注入；Test 在约 40%～70% 的位置开始，对 `field.measured.roll` 施加持续到飞行结束的 **+8 原始单位偏置**（每个测试架次至少保留 20 个异常点，与模型窗口参数无关），并将逐点 `label` 标为 1。`time` 和其余 17 个特征保持不变。不拼接架次、不提前切窗、不改动源目录。

输出目录如下（保留各架次的原始文件名）：

```text
datasets/ALFA/forecast/roll_bias_v1/
├── train/*.csv
├── val/*.csv
├── test/*.csv
└── manifest.json
```

`manifest.json` 记录原始文件名、集合、行数、异常起止索引（左闭右开）、异常比例、时间和随机种子。若输出目录已存在，脚本会报错以避免误覆盖；重新生成时请指定新目录或先自行移走旧目录。

## 预测式 DataLoader

```python
from data_provider.alfa_forecast_loader import make_alfa_loader

root = "datasets/ALFA/forecast/roll_bias_v1"
train_ds, train_loader = make_alfa_loader(root, "train", seq_len=100, pred_len=1)
val_ds, val_loader = make_alfa_loader(root, "val", seq_len=100, pred_len=1)
test_ds, test_loader = make_alfa_loader(root, "test", seq_len=100, pred_len=1)

seq_x, lab_x, seq_y, lab_y = next(iter(test_loader))
print(seq_x.shape, lab_x.shape, seq_y.shape, lab_y.shape)
# [B,100,18] [B,100,1] [B,1,18] [B,1,1]
```

只有 DataLoader 接受 `seq_len`、`pred_len`、`stride`，注入器不接收这些参数。更改窗口参数无需重新生成 CSV，只需要重新构建 DataLoader；但如果窗口长于架次可形成的样本长度，会报错。

与 TimeRadar 的 `TrainSegLoaderAddPre` 一样，返回 `(seq_x, lab_x, seq_y, lab_y)`。不同之处是我们直接使用已经按飞行架次划分的 CSV；每个飞行文件独立切窗，绝不跨飞行边界。`time` 与 `label` 不参与预测。`StandardScaler` 只使用 Train 中的正常观测点拟合，Val/Test 使用同一组参数。

`dataset.window_info(index)` 可取得目标区间所属飞行、原始点索引与时间戳，供之后将预测误差聚合成时间点级异常分数。对于 `pred_len > 1`，同一时间点可能被多次预测，评估前需定义聚合规则。

**实验注意**：Val 全正常，因此适合早停或校准正常误差分布，但不能用来直接调优异常 F1 阈值。Test 的异常点占比会随注入位置变化；报告 PR-AUC 时应同时报告异常占比。不要以 Test 标签选阈值后再把同一 Test F1 当作无偏结果。

**评估提醒**：改变窗口参数会改变可被预测覆盖的时间点，因此可能改变测试集预测目标中的正常/异常点数量。如果模型窗口太长，预测目标可能仅有异常点，无法计算 ROC-AUC/PR-AUC；评估前应统计逐点标签覆盖情况。这是评分范围的限制，而不是数据注入阶段的参数依赖。