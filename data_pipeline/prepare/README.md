

# 数据集构建

## build_alfa_bias_dataset.py

从 ALFA 预处理数据中提取正常架次，构建带有人工偏移异常的实验数据集。

**处理流程：**

1. 读取所有 `*_no_failure.csv` 文件。
2. 每个架次按时间顺序以 6:2:2 划分训练集、验证集和测试集。
3. 分别拼接得到 `train.csv`、`val.csv` 和 `test.csv`。
4. 在 `test.csv` 最后 N 个点的指定特征上注入加性偏移异常（`x' = x + bias`），对应标签设为 1。
5. 生成 `boundaries.csv`，记录各架次在拼接数据中的边界。

**运行示例（项目根目录）：**

```bash
python data_pipeline/prepare/build_alfa_bias_dataset.py \
    --input-dir datasets/ALFA/intermediate/alfa_20hz_trim10s_18features \
    --output-dir datasets/ALFA/forecast/bias_roll_1000_8 \
    --anomaly-length 1000 \
    --feature field.measured.roll \
    --bias 8
```

**参数说明：**

| 参数 | 说明 |
|---|---|
| `--input-dir` | 输入数据目录（必填） |
| `--output-dir` | 输出数据目录（必填） |
| `--anomaly-length` | 注入异常的点数，默认 1000 |
| `--feature` | 注入异常的特征，默认 `field.measured.roll` |
| `--bias` | 加性偏移量，默认 8 |

**输出文件：** `train.csv`、`val.csv`、`test.csv`、`boundaries.csv`。

注意：异常注入发生在**拼接后的测试集尾部**，可能跨越多个架次。后续滑动窗口需根据 `boundaries.csv` 避免跨架次取样。
