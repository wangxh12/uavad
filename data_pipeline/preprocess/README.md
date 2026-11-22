# 数据预处理（Preprocess）

用于将原始数据转换为按架次组织的初步处理数据集。

## 1. process_alfa_20hz.py

**功能：** 将 ALFA 原始话题数据按时间戳对齐，生成每个架次一份的 CSV。

- 以 `mavros-nav_info-errors` 的原始时间戳为基准（约 20 Hz）。
- 裁剪共同有效时间范围的前 20 秒。
- 对 18 个特征进行插值，并生成异常标签。
- 排除 `no_ground_truth` 架次。

**运行：** 在项目根目录执行：

```bash
python data_pipeline/preprocess/process_alfa_20hz.py
```

可选参数：`--source`、`--output`、`--trim-seconds`。

**默认路径：**

- 输入：`/mnt/d/datasets/ALFA/processed`
- 输出：`data/processed/alfa_20hz_trim20s_18features/`

示例：
```python
python data_pipeline/preprocess/process_alfa_20hz.py \
    --source /mnt/d/datasets/ALFA/processed \
    --output datasets/ALFA/intermediate/alfa_20hz_trim10s_18features \
    --trim-seconds 10
```

**输出内容：** 每个架次一个 CSV（`time` + 18 个特征 + `label`），以及记录处理参数与架次统计信息的 `manifest.json`。

注意：输出目录必须不存在，脚本不会覆盖已有数据。