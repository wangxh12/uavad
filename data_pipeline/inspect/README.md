
inspect_normal_flights.py
统计给定目录的正常架次数据集信息

## 1. inspect_normal_flights.py

**功能：** 统计给定目录的正常架次数据集信息。

**运行：** 在项目根目录执行：

```bash
python data_pipeline/inspect/inspect_normal_flights.py
```

可选参数：`--data-dir`

**默认路径：**

- data-dir：`CSV 所在文件夹（默认当前目`

示例：
```python
python data_pipeline/inspect/inspect_normal_flights.py \
    --data-dir datasets/ALFA/intermediate/alfa_20hz_trim10s_18features
```


## plot_injected_anomaly.py

用于可视化人工注入异常前后的测试集差异。

功能：
1. 对比未注入异常和已注入异常的同一特征曲线；
2. 用阴影标记异常区间；
3. 可选显示各架次在拼接测试集中的边界；
4. 输出全局图和异常局部放大图。

运行示例：

```bash
python data_pipeline/inspect/plot_injected_anomaly.py \
    --clean-csv datasets/ALFA/forecast/bias_roll_normal/test.csv \
    --anomaly-csv datasets/ALFA/forecast/bias_roll_1000_8/test.csv \
    --output-dir datasets/ALFA/forecast/bias_roll_1000_8/figures \
    --feature field.measured.roll \
    --boundaries-csv datasets/ALFA/forecast/bias_roll_1000_8/boundaries.csv
```