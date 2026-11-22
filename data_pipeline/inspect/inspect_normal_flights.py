#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计 ALFA 已预处理的 *_no_failure.csv 的数据点数和字段信息。"""

import argparse
from pathlib import Path

import pandas as pd


def inspect_one(csv_path: Path, chunk_size: int):
    # 只读表头，保留原始字段名和原始顺序
    columns = pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig").columns.tolist()
    if not columns:
        raise ValueError(f"CSV 没有字段：{csv_path}")

    # 分块读取一列来统计有效数据行，避免加载全部特征到内存
    num_rows = sum(
        len(chunk)
        for chunk in pd.read_csv(
            csv_path,
            usecols=[columns[0]],
            chunksize=chunk_size,
            encoding="utf-8-sig",
        )
    )
    return {
        "flight": csv_path.stem,
        "file": csv_path.name,
        "rows": num_rows,
        "columns": len(columns),
        "fields": columns,
    }


def main():
    parser = argparse.ArgumentParser(description="统计 ALFA 正常飞行 CSV 的数据点和字段")
    parser.add_argument("--data-dir", type=Path, default=Path.cwd(),
                        help="CSV 所在文件夹（默认当前目录）")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="报告保存目录（默认与 CSV 同目录）")
    parser.add_argument("--chunk-size", type=int, default=100_000,
                        help="分块读取行数（默认 100000）")
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size 必须大于 0")

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = (args.output_dir or data_dir).expanduser().resolve()
    files = sorted(data_dir.glob("*_no_failure.csv"))
    if not files:
        parser.error(f"未在 {data_dir} 找到 *_no_failure.csv 文件")

    results = []
    for file in files:
        try:
            results.append(inspect_one(file, args.chunk_size))
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            print(f"[跳过] {file.name}: {exc}")

    if not results:
        parser.error("没有成功读取的 CSV 文件")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "normal_flights_summary.csv"
    field_csv_path = output_dir / "normal_flights_fields.csv"
    report_path = output_dir / "normal_flights_report.txt"

    # pd.DataFrame([
    #     {"flight": r["flight"], "file": r["file"],
    #      "num_points": r["rows"], "num_fields": r["columns"]}
    #     for r in results
    # ]).to_csv(summary_path, index=False, encoding="utf-8-sig")

    # pd.DataFrame([
    #     {"flight": r["flight"], "field_order": i, "field_name": field}
    #     for r in results for i, field in enumerate(r["fields"], start=1)
    # ]).to_csv(field_csv_path, index=False, encoding="utf-8-sig")

    reference = results[0]["fields"]
    same_fields = all(r["fields"] == reference for r in results)
    common_fields = set.intersection(*(set(r["fields"]) for r in results))
    union_fields = set.union(*(set(r["fields"]) for r in results))
    total_points = sum(r["rows"] for r in results)

    lines = [
        "ALFA 正常架次 CSV 统计报告",
        f"数据目录: {data_dir}",
        f"成功读取的正常架次: {len(results)} (预期 10)",
        f"各架次数据点合计: {total_points:,}",
        f"所有架次字段及顺序完全一致: {'是' if same_fields else '否'}",
        f"共有字段数量（交集）: {len(common_fields)}",
        f"出现过的字段数量（并集）: {len(union_fields)}",
        "",
        "=== 每个架次：数据点数与字段列表 ===",
    ]
    for r in results:
        lines += [
            "",
            f"文件: {r['file']}",
            f"数据点数: {r['rows']:,}",
            f"字段数量: {r['columns']}",
            "字段（原始顺序）:",
        ]
        lines += [f"  {i:02d}. {field}" for i, field in enumerate(r["fields"], start=1)]
        if r["fields"] != reference:
            missing = [f for f in reference if f not in r["fields"]]
            extra = [f for f in r["fields"] if f not in reference]
            lines += [f"相对首个架次缺少字段: {missing}",
                      f"相对首个架次额外字段: {extra}"]
            if not missing and not extra:
                lines.append("提示: 字段集合相同，但字段顺序不同")
    lines += ["", "=== 所有架次共有的字段（按首个架次顺序） ==="]
    lines += [f"  {field}" for field in reference if field in common_fields]
    # report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"正常架次数量: {len(results)} (预期 10)")
    print(f"所有正常架次合计数据点数: {total_points:,}")
    print(f"字段及顺序完全一致: {'是' if same_fields else '否'}")
    print("\n{:<67} {:>12} {:>10}".format("文件名", "数据点数", "字段数"))
    for r in results:
        print("{:<67} {:>12,} {:>10}".format(r["file"], r["rows"], r["columns"]))
    print(f"\n共有字段: {len(common_fields)} 个；出现过的字段: {len(union_fields)} 个")
    if same_fields:
        print("\n全部字段（各架次相同）：")
        for i, col in enumerate(reference, 1):
            print(f"  {i:02d}. {col}")
    else:
        print("各架次字段存在差异，请查看完整报告。")
    # print("\n已保存:")
    # print(f"  {summary_path}")
    # print(f"  {field_csv_path}")
    # print(f"  {report_path}")
    # if len(results) != 10:
    #     print("[注意] 读取到的正常架次数不是 10，请检查文件命名及读取错误信息。")


if __name__ == "__main__":
    main()
