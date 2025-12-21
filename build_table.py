import pandas as pd
import glob
import os
from tabulate import tabulate


def build_summary_table(input_dir, output_csv, ours_method="DMMA", dataset_names=None):
    """
    input_dir: directory containing multiple tta-method CSV files
    output_csv: final merged pivot-table csv path
    ours_method: 保留该参数以兼容调用，但不再仅对其着色
    dataset_names: 如果不为 None，则只显示这些 dataset_name（支持单个字符串或列表）
    """

    # ---- 1. 读取所有 TTA-method 的 csv 文件 ----
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    df_list = []

    for f in csv_files:
        tta_method = os.path.splitext(os.path.basename(f))[0]
        tta_method = tta_method.split("_")[0]
        tmp = pd.read_csv(f)

        rename_map = {
            "pre_len": "pred_len",
            "mse_after_tta": "mse"
        }
        tmp = tmp.rename(columns=rename_map)

        tmp["tta_method"] = tta_method
        df_list.append(tmp)

    df = pd.concat(df_list, ignore_index=True)

    if dataset_names is not None:
        if "dataset_name" not in df.columns:
            print("No 'dataset_name' column found in input CSVs.")
            return
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        df = df[df["dataset_name"].isin(dataset_names)]
        if df.empty:
            print(f"No rows found for dataset_names = {dataset_names}.")
            return

    # ---- 2. 对同组 seed 平均 ----
    df = (
        df.groupby(["model", "dataset_name", "pred_len", "tta_method"], as_index=False)
          .agg({"mse": "mean"})
    )

    # ---- 3. pivot ----
    table = df.pivot_table(
        index=["dataset_name", "pred_len"],
        columns=["model", "tta_method"],
        values="mse"
    )

    table = table.sort_index()
    table = table.sort_index(axis=1)

    # ---- 4. 保存原始 CSV ----
    table.to_csv(output_csv)
    print(f"Saved → {output_csv}")

    # ---- 5. 创建颜色版副本用于打印 ----
    colored_table = table.copy().astype(object)

    # ---- 6. 统计每个 method 的 top-1 次数 ----
    top1_count = {m: 0 for m in df["tta_method"].unique()}

    # 颜色代码
    RED = "\033[91m"
    BLUE = "\033[94m"
    END = "\033[0m"

    # 遍历每个 (dataset, pred_len)
    for idx in table.index:
        # 每个 model 单独一组
        for model_name in table.columns.levels[0]:
            sub_cols = [col for col in table.columns if col[0] == model_name]
            if not sub_cols:
                continue

            row_values = table.loc[idx, sub_cols]
            # 忽略 NaN，获取升序的唯一值以确定 top1/top2
            vals = row_values.dropna().astype(float).values
            if vals.size == 0:
                continue
            unique_sorted = sorted(set(vals))
            top1_value = unique_sorted[0]
            top2_value = unique_sorted[1] if len(unique_sorted) > 1 else None

            # 找到所有并列第一和第二的 method，并着色
            for (mod, method_name), val in row_values.items():
                if pd.isna(val):
                    continue

                if val == top1_value:
                    # top-1 标红
                    colored_table.loc[idx, (mod, method_name)] = f"{RED}{val}{END}"
                    top1_count[method_name] += 1
                elif top2_value is not None and val == top2_value:
                    # top-2 标蓝
                    colored_table.loc[idx, (mod, method_name)] = f"{BLUE}{val}{END}"
                else:
                    # 保持原始数值（作为字符串以便 tabulate 正常显示）
                    colored_table.loc[idx, (mod, method_name)] = val

    # ---- 7. 构造 Top-1 次数表并按 model 拆分打印 ----
    top1_series = pd.Series(top1_count, name="Top1_Count")

    # 按 model 拆分小表打印，避免表过宽
    for model_name in table.columns.levels[0]:
        sub_cols = [col for col in colored_table.columns if col[0] == model_name]
        if not sub_cols:
            continue
        sub_df = colored_table.loc[:, sub_cols].copy()
        # 将 MultiIndex 的列扁平化为 method 名称，便于显示
        sub_df.columns = [method for (_, method) in sub_df.columns]
        # 将索引转为列（dataset_name, pred_len）
        display_df = sub_df.reset_index()
        print(f"\n=== Model: {model_name} ===")
        print(tabulate(display_df, headers="keys", tablefmt="github"))

    # 打印 Top-1 统计（整体）
    print("\n=== Top-1 Counts (per method) ===")
    print(tabulate(top1_series.to_frame(), headers="keys", tablefmt="github"))


# 示例：
build_summary_table("./results", "./results/final_tta_summary.csv", ours_method="DMMA", dataset_names=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "exchange_rate", "weather"])
