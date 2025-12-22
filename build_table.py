import pandas as pd
import glob
import os
from tabulate import tabulate
import string

def excel_colname(n):
    s = ''
    while n >= 0:
        s = chr(n % 26 + ord('A')) + s
        n = n // 26 - 1
    return s

def build_summary_table(input_dir, output_csv, dataset_names=None):
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

    df = (
        df.groupby(["model", "dataset_name", "pred_len", "tta_method"], as_index=False)
          .agg({"mse": "mean"})
    )
    all_methods = sorted([m for m in df["tta_method"].unique() if m != "None"])
        
    method_mapping = {m: excel_colname(i) for i, m in enumerate(all_methods)}
    method_mapping["None"] = "None" # 保持 None 原样
    # 反向映射用于最后打印对照表
    reverse_mapping = {v: k for k, v in method_mapping.items() if v != "None"}

    # 将 dataframe 里的名称全部替换为代号
    df["tta_method"] = df["tta_method"].map(method_mapping)

    # ----------------------------------
    table = df.pivot_table(
        index=["dataset_name", "pred_len"],
        columns=["model", "tta_method"],
        values="mse"
    )

    table = table.sort_index()
    table = table.sort_index(axis=1)

    table.to_csv(output_csv)
    print(f"Saved → {output_csv}")

    colored_table = table.copy().astype(object)

    top1_count = {m: 0 for m in df["tta_method"].unique()}

    # 颜色定义
    RED = "\033[91m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"  # 新增绿色
    END = "\033[0m"

    for idx in table.index:
        for model_name in table.columns.levels[0]:
            sub_cols = [col for col in table.columns if col[0] == model_name]
            if not sub_cols:
                continue

            row_values = table.loc[idx, sub_cols]
            vals = row_values.dropna().astype(float).values
            if vals.size == 0:
                continue
            
            # --- 需求实现核心逻辑 ---
            # 1. 获取该模型下的 None 方法的值作为基准
            none_val = None
            if (model_name, "None") in row_values.index:
                none_val = row_values[(model_name, "None")]
            
            # 2. 计算 Top1 和 Top2
            unique_sorted = sorted(set(vals))
            top1_value = unique_sorted[0]
            top2_value = unique_sorted[1] if len(unique_sorted) > 1 else None

            # 3. 着色
            for (mod, method_name), val in row_values.items():
                if pd.isna(val):
                    continue

                formatted_val = val
                
                # 逻辑：Top1 红 > Top2 蓝 > 比 None 差 绿
                if val == top1_value:
                    formatted_val = f"{RED}{val}{END}"
                    top1_count[method_name] += 1
                elif top2_value is not None and val == top2_value:
                    formatted_val = f"{BLUE}{val}{END}"
                # 如果比 None 方法的值大（且当前不是 None 自身），标绿
                elif none_val is not None and method_name != "None" and val > none_val:
                    formatted_val = f"{GREEN}{val}{END}"
                
                colored_table.loc[idx, (mod, method_name)] = formatted_val

    top1_series = pd.Series(top1_count, name="Top1_Count")

    for model_name in table.columns.levels[0]:
        sub_cols = [col for col in colored_table.columns if col[0] == model_name]
        if not sub_cols:
            continue
        sub_df = colored_table.loc[:, sub_cols].copy()
        sub_df.columns = [method for (_, method) in sub_df.columns]
        display_df = sub_df.reset_index()
        print(f"\n=== Model: {model_name} ===")
        print(tabulate(display_df, headers="keys", tablefmt="github"))

    # --- 新增：打印对照表 ---
    print("\n=== Method Mapping (Legend) ===")
    legend_data = [[key, val] for key, val in reverse_mapping.items()]
    print(tabulate(legend_data, headers=["ID", "Original Method Name"], tablefmt="github"))

    print("\n=== Top-1 Counts (per method ID) ===")
    # 注意：这里的 top1_series 现在的 index 已经是 ID 了
    print(tabulate(top1_series.to_frame(), headers="keys", tablefmt="github"))
# 运行
if __name__ == "__main__":
    build_summary_table("./results", "./results/final_tta_summary.csv", 
                        dataset_names=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "exchange_rate", "weather"])