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

# --- 修改点 1: 在函数定义中增加了 model_names 参数 ---
def build_summary_table(input_dir, output_csv, dataset_names=None, model_names=None):
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    df_list = []

    for f in csv_files:
        # 文件名假设: MethodName_Rest.csv
        tta_method = os.path.splitext(os.path.basename(f))[0]
        tta_method = tta_method.split("_")[0]
        tmp = pd.read_csv(f)

        rename_map = {
            "pre_len": "pred_len",
            "mse_after_tta": "mse"
        }
        tmp = tmp.rename(columns=rename_map)
        
        if "mse" not in tmp.columns:
            continue

        tmp["tta_method"] = tta_method
        df_list.append(tmp)

    if not df_list:
        print("No valid data found.")
        return

    df = pd.concat(df_list, ignore_index=True)

    # --- 原有的 dataset_names 过滤 ---
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

    # --- 修改点 2: 新增 model_names 过滤逻辑 ---
    if model_names is not None:
        if "model" not in df.columns:
            print("No 'model' column found in input CSVs.")
            return
        # 允许传入单个字符串或列表
        if isinstance(model_names, str):
            model_names = [model_names]
        
        df = df[df["model"].isin(model_names)]
        
        if df.empty:
            print(f"No rows found for model_names = {model_names}.")
            return

    # 聚合
    df = (
        df.groupby(["model", "dataset_name", "pred_len", "tta_method"], as_index=False)
          .agg({"mse": "mean"})
    )
    
    # 建立映射
    all_methods = sorted([m for m in df["tta_method"].unique() if m != "None"])
    method_mapping = {m: excel_colname(i) for i, m in enumerate(all_methods)}
    method_mapping["None"] = "None"
    
    reverse_mapping = {v: k for k, v in method_mapping.items() if v != "None"}

    df["tta_method"] = df["tta_method"].map(method_mapping)

    # 透视
    table = df.pivot_table(
        index=["dataset_name", "pred_len"],
        columns=["model", "tta_method"],
        values="mse"
    )

    table = table.sort_index()
    table = table.sort_index(axis=1)

    table.to_csv(output_csv)
    print(f"Saved raw summary → {output_csv}")

    # --- 着色逻辑 (Top1:红, Top2:蓝, 比None差:绿) ---
    colored_table = table.copy().astype(object)
    top1_count = {m: 0 for m in df["tta_method"].unique()}

    RED = "\033[91m"
    BLUE = "\033[94m"
    GREEN = "\033[92m" 
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
            
            # 获取 None 基准值
            none_val = None
            if (model_name, "None") in row_values.index:
                none_val = row_values[(model_name, "None")]
            
            unique_sorted = sorted(set(vals))
            top1_value = unique_sorted[0]
            top2_value = unique_sorted[1] if len(unique_sorted) > 1 else None

            for (mod, method_name), val in row_values.items():
                if pd.isna(val):
                    continue

                val_str = f"{val:.4f}"
                formatted_val = val_str
                is_colored = False
                
                if val == top1_value:
                    formatted_val = f"{RED}{val_str}{END}"
                    top1_count[method_name] += 1
                    is_colored = True
                elif top2_value is not None and val == top2_value:
                    formatted_val = f"{BLUE}{val_str}{END}"
                    is_colored = True
                
                if not is_colored and none_val is not None and method_name != "None":
                    if val > none_val:
                        formatted_val = f"{GREEN}{val_str}{END}"
                
                colored_table.loc[idx, (mod, method_name)] = formatted_val

    # --- 打印 ---
    # 这里的循环会自动只打印剩余的 model (因为前面已经 filter 过了)
    for model_name in table.columns.levels[0]:
        sub_cols = [col for col in colored_table.columns if col[0] == model_name]
        if not sub_cols:
            continue
        
        sub_df = colored_table.loc[:, sub_cols].copy()
        sub_df.columns = [method for (_, method) in sub_df.columns]
        
        display_df = sub_df.reset_index()
        
        print(f"\n{'='*20}")
        print(f" Model: {model_name} ")
        print(f"{'='*20}")
        print(tabulate(display_df, headers="keys", tablefmt="simple", stralign="right"))

    print("\n\n=== Method Mapping (Legend) ===")
    legend_data = [[key, val] for key, val in reverse_mapping.items()]
    print(tabulate(legend_data, headers=["ID", "Original Method Name"], tablefmt="simple"))

    print("\n=== Top-1 Counts (Lower MSE is better) ===")
    top1_df = pd.DataFrame(list(top1_count.items()), columns=["Method ID", "Count"])
    top1_df = top1_df.sort_values(by="Count", ascending=False)
    print(tabulate(top1_df, headers="keys", tablefmt="simple"))

if __name__ == "__main__":
    if not os.path.exists("./results"):
        os.makedirs("./results", exist_ok=True)
        
    build_summary_table(
        input_dir="./results", 
        output_csv="./results/final_tta_summary.csv", 
        
        # 1. 筛选数据集
        # dataset_names=["ETTh1", "ETTh2", "ETTm1", "weather"],
        dataset_names=["ETTh1_2_ETTh2"],
        
        # 2. 筛选模型 (可以是列表，也可以是单个字符串)
        # model_names=["DLinear",] 
        # model_names="DLinear"
        model_names=None # 设为 None 则显示所有模型
    )