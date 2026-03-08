"""
Resample_Per_Second.py

按要求对 eVED 数据按 (VehId, Trip) 分组并按 1 秒窗口重采样。

主要行为（与你的要求一致）：
- 对每个 (VehId, Trip) 取最小时间戳 t0 = min(Timestamp(ms))
- 计算 window_index = floor((Timestamp(ms) - t0) / 1000)
- 每个 (VehId, Trip, window_index) 为一个 1 秒窗口
- 连续数值列采用 mean；标志位采用 max；gps_match_status 按优先级映射取 max 再映回；限速相关取众数或按优先级
- 如果某秒窗口完全没有原始记录，当前实现会跳过（不插入缺失窗口）。代码中有注释说明如何改为插值/填补

用法示例:
    python Resample_Per_Second.py --input input.csv --output output_resampled.csv

依赖: pandas, numpy

作者: 自动生成（含注释以便修改）
"""
import argparse
import glob
import math
import os
from collections import Counter
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# 默认的输入/输出路径与时间列（已写死，方便直接运行）
# 用户指定的输入路径（注意这里按你的要求使用 data/eVE）
input="/lichenghao/lzh/workspace/EnergyPrediction/data/eved-dataset/data/eVE"
output="/lichenghao/lzh/workspace/EnergyPrediction/data/eved-dataset/data/filtered_second"
ts_col = "Timestamp(ms)"

def _safe_mode(series: pd.Series):
    """返回众数，如果有多个众数则返回出现次数最多的一个（取第一个）。"""
    if series.empty:
        return np.nan
    modes = series.mode(dropna=True)
    if modes.empty:
        # 没有非空值，返回 NaN
        return np.nan
    return modes.iloc[0]


def _speed_limit_class_priority(series: pd.Series):
    """
    对 speed_limit_class 采用优先级选择：0 > 1 > 2 > 3 > -1
    如果 series 中存在多个值，则按优先级返回第一个找到的；否则返回众数或第一个非空。
    """
    if series is None or len(series) == 0:
        return np.nan
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    try:
        vals = non_null.astype(int).unique().tolist()
    except Exception:
        # 如果无法转为 int（奇怪的格式），尝试按众数返回
        mode_val = _safe_mode(non_null)
        return int(mode_val) if not pd.isna(mode_val) else np.nan

    priority = [0, 1, 2, 3, -1]
    for p in priority:
        if p in vals:
            return p
    # 否则返回众数或第一个非空
    mode_val = _safe_mode(non_null)
    return int(mode_val) if not pd.isna(mode_val) else np.nan


def _gps_status_agg(series: pd.Series):
    """
    将 gps_match_status 按优先级映射：matched->2, interpolated->1, unmatched->0
    在窗口内取最大值再映回字符串。
    """
    if series.empty:
        return np.nan
    mapping = {"unmatched": 0, "interpolated": 1, "matched": 2}
    inv = {v: k for k, v in mapping.items()}
    nums = series.map(lambda x: mapping.get(str(x).lower(), -1)).dropna()
    if nums.empty:
        return np.nan
    m = int(nums.max())
    return inv.get(m, np.nan)


def process_trip_group(df_trip: pd.DataFrame, timestamp_col: str = "Timestamp(ms)") -> pd.DataFrame:
    """
    对单个 (VehId, Trip) 的 DataFrame 做重采样并返回聚合后的 DataFrame。

    注意：中间缺失的 1 秒窗口（即该秒没有任何原始行）不会被插入；如果需要插入/插值，请在此函数后补充插值逻辑。
    """
    # 确保时间列为数值（毫秒）
    if timestamp_col not in df_trip.columns:
        raise KeyError(f"missing timestamp column: {timestamp_col}")
    df_trip = df_trip.copy()
    df_trip[timestamp_col] = pd.to_numeric(df_trip[timestamp_col], errors="coerce")
    df_trip = df_trip.dropna(subset=[timestamp_col])

    if df_trip.empty:
        return pd.DataFrame()

    t0 = int(df_trip[timestamp_col].min())
    # window_index 为 int
    df_trip["window_index"] = ((df_trip[timestamp_col] - t0) // 1000).astype(int)

    # 预定义列集合（可按需扩展）
    # 连续数值列（如有缺失会自动检测）
    numeric_cols = df_trip.select_dtypes(include=[np.number]).columns.tolist()
    # 排除不应被 mean 的字段
    for ex in ["VehId", "Trip", "window_index", timestamp_col]:
        if ex in numeric_cols:
            numeric_cols.remove(ex)

    # 一些用户明确列出要 mean 的列（优先），其余数值列也会被 mean
    mean_cols_preferred = [
        "Vehicle Speed[km/h]",
        "Engine RPM[RPM]",
        "Fuel Rate[L/hr]",
        "HV Battery Current[A]",
        "HV Battery Voltage[V]",
        "HV Battery SOC[%]",
        "Outside Air Temperature[DegC]",
        "elevation",
        "gradient",
    ]

    # 标志位（0/1）
    flag_cols_known = ["intersection", "bus_stop", "traffic_signal", "crossing", "stop_sign"]
    # 可能存在的 gps 匹配列名
    gps_col_candidates = ["gps_match_status", "GPS_Match_Status", "gps_status"]
    gps_col = next((c for c in gps_col_candidates if c in df_trip.columns), None)

    # 限速字段名检测
    speed_limit_candidates = ["speed_limit", "speed_limit_directional", "Speed Limit with Direction[km/h]", "speed_limit_with_direction"]
    speed_limit_col = next((c for c in speed_limit_candidates if c in df_trip.columns), None)
    speed_limit_class_candidates = ["class_speed_limit", "speed_limit_class", "Class of Speed Limit"]
    speed_limit_class_col = next((c for c in speed_limit_class_candidates if c in df_trip.columns), None)

    # 自动检测实际存在的标志位（优先使用已知列表，然后补充以 bool/0-1 整数列）
    flag_cols: List[str] = [c for c in flag_cols_known if c in df_trip.columns]
    # 另外把只有 0/1 的 numeric 列当作标志位的候选补充（但排除 mean 列）
    for c in df_trip.select_dtypes(include=[np.number]).columns:
        if c in ["VehId", "Trip", "window_index", timestamp_col]:
            continue
        if c in flag_cols:
            continue
        vals = df_trip[c].dropna().unique()
        if len(vals) > 0 and set(np.unique(vals)).issubset({0, 1}):
            # 视为标志位
            if c not in flag_cols:
                flag_cols.append(c)

    # 构造聚合字典
    agg: Dict[str, Any] = {}

    # 连续数值默认 mean（优先包含用户指定的列）
    for c in numeric_cols:
        agg[c] = "mean"
    for c in mean_cols_preferred:
        if c in df_trip.columns:
            agg[c] = "mean"

    # 标志位取 max（逻辑 OR）
    for c in flag_cols:
        agg[c] = "max"

    # gps 采用自定义映射
    if gps_col is not None:
        agg[gps_col] = _gps_status_agg

    # speed_limit 取众数
    if speed_limit_col is not None:
        agg[speed_limit_col] = lambda s: _safe_mode(s)

    # speed_limit_class 采用优先级
    if speed_limit_class_col is not None:
        agg[speed_limit_class_col] = lambda s: _speed_limit_class_priority(s)

    # 其他非数值、非已定义列：取第一个（保留示例行的非聚合信息）
    for c in df_trip.columns:
        if c in ["VehId", "Trip", "window_index", timestamp_col]:
            continue
        if c in agg:
            continue
        # 避免覆盖已经设置了 agg 的列
        if c not in agg:
            # 对于非数值列，默认取 first
            if not pd.api.types.is_numeric_dtype(df_trip[c].dtype):
                agg[c] = lambda s: s.iloc[0]

    # 执行聚合
    grouped = df_trip.groupby("window_index", sort=True).agg(agg).reset_index()

    # 补回 VehId 和 Trip 列
    if "VehId" in df_trip.columns:
        grouped["VehId"] = df_trip["VehId"].iat[0]
    if "Trip" in df_trip.columns:
        grouped["Trip"] = df_trip["Trip"].iat[0]

    # 生成 SecTime（毫秒）与 SecTime_dt（datetime）
    grouped["SecTime_ms"] = t0 + grouped["window_index"].astype(int) * 1000
    grouped["SecTime"] = pd.to_datetime(grouped["SecTime_ms"], unit="ms")

    # 将列顺序调整为：VehId, Trip, window_index, SecTime_ms, SecTime, ...
    cols_order = [c for c in ["VehId", "Trip", "window_index", "SecTime_ms", "SecTime"] if c in grouped.columns]
    rest = [c for c in grouped.columns if c not in cols_order]
    grouped = grouped[cols_order + rest]

    return grouped


def resample_dataframe(df: pd.DataFrame, timestamp_col: str = "Timestamp(ms)") -> pd.DataFrame:
    """
    对整张表按 VehId, Trip 分组，并对每个 trip 调用 process_trip_group，合并返回最终结果。
    中间缺失窗口不会被插入（按你的要求）。
    """
    out_frames = []
    # 确保分组键存在
    if "VehId" not in df.columns or "Trip" not in df.columns:
        raise KeyError("输入数据必须包含 VehId 和 Trip 列")

    # 逐 trip 处理（便于每个 trip 使用自己的 t0）
    grouped = df.groupby(["VehId", "Trip"] , sort=True)
    for (veh, trip), grp in grouped:
        res = process_trip_group(grp, timestamp_col=timestamp_col)
        if not res.empty:
            out_frames.append(res)

    if out_frames:
        result = pd.concat(out_frames, ignore_index=True)
    else:
        result = pd.DataFrame()

    return result


def batch_process_input_dir(input_dir: str, output_root: str, timestamp_col: str = "Timestamp(ms)"):
    """
    遍历 input_dir 下的所有 CSV 文件（不递归），对每个文件按 (VehId, Trip) 分组并把每个 trip 的重采样结果写到
    output_root/{category}/{VehId}/{Trip}/{VehId}_{Trip}_resampled.csv

    category: 根据 VehId 分类为 EV 或 PHEV（脚本内预定义列表），如果未命中则放到 Unknown 文件夹
    """
    # 车辆分类列表（与仓库中其他脚本一致）
    EVs_ids = [10, 455, 541]
    PHEVs_ids = [9, 11, 371, 379, 388, 398, 417, 431, 443, 449, 453, 457, 492, 497, 536, 537, 542, 545, 550, 554, 560, 561, 567, 569]

    os.makedirs(output_root, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV files in {input_dir}, processing...")
    for file_path in csv_files:
        print(f"Processing file: {file_path}")
        # 使用 low_memory=False 减少 dtype 警告
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            continue

        if df.empty:
            print(f"Skipping empty file: {file_path}")
            continue

        if "VehId" not in df.columns or "Trip" not in df.columns:
            print(f"File {file_path} missing VehId or Trip column, skipping")
            continue

        # 按 vehicle/trip 分组并对每个 trip 写入单独文件
        grouped = df.groupby(["VehId", "Trip"], sort=True)
        for (veh, trip), grp in grouped:
            try:
                out_df = process_trip_group(grp, timestamp_col=timestamp_col)
            except Exception as e:
                print(f"Error processing VehId={veh}, Trip={trip} in {file_path}: {e}")
                continue

            if out_df.empty:
                # 如果该 trip 全部被过滤掉或无有效时间戳则跳过
                continue

            # 分类
            category = "Unknown"
            try:
                veh_int = int(veh)
                if veh_int in EVs_ids:
                    category = "EV"
                elif veh_int in PHEVs_ids:
                    category = "PHEV"
            except Exception:
                # 若 VehId 不是整型则保留 Unknown
                pass

            # 不再为每个 Trip 创建子目录，只在 VehId 目录下写入以 Trip 编号命名的 CSV 文件
            veh_folder = str(veh_int if isinstance(veh, (int, np.integer)) else veh)
            out_dir = os.path.join(output_root, category, veh_folder)
            os.makedirs(out_dir, exist_ok=True)
            # 文件名只用 Trip 编号（按需求）。若冲突（不同源文件含相同 VehId+Trip）会被覆盖。
            out_file = os.path.join(out_dir, f"{trip}.csv")
            try:
                out_df.to_csv(out_file, index=False)
            except Exception as e:
                print(f"Failed to write {out_file}: {e}")
                continue

    print("Batch processing complete.")


def split_then_resample_by_vehicle(input_dir: str, output_root: str, timestamp_col: str = "Timestamp(ms)"):
    """
    参考 Filter_VehId.py 的做法：
    - 先按 category (EV/PHEV) 与 VehId 列表遍历所有输入 CSV，把同一 VehId 的行按 Trip 合并（跨文件合并）
    - 对每个合并后的 Trip 调用 process_trip_group 做 per-second 重采样
    - 输出到 output_root/{category}/{VehId}/{Trip}.csv （文件名为 Trip 编号）

    注意：若某个 Trip 在所有输入文件里都没有记录，会被跳过；中间缺失的 1 秒窗口不会被插入（可后续改为插值）。
    """
    # 使用与 Filter_VehId.py 一致的车辆分类列表
    EVs_ids = [10, 455, 541]
    PHEVs_ids = [9, 11, 371, 379, 388, 398, 417, 431, 443, 449, 453, 457, 492, 497, 536, 537, 542, 545, 550, 554, 560, 561, 567, 569]
    categories = {
        "EV": EVs_ids,
        "PHEV": PHEVs_ids,
    }

    os.makedirs(output_root, exist_ok=True)

    # 列出输入目录下的 CSV 文件
    csv_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.csv')])
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV files in {input_dir}. Starting split+resample by vehicle...")

    for category, veh_ids in categories.items():
        for veh_id in veh_ids:
            trip_data = {}
            print(f"Gathering rows for category={category}, VehId={veh_id} ...")
            for file_path in csv_files:
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
                    continue

                if "VehId" not in df.columns or "Trip" not in df.columns:
                    # 如果文件缺少关键列，跳过
                    continue

                # 选出该车辆的所有行
                filtered_df = df[df["VehId"] == veh_id]
                if filtered_df.empty:
                    continue

                # 按 Trip 聚合到 trip_data（跨文件合并）
                for trip_id, group in filtered_df.groupby("Trip"):
                    trip_data.setdefault(trip_id, []).append(group)

            # 对每个 trip 合并并重采样
            veh_folder = os.path.join(output_root, category, str(veh_id))
            os.makedirs(veh_folder, exist_ok=True)

            if not trip_data:
                print(f"No rows found for VehId={veh_id} (category={category}).")
                continue

            print(f"Processing {len(trip_data)} trips for VehId={veh_id} ...")
            for trip_id, dfs in trip_data.items():
                try:
                    combined = pd.concat(dfs, ignore_index=True)
                except Exception as e:
                    print(f"Failed to concat trip {trip_id} for VehId={veh_id}: {e}")
                    continue

                try:
                    out_df = process_trip_group(combined, timestamp_col=timestamp_col)
                except Exception as e:
                    print(f"Error processing VehId={veh_id}, Trip={trip_id}: {e}")
                    continue

                if out_df.empty:
                    continue

                out_file = os.path.join(veh_folder, f"{trip_id}.csv")
                try:
                    out_df.to_csv(out_file, index=False)
                except Exception as e:
                    print(f"Failed to write {out_file}: {e}")
                    continue

            print(f"Completed category={category}, VehId={veh_id}.")

    print("All vehicles processed.")


def main():
    parser = argparse.ArgumentParser(description="Per-second resampling for eVED CSV data")
    parser.add_argument("--input", help="输入 CSV 文件路径，例如 input.csv，和 --input-dir 二选一")
    parser.add_argument("--input-dir", help="输入 CSV 文件夹路径，脚本会对该文件夹下所有 CSV 批量处理（非递归）")
    parser.add_argument("--output", help="当使用 --input 时写出单个输出 CSV 的路径例如 output_resampled.csv")
    parser.add_argument("--output-root", help="当使用 --input-dir 时写出到的根目录，例如 ./data/.../filtered_second/")
    parser.add_argument("--ts-col", default="Timestamp(ms)", help="时间列名（默认 Timestamp(ms)）")
    args = parser.parse_args()
    if args.input and args.input_dir:
        raise ValueError("请只指定 --input 或 --input-dir 中的一个")

    if args.input_dir:
        out_root = args.output_root or os.path.join(os.path.dirname(args.input_dir), "filtered_second")
        print(f"Batch processing directory: {args.input_dir} -> {out_root}")
        batch_process_input_dir(args.input_dir, out_root, timestamp_col=args.ts_col)
        return

    if args.input:
        if not args.output:
            raise ValueError("使用 --input 时必须指定 --output")
        print(f"Reading {args.input} ...")
        df = pd.read_csv(args.input, low_memory=False)
        print(f"Rows loaded: {len(df)}")

        print("开始重采样...（按 VehId, Trip，每个 trip 使用自身 t0）")
        out = resample_dataframe(df, timestamp_col=args.ts_col)
        print(f"重采样后总行数: {len(out)}")

        print(f"写出到 {args.output}")
        out.to_csv(args.output, index=False)
        print("完成。说明：脚本当前不会填补缺失的 1 秒窗口，若需插值请修改 process_trip_group。")
        return

    parser.print_help()


if __name__ == "__main__":
    # 直接使用脚本顶部写死的 input/output/ts_col 运行批量处理，避免每次在控制台传参。
    # 如果需要用 CLI，可以在 main() 中启动；当前行为为默认自动运行。
    try:
        # 自动检测输入目录是否存在；若不存在，尝试常见变体（eVED vs eVE）以提高鲁棒性
        input_path = input
        if not os.path.isdir(input_path):
            alt = input_path.replace('/data/eVE', '/data/eVED')
            if alt != input_path and os.path.isdir(alt):
                print(f"Input path {input_path} not found, using fallback {alt}")
                input_path = alt
            else:
                raise FileNotFoundError(f"Input directory not found: {input_path}")

        # 使用先 split 再 resample 的流程，参考 Filter_VehId 的划分方式
        split_then_resample_by_vehicle(input_path, output, timestamp_col=ts_col)
    except Exception as e:
        print(f"Processing failed: {e}")
