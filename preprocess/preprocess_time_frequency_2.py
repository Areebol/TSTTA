"""
Resample_Full_Pipeline.py

完整可运行脚本：按 EngineType -> VehId -> Trip 划分，并在每个 Trip 内以 1000ms 窗口重采样（1s），
按用户规则聚合字段并输出 CSV。

目录与命名约定（输出）
  output_root/
    EV/
      Veh_<VehId>/
        Trip_<TripId>_resampled_1s.csv
    PHEV/
      Veh_<VehId>/
        Trip_<TripId>_resampled_1s.csv

脚本中已把输入/输出路径写死（按你的要求）：
  INPUT_DIR = "/lichenghao/lzh/workspace/EnergyPrediction/data/eved-dataset/data/eVED"
  OUTPUT_ROOT = "/lichenghao/lzh/workspace/EnergyPrediction/data/eved-dataset/data/segmented_1s_eVED_v3"

依赖：pandas, numpy

使用说明：直接运行该脚本（会遍历输入目录下所有 CSV 并处理）。
"""
import os
import math
from collections import Counter
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from geopy.distance import geodesic


# ============== 配置（可按需修改列名列表） ==============
INPUT_DIR = "/lichenghao/lzh/workspace/EnergyPrediction/data/eved-dataset/data/filtered_vehID_eVED"
OUTPUT_ROOT = "/lichenghao/lzh/workspace/EnergyPrediction/data/eved-dataset/data/segmented_1s_eVED_v9"
TS_COL = "Timestamp(ms)"

# MATCH TYPE 候选列名（用于判断 match type == 2）
MATCH_TYPE_CANDIDATES = ["Match Type", "MatchType", "Match_Type", "match_type", "Match_Type"]

# 需要在 match type == 2 时进行线性插值填补的列
SPEED_FILL_COLS = ["Class of Speed Limit", "Speed Limit[km/h]", "Speed Limit with Direction[km/h]"]

# 明确的连续数值列（优先使用；其余数值列也会被 mean）
CONTINUOUS_COLS = [
    "Vehicle Speed[km/h]",
    "Engine RPM[RPM]",
    "Fuel Rate[L/hr]",
    "HV Battery Current[A]",
    "HV Battery Voltage[V]",
    "HV Battery SOC[%]",
    "Outside Air Temperature[DegC]",
    "elevation",
    "elevation_raw",
    "elevation_smoothed",
    "gradient",
]

# 添加短期/长期燃油修正列，确保在输出中计算平均值
FUEL_TRIM_COLS = [
    "Short Term Fuel Trim Bank 1[%]",
    "Short Term Fuel Trim Bank 2[%]",
    "Long Term Fuel Trim Bank 1[%]",
    "Long Term Fuel Trim Bank 2[%]",
]

# 对于 Intersection/Bus Stops/Focus Points 的空白填 0 的列名候选
FILL_ZERO_COLS = [
    "Intersection",
    "intersection",
    "Bus Stops",
    "Bus Stops",
    "Bus Stops",
    "Bus Stops",
    "Focus Points",
    "Focus Points;",
    "Focus Points;",
    "FocusPoints",
    "focus",
]

# 明确的 0/1 标志列或 focus point 列（如果数据名不同，请修改）
FLAG_COLS_KNOWN = [
    "intersection",
    "bus_stop",
    "traffic_signal",
    "crossing",
    "stop_sign",
]

# gps 匹配状态候选列名
GPS_COL_CANDIDATES = ["gps_match_status", "GPS_Match_Status", "gps_status"]

# 限速字段候选
SPEED_LIMIT_CANDIDATES = ["speed_limit", "speed_limit_directional", "Speed Limit with Direction[km/h]", "speed_limit_with_direction"]
SPEED_LIMIT_CLASS_CANDIDATES = ["class_speed_limit", "speed_limit_class", "Class of Speed Limit"]


def _safe_mode(series: pd.Series):
    """返回众数；若没有非空值返回 NaN；若有多个众数取第一个。"""
    if series is None or len(series) == 0:
        return np.nan
    modes = series.mode(dropna=True)
    if modes is None or modes.empty:
        return np.nan
    return modes.iloc[0]


def _speed_limit_class_priority(series: pd.Series):
    """按优先级选择 speed_limit_class：0 > 1 > 2 > 3 > -1；否则取众数。"""
    if series is None or len(series) == 0:
        return np.nan
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    # 尝试转为整数集
    try:
        vals = list(set(int(float(x)) for x in non_null))
    except Exception:
        mode_val = _safe_mode(non_null)
        return int(mode_val) if not pd.isna(mode_val) else np.nan

    priority = [0, 1, 2, 3, -1]
    for p in priority:
        if p in vals:
            return p
    # 否则返回众数
    mode_val = _safe_mode(non_null)
    return int(mode_val) if not pd.isna(mode_val) else np.nan


def _gps_status_agg(series: pd.Series):
    """gps status 聚合：matched->2, interpolated->1, unmatched->0，窗口内取 max 再映回字符串。"""
    if series is None or len(series) == 0:
        return np.nan
    mapping = {"unmatched": 0, "interpolated": 1, "matched": 2}
    inv = {v: k for k, v in mapping.items()}
    nums = []
    for x in series.dropna():
        try:
            nums.append(mapping.get(str(x).strip().lower(), -1))
        except Exception:
            nums.append(-1)
    nums = [n for n in nums if n >= 0]
    if not nums:
        return np.nan
    m = max(nums)
    return inv.get(m, np.nan)


def _mode_or_nan(series: pd.Series):
    return _safe_mode(series)


def _agg_first_nonzero_or_zero(series: pd.Series):
    """如果窗口内存在任意非0（数值或文本）值，返回第一个非0值；否则返回 0。
    这样能支持既有数值又有文本标签的列（比如 Focus Points）作为 flag 信号处理。
    """
    if series is None or len(series) == 0:
        return 0
    s = series.dropna()
    if s.empty:
        return 0
    for v in s:
        # 先尝试按数字判断是否为 0
        try:
            if float(v) != 0.0:
                return v
        except Exception:
            # 非数值（比如文本标签），视为非0，直接返回
            return v
    return 0


def _first_nonzero_value(series: pd.Series):
    """返回序列中第一个非0（数值）或非空文本值；若没有则返回 0。
    与 _agg_first_nonzero_or_zero 功能类似，但专为在 groupby 之前计算每个 window 的首个非0值使用。
    """
    if series is None or len(series) == 0:
        return 0
    s = series.dropna()
    if s.empty:
        return 0
    for v in s:
        try:
            if float(v) != 0.0:
                return v
        except Exception:
            # 非数值（比如文本标签），视为非0，直接返回
            if str(v).strip() != "":
                return v
    return 0


def _collect_ordered_values(series: pd.Series):
    """返回窗口内按时间顺序出现的所有非0数值或非空文本值的列表（保留出现顺序）。
    例：输入 [nan, 50, 50, 0, 60] -> [50, 60]
    """
    if series is None or len(series) == 0:
        return []
    res = []
    for v in series:
        if pd.isna(v):
            continue
        # 尝试数值化
        try:
            fv = float(v)
            if fv != 0.0:
                iv = int(round(fv))
                if not res or res[-1] != iv:
                    res.append(iv)
        except Exception:
            # 非数值文本
            sv = str(v).strip()
            if sv != "":
                if not res or res[-1] != sv:
                    res.append(sv)
    return res


def _is_01_series(series: pd.Series) -> bool:
    """检测 series 是否只有 0/1（或 NaN）值。"""
    if series is None or len(series) == 0:
        return False
    vals = pd.Series(series.dropna().unique())
    try:
        vals = vals.astype(float)
    except Exception:
        return False
    s = set(vals.tolist())
    return s.issubset({0.0, 1.0})


def _gradient_agg_keep_single_nonzero(series: pd.Series):
    """
    Gradient 专用聚合：
    - 去掉 NaN 后，如果只有一个非零值，其余为 0，则返回该非零值；
    - 其他情况退回到 mean。
    """
    if series is None or len(series) == 0:
        return np.nan
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan

    non_zero = s[s != 0]
    if len(non_zero) == 0:
        # 全是 0
        return 0.0

    # 唯一非零值
    uniq_non_zero = np.unique(non_zero.values)
    if len(uniq_non_zero) == 1:
        # 这一秒内只有这一个非零，其余都为 0
        if (s == 0).sum() + len(non_zero) == len(s):
            return float(uniq_non_zero[0])

    # 其他情况：按 mean
    return float(s.mean())


def process_trip_group(df_trip: pd.DataFrame, timestamp_col: str = TS_COL) -> pd.DataFrame:
    """
    对单个 trip 做 1000ms 的重采样并按规则聚合，返回聚合后的 DataFrame。

    注意：如果某些 1 秒窗口完全没有原始记录，当前实现不会插入这些缺失窗口（时间序列不连续）。
    """
    if timestamp_col not in df_trip.columns:
        raise KeyError(f"Timestamp column not found: {timestamp_col}")

    df = df_trip.copy()
    # 确保 timestamp 为数值（毫秒）
    df[timestamp_col] = pd.to_numeric(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    if df.empty:
        return pd.DataFrame()

    # 按时间排序并计算 t0
    df = df.sort_values(by=timestamp_col)
    t0 = int(df[timestamp_col].min())
    df["window_index"] = ((df[timestamp_col] - t0) // 1000).astype(int)

    # --- 修正梯度：使用平滑海拔与匹配经纬度计算 Gradient Smoothed，避免 Raw 高度引起的跳变 ---
    def _recompute_gradient_from_smoothed(df_src: pd.DataFrame) -> pd.DataFrame:
        # 优先使用窗口级等差平滑后的海拔 Evevation Smoothed_2（若已存在），否则退回原始平滑海拔
        elev_cols = ["Evevation Smoothed_2"]
        lat_cols = ["Matched Latitude[deg]", "Matchted Latitude[deg]"]
        lon_cols = ["Matched Longitude[deg]", "Matchted Longitude[deg]"]
        elev_col = next((c for c in elev_cols if c in df_src.columns), None)
        lat_col = next((c for c in lat_cols if c in df_src.columns), None)
        lon_col = next((c for c in lon_cols if c in df_src.columns), None)
        if elev_col is None or lat_col is None or lon_col is None:
            return df_src  # 缺列则跳过

        elev = pd.to_numeric(df_src[elev_col], errors="coerce")
        lat = pd.to_numeric(df_src[lat_col], errors="coerce")
        lon = pd.to_numeric(df_src[lon_col], errors="coerce")

        # 段定义：使用 "当前点 -> 下一点" (i -> i+1) 的坡度，并将结果挂在样本 i 上
        lat_cur = lat.values              # i
        lat_next = lat.shift(-1).values   # i+1
        lon_cur = lon.values
        lon_next = lon.shift(-1).values

        # 使用 GeoPy (Karney) 的 Geodesic 算法计算椭球体上的测地线距离（米）
        dist_list = []
        for la1, lo1, la2, lo2 in zip(lat_cur, lon_cur, lat_next, lon_next):
            if (pd.isna(la1) or pd.isna(lo1) or pd.isna(la2) or pd.isna(lo2)):
                dist_list.append(np.nan)
            else:
                try:
                    dist_list.append(geodesic((la1, lo1), (la2, lo2)).meters)
                except Exception:
                    dist_list.append(np.nan)
        dist = np.array(dist_list, dtype=float)  # 段 i: 从 i 到 i+1 的水平距离

        # 竖直变化：elev[i+1] - elev[i]
        dh = (elev.shift(-1) - elev).values

        # 计算坡度（单位：比值），避免除 0/极小距离
        min_d = 0.001  # m，极小距离阈值
        with np.errstate(divide="ignore", invalid="ignore"):
            grad_raw = np.where(
                (dist >= min_d) & np.isfinite(dist) & np.isfinite(dh),
                dh / dist,
                np.nan,
            )

        # 对梯度应用与 Evevation Smoothed_2 类似的等差前向填充逻辑：
        # 当在某点 i 发现梯度与前一时刻不同（且都有效）时，
        # 将梯度差按等差分配到该变化点前面连续等于旧值的那段索引中。
        grad = pd.to_numeric(pd.Series(grad_raw, index=df_src.index), errors="coerce").values
        n = len(grad)
        i = 1
        while i < n:
            if not np.isfinite(grad[i]) or not np.isfinite(grad[i-1]) or grad[i] == grad[i-1]:
                i += 1
                continue

            old = grad[i-1]
            new = grad[i]
            delta = new - old

            j = i - 1
            while j - 1 >= 0 and np.isfinite(grad[j-1]) and grad[j-1] == old:
                j -= 1
            k = i - j

            if k > 0:
                for t in range(1, k + 1):
                    idx = j + t - 1
                    try:
                        grad[idx] = old + delta * (t / float(k))
                    except Exception:
                        grad[idx] = old

            i += 1

        grad_series = pd.Series(grad, index=df_src.index)
        grad_series = grad_series.ffill()

        df_src["Gradient Smoothed"] = grad_series
        return df_src

    df = _recompute_gradient_from_smoothed(df)

    # 检测列集合
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 排除索引/键列
    for ex in ["VehId", "Trip", "window_index", timestamp_col]:
        if ex in numeric_cols:
            numeric_cols.remove(ex)

    # 连续值字段：在 numeric_cols 和 CONTINUOUS_COLS 的交集/并集中处理
    continuous = [c for c in (CONTINUOUS_COLS + FUEL_TRIM_COLS) if c in df.columns]
    # 其它数值列也当作连续字段处理（若不是 flag）
    extra_numeric = [c for c in numeric_cols if c not in continuous]

    # 标志位：已知列表（case-insensitive） + 自动检测只有 0/1 的数值列
    # 进行不区分大小写的匹配，以便捕获像 "Intersection" 这种大小写不同但语义相同的列名
    lower_known_flags = {f.lower() for f in FLAG_COLS_KNOWN}
    flag_cols = [c for c in df.columns if c.lower() in lower_known_flags]
    for c in extra_numeric:
        if _is_01_series(df[c]):
            if c not in flag_cols:
                flag_cols.append(c)

    # gps 列检测
    gps_col = next((c for c in GPS_COL_CANDIDATES if c in df.columns), None)

    # speed limit 列检测
    speed_limit_col = next((c for c in SPEED_LIMIT_CANDIDATES if c in df.columns), None)
    speed_limit_class_col = next((c for c in SPEED_LIMIT_CLASS_CANDIDATES if c in df.columns), None)

    # 构造聚合字典
    agg: Dict[str, Any] = {}

    # 连续数值使用 mean
    for c in continuous + extra_numeric:
        agg[c] = "mean"

    # 标志位使用 max
    for c in flag_cols:
        agg[c] = "max"

    # 对于 Intersection / Bus Stops / Focus Points 三类（可能为文本或数值）的特殊处理：
    # 我们希望 NaN 已被填为 0；只要窗口内出现任意非0（数值或文本）就保留该非0值，
    # 否则结果为 0。使用自定义聚合器实现该行为。
    FLAG_PRESERVE_LOWER = {"intersection", "bus stops", "bus_stop", "focus points", "focuspoints", "focus"}
    for c in df.columns:
        if c.lower().strip().rstrip(';') in FLAG_PRESERVE_LOWER:
            # 覆盖之前对该列的 agg，使用 first-nonzero-or-zero 的策略
            agg[c] = _agg_first_nonzero_or_zero

    # gps 自定义
    if gps_col is not None:
        agg[gps_col] = _gps_status_agg

    # speed_limit 聚合：收集窗口内按时间顺序出现的所有非0值（列表），后续根据 Intersection 选择
    if speed_limit_col is not None:
        agg[speed_limit_col] = _collect_ordered_values

    # speed_limit_class 自定义优先级
    if speed_limit_class_col is not None:
        agg[speed_limit_class_col] = _speed_limit_class_priority

    # 对 SPEED_FILL_COLS 中明确的数值列（如 "Speed Limit[km/h]", "Speed Limit with Direction[km/h]"）
    # 使用 mean 聚合（插值后取均值）以保留插值效果
    for name in ["Speed Limit[km/h]", "Speed Limit with Direction[km/h]"]:
        if name in df.columns:
            agg[name] = _collect_ordered_values

    # 其它非数值列取 first
    for c in df.columns:
        if c in ["VehId", "Trip", "window_index", timestamp_col]:
            continue
        if c in agg:
            continue
        if not pd.api.types.is_numeric_dtype(df[c].dtype):
            agg[c] = lambda s: s.iloc[0]

    # 确保经纬度列不被 mean 聚合：优先保留窗口内的第一个原始观测（或第一个非空）
    lat_candidates = ["Matched Latitude[deg]", "Matchted Latitude[deg]", "Latitude[deg]"]
    lon_candidates = ["Matched Longitude[deg]", "Matchted Longitude[deg]", "Longitude[deg]"]
    for name in lat_candidates:
        if name in df.columns:
            agg[name] = lambda s: s.dropna().iloc[0] if not s.dropna().empty else (s.iloc[0] if len(s) else np.nan)
    for name in lon_candidates:
        if name in df.columns:
            agg[name] = lambda s: s.dropna().iloc[0] if not s.dropna().empty else (s.iloc[0] if len(s) else np.nan)

    # <<< 在这里追加 Gradient 的专用聚合覆盖 >>>
    if "Gradient" in df.columns:
        agg["Gradient"] = _gradient_agg_keep_single_nonzero

    # 执行 groupby 聚合
    grouped = df.groupby("window_index", sort=True).agg(agg).reset_index()

    # --- 匹配经纬度：按窗口中点时间选取最近的原始观测值，而不是均值 ---
    # 支持列名变体：Matched / Matchted
    lat_src_col = next((c for c in ["Matched Latitude[deg]", "Matchted Latitude[deg]", "Latitude[deg]"] if c in df.columns), None)
    lon_src_col = next((c for c in ["Matched Longitude[deg]", "Matchted Longitude[deg]", "Longitude[deg]"] if c in df.columns), None)
    lat_grp_col = next((c for c in ["Matched Latitude[deg]", "Matchted Latitude[deg]", "Latitude[deg]"] if c in grouped.columns), None)
    lon_grp_col = next((c for c in ["Matched Longitude[deg]", "Matchted Longitude[deg]", "Longitude[deg]"] if c in grouped.columns), None)

    if lat_src_col is not None and lon_src_col is not None and lat_grp_col is not None and lon_grp_col is not None:
        try:
            # 确保原始数据按时间排序
            df_sorted = df.sort_values(timestamp_col).copy()
            df_sorted[timestamp_col] = pd.to_numeric(df_sorted[timestamp_col], errors="coerce")

            # 逐窗口选择：中点时间最近的观测值
            for idx, row in grouped.iterrows():
                win = int(row["window_index"]) if "window_index" in grouped.columns else int(row.name)
                w_start = int(t0 + win * 1000)
                mid_t = w_start + 500

                sub = df_sorted[df_sorted["window_index"] == win]
                if sub.empty:
                    continue

                ts = pd.to_numeric(sub[timestamp_col], errors="coerce")
                valid = ~ts.isna()
                if not valid.any():
                    continue

                ts_v = ts[valid].values.astype(float)
                # 找到时间上距离中点最近的采样
                idx_min = int(np.argmin(np.abs(ts_v - mid_t)))
                chosen = sub.loc[valid].iloc[idx_min]

                grouped.at[idx, lat_grp_col] = pd.to_numeric(chosen[lat_src_col], errors="coerce")
                grouped.at[idx, lon_grp_col] = pd.to_numeric(chosen[lon_src_col], errors="coerce")
        except Exception:
            # 任何异常都不影响整体流程，保持 groupby 结果
            pass

    # 基于时间占比选择该 1s 内持续时间最长的原始 (V,I) 采样，功率直接 -V*I，能耗取该采样行原值
    def _dominant_vi_selection(df_local: pd.DataFrame, ts_col: str = timestamp_col):
        v_map, i_map, p_map, e_map = {}, {}, {}, {}
        v_col = 'HV Battery Voltage[V]'
        i_col = 'HV Battery Current[A]'
        e_col = 'Energy_Consumption' if 'Energy_Consumption' in df_local.columns else None
        if v_col not in df_local.columns or i_col not in df_local.columns:
            return v_map, i_map, p_map, e_map

        # 按窗口分组
        for win, sub in df_local.groupby('window_index'):
            sub = sub.sort_values(ts_col)
            if sub.empty:
                continue
            w_start = int(t0 + int(win) * 1000)
            w_end = w_start + 1000

            times = pd.to_numeric(sub[ts_col], errors='coerce').astype(float).values
            Vs = pd.to_numeric(sub[v_col], errors='coerce').values
            Is = pd.to_numeric(sub[i_col], errors='coerce').values
            Es = pd.to_numeric(sub[e_col], errors='coerce').values if e_col else None

            # 仅使用采样点之间的时间段：每个采样值从其时间戳持续到下一个采样时间或窗口结束
            duration_acc = {}  # key=(V,I) -> total duration
            src_index_for_key = {}  # 保存代表采样行索引，用于能耗取值

            for idx in range(len(times)):
                t_start = times[idx]
                t_next = times[idx + 1] if idx + 1 < len(times) else w_end
                # 若采样点在窗口外（理论不应发生），跳过
                if t_start < w_start or t_start >= w_end:
                    continue
                # 持续时间只能从自身时间戳算起
                dt = (t_next - t_start) / 1000.0
                if dt <= 0:
                    continue
                v_val = Vs[idx]
                i_val = Is[idx]
                if pd.isna(v_val) or pd.isna(i_val):
                    continue
                key = (float(v_val), float(i_val))
                duration_acc[key] = duration_acc.get(key, 0.0) + dt
                # 只在首次出现时记录其源行索引（保证能耗来源稳定）
                if key not in src_index_for_key:
                    src_index_for_key[key] = idx

            if not duration_acc:
                # 兜底：取最后一个非 NaN 采样
                valid_idx = np.where(~(pd.isna(Vs) | pd.isna(Is)))[0]
                if len(valid_idx):
                    last = valid_idx[-1]
                    v_map[win] = float(Vs[last])
                    i_map[win] = float(Is[last])
                    p_map[win] = -v_map[win] * i_map[win]
                    if e_col and Es is not None and not pd.isna(Es[last]):
                        e_map[win] = float(Es[last])
                continue

            # 选择持续时间最大 key
            best_key = max(duration_acc.items(), key=lambda kv: kv[1])[0]
            rep_idx = src_index_for_key[best_key]
            v_sel, i_sel = best_key
            v_map[win] = v_sel
            i_map[win] = i_sel
            p_map[win] = -v_sel * i_sel
            if e_col and Es is not None and rep_idx < len(Es):
                e_val = Es[rep_idx]
                if not pd.isna(e_val):
                    e_map[win] = float(e_val)

        return v_map, i_map, p_map, e_map

    # 应用时间占比选择结果覆盖 grouped
    try:
        v_map, i_map, p_map, e_map = _dominant_vi_selection(df, timestamp_col)
        if 'HV Battery Power[W]' not in grouped.columns:
            grouped['HV Battery Power[W]'] = np.nan
        if 'Energy_Consumption' not in grouped.columns and 'Energy_Consumption' in df.columns:
            grouped['Energy_Consumption'] = np.nan
        for idx, row in grouped.iterrows():
            win = int(row['window_index'])
            if 'HV Battery Voltage[V]' in grouped.columns and win in v_map:
                grouped.at[idx, 'HV Battery Voltage[V]'] = v_map[win]
            if 'HV Battery Current[A]' in grouped.columns and win in i_map:
                grouped.at[idx, 'HV Battery Current[A]'] = i_map[win]
            if win in p_map:
                grouped.at[idx, 'HV Battery Power[W]'] = p_map[win]
            if 'Energy_Consumption' in grouped.columns and win in e_map:
                grouped.at[idx, 'Energy_Consumption'] = e_map[win]
    except Exception:
        pass

    # --- 新增：SOC（HV Battery SOC[%]）按“中点取值”的阶梯规则回填 ---
    def _soc_midpoint_map(df_local: pd.DataFrame, ts_col: str = timestamp_col):
        soc_map = {}
        soc_col = "HV Battery SOC[%]"
        if soc_col not in df_local.columns:
            return soc_map
        by = df_local.groupby("window_index")
        for win, sub in by:
            sub = sub.sort_values(ts_col)
            w_start = int(t0 + int(win) * 1000)
            mid_t = w_start + 500
            soc = pd.to_numeric(sub[soc_col], errors="coerce")
            ts = pd.to_numeric(sub[ts_col], errors="coerce")
            valid = (~soc.isna()) & (~ts.isna())
            if not valid.any():
                continue
            soc_v = soc[valid].values
            ts_v = ts[valid].values.astype(float)

            # 全部相同则直接取该值
            if len(pd.unique(soc_v)) == 1:
                soc_map[win] = float(soc_v[0])
                continue

            # 阶梯：选取中点时刻左侧最后一个观测值（若中点前无观测，则取最早一个）
            idx = np.searchsorted(ts_v, mid_t, side="right") - 1
            idx = max(0, min(idx, len(ts_v) - 1))
            soc_map[win] = float(soc_v[idx])
        return soc_map

    try:
        soc_map = _soc_midpoint_map(df, timestamp_col)
        if soc_map:
            if "HV Battery SOC[%]" in grouped.columns:
                for idx, row in grouped.iterrows():
                    win_idx = int(row["window_index"])
                    if win_idx in soc_map:
                        grouped.at[idx, "HV Battery SOC[%]"] = soc_map[win_idx]
    except Exception:
        pass

    # --- 在聚合结果上补齐所有缺失的整秒，并对速度/坡度/其他列进行填补 ---
    # 规则：
    #  - 以 1000ms 为间隔补齐所有秒；
    #  - 速度列使用时间线性插值；
    #  - 原始梯度列 Gradient 缺失秒填 0；
    #  - 其他列对缺失秒采用前向+后向最近值填补；
    #  - Evevation Smoothed_2 / Gradient Smoothed 的计算逻辑保持不变，但基于补全后的秒级数据进行。
    if not grouped.empty:
        # grouped 已包含 window_index（来自原始 t0 计算），使用它来补齐所有整秒窗口
        if "window_index" not in grouped.columns:
            # 若没有 window_index（极少见），尝试通过 SecTime_ms 构造
            if "SecTime_ms" in grouped.columns:
                grouped["window_index"] = ((grouped["SecTime_ms"] - grouped["SecTime_ms"].min()) // 1000).astype(int)
            else:
                # 无法补齐，跳过该步骤
                pass

        if "window_index" in grouped.columns:
            win_min = int(grouped["window_index"].min())
            win_max = int(grouped["window_index"].max())
            full_win = np.arange(win_min, win_max + 1, dtype=int)

            # 先保存原始存在的 window indices，用于标注哪些行是真实存在的
            orig_windows = set(grouped["window_index"].astype(int).tolist())

            grouped = grouped.set_index("window_index").reindex(full_win)
            grouped.index.name = "window_index"

            # 标记哪些秒在原始聚合结果中真实存在（基于原始 window_index）
            grouped["has_orig"] = grouped.index.to_series().isin(orig_windows).astype(bool)

            # 为新补出来的秒填充 SecTime_ms（保证基准 t0 可用）
            base_t0 = t0
            if "SecTime_ms" in grouped.columns and grouped["SecTime_ms"].notna().any():
                try:
                    base_t0 = int(grouped["SecTime_ms"].min())
                except Exception:
                    base_t0 = t0

            grouped["SecTime_ms"] = base_t0 + grouped.index.astype(int) * 1000

            SPEED_COLS = [
                "Vehicle Speed[km/h]",
            ]

            # 使用原始子秒级采样点作为辅助锚点，但保留每秒聚合值作为该秒的主锚点。
            # 构造（时间，速度）锚点集合：先加入原始采样点（timestamp_col），再加入每秒的聚合点（SecTime_ms），
            # 对这些锚点按时间排序并做一次插值，然后在每个 SecTime_ms 处取插值结果作为最终该秒的速度值。
            for col in SPEED_COLS:
                if col in grouped.columns and col in df.columns:
                    try:
                        # 原始样本点
                        df_sorted = df.sort_values(timestamp_col).copy()
                        df_sorted[timestamp_col] = pd.to_numeric(df_sorted[timestamp_col], errors="coerce")
                        raw_ts = pd.to_numeric(df_sorted[timestamp_col], errors="coerce").values.astype(float)
                        raw_v = pd.to_numeric(df_sorted[col], errors="coerce").values.astype(float)

                        # 秒级聚合点（保留原有聚合值）
                        sec_ts = pd.to_numeric(grouped["SecTime_ms"], errors="coerce").values.astype(float)
                        sec_v = pd.to_numeric(grouped[col], errors="coerce").values.astype(float)

                        # 合并点：先放原始采样点，再放每秒聚合点以覆盖同一时间点的原始值
                        times = np.concatenate([raw_ts, sec_ts])
                        vals = np.concatenate([raw_v, sec_v])

                        # 去除 NaN
                        mask = np.isfinite(times) & np.isfinite(vals)
                        times = times[mask]
                        vals = vals[mask]

                        if times.size >= 2:
                            # 对重复时间点取最后一个（因 sec_v 是后加入，会覆盖原始采样点）
                            order = np.argsort(times)
                            times_s = times[order]
                            vals_s = vals[order]
                            # 去重：保留最后一个出现的值
                            uniq_times, idx_last = np.unique(times_s, return_index=False, return_inverse=False), None
                            # numpy.unique with return_index gives first index; we need last -> use dict
                            tm_to_val = {}
                            for t_, v_ in zip(times_s, vals_s):
                                tm_to_val[t_] = v_
                            times_u = np.array(sorted(tm_to_val.keys()), dtype=float)
                            vals_u = np.array([tm_to_val[t_] for t_ in times_u], dtype=float)

                            # 插值到每秒时间点
                            target_ts = sec_ts
                            v_interp = np.interp(target_ts, times_u, vals_u)

                            # 只替换新补的秒的速度值；保留原始存在的秒的聚合值
                            has_orig_arr = grouped["has_orig"].to_numpy()
                            v_sec = sec_v.copy()
                            # 先用插值结果填充新补秒
                            v_sec[~has_orig_arr] = v_interp[~has_orig_arr]

                            # 额外规则：如果某个被填充的秒，其左侧最近的原始采样（时间 <= 该秒时间）存在且速度为 0，
                            # 则把该被填充秒设置为 0（从 0 开始插值）。
                            if raw_ts.size > 0:
                                # raw_ts 已按时间排序
                                for ii, tt in enumerate(target_ts):
                                    if has_orig_arr[ii]:
                                        continue
                                    # 找到最后一个 raw_ts <= tt
                                    idx_left = np.searchsorted(raw_ts, tt, side='right') - 1
                                    if idx_left >= 0 and idx_left < raw_ts.size:
                                        try:
                                            left_val = raw_v[idx_left]
                                            if np.isfinite(left_val) and float(left_val) == 0.0:
                                                v_sec[ii] = 0.0
                                        except Exception:
                                            pass

                            grouped[col] = v_sec
                    except Exception:
                        # 失败则保持原来的聚合结果
                        grouped[col] = grouped[col]
                else:
                    # 若没有原始列，按原逻辑做插值（使用已有秒级点）
                    if col in grouped.columns:
                        ts_sec = pd.to_numeric(grouped["SecTime_ms"], errors="coerce").to_numpy(dtype=float)
                        v_sec = pd.to_numeric(grouped[col], errors="coerce").to_numpy(dtype=float)
                        mask_valid = np.isfinite(ts_sec) & np.isfinite(v_sec)
                        if mask_valid.sum() >= 2:
                            v_interp = np.interp(ts_sec, ts_sec[mask_valid], v_sec[mask_valid])
                            has_orig_arr = grouped["has_orig"].to_numpy()
                            v_sec[~has_orig_arr] = v_interp[~has_orig_arr]
                            grouped[col] = v_sec

            # 其他列：最近值填补；Gradient 缺失秒置 0
            for c in list(grouped.columns):
                if c in ["has_orig", "SecTime_ms"]:
                    continue
                if c in SPEED_COLS:
                    continue
                if c == "Gradient":
                    if "has_orig" in grouped.columns:
                        mask_missing = ~grouped["has_orig"]
                        grouped.loc[mask_missing, c] = 0.0
                    # 仅对新补的秒填 0；保留原始存在秒的 Gradient 值（不做全局 fillna）
                else:
                    grouped[c] = grouped[c].ffill().bfill()

            grouped = grouped.reset_index(drop=False)

    # --- 新增列：Acceleration（基于每秒速度差，单位 km/h/s） ---
    try:
        acc_col = "Acceleration[m/s^2]"
        speed_col = "Vehicle Speed[km/h]"
        if speed_col in grouped.columns and "SecTime_ms" in grouped.columns:
            sp = pd.to_numeric(grouped[speed_col], errors="coerce")
            # convert km/h to m/s
            sp_m_s = sp * (1000.0 / 3600.0)
            ts_sec = pd.to_numeric(grouped["SecTime_ms"], errors="coerce").astype(float)
            # 以秒为单位计算时间差（保底 1s）
            dt = ts_sec.diff().fillna(1000.0) / 1000.0
            # 速度差除以时间差 -> m/s^2
            acc = sp_m_s.diff() / dt
            # 首行没有差值，设为 0.0
            if len(acc) > 0:
                acc.iloc[0] = 0.0

            # 将列插入到速度列之后
            cols = list(grouped.columns)
            insert_pos = 0
            if speed_col in cols:
                insert_pos = cols.index(speed_col) + 1
            grouped.insert(insert_pos, acc_col, acc)
    except Exception:
        pass

    # --- 新增列：Evevation Smoothed_2（基于窗口级 Elevation Raw[m] 的等差平滑） ---
    # 规则（按你的描述）：
    #   从 j[0] 开始遍历检测，当出现“新值”时，记该行索引为 j[i]；
    #   - j[0] 与 j[i] 行保持 Elevation Raw[m] 原值，直接写入 Evevation Smoothed_2；
    #   - j[1] 到 j[i-1] 行在 j[0] 与 j[i] 之间做等差填充（包含端点 j[0] 与 j[i] 的线性插值）。
    #   后续继续从 j[i] 作为新的 j[0]，重复上述逻辑。
    try:
        # 以秒级聚合后的 Elevation Raw[m] 为基础，如果不存在则不生成 Evevation Smoothed_2
        elev_raw_col = next((c for c in ["Elevation Raw[m]", "elevation_raw"] if c in grouped.columns), None)
        if elev_raw_col is not None:
            # 确保按时间顺序
            if 'window_index' in grouped.columns:
                gs = grouped.sort_values('window_index').reset_index(drop=True)
            else:
                gs = grouped.reset_index(drop=True)

            vals = pd.to_numeric(gs[elev_raw_col], errors='coerce').values
            sm2 = vals.copy()
            n = len(vals)
            if n > 0:
                # 当前段起点索引 j0 以及其值 v0
                j0 = 0
                v0 = vals[j0]
                # 遍历后续索引，寻找“新值”出现的位置 j[i]
                for idx in range(1, n):
                    v = vals[idx]
                    # 若当前值与段起点值不同，且两者都不是 NaN，则认为出现新值
                    if pd.isna(v) or pd.isna(v0) or v == v0:
                        continue

                    j1 = idx
                    v1 = v

                    # 在 j0 与 j1 之间做线性插值：
                    # j0 与 j1 位置保持原值 v0 / v1，其间的点按等差填充
                    length = j1 - j0
                    if length >= 1:
                        for k in range(0, length + 1):
                            alpha = k / float(length)
                            sm2[j0 + k] = v0 + (v1 - v0) * alpha

                    # 更新新的段起点
                    j0 = j1
                    v0 = v1

            # 将结果按照 window_index 的顺序写回 grouped
            if 'window_index' in grouped.columns:
                grouped = grouped.sort_values('window_index').reset_index(drop=True)
            grouped['Evevation Smoothed_2'] = sm2
    except Exception:
        # 任何异常不影响整体流程
        pass

    # --- 在秒级上对 Gradient Smoothed 做段内平滑 ---
    # 新规则：当在某秒 k 处检测到有效的经纬度变化（即 k->k+1 的水平距离可计算且大于阈值），
    # 将该变化的测地距离视为该段的总水平位移（total_dist），并平均分配到从上一个变化后第一个秒
    # 到当前秒 k 的每一秒上（包含 k）。随后对于该段内的每一秒 i，使用该秒的平滑海拔差值
    # dh_i = elev[i+1] - elev[i]（注意段内最后一个 dh 取 i=k -> k+1），并按：grad_i = dh_i / (total_dist / L)
    # 计算每秒坡度，从而保证分母相同，段内由水平位移均匀分配。
    try:
        # 确保输出中始终包含 `Gradient Smoothed` 列（若不存在则先创建），随后对其做段内平滑计算。
        if 'Gradient Smoothed' not in grouped.columns:
            grouped['Gradient Smoothed'] = np.nan

        # 需要经纬度与平滑海拔列
        lat_col = next((c for c in ["Matched Latitude[deg]", "Matchted Latitude[deg]", "Latitude[deg]"] if c in grouped.columns), None)
        lon_col = next((c for c in ["Matched Longitude[deg]", "Matchted Longitude[deg]", "Longitude[deg]"] if c in grouped.columns), None)
        elev_col = next((c for c in ["Evevation Smoothed_2", "Elevation Smoothed[m]", "elevation_smoothed", "Elevation Raw[m]", "elevation_raw"] if c in grouped.columns), None)

        # 若缺少必要列，则保留原有值并把首行为 0（兼容之前行为）
        if lat_col is None or lon_col is None or elev_col is None:
            try:
                grouped = grouped.reset_index(drop=True)
            except Exception:
                pass
        else:
            gs = grouped.sort_values('SecTime_ms').reset_index(drop=True) if 'SecTime_ms' in grouped.columns else grouped.reset_index(drop=True)
            n = len(gs)
            # 计算相邻秒之间的水平距离（meters）
            dists = np.full(n - 1, np.nan, dtype=float)
            for i in range(n - 1):
                try:
                    la1 = float(gs.at[i, lat_col])
                    lo1 = float(gs.at[i, lon_col])
                    la2 = float(gs.at[i + 1, lat_col])
                    lo2 = float(gs.at[i + 1, lon_col])
                    dists[i] = geodesic((la1, lo1), (la2, lo2)).meters
                except Exception:
                    dists[i] = np.nan

            # 平滑海拔数值
            elev = pd.to_numeric(gs[elev_col], errors='coerce').to_numpy(dtype=float)

            # 阈值（与样本级保持一致的小距离阈）
            min_d = 0.001

            # 初始化结果梯度数组（以秒为单位，长度 n；对最后一秒无法计算 dh 时填 0）
            grad_seconds = np.zeros(n, dtype=float)

            last_change = -1
            # 遍历每个可能的变化点（对应段内最后一秒 k，使用 dists[k] 表示 k->k+1 的水平位移）
            for k in range(n - 1):
                dist_k = dists[k]
                # 只有当该段水平位移可用并且大于阈值时视为变化点
                if not np.isfinite(dist_k) or dist_k < min_d:
                    continue

                # 定义当前段的秒范围：从 last_change+1 到 k （包含 k）
                seg_start = last_change + 1
                seg_end = k
                L = seg_end - seg_start + 1
                if L <= 0:
                    last_change = k
                    continue

                total_dist = float(dist_k)
                # 平均分配到段内每秒
                avg_dist_per_sec = total_dist / float(L)

                # 计算段内每一秒的海拔差 dh_i: dh_i = elev[i+1] - elev[i] for i in seg_start..seg_end
                dh = []
                valid = True
                for i in range(seg_start, seg_end + 1):
                    if i + 1 >= len(elev):
                        valid = False
                        break
                    v1 = elev[i]
                    v2 = elev[i + 1]
                    if not (np.isfinite(v1) and np.isfinite(v2)):
                        # 若遇到缺失海拔，则标记该段无效，跳过
                        valid = False
                        break
                    dh.append(v2 - v1)

                if not valid or avg_dist_per_sec <= 0:
                    # 无法计算该段，跳过并继续
                    last_change = k
                    continue

                # 计算并写入段内每秒的梯度
                for idx_offset, dh_val in enumerate(dh):
                    i_sec = seg_start + idx_offset
                    try:
                        grad_seconds[i_sec] = float(dh_val) / float(avg_dist_per_sec)
                    except Exception:
                        grad_seconds[i_sec] = 0.0

                # 更新 last_change 到当前 k
                last_change = k

            # 对未被上面覆盖到的秒（例如最后一秒或没有有效段的秒）保留原来的 grouped 值或设为 0
            # 先把原有的 Gradient Smoothed 值尽量保留
            orig_grad = pd.to_numeric(gs.get('Gradient Smoothed', pd.Series([np.nan]*n)), errors='coerce').to_numpy(dtype=float)
            for i in range(n):
                if grad_seconds[i] == 0.0 and (not np.isfinite(orig_grad[i])):
                    # 保底 0
                    grad_seconds[i] = 0.0
                elif grad_seconds[i] == 0.0 and np.isfinite(orig_grad[i]):
                    # 若原来有值且我们没有覆盖，保留原始
                    grad_seconds[i] = orig_grad[i]

            # 将计算结果写回 grouped
            try:
                gs['Gradient Smoothed'] = grad_seconds
                # 按原 window 顺序映回 grouped
                # 如果 grouped 原本有 SecTime_ms 顺序则以该顺序覆盖
                if 'SecTime_ms' in grouped.columns:
                    grouped = grouped.set_index('SecTime_ms')
                    gs2 = gs.set_index('SecTime_ms') if 'SecTime_ms' in gs.columns else gs
                    # 依据索引对齐并覆盖 Gradient Smoothed
                    for idx in gs2.index:
                        try:
                            grouped.at[idx, 'Gradient Smoothed'] = gs2.at[idx, 'Gradient Smoothed']
                        except Exception:
                            continue
                    grouped = grouped.reset_index()
                else:
                    # 否则直接用 gs 覆盖
                    grouped = gs
            except Exception:
                # 出现任何错误时退回到原值（不破坏流程）
                try:
                    grouped = grouped.reset_index(drop=True)
                except Exception:
                    pass
    except Exception:
        pass

    # --- Post-processing for speed limit selection:
    # We aggregated speed-limit columns using _collect_ordered_values which returns a
    # list of values seen in the window in temporal order. Now choose a single value
    # per window using Intersection: if Intersection==1 use the last-seen value,
    # otherwise use the first-seen value. If no values present -> 0.
    speed_fix_cols = [name for name in ["Speed Limit[km/h]", "Speed Limit with Direction[km/h]"] if name in grouped.columns]
    intersection_col = next((c for c in grouped.columns if c.lower().strip().rstrip(';') == 'intersection'), None)
    if speed_fix_cols:
        for idx, row in grouped.iterrows():
            # determine intersection flag for this window (compatible with numeric or string)
            is_inter = False
            if intersection_col is not None:
                try:
                    ival = row.get(intersection_col, 0)
                    try:
                        is_inter = float(ival) == 1.0
                    except Exception:
                        is_inter = str(ival).strip() in ('1', '1.0', 'True', 'true')
                except Exception:
                    is_inter = False

            for col in speed_fix_cols:
                try:
                    val = row.get(col, None)
                    chosen = 0
                    # val expected to be a list from _collect_ordered_values; handle scalar too
                    if isinstance(val, list):
                        if len(val) == 0:
                            chosen = 0
                        else:
                            chosen = val[-1] if is_inter else val[0]
                    else:
                        # scalar (single value) - keep it or convert to int if numeric
                        if pd.isna(val):
                            chosen = 0
                        else:
                            chosen = val
                    # normalize numeric to integer where possible
                    try:
                        if not (pd.isna(chosen)):
                            fv = float(chosen)
                            grouped.at[idx, col] = int(round(fv))
                        else:
                            grouped.at[idx, col] = chosen
                    except Exception:
                        grouped.at[idx, col] = chosen
                except Exception:
                    continue

        # 如果某个窗口的最终速限为 0，则使用该窗口之前最后一次出现的非0速限值进行填充（向前填充）
        for col in speed_fix_cols:
            if col not in grouped.columns:
                continue
            last_nonzero = None
            # 按 window_index 的时间顺序遍历
            for ridx in grouped.sort_values('window_index').index:
                try:
                    val = grouped.at[ridx, col]
                except Exception:
                    val = None
                if pd.isna(val):
                    # 保持 NaN
                    continue
                try:
                    fv = float(val)
                    if fv != 0.0:
                        # 记录为最近非0
                        last_nonzero = int(round(fv))
                    else:
                        if last_nonzero is not None:
                            grouped.at[ridx, col] = last_nonzero
                except Exception:
                    # 非数值（文本），如果非空则视为非0并记录
                    try:
                        s = str(val).strip()
                        if s != "":
                            last_nonzero = s
                    except Exception:
                        continue

    # 恢复 VehId, Trip
    if "VehId" in df.columns:
        grouped["VehId"] = df["VehId"].iat[0]
    if "Trip" in df.columns:
        grouped["Trip"] = df["Trip"].iat[0]

    # 生成 SecTime_ms（毫秒）
    grouped["SecTime_ms"] = t0 + grouped["window_index"].astype(int) * 1000
    if "window_index" in grouped.columns:
        grouped = grouped.drop(columns=["window_index"])

    # 移除不需要的辅助列
    to_remove = []
    for cand in MATCH_TYPE_CANDIDATES:
        if cand in grouped.columns:
            to_remove.append(cand)
    if "Class of Speed Limit" in grouped.columns:
        to_remove.append("Class of Speed Limit")
    if to_remove:
        grouped = grouped.drop(columns=list(set(to_remove)))

    # --- 列顺序整理 ---
    # 目标：
    # 1) 尽量沿用原始列顺序（以 df_trip.columns 为准）
    # 2) 用 SecTime_ms 替换原 Timestamp(ms) 的位置
    # 3) 若存在 Gradient/gradient，则将 Gradient Smoothed 插在其后
    # 4) 将 [I, V, Power, SOC, Energy] 相邻且按此顺序排列
    def _final_column_order(orig_cols: List[str], current_cols: List[str]) -> List[str]:
        present = [c for c in current_cols]
        # 基于原始顺序构建基础顺序
        base = []
        for c in orig_cols:
            if c == timestamp_col:
                if "SecTime_ms" in present and "SecTime_ms" not in base:
                    base.append("SecTime_ms")
                continue
            if c in present and c not in base:
                base.append(c)
        # 原始列中没有 SecTime_ms 时，尽早插入它（在 VehId/Trip 之后）
        if "SecTime_ms" in present and "SecTime_ms" not in base:
            insert_pos = 0
            for key in ["VehId", "Trip"]:
                if key in base:
                    insert_pos = max(insert_pos, base.index(key) + 1)
            base.insert(insert_pos, "SecTime_ms")

        # 在 Gradient/gradient/Gradient 后插入 Gradient Smoothed
        gs = "Gradient Smoothed"
        if gs in present:
            anchor_names = ["Gradient", "gradient"]
            anchor_pos = None
            for anc in anchor_names:
                if anc in base:
                    anchor_pos = base.index(anc) + 1
                    break
            # 若基础顺序中不存在，则尽量在原 gradient 的邻近位置插入，否则保持末尾追加
            if gs in base:
                base.remove(gs)
            if anchor_pos is None:
                base.append(gs)
            else:
                base.insert(anchor_pos, gs)

        # I/V/Power/SOC/Energy 相邻，并按指定顺序
        block = [
            "HV Battery Current[A]",
            "HV Battery Voltage[V]",
            "HV Battery Power[W]",
            "HV Battery SOC[%]",
            "Energy_Consumption",
        ]
        exist_block = [c for c in block if c in present]
        if exist_block:
            # 找一个锚位：这些列在 base 中出现的最小索引
            candidates = [idx for idx, c in enumerate(base) if c in exist_block]
            if candidates:
                anchor = min(candidates)
            else:
                # 如果 base 没有这些列（例如它们是新增列），放到 SecTime_ms 后面
                anchor = base.index("SecTime_ms") + 1 if "SecTime_ms" in base else 0
            # 移除老的位置
            base = [c for c in base if c not in exist_block]
            # 按固定顺序插入存在的列
            ordered_block = [c for c in block if c in exist_block]
            for offset, name in enumerate(ordered_block):
                base.insert(anchor + offset, name)

        # 在 Elevation Smoothed 之后插入 Evevation Smoothed_2（若两者都存在）
        if "Evevation Smoothed_2" in present:
            # 先从 base 中移除，以便重新插入
            if "Evevation Smoothed_2" in base:
                base.remove("Evevation Smoothed_2")
            insert_pos = None
            for name in ["Elevation Smoothed[m]", "elevation_smoothed"]:
                if name in base:
                    insert_pos = base.index(name) + 1
                    break
            if insert_pos is None:
                # 若找不到 Elevation Smoothed，则保持默认附加顺序
                base.append("Evevation Smoothed_2")
            else:
                base.insert(insert_pos, "Evevation Smoothed_2")

        # 在 Vehicle Speed 后插入 Acceleration（若存在）
        acc_name = "Acceleration[km/h/s]"
        if acc_name in present:
            if acc_name in base:
                base.remove(acc_name)
            insert_pos = None
            if "Vehicle Speed[km/h]" in base:
                insert_pos = base.index("Vehicle Speed[km/h]") + 1
            if insert_pos is None:
                base.append(acc_name)
            else:
                base.insert(insert_pos, acc_name)

        # 补上未被覆盖到的其余列（例如聚合产生的新列）
        leftovers = [c for c in present if c not in base]
        return base + leftovers

    try:
        # 确保不输出原始 timestamp 列（若聚合侧误入）
        if timestamp_col in grouped.columns:
            grouped = grouped.drop(columns=[timestamp_col])
        # 计算最终列顺序并重排
        desired_order = _final_column_order(list(df_trip.columns), list(grouped.columns))
        grouped = grouped[[c for c in desired_order if c in grouped.columns]]
    except Exception:
        # 兜底：若排序失败，退回原逻辑（VehId, Trip, SecTime_ms, 其余）
        cols_order = [c for c in ["VehId", "Trip", "SecTime_ms"] if c in grouped.columns]
        rest = [c for c in grouped.columns if c not in cols_order]
        grouped = grouped[cols_order + rest]

    # 强制将 Acceleration 列放到 Vehicle Speed 之后（作为最终保险措施）
    acc_name = "Acceleration[m/s^2]"
    speed_name = "Vehicle Speed[km/h]"
    if acc_name in grouped.columns and speed_name in grouped.columns:
        cols = list(grouped.columns)
        # remove acc and re-insert after speed
        if acc_name in cols:
            cols.remove(acc_name)
        try:
            insert_pos = cols.index(speed_name) + 1
        except ValueError:
            insert_pos = None
        if insert_pos is None or insert_pos < 0 or insert_pos > len(cols):
            cols.append(acc_name)
        else:
            cols.insert(insert_pos, acc_name)
        grouped = grouped[cols]

    return grouped


def ensure_int_str(x):
    """把数字型或浮点字符串规范为整数字符串（去掉 .0），否则返回原字符串。"""
    try:
        return str(int(float(x)))
    except Exception:
        return str(x)


def run_pipeline(input_dir: str = INPUT_DIR, output_root: str = OUTPUT_ROOT, timestamp_col: str = TS_COL):
    """主流程：参考 Filter_VehId.py 的划分方式，按 category 列表和 veh_id 列表逐辆车聚合并处理。

    该实现不会一次性把所有 CSV 合并到内存，而是按 vehicle 聚合（跨文件合并同一 VehId 的 Trip 行）。
    目录结构按 Filter_VehId.py 的分类：output_root/EV/Veh_<VehId>/Trip_<TripId>_resampled_1s.csv
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    os.makedirs(output_root, exist_ok=True)

    # 输入数据已经按 Filter_VehId.py 的方式分好了目录：input_dir/{category}/{veh_id}/*.csv
    # 直接遍历 category (EV/PHEV) 下每个 VehId 目录，并对目录下的每个 trip CSV 进行处理
    categories = ["EV"]
    for category in categories:
        cat_dir = os.path.join(input_dir, category)
        if not os.path.isdir(cat_dir):
            print(f"Category folder not found, skipping: {cat_dir}")
            continue
        print(f"Processing category folder: {cat_dir}")

        for veh_folder_name in sorted(os.listdir(cat_dir)):
            veh_folder_path = os.path.join(cat_dir, veh_folder_name)
            if not os.path.isdir(veh_folder_path):
                continue

            # veh_folder_name 期望为车辆 id（Filter_VehId.py 使用数字文件夹名），规范为字符串
            veh_id_str = ensure_int_str(veh_folder_name)
            # 输出目录改为 output_root/{category}/{VehId}/
            out_veh_folder = os.path.join(output_root, category, f"{veh_id_str}")
            os.makedirs(out_veh_folder, exist_ok=True)

            # 遍历该车辆目录下的 trip CSV 文件
            trip_files = sorted([f for f in os.listdir(veh_folder_path) if f.lower().endswith('.csv')])
            for trip_file_name in trip_files:
                trip_file = os.path.join(veh_folder_path, trip_file_name)
                print(f"  Processing trip file: {trip_file}")
                try:
                    trip_df = pd.read_csv(trip_file, low_memory=False)
                except Exception as e:
                    print(f"    Failed to read {trip_file}: {e}")
                    continue

                if trip_df.empty:
                    print(f"    Empty file {trip_file}, skipping")
                    continue

                # 找到 match type 列并识别 match==2 的行
                match_col = next((c for c in MATCH_TYPE_CANDIDATES if c in trip_df.columns), None)
                if match_col is not None:
                    try:
                        match_vals = pd.to_numeric(trip_df[match_col], errors='coerce')
                    except Exception:
                        match_vals = None
                else:
                    match_vals = None

                mask_interp = match_vals == 2 if match_vals is not None else pd.Series([False] * len(trip_df))

                # 先将 Intersection / Bus Stops / Focus Points 的空白填充为 0（按需保留其他非空值）
                # 新规则：把这三类列视为 flag 信号，NaN 填补为 0；在聚合时只要出现非0情况就保留（否则为0）。
                # 支持列名变体（不区分大小写，允许末尾分号等）。
                FLAG_PRESERVE_LOWER = {"intersection", "bus stops", "bus_stop", "focus points", "focuspoints", "focus"}
                for col in FILL_ZERO_COLS:
                    if col in trip_df.columns:
                        # 把空字符串视为缺失
                        trip_df[col] = trip_df[col].replace("", np.nan)
                        lname = str(col).lower().strip().rstrip(';')
                        if lname in FLAG_PRESERVE_LOWER:
                            # 对这些列，直接把 NaN 填为 0（保留原有非0/文本值）
                            trip_df[col] = trip_df[col].where(trip_df[col].notna(), 0)
                        else:
                            # 其它候选项（非我们判定的三类）保持原样（保留 NaN，不填 0）
                            trip_df[col] = trip_df[col].where(trip_df[col].notna(), np.nan)

                # 对指定的三个限速相关列在 match==2 且原值为空的位置使用线性插值填补
                for col in SPEED_FILL_COLS:
                    if col in trip_df.columns:
                        try:
                            s = pd.to_numeric(trip_df[col], errors='coerce')
                            if mask_interp.any() and s.isna().any():
                                s_interp = s.interpolate(method='linear', limit_direction='both')
                                fill_idx = mask_interp & s.isna()
                                trip_df.loc[fill_idx, col] = s_interp[fill_idx]
                        except Exception as e:
                            print(f"    Warning: failed to interpolate {col} in {trip_file}: {e}")

                # 将限速相关列在插值后四舍五入为整数，确保后续聚合不会产生小数
                for col in SPEED_FILL_COLS:
                    if col in trip_df.columns:
                        try:
                            # 原先的写法会触发 pandas 的 DeprecationWarning（对列子集赋值）
                            # trip_df[col] = pd.to_numeric(trip_df[col], errors='coerce')
                            # mask = trip_df[col].notna()
                            # trip_df.loc[mask, col] = trip_df.loc[mask, col].round().astype('Int64')

                            # 统一对整列进行转换与四舍五入，并一次性赋值，避免警告
                            trip_df[col] = pd.to_numeric(trip_df[col], errors='coerce').round().astype('Int64')
                        except Exception:
                            # 忽略转换问题，保持原状
                            continue

                # 重采样处理
                try:
                    out_df = process_trip_group(trip_df, timestamp_col=timestamp_col)
                except Exception as e:
                    print(f"    Error processing trip file {trip_file}: {e}")
                    continue

                if out_df.empty:
                    print(f"    Trip produced no output, skipping")
                    continue

                # 输出文件名：使用 Trip 列（若存在）否则用原文件名（去扩展名）
                if 'Trip' in trip_df.columns:
                    trip_id_val = ensure_int_str(trip_df['Trip'].iat[0])
                else:
                    trip_id_val = ensure_int_str(os.path.splitext(trip_file_name)[0])

                # 输出文件名为 {TripId}.csv
                out_file = os.path.join(out_veh_folder, f"{trip_id_val}.csv")
                try:
                    out_df.to_csv(out_file, index=False)
                    print(f"    Wrote {out_file}, rows={len(out_df)}")
                except Exception as e:
                    print(f"    Failed to write {out_file}: {e}")

    print("Pipeline complete.")


if __name__ == "__main__":
    run_pipeline()