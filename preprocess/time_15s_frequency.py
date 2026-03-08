"""time_15s_frequency.py

从已经按 1s 分段的 CSV (`segmented_1s_eVED_v9`) 生成 15s 窗口聚合输出到
`segmented_15s_eVED_v2`。

当前实现要点（可后续再细化规则）：
- 时间列：优先使用 `SecTime_ms`（毫秒）或 `Timestamp(ms)`。
- 数值列（白名单）聚合：
    - `Vehicle Speed[km/h]`：窗口均值
    - `HV Battery Current[A]` / `HV Battery Voltage[V]` / `HV Battery Power[W]`：窗口均值
    - `Energy_Consumption`：窗口求和（输入为每秒瞬时能耗）
- 经纬度（原始与 matched）：按窗口采样时刻（默认窗口结束时刻）取当时值，仅输出单列。
- 海拔（raw/smoothed）：按窗口采样时刻取当时值（单列）。
- `Air Conditioning Power` / `Heater Power`：窗口均值。
- `Gradient`：使用 15s 间平滑海拔起止高度差 + matched 起止经纬度的测地线距离（geodesic）计算。

依赖: pandas, numpy, geopy
"""

import os
from typing import Dict, Any

import numpy as np
import pandas as pd
from geopy.distance import geodesic


INPUT_ROOT = "/wenzhiquan/dengzeshuai/datasets/eved-dataset/data/segmented_1s_eVED_v9"
OUTPUT_ROOT = "/wenzhiquan/dengzeshuai/datasets/eved-dataset/data/segmented_15s_eVED_v5"
WINDOW_MS = 15000
COVERAGE_THRESHOLD = 0.6
TS_COL_CANDIDATES = ["SecTime_ms", "Timestamp(ms)"]
SAMPLE_TS_IN_WINDOW = "start"  # one of: "start" | "mid" | "end"

# Important electrical column names used by default
I_COL = "HV Battery Current[A]"
V_COL = "HV Battery Voltage[V]"
P_COL = "HV Battery Power[W]"
E_COL = "Energy_Consumption"

# candidate column names for special handling
# split candidates into original vs matched sets so we can produce start/mid/end for both
ORIG_LAT_CANDIDATES = ["Latitude[deg]", "lat"]
ORIG_LON_CANDIDATES = ["Longitude[deg]", "lon"]
MATCHED_LAT_CANDIDATES = ["Matched Latitude[deg]", "Matchted Latitude[deg]"]
MATCHED_LON_CANDIDATES = ["Matched Longitude[deg]", "Matchted Longitude[deg]"]
ELEV_RAW_CANDIDATES = ["Elevation Raw[m]", "elevation_raw", "Elevation Raw"]
ELEV_SMOOTH2_CANDIDATES = ["Evevation Smoothed_2", "Evevation Smoothed_2", "Elevation Smoothed[m]", "elevation_smoothed"]
OAT_CANDIDATES = ["Outside Air Temperature[DegC]", "Outside Air Temperature", "OAT", "OAT[DegC]"]
SOC_CANDIDATES = ["HV Battery SOC[%]", "HV Battery SOC", "SOC"]
SPEED_LIMIT_CANDIDATES = ["Speed Limit[km/h]", "Speed Limit with Direction[km/h]", "speed_limit", "speed_limit_with_direction"]
AC_ENERGY_CANDIDATES = ["AC Energy", "AC_Energy", "HVAC_Energy", "ACEnergy"]
HEATER_ENERGY_CANDIDATES = ["Heater Energy", "Heater_Energy", "HTR_Energy", "HeaterEnergy"]

# flag-like place names
FLAG_PLACE_CANDIDATES = ["Intersection", "intersection", "Bus Stops", "Focus Points", "Focus Points;", "FocusPoints", "focus"]



def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def find_time_col(df: pd.DataFrame) -> str:
    for c in TS_COL_CANDIDATES:
        if c in df.columns:
            return c
    # fallback: try any column with 'time' in name
    for c in df.columns:
        if 'time' in c.lower():
            return c
    raise KeyError("No time column found (expect SecTime_ms or Timestamp(ms))")


def process_trip_file(in_file: str, out_file: str):
    df = pd.read_csv(in_file, low_memory=False)
    if df.empty:
        return

    # locate time column and produce SecTime_ms numeric
    time_col = find_time_col(df)
    df['SecTime_ms'] = pd.to_numeric(df[time_col], errors='coerce')
    df = df.dropna(subset=['SecTime_ms'])
    if df.empty:
        return

    # Align window origin to the lower multiple of WINDOW_MS so that 0s is included.
    # Windows are forward-looking: [w_start, w_end) and each output row is labeled
    # by the sampling time point within the window (default: w_start).
    t0 = int(np.floor(df['SecTime_ms'].min() / WINDOW_MS) * WINDOW_MS)
    df['window_index'] = ((df['SecTime_ms'] - t0) // WINDOW_MS).astype(int)

    # 覆盖率过滤：对于末尾不完整窗口，不再强制要求存在 w_end 时间点。
    # 改为：窗口内实际覆盖的 1s 采样点数 / 15s >= COVERAGE_THRESHOLD 才保留。
    # 例如原始末尾为 53s，则窗口 [45,60) 覆盖 45..53 共 9s，覆盖率 9/15=0.6，仍保留并输出 45s。

    # 明确的数值白名单：仅包含你明确指定要聚合的列
    NUMERIC_MEAN_COLS = [
        'Vehicle Speed[km/h]',
        I_COL,
        V_COL,
        P_COL,
        E_COL,
    ]
    # 保留实际存在于文件中的列
    numeric_cols = [c for c in NUMERIC_MEAN_COLS if c in df.columns]

    def mode_of(series):
        s = series.dropna()
        if s.empty:
            return np.nan
        try:
            m = s.mode()
            return m.iloc[0]
        except Exception:
            # fallback: value_counts
            vc = s.value_counts()
            if vc.empty:
                return np.nan
            return vc.index[0]

    def get_val_at_exact_time(full_df: pd.DataFrame, target_ts: float, col: str):
        if col not in full_df.columns:
            return np.nan
        ts_int = int(round(target_ts))
        s = full_df.loc[full_df['SecTime_ms'].astype(int) == ts_int, col]
        if s.empty:
            return np.nan
        # pick first non-null if possible
        s2 = s.dropna()
        if s2.empty:
            return np.nan
        return s2.iloc[0]

    def get_val_in_window(sub_df: pd.DataFrame, target_ts: float, col: str, how: str):
        """Window-safe sampling.

        - Try exact timestamp first.
        - If missing, fall back to a reasonable point within this window.
          how: 'start'|'mid'|'end'
        """
        if col not in sub_df.columns:
            return np.nan
        exact = get_val_at_exact_time(df, target_ts, col)
        if not (isinstance(exact, float) and np.isnan(exact)):
            return exact

        # fallback within window
        s = sub_df[['SecTime_ms', col]].copy()
        s[col] = pd.to_numeric(s[col], errors='ignore')
        s = s.dropna(subset=['SecTime_ms', col]).sort_values('SecTime_ms')
        if s.empty:
            return np.nan

        if how == 'start':
            return s.iloc[0][col]
        if how == 'end':
            return s.iloc[-1][col]

        # mid: pick nearest by time
        t = pd.to_numeric(s['SecTime_ms'], errors='coerce').to_numpy(dtype=float)
        idx = int(np.argmin(np.abs(t - float(target_ts))))
        return s.iloc[idx][col]

    groups = df.groupby('window_index', sort=True)

    rows = []
    for win, sub in groups:
        sub = sub.sort_values('SecTime_ms')
        row: Dict[str, Any] = {}
        row['window_index'] = int(win)
        
        # Time points for sampling
        w_start = t0 + win * WINDOW_MS
        w_end = t0 + (win + 1) * WINDOW_MS
        w_mid = w_start + WINDOW_MS / 2.0

        if SAMPLE_TS_IN_WINDOW == "start":
            w_sample = w_start
            sample_how = 'start'
        elif SAMPLE_TS_IN_WINDOW == "mid":
            w_sample = w_mid
            sample_how = 'mid'
        else:
            w_sample = w_end
            sample_how = 'end'

        # 覆盖率：按 1s 分段数据，窗口期望 15 个采样点。这里使用窗口内独立时间戳数量近似覆盖秒数。
        expected = int(WINDOW_MS // 1000)
        coverage_seconds = int(sub['SecTime_ms'].astype(int).nunique())
        coverage_ratio = coverage_seconds / float(expected) if expected > 0 else 0.0
        if coverage_ratio < COVERAGE_THRESHOLD:
            continue

        # label row by sampling time point so first row keeps 0s
        row['SecTime_ms'] = int(round(w_sample))

        # (已移除) 不再在输出中包含采样计数与置信度字段

        # numeric aggregates: only keep explicitly requested outputs
        for col in numeric_cols:
            col_vals = pd.to_numeric(sub[col], errors='coerce').dropna().to_numpy()
            # empty handling
            if col_vals.size == 0:
                if col == E_COL:
                    row[f"{col}_sum"] = np.nan
                elif col == 'Vehicle Speed[km/h]':
                    row[f"{col}_mean"] = np.nan
                elif col in [I_COL, V_COL, P_COL]:
                    row[f"{col}_mean"] = np.nan
                continue

            if col == 'Vehicle Speed[km/h]':
                row[f"{col}_mean"] = float(np.mean(col_vals))
            elif col in [I_COL, V_COL, P_COL]:
                row[f"{col}_mean"] = float(np.mean(col_vals))
            elif col == E_COL:
                # Energy: sum per-window (输入为每秒瞬时能耗)
                row[f"{col}_sum"] = float(np.nansum(col_vals))

        # 1) 原始经纬度：取窗口采样时刻当时值（单列）
        orig_lat = None
        for c in ORIG_LAT_CANDIDATES:
            if c in sub.columns:
                orig_lat = c
                break
        if orig_lat:
            row['Latitude'] = get_val_in_window(sub, w_sample, orig_lat, sample_how)
        else:
            row['Latitude'] = np.nan

        orig_lon = None
        for c in ORIG_LON_CANDIDATES:
            if c in sub.columns:
                orig_lon = c
                break
        if orig_lon:
            row['Longitude'] = get_val_in_window(sub, w_sample, orig_lon, sample_how)
        else:
            row['Longitude'] = np.nan

        # 1b) 匹配经纬度（matched）：取窗口采样时刻当时值（单列）
        matched_lat = None
        for c in MATCHED_LAT_CANDIDATES:
            if c in sub.columns:
                matched_lat = c
                break
        if matched_lat:
            row['Matched_Latitude'] = get_val_in_window(sub, w_sample, matched_lat, sample_how)
        else:
            row['Matched_Latitude'] = np.nan

        matched_lon = None
        for c in MATCHED_LON_CANDIDATES:
            if c in sub.columns:
                matched_lon = c
                break
        if matched_lon:
            row['Matched_Longitude'] = get_val_in_window(sub, w_sample, matched_lon, sample_how)
        else:
            row['Matched_Longitude'] = np.nan

        # 4) 海拔类字段：与经纬度一致，取窗口采样时刻当时值（单列）
        for elev_c in ELEV_RAW_CANDIDATES:
            if elev_c in sub.columns:
                row[f"{elev_c}_sample"] = get_val_in_window(sub, w_sample, elev_c, sample_how)
        for elev_c in ELEV_SMOOTH2_CANDIDATES:
            if elev_c in sub.columns:
                row[f"{elev_c}_sample"] = get_val_in_window(sub, w_sample, elev_c, sample_how)

        # 2) 速度取窗口内均值（已由 numeric mean 生成，但 ensure naming)
        # nothing further needed; Vehicle Speed[km/h]_mean present if exists

        # 3) OAT, AC energy, Heater energy, SOC, Speed Limit -> 窗口众数
        for cands in [OAT_CANDIDATES, AC_ENERGY_CANDIDATES, HEATER_ENERGY_CANDIDATES, SOC_CANDIDATES, SPEED_LIMIT_CANDIDATES]:
            for c in cands:
                if c in sub.columns:
                    m = mode_of(sub[c])
                    row[f"{c}_mode"] = m
                    break

        # 确保为 OAT 生成标准列名 `OAT_mode`（有些文件列名多样）
        for c in OAT_CANDIDATES:
            if c in sub.columns:
                row['OAT_mode'] = mode_of(sub[c])
                break

        # 确保为 Speed Limit with Direction 生成标准列名
        for c in SPEED_LIMIT_CANDIDATES:
            if c in sub.columns:
                # use explicit standardized output name
                row['Speed Limit with Direction[km/h]_mode'] = mode_of(sub[c])
                break

        # Air Conditioning Power and Heater Power: use mean in window if present
        AC_POWER_CANDS = ['Air Conditioning Power[Watts]', 'Air Conditioning Power[kW]', 'Air Conditioning Power[kW]']
        HEATER_POWER_CANDS = ['Heater Power[Watts]', 'Heater Power[kW]']
        for c in AC_POWER_CANDS:
            if c in sub.columns:
                vals = pd.to_numeric(sub[c], errors='coerce').dropna().to_numpy()
                row[f"{c}_mean"] = float(np.mean(vals)) if vals.size else np.nan
                break
        for c in HEATER_POWER_CANDS:
            if c in sub.columns:
                vals = pd.to_numeric(sub[c], errors='coerce').dropna().to_numpy()
                row[f"{c}_mean"] = float(np.mean(vals)) if vals.size else np.nan
                break

        # 5) intersection/bus stops/focus points -> 有效值众数（非0/非空）
        for fp in FLAG_PLACE_CANDIDATES:
            if fp in sub.columns:
                s = sub[fp].dropna()
                # consider valid values as those not equal to 0 and not empty
                try:
                    valid = s[s.astype(str).str.strip() != ""].copy()
                except Exception:
                    valid = s
                try:
                    # filter numeric zeros
                    valid_num = pd.to_numeric(valid, errors='coerce')
                    valid = valid[~(pd.isna(valid_num) & (valid.astype(str).str.strip() == ""))]
                    valid = valid[~(valid_num == 0)]
                except Exception:
                    pass
                if valid.dropna().empty:
                    row[f"{fp}_mode"] = 0
                else:
                    row[f"{fp}_mode"] = mode_of(valid)

        # non-numeric: take first non-null for common identifiers
        for key in ['VehId', 'Trip']:
            if key in sub.columns:
                try:
                    val = sub[key].dropna().iloc[0]
                    row[key] = val
                except Exception:
                    row[key] = np.nan

        # Gradient: 15s 间平滑海拔起止高度差 / matched 起止经纬度测地线距离
        elev_col_for_grad = None
        for c in ELEV_SMOOTH2_CANDIDATES:
            if c in sub.columns:
                elev_col_for_grad = c
                break
        
        row['Gradient'] = np.nan
        if elev_col_for_grad and matched_lat and matched_lon:
            # 使用窗口内实际首尾点（更稳健，支持末尾不完整窗口）
            t_first = float(sub['SecTime_ms'].min())
            t_last = float(sub['SecTime_ms'].max())
            elev_s = get_val_in_window(sub, t_first, elev_col_for_grad, 'start')
            elev_e = get_val_in_window(sub, t_last, elev_col_for_grad, 'end')

            lat_s = get_val_in_window(sub, t_first, matched_lat, 'start')
            lon_s = get_val_in_window(sub, t_first, matched_lon, 'start')
            lat_e = get_val_in_window(sub, t_last, matched_lat, 'end')
            lon_e = get_val_in_window(sub, t_last, matched_lon, 'end')

            if not (np.isnan(lat_s) or np.isnan(lon_s) or np.isnan(lat_e) or np.isnan(lon_e) or np.isnan(elev_s) or np.isnan(elev_e)):
                try:
                    dist = geodesic((lat_s, lon_s), (lat_e, lon_e)).meters
                    diff = elev_e - elev_s
                    if dist > 0.1:
                        row['Gradient'] = diff / dist
                    else:
                        row['Gradient'] = 0.0
                except Exception:
                    pass

        rows.append(row)

    if not rows:
        return

    out_df = pd.DataFrame(rows)
    # 保留统一的 `OAT_mode`，删除由候选名生成的冗余 `OAT[DegC]_mode`（如果存在）
    if 'OAT[DegC]_mode' in out_df.columns and 'OAT_mode' in out_df.columns:
        out_df.drop(columns=['OAT[DegC]_mode'], inplace=True)
    # 移除不需要输出的采样与置信度字段（以防中间产生）以及 window_index
    for _c in ['sample_count', 'coverage_seconds', 'coverage_ratio', 'low_confidence', 'window_index']:
        if _c in out_df.columns:
            out_df.drop(columns=[_c], inplace=True)

    # 移除用户不想要的其它聚合列（Engine RPM, Fuel Rate）以防残留
    for bad in ['Engine RPM', 'Fuel Rate']:
        drop_cols = [c for c in out_df.columns if bad in c]
        if drop_cols:
            out_df.drop(columns=drop_cols, inplace=True)

    # reorder columns to a sensible, consistent order:
    # 1) identifiers and time
    desired = []
    for k in ['VehId', 'Trip', 'SecTime_ms']:
        if k in out_df.columns:
            desired.append(k)

    # 2) 经纬度（单列）
    for k in ['Longitude', 'Latitude', 'Matched_Longitude', 'Matched_Latitude']:
        if k in out_df.columns:
            desired.append(k)

    # 3) speed & acceleration
    for k in ['Vehicle Speed[km/h]_mean', 'Acceleration[m/s^2]_mean']:
        if k in out_df.columns:
            desired.append(k)

    # 4) electrical core metrics: I, V, P (mean then rms), SOC mode, Energy
    # look for common suffixes
    def pick(cols, substr):
        return [c for c in cols if substr in c]

    all_cols = list(out_df.columns)
    # HV Battery Current
    for k in pick(all_cols, 'HV Battery Current[A]'):
        if k not in desired:
            desired.append(k)
    # HV Battery Voltage
    for k in pick(all_cols, 'HV Battery Voltage[V]'):
        if k not in desired:
            desired.append(k)
    # HV Battery Power
    for k in pick(all_cols, 'HV Battery Power[W]'):
        if k not in desired:
            desired.append(k)

    # SOC mode
    for k in all_cols:
        if '_mode' in k and ('SOC' in k or 'HV Battery SOC' in k):
            if k not in desired:
                desired.append(k)

    # Energy_Consumption_sum
    if f"{E_COL}_sum" in all_cols:
        desired.append(f"{E_COL}_sum")

    # 5) speed limit / flags modes
    for k in all_cols:
        if k.endswith('_mode') and k not in desired:
            desired.append(k)

    # remove unwanted elevation column if present (legacy name)
    for drop_elev in ['Elevation Smoothed[m]_mean']:
        if drop_elev in all_cols:
            all_cols.remove(drop_elev)

    # finally append remaining columns in original order
    for c in all_cols:
        if c not in desired:
            desired.append(c)

    # apply ordering
    out_df = out_df[[c for c in desired if c in out_df.columns]]

    # write
    ensure_output_dir(os.path.dirname(out_file))
    out_df.to_csv(out_file, index=False)


def run(input_root: str = INPUT_ROOT, output_root: str = OUTPUT_ROOT):
    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Input root not found: {input_root}")
    os.makedirs(output_root, exist_ok=True)

    categories = [d for d in sorted(os.listdir(input_root)) if os.path.isdir(os.path.join(input_root, d))]
    if not categories:
        # fallback to single-level input
        categories = [""]

    for cat in categories:
        cat_in = os.path.join(input_root, cat) if cat else input_root
        cat_out = os.path.join(output_root, cat) if cat else output_root
        os.makedirs(cat_out, exist_ok=True)
        print(f"Processing category: {cat_in} -> {cat_out}")

        for veh in sorted(os.listdir(cat_in)):
            veh_in = os.path.join(cat_in, veh)
            if not os.path.isdir(veh_in):
                continue
            veh_out = os.path.join(cat_out, veh)
            os.makedirs(veh_out, exist_ok=True)

            trip_files = sorted([f for f in os.listdir(veh_in) if f.lower().endswith('.csv')])
            for fname in trip_files:
                in_file = os.path.join(veh_in, fname)
                out_file = os.path.join(veh_out, fname)
                try:
                    process_trip_file(in_file, out_file)
                    print(f" Wrote {out_file}")
                except Exception as e:
                    print(f" Failed {in_file}: {e}")


if __name__ == '__main__':
    run()
