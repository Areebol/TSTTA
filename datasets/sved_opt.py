import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from torch.utils.data import Dataset
from utils.timefeatures import time_features

class sVED(Dataset):
    """
    sVED Dataset with One-Hot Encoded Focus Points.
    Focus Points are expanded from 1 categorical column into 16 binary columns (0.0/1.0).
    """
    VEHICLE_IDS = ("10", "455")
    DEFAULT_EV_ROOT = Path("./data/sved")

    # 定义映射关系
    FOCUS_POINTS_MAP = {
        "crossing": 1,
        "traffic_signals": 2,
        "stop": 3,
        "turning_loop": 4,
        "bump": 5,
        "turning_circle": 6,
        "motorway_junction": 7,
        "hump": 8,
        "lift_gate": 9,
        "gate": 10,
        "give_way": 11,
        "bollard": 12,
        "level_crossing": 13,
        "roundabout": 14,
        "mini_roundabout": 15,
        "swing_gate": 16,
    }
    
    # 获取有序的 Focus Point 名称列表 (按 ID 排序 1-16)
    _SORTED_FP_NAMES = sorted(FOCUS_POINTS_MAP, key=FOCUS_POINTS_MAP.get)
    # 生成对应的列名，例如: "FP_crossing", "FP_traffic_signals", ...
    FP_COLUMNS = [f"FP_{name}" for name in _SORTED_FP_NAMES]

    # --- 1. 修改 FEATURE_COLUMNS ---
    # 移除了原始的 "Focus Points"，加入了 16 个展开后的列
    FEATURE_COLUMNS = [
        "OAT[DegC]",                    # 0: Static
        "Air Conditioning Power[Watts]",# 1: Static
        "Heater Power[Watts]",          # 2: Static
        "Elevation Smoothed[m]",        # 3: Static
        "Gradient Smoothed",            # 4: Dynamic
        "Speed Limit[km/h]",            # 5: Static
        "Intersection",                 # 6: Static
        "Bus Stops",                    # 7: Static
        # --- Focus Points Expansion Start ---
    ] + FP_COLUMNS + [                  # 8-23: FP_crossing ... FP_swing_gate
        # --- Focus Points Expansion End ---
        "HV Battery SOC[%]",            # 24: Static
        "HV Battery Current[A]",        # 25: Dynamic
        "HV Battery Voltage[V]",        # 26: Dynamic
        "Vehicle Speed[km/h]",          # 27: Dynamic (Target)
        "Energy_Consumption",           # 28: Dynamic (Target)
    ]

    def __init__(
        self,
        data_dir: str,
        n_var: int,
        seq_len: int,
        label_len: int,
        pred_len: int,
        features: str,
        timeenc: int,
        freq: str,
        date_idx: int,
        target_start_idx: int,
        scale="standard",
        split="train",
        train_ratio=0.7,
        test_ratio=0.2,
        train_vehicle_ids=None,
        val_vehicle_ids=None,
        test_vehicle_ids=None,
        min_test_len=300,
    ):
        assert split in ("train", "val", "test")
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.features = features
        self.timeenc = timeenc
        self.freq = freq
        self.date_idx = date_idx
        self.target_start_idx = target_start_idx
        self.scale = scale
        self.split = split
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.train_vehicle_ids = train_vehicle_ids
        self.val_vehicle_ids = val_vehicle_ids if train_vehicle_ids is not None else train_vehicle_ids
        self.test_vehicle_ids = test_vehicle_ids
        self.min_test_len = min_test_len

        # --- 2. 修改 future_known ---
        # 同样移除 "Focus Points"，加入展开后的 FP 列
        # 这些信息在推理时是未来已知的 (Future Known)
        base_future_known = [
            "Elevation Smoothed[m]",
            "Gradient Smoothed",
            "Speed Limit[km/h]",
            "Intersection",
            "Bus Stops",
        ]
        self._future_known_names = base_future_known + self.FP_COLUMNS
        
        # 重新计算索引
        self._future_known_idx = [self.FEATURE_COLUMNS.index(n) for n in self._future_known_names]
        self._aug_per_step = len(self._future_known_idx)
        self._aug_dim = self._aug_per_step
        
        # 自动计算总维度 (Feature数 + 扩展的未来特征数)
        self.n_var = len(self.FEATURE_COLUMNS) + self._aug_dim

        # read data
        (
            self.train,
            self.val,
            self.test,
            self.train_stamp,
            self.val_stamp,
            self.test_stamp,
        ) = self._load_data()

        assert self.train.shape[1] == self.n_var

        self._normalize_data()

        self._subsets_cache = {"train": None, "val": None, "test": None}
        self._subsets_meta = {"train": {}, "val": {}, "test": {}}

    @staticmethod
    def _to_binary_flag(v) -> float:
        if pd.isna(v):
            return 0.0
        s = str(v).strip().lower()
        if s in ("", "0", "0.0", "nan", "none"):
            return 0.0
        try:
            return 0.0 if float(s) == 0.0 else 1.0
        except Exception:
            return 1.0

    def _map_focus_point_to_id(self, v) -> int:
        if pd.isna(v):
            return 0
        key = str(v).strip().lower()
        return self.FOCUS_POINTS_MAP.get(key, 0)

    def _resolve_ev_root(self) -> Path:
        base = Path(self.data_dir)
        if (base / "EV").exists():
            return base / "EV"
        if base.name == "EV" and base.exists():
            return base
        if self.DEFAULT_EV_ROOT.exists():
            return self.DEFAULT_EV_ROOT
        raise FileNotFoundError(f"EV root not found: {base}")

    def _collect_split_file_lists(self, ev_root: Path):
        train_files, val_files, test_files = [], [], []
        def _resolve_ids(ids):
            if ids is None: return self.VEHICLE_IDS
            if isinstance(ids, str): return (ids,)
            return tuple(ids)
        train_vids = _resolve_ids(self.train_vehicle_ids)
        val_vids = _resolve_ids(self.val_vehicle_ids)
        test_vids = _resolve_ids(self.test_vehicle_ids)
        all_vids = sorted(list(set(train_vids) | set(val_vids) | set(test_vids)))
        for vid in all_vids:
            vdir = ev_root / str(vid)
            if not vdir.exists(): continue
            files = sorted(vdir.glob("*.csv"))
            n = len(files)
            if n == 0: continue
            n_train = int(n * self.train_ratio)
            n_test = int(n * self.test_ratio)
            n_val = n - n_train - n_test
            n_train = max(0, min(n_train, n))
            n_val = max(0, min(n_val, n - n_train))
            n_test = max(0, n - n_train - n_val)
            if vid in train_vids: train_files.extend(files[:n_train])
            if vid in val_vids: val_files.extend(files[n_train:n_train + n_val])
            if vid in test_vids: test_files.extend(files[n_train + n_val:])
        self.test_files = test_files
        return {"train": train_files, "val": val_files, "test": test_files}

    def _load_data(self):
        ev_root = self._resolve_ev_root()
        
        # --- 3. 修改读取列的逻辑 ---
        # 原始 CSV 中并没有 "FP_crossing" 等列，只有 "Focus Points"
        # 所以我们需要：
        #   1. 找出 FEATURE_COLUMNS 中那些存在于 CSV 的列 (非 FP 列)
        #   2. 显式添加 "Focus Points" 以便后续处理
        csv_raw_cols = [c for c in self.FEATURE_COLUMNS if not c.startswith("FP_")]
        cols_needed_read = ["VehId", "Trip", "SecTime_ms", "Focus Points"] + csv_raw_cols
        # 去重 (防止 Focus Points 本身在 csv_raw_cols 里)
        cols_needed_read = list(set(cols_needed_read))

        base_ts = pd.Timestamp("1970-01-01 00:00:00")
        split_files = self._collect_split_file_lists(ev_root)

        train_chunks, val_chunks, test_chunks = [], [], []
        train_stamp_chunks, val_stamp_chunks, test_stamp_chunks = [], [], []
        
        self._segments = {"train": [], "val": [], "test": []}
        self._csv_lengths = {"train": [], "val": [], "test": []}
        running = {"train": 0, "val": 0, "test": 0}

        def _read_trip(p: Path):
            try:
                # 只读取必要的原始列
                df = pd.read_csv(p, usecols=lambda c: c in cols_needed_read)
            except Exception:
                df_full = pd.read_csv(p)
                df = df_full[[c for c in cols_needed_read if c in df_full.columns]].copy()

            if "SecTime_ms" in df.columns:
                df = df.sort_values("SecTime_ms").reset_index(drop=True)

            # --- 4. 核心处理：Focus Points One-Hot 展开 ---
            if "Focus Points" in df.columns:
                # 先映射成 ID (0-16)
                fp_ids = df["Focus Points"].apply(self._map_focus_point_to_id).astype(int)
                
                # 循环生成 16 个 binary 列
                # name 是 'crossing', code 是 1
                for name, code in self.FOCUS_POINTS_MAP.items():
                    col_name = f"FP_{name}"
                    # 生成 0.0 或 1.0
                    df[col_name] = (fp_ids == code).astype(float)
            else:
                # 如果缺失该列，全补 0
                for col in self.FP_COLUMNS:
                    df[col] = 0.0

            # 处理 Intersection, Bus Stops (转 0/1)
            for c in ["Intersection", "Bus Stops"]:
                if c in df.columns:
                    df[c] = df[c].apply(self._to_binary_flag).astype(float)

            # numeric coercion (不包括生成的 FP 列，因为已经是 float 了)
            raw_numeric_cols = [c for c in csv_raw_cols if c != "Focus Points"] + ["SecTime_ms"]
            for c in raw_numeric_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            # interpolation
            all_numeric = raw_numeric_cols + self.FP_COLUMNS
            # 只对原始数值列做插值，FP 列不需要（本身是离散的，且刚生成）
            # 不过做一下也无妨，只要别把 0/1 搞乱。这里只对 raw 做插值更安全
            interp_cols = [c for c in raw_numeric_cols if c in df.columns]
            if interp_cols:
                df[interp_cols] = df[interp_cols].interpolate(method="linear", limit_direction="both")
                df[interp_cols] = df[interp_cols].fillna(0.0)

            # --- 5. 整理列顺序 ---
            # 确保 DataFrame 拥有 FEATURE_COLUMNS 中定义的所有列，且顺序一致
            # 如果某些列在 CSV 没读到（比如 CSV 缺列），则补 0
            for c in self.FEATURE_COLUMNS:
                if c not in df.columns:
                    df[c] = 0.0
            
            # 这里的 feat_df 严格按照 FEATURE_COLUMNS 排序
            # SecTime_ms 用于生成 date，不在最终特征里
            secs = df["SecTime_ms"].astype(float) / 1000.0
            df["date"] = base_ts + pd.to_timedelta(secs, unit="s")

            dates = pd.to_datetime(df["date"])
            if self.timeenc == 0:
                stamp_df = pd.DataFrame({
                    "month": dates.dt.month,
                    "day": dates.dt.day,
                    "weekday": dates.dt.weekday,
                    "hour": dates.dt.hour,
                })
                stamp_np = stamp_df.to_numpy()
            else:
                stamp_np = time_features(dates.values, freq=self.freq).transpose(1, 0)

            # 提取最终数据矩阵
            data_np = df[self.FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)

            # future known append (现在包含了展开后的 FP 列)
            L = data_np.shape[0]
            fk = data_np[:, self._future_known_idx]  # [L, 22] (6 + 16)
            
            idx_next = np.arange(L) + self.seq_len
            idx_next[idx_next >= L] = L - 1
            next_step_fk = fk[idx_next]
            
            # 拼接到最后
            data_np = np.concatenate([data_np, next_step_fk.astype(np.float32)], axis=1)

            data_np = np.nan_to_num(data_np, nan=0.0, posinf=0.0, neginf=0.0)
            return data_np, stamp_np

        # ... (后续循环读取文件的逻辑保持不变) ...
        for split_name in ["train", "val", "test"]:
            files = split_files[split_name]
            for p in files:
                data_np, stamp_np = _read_trip(p)
                if data_np is None: continue
                L = len(data_np)
                if L < self.seq_len + self.pred_len: continue
                if split_name == "test" and L < self.min_test_len: continue

                if split_name == "train":
                    train_chunks.append(data_np)
                    train_stamp_chunks.append(stamp_np)
                elif split_name == "val":
                    val_chunks.append(data_np)
                    val_stamp_chunks.append(stamp_np)
                else:
                    test_chunks.append(data_np)
                    test_stamp_chunks.append(stamp_np)

                self._segments[split_name].append((running[split_name], L))
                self._csv_lengths[split_name].append(L)
                running[split_name] += L

        if not train_chunks and not val_chunks and not test_chunks:
            raise RuntimeError("No usable eVED trips found after filtering.")

        def _concat_or_empty(chunks, n_var):
            if chunks: return np.concatenate(chunks, axis=0)
            return np.zeros((0, n_var), dtype=np.float32)

        train = _concat_or_empty(train_chunks, self.n_var)
        val = _concat_or_empty(val_chunks, self.n_var)
        test = _concat_or_empty(test_chunks, self.n_var)
        
        train_stamp = _concat_or_empty(train_stamp_chunks, train_stamp_chunks[0].shape[1] if train_stamp_chunks else 4)
        val_stamp = _concat_or_empty(val_stamp_chunks, val_stamp_chunks[0].shape[1] if val_stamp_chunks else 4)
        test_stamp = _concat_or_empty(test_stamp_chunks, test_stamp_chunks[0].shape[1] if test_stamp_chunks else 4)

        self._valid_starts = {}
        total_needed = self.seq_len + self.pred_len
        for split_name in ["train", "val", "test"]:
            starts = []
            for seg_start, seg_len in self._segments[split_name]:
                max_start = seg_len - total_needed
                if max_start >= 0:
                    starts.extend(seg_start + np.arange(max_start + 1))
            self._valid_starts[split_name] = list(map(int, starts))

        self._build_test_csv_window_indices(total_needed)

        return train, val, test, train_stamp, val_stamp, test_stamp

    def _build_test_csv_window_indices(self, total_needed: int):
        test_starts = self._valid_starts.get("test", [])
        csv_lengths = self._csv_lengths.get("test", [])
        self._test_csv_windows = []
        if not test_starts or not csv_lengths: return
        csv_row_starts = []
        cur = 0
        for L in csv_lengths:
            csv_row_starts.append(cur)
            cur += L
        start_ptr = 0
        n_starts = len(test_starts)
        for csv_i, row_start in enumerate(csv_row_starts):
            row_end = row_start + csv_lengths[csv_i]
            left = row_start
            right = row_end - total_needed
            if right < left:
                self._test_csv_windows.append((start_ptr, start_ptr))
                continue
            csv_win_start = start_ptr
            while csv_win_start < n_starts and test_starts[csv_win_start] < left:
                csv_win_start += 1
            csv_win_end = csv_win_start
            while csv_win_end < n_starts and left <= test_starts[csv_win_end] <= right:
                csv_win_end += 1
            self._test_csv_windows.append((csv_win_start, csv_win_end))
            start_ptr = csv_win_end

    def __len__(self):
        starts = self._valid_starts.get(self.split, [])
        return len(starts)

    def __getitem__(self, index):
        if self.split == "train":
            data, stamp = self.train, self.train_stamp
        elif self.split == "val":
            data, stamp = self.val, self.val_stamp
        else:
            data, stamp = self.test, self.test_stamp
        valid_starts = self._valid_starts[self.split]
        enc_start_idx = int(valid_starts[index])
        enc_end_idx = enc_start_idx + self.seq_len
        dec_start_idx = enc_end_idx - self.label_len
        dec_end_idx = dec_start_idx + self.label_len + self.pred_len
        enc_window = data[enc_start_idx:enc_end_idx]
        enc_window_stamp = stamp[enc_start_idx:enc_end_idx]
        dec_window = data[dec_start_idx:dec_end_idx]
        dec_window_stamp = stamp[dec_start_idx:dec_end_idx]
        return enc_window, enc_window_stamp, dec_window, dec_window_stamp

    def _normalize_data(self):
        Xtr, Xva, Xte = self.train.astype(np.float32), self.val.astype(np.float32), self.test.astype(np.float32)
        if self.scale == "standard":
            mu = np.nanmean(Xtr, axis=0)
            sd = np.nanstd(Xtr, axis=0)
            sd_safe = np.where(sd < 1e-8, 1.0, sd)
            self.train = (Xtr - mu) / sd_safe
            self.val = (Xva - mu) / sd_safe
            self.test = (Xte - mu) / sd_safe
            self.mu = mu
            self.sd = sd_safe
        elif self.scale == "min-max":
            mn = np.nanmin(Xtr, axis=0)
            mx = np.nanmax(Xtr, axis=0)
            denom = np.where((mx - mn) < 1e-8, 1.0, (mx - mn))
            self.train = (Xtr - mn) / denom
            self.val = (Xva - mn) / denom
            self.test = (Xte - mn) / denom
            self.mn = mn
            self.denom = denom
        elif self.scale == "min-max_fixed":
            return
        else:
            raise ValueError

    def inverse_transform(self, data):
        # 原来 Target 在 12, 13
        # 现在前面多了 16 列 Focus Points, 且去掉了 1 列 Raw FP
        # 净增加 15 列
        # 原索引: 12 (Vehicle Speed), 13 (Energy)
        # 新索引: 12 - 1 + 16 = 27, 28
        start = 27 
        end = 29 # 27, 28
        
        if self.scale == "standard":
            if hasattr(self, 'mu') and hasattr(self, 'sd'):
                return data * self.sd[start:end] + self.mu[start:end]
        elif self.scale == "min-max":
            if hasattr(self, 'mn') and hasattr(self, 'denom'):
                return data * self.denom[start:end] + self.mn[start:end]
        return data

    def get_test_num_windows(self) -> int:
        return len(self._valid_starts.get("test", []))
    def get_num_test_csvs(self) -> int:
        return len(self._csv_lengths.get("test", []))
    def get_test_csv_window_range(self, csv_idx: int):
        if not hasattr(self, "_test_csv_windows"): return 0, 0
        return self._test_csv_windows[csv_idx]
    def get_test_windows_for_csv(self, csv_idx: int):
        start_idx, end_idx = self.get_test_csv_window_range(csv_idx)
        if start_idx >= end_idx: return None
        indices = list(range(start_idx, end_idx))
        return indices