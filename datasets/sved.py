import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
# 不再继承 ForecastingDataset，只保留接口风格
from torch.utils.data import Dataset
from utils.timefeatures import time_features


class sVED(Dataset):
    """
    eVED: trip-level split per vehicle (by CSV count ratios). Windows never cross trip boundaries.
    Focus Points mapped to categorical codes (1..16), empty/unknown -> 0.
    Future road info (OAT, Elevation Smoothed[m], Gradient, Matchted Latitude[deg],
    Matched Longitude[deg], Speed Limit[km/h], Intersection, Bus Stops, Focus Points) is
    included in decoder inputs for the future horizon.
    """
    VEHICLE_IDS = ("10", "455")
    DEFAULT_EV_ROOT = Path("./data/sved")

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

    # 12 covariates + 2 target = 14 variables
    FEATURE_COLUMNS = [
        "OAT[DegC]",
        "Air Conditioning Power[Watts]",
        "Heater Power[Watts]",
        "Elevation Smoothed[m]",
        "Gradient Smoothed",
        # "Gradient Smoothed_2",
        "Speed Limit[km/h]",
        "Intersection",
        "Bus Stops",
        "Focus Points",
        "HV Battery SOC[%]",
        "HV Battery Current[A]",
        "HV Battery Voltage[V]",
        "Vehicle Speed[km/h]",
        "Energy_Consumption",
    ]

    # FEATURE_COLUMNS = [
    #     "Vehicle Speed[km/h]",
    #     "Energy_Consumption",
    # ]

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

        ### 准备 future_known 信息
        self._future_known_names = [
            "Elevation Smoothed[m]",
            "Gradient Smoothed",
            # "Gradient Smoothed_2",
            "Speed Limit[km/h]",
            "Intersection",
            "Bus Stops",
            "Focus Points",
        ]
        self._future_known_idx = [self.FEATURE_COLUMNS.index(n) for n in self._future_known_names]
        self._aug_per_step = len(self._future_known_idx)  # 6
        self._aug_dim = self._aug_per_step
        ## 用实际维度覆盖 n_var（14 + 6）
        self.n_var = len(self.FEATURE_COLUMNS) + self._aug_dim
        
        # self.n_var = len(self.FEATURE_COLUMNS)

        # read data and obtain the start indinces of each samples
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

        # 缓存 subsets
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

    def _map_focus_point(self, v) -> int:
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
            if ids is None:
                return self.VEHICLE_IDS
            if isinstance(ids, str):
                return (ids,)
            return tuple(ids)

        train_vids = _resolve_ids(self.train_vehicle_ids)
        val_vids = _resolve_ids(self.val_vehicle_ids)
        test_vids = _resolve_ids(self.test_vehicle_ids)
        
        # Collect all unique vehicle IDs involved
        all_vids = sorted(list(set(train_vids) | set(val_vids) | set(test_vids)))
        # print('all_vids', all_vids, train_vids, test_vids)
        for vid in all_vids:
            vdir = ev_root / str(vid)
            if not vdir.exists():
                continue
            files = sorted(vdir.glob("*.csv"))
            n = len(files)
            if n == 0:
                continue
            n_train = int(n * self.train_ratio)
            n_test = int(n * self.test_ratio)
            n_val = n - n_train - n_test
            n_train = max(0, min(n_train, n))  # clamp
            n_val = max(0, min(n_val, n - n_train))
            n_test = max(0, n - n_train - n_val)
            
            if vid in train_vids:
                train_files.extend(files[:n_train])
            if vid in val_vids:
                val_files.extend(files[n_train:n_train + n_val])
            if vid in test_vids:
                test_files.extend(files[n_train + n_val:])

        self.test_files = test_files
        return {"train": train_files, "val": val_files, "test": test_files}

    def _load_data(self):
        ev_root = self._resolve_ev_root()
        cols_needed = ["VehId", "Trip", "SecTime_ms"] + self.FEATURE_COLUMNS
        base_ts = pd.Timestamp("1970-01-01 00:00:00")

        split_files = self._collect_split_file_lists(ev_root)

        train_chunks, val_chunks, test_chunks = [], [], []
        train_stamp_chunks, val_stamp_chunks, test_stamp_chunks = [], [], []

        # Segment boundaries per split for windowing (global_row_start, length)
        self._segments = {"train": [], "val": [], "test": []}
        # 每个 csv 的长度
        self._csv_lengths = {"train": [], "val": [], "test": []}

        # running offsets per split（全局起点）
        running = {"train": 0, "val": 0, "test": 0}

        def _read_trip(p: Path):
            try:
                df = pd.read_csv(p, usecols=cols_needed)
            except Exception:
                df_full = pd.read_csv(p)
                if any(c not in df_full.columns for c in cols_needed):
                    return None, None
                df = df_full[cols_needed]

            # sort in time within trip
            if "SecTime_ms" in df.columns:
                df = df.sort_values("SecTime_ms").reset_index(drop=True)

            # flags and categorical mapping
            for c in ["Intersection", "Bus Stops"]:
                if c in df.columns:
                    df[c] = df[c].apply(self._to_binary_flag).astype(float)
            if "Focus Points" in df.columns:
                df["Focus Points"] = df["Focus Points"].apply(self._map_focus_point).astype(int)

            # numeric coercion for all feature columns + SecTime_ms
            numeric_cols = [c for c in self.FEATURE_COLUMNS if c != "Focus Points"] + ["SecTime_ms"]
            for c in numeric_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            # per-trip interpolation and fill for missing values
            if numeric_cols:
                df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
                df[numeric_cols] = df[numeric_cols].fillna(0.0)

            # per-trip synthetic timestamp from seconds
            secs = df["SecTime_ms"].astype(float) / 1000.0
            df["date"] = base_ts + pd.to_timedelta(secs, unit="s")

            # stamp
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

            # features (exclude date column)
            feat_df = df[["date"] + self.FEATURE_COLUMNS]
            assert feat_df.columns[self.date_idx] == "date"
            data_np = feat_df.iloc[:, 1:].to_numpy(dtype=np.float32, copy=False)

            # append only the next-step (t+1s) future-known covariates (6 dims)
            L = data_np.shape[0]
            fk = data_np[:, self._future_known_idx]  # [L, 6]
            # use the next seq_len features
            idx_next = np.arange(L) + self.seq_len
            idx_next[idx_next >= L] = L - 1  # clamp at trip end
            next_step_fk = fk[idx_next]      # [L, 6]
            data_np = np.concatenate([data_np, next_step_fk.astype(np.float32)], axis=1)  # [L, 16 + 6]

            # final NaN/Inf guard
            data_np = np.nan_to_num(data_np, nan=0.0, posinf=0.0, neginf=0.0)
            return data_np, stamp_np

        # 逐 split 读入并拼接
        for split_name in ["train", "val", "test"]:
            files = split_files[split_name]
            for p in files:
                data_np, stamp_np = _read_trip(p)
                if data_np is None:
                    continue
                L = len(data_np)
                if L < self.seq_len + self.pred_len:
                    continue  # too short
                
                # Filter out short trips for testing
                if split_name == "test" and L < self.min_test_len:
                    continue

                if split_name == "train":
                    train_chunks.append(data_np)
                    train_stamp_chunks.append(stamp_np)
                elif split_name == "val":
                    val_chunks.append(data_np)
                    val_stamp_chunks.append(stamp_np)
                else:
                    test_chunks.append(data_np)
                    test_stamp_chunks.append(stamp_np)

                # record (global_start, length) for each segment
                self._segments[split_name].append((running[split_name], L))
                self._csv_lengths[split_name].append(L)
                running[split_name] += L

        if not train_chunks and not val_chunks and not test_chunks:
            raise RuntimeError("No usable eVED trips found after filtering.")

        def _concat_or_empty(chunks, n_var: int):
            if chunks:
                return np.concatenate(chunks, axis=0)
            return np.zeros((0, n_var), dtype=np.float32)

        train = _concat_or_empty(train_chunks, self.n_var)
        val = _concat_or_empty(val_chunks, self.n_var)
        test = _concat_or_empty(test_chunks, self.n_var)
        train_stamp = _concat_or_empty(train_stamp_chunks, train_stamp_chunks[0].shape[1] if train_stamp_chunks else 4)
        val_stamp = _concat_or_empty(val_stamp_chunks, val_stamp_chunks[0].shape[1] if val_stamp_chunks else 4)
        test_stamp = _concat_or_empty(test_stamp_chunks, test_stamp_chunks[0].shape[1] if test_stamp_chunks else 4)

        # valid window starts per split (no cross-trip windows)
        self._valid_starts = {}
        total_needed = self.seq_len + self.pred_len
        for split_name in ["train", "val", "test"]:
            starts = []
            for seg_start, seg_len in self._segments[split_name]:
                max_start = seg_len - total_needed
                if max_start >= 0: # ignore those trips with short trip length
                    starts.extend(seg_start + np.arange(max_start + 1))
            self._valid_starts[split_name] = list(map(int, starts))

        # 基于 valid_starts['test'] 与 csv_lengths 构造每个 test csv 的 window 范围
        self._build_test_csv_window_indices(total_needed)

        return train, val, test, train_stamp, val_stamp, test_stamp

    def _build_test_csv_window_indices(self, total_needed: int):
        """
        为 test split 中的每个 CSV 构造其 window 起点在 self._valid_starts['test'] 中的下标范围 [start, end)。
        """
        test_starts = self._valid_starts.get("test", [])
        csv_lengths = self._csv_lengths.get("test", [])
        self._test_csv_windows = []  # list of (start_idx_in_valid_starts, end_idx_exclusive)

        if not test_starts or not csv_lengths:
            return

        # 每个 csv 在 test 数据中的行位置区间
        csv_row_starts = []
        cur = 0
        for L in csv_lengths:
            csv_row_starts.append(cur)
            cur += L

        # 对每个 csv，找到属于该 csv 的所有 window 起点的下标范围
        start_ptr = 0
        n_starts = len(test_starts)
        for csv_i, row_start in enumerate(csv_row_starts):
            row_end = row_start + csv_lengths[csv_i]
            # window 起点必须满足：row_start <= s <= row_end - total_needed
            left = row_start
            right = row_end - total_needed
            if right < left:
                # 该 csv 太短，没有任何有效 window
                self._test_csv_windows.append((start_ptr, start_ptr))
                continue

            # 找到 test_starts 中属于 [left, right] 的 index 范围
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
        """
        使用预先计算好的 valid_starts[split][index] 作为该样本的历史起点，
        保证窗口不跨 CSV / trip。
        """
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
            # keep raw
            return
        else:
            raise ValueError

    def inverse_transform(self, data):
        start = 12
        end = start + 2
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
        """
        返回 test split 中 CSV 的数量。
        """
        return len(self._csv_lengths.get("test", []))

    def get_test_csv_window_range(self, csv_idx: int):
        """
        返回第 csv_idx 个 CSV 对应的 window 下标范围 [start, end)，
        其中 start/end 是在 self._valid_starts['test'] 上的下标。
        """
        if not hasattr(self, "_test_csv_windows"):
            # 兼容旧模型：没有构建则返回空
            return 0, 0
        return self._test_csv_windows[csv_idx]

    def get_test_windows_for_csv(self, csv_idx: int):
        """
        返回某个 CSV 所有 test window 对应的 (enc_window, enc_stamp, dec_window, dec_stamp)。
        这里复用数据集原有 __getitem__ / index 到 window 的映射逻辑。
        """
        from torch.utils.data import Subset  # 仅在需要时导入

        # 利用 get_test_csv_window_range 得到该 csv 所有 window 的索引范围
        start_idx, end_idx = self.get_test_csv_window_range(csv_idx)
        if start_idx >= end_idx:
            return None  # 没有有效 window

        # 假设 ForecastingDataset.__getitem__(idx) 是基于全局 test window 索引
        indices = list(range(start_idx, end_idx))
        # 外部可以创建 Subset(self, indices) 和 DataLoader；此处为了不依赖 torch，这里只返回下标列表
        return indices