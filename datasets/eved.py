import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
# 不再继承 ForecastingDataset，只保留接口风格
from torch.utils.data import Dataset
from utils.timefeatures import time_features


class EVED(Dataset):
    """
    eVED: trip-level split per vehicle (by CSV count ratios). Windows never cross trip boundaries.
    Focus Points mapped to categorical codes (1..16), empty/unknown -> 0.
    Future road info (OAT, Elevation Smoothed[m], Gradient, Matchted Latitude[deg],
    Matched Longitude[deg], Speed Limit[km/h], Intersection, Bus Stops, Focus Points) is
    included in decoder inputs for the future horizon.
    """
    VEHICLE_IDS = ("10", "455", "541")
    DEFAULT_EV_ROOT = Path("./data/segmented_1s_eVED_v9/EV")

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

    # 13 covariates + 1 target = 14 variables
    FEATURE_COLUMNS = [
        "OAT[DegC]",
        "Air Conditioning Power[Watts]",
        "Heater Power[Watts]",
        "Elevation Smoothed[m]",
        "Gradient Smoothed",
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

        # 准备 future_known 信息
        self._future_known_names = [
            "Elevation Smoothed[m]",
            "Gradient Smoothed",
            "Speed Limit[km/h]",
            "Intersection",
            "Bus Stops",
            "Focus Points",
        ]
        self._future_known_idx = [self.FEATURE_COLUMNS.index(n) for n in self._future_known_names]
        self._aug_per_step = len(self._future_known_idx)  # 6
        self._aug_dim = self._aug_per_step
        # 用实际维度覆盖 n_var（14 + 6）
        self.n_var = len(self.FEATURE_COLUMNS) + self._aug_dim

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
        for vid in self.VEHICLE_IDS:
            vdir = ev_root / vid
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
            train_files.extend(files[:n_train])
            val_files.extend(files[n_train:n_train + n_val])
            test_files.extend(files[n_train + n_val:])
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
        elif self.scale == "min-max":
            mn = np.nanmin(Xtr, axis=0)
            mx = np.nanmax(Xtr, axis=0)
            denom = np.where((mx - mn) < 1e-8, 1.0, (mx - mn))
            self.train = (Xtr - mn) / denom
            self.val = (Xva - mn) / denom
            self.test = (Xte - mn) / denom
        elif self.scale == "min-max_fixed":
            # keep raw
            return
        else:
            raise ValueError

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

    def compute_window_stats(self, split: str, metrics=None):
        """
        计算每个有效窗口的统计量（可插拔）。
        - split: "train"/"val"/"test"
        - metrics: dict 名称->函数, 每个函数签名 (hist, fut, feat_names) -> number
          hist: ndarray [seq_len, n_base_feats]
          fut:  ndarray [pred_len, n_base_feats]
          feat_names: list of base feature names (self.FEATURE_COLUMNS)
        如果 metrics 为 None，则使用一组默认统计量（见下）。
        返回 DataFrame，每行对应一个 start（列包含 "start" 与各统计量）。
        """
        assert split in ("train", "val", "test")
        starts = self._valid_starts.get(split, [])
        if not starts:
            return pd.DataFrame()

        # base arrays: choose split array (un-normalized already stored in self.train/val/test)
        data = getattr(self, split)
        # base feature column count (exclude appended future-known)
        base_n = len(self.FEATURE_COLUMNS)

        # default metric functions
        def _default_metrics():
            FC = self.FEATURE_COLUMNS
            idx = lambda name: FC.index(name)
            i_speed = idx("Vehicle Speed[km/h]")
            i_speed_limit = idx("Speed Limit[km/h]")
            i_grad = idx("Gradient Smoothed")
            i_ele = idx("Elevation Smoothed[m]")
            i_inter = idx("Intersection")
            i_bus = idx("Bus Stops")
            i_focus = idx("Focus Points")
            i_ac = idx("Air Conditioning Power[Watts]")
            i_heater = idx("Heater Power[Watts]")
            i_oat = idx("OAT[DegC]")
            i_current = idx("HV Battery Current[A]")
            i_voltage = idx("HV Battery Voltage[V]")
            i_soc = idx("HV Battery SOC[%]")

            def stop_ratio(hist, fut, feat_names):
                return float(np.mean(hist[:, i_speed] < 5.0))
            def freeflow_ratio(hist, fut, feat_names):
                # forward-fill speed limit in history (simple)
                sl = hist[:, i_speed_limit].copy().astype(float)
                last = None
                for k in range(len(sl)):
                    if sl[k] == 0 or np.isnan(sl[k]):
                        if last is not None:
                            sl[k] = last
                    else:
                        last = sl[k]
                with np.errstate(divide='ignore', invalid='ignore'):
                    r = np.where(sl > 0, hist[:, i_speed] / sl, 0.0)
                return float(np.nanmean(r))
            def speed_std(hist, fut, feat_names):
                s = float(np.nanstd(hist[:, i_speed]))
                return s
            def std_range(hist, fut, feat_names):
                return float(np.nanpercentile(hist[:, i_speed], 90) - np.nanpercentile(hist[:, i_speed], 10))
            def grad_mean_fut(hist, fut, feat_names):
                return float(np.nanmean(fut[:, i_grad]))
            def ascent_fut(hist, fut, feat_names):
                g = fut[:, i_grad]
                return float(np.sum(np.clip(g, 0.0, None)))
            def descent_fut(hist, fut, feat_names):
                g = fut[:, i_grad]
                return float(np.sum(np.abs(np.clip(g, None, 0.0))))
            def ele_range_fut(hist, fut, feat_names):
                return float(np.nanmax(fut[:, i_ele]) - np.nanmin(fut[:, i_ele]))
            def inter_cnt(hist, fut, feat_names):
                return int(np.nansum(fut[:, i_inter]))
            def bus_cnt(hist, fut, feat_names):
                return int(np.nansum(fut[:, i_bus]))
            def focus_cnt(hist, fut, feat_names):
                return int(np.nansum(fut[:, i_focus]))
            def limit_std(hist, fut, feat_names):
                return float(np.nanstd(fut[:, i_speed_limit]))
            def limit_changes(hist, fut, feat_names):
                return int(np.sum(np.diff(fut[:, i_speed_limit]) != 0))
            def hvac_mean(hist, fut, feat_names):
                return float(np.nanmean(hist[:, i_ac] + hist[:, i_heater]))
            def oat_mean(hist, fut, feat_names):
                return float(np.nanmean(hist[:, i_oat]))
            def power_mean(hist, fut, feat_names):
                power = hist[:, i_current] * hist[:, i_voltage]
                return float(np.nanmean(power))
            def power_p90(hist, fut, feat_names):
                power = hist[:, i_current] * hist[:, i_voltage]
                return float(np.nanpercentile(power, 90))
            def soc_slope(hist, fut, feat_names):
                soc = hist[:, i_soc]
                if soc.size >= 2 and np.all(np.isfinite(soc)):
                    try:
                        return float(np.polyfit(np.arange(soc.size), soc, 1)[0])
                    except Exception:
                        return 0.0
                return 0.0

            return {
                "stop_ratio": stop_ratio,
                "freeflow_ratio": freeflow_ratio,
                "speed_std": speed_std,
                "std_range": std_range,
                "grad_mean_fut": grad_mean_fut,
                "ascent_fut": ascent_fut,
                "descent_fut": descent_fut,
                "ele_range_fut": ele_range_fut,
                "inter_cnt": inter_cnt,
                "bus_cnt": bus_cnt,
                "focus_cnt": focus_cnt,
                "limit_std": limit_std,
                "limit_changes": limit_changes,
                "hvac_mean": hvac_mean,
                "OAT_mean": oat_mean,
                "power_mean": power_mean,
                "power_p90": power_p90,
                "soc_slope": soc_slope,
            }

        metrics_map = metrics if metrics is not None else _default_metrics()
        records = []
        total_needed = self.seq_len + self.pred_len
        for start in starts:
            h0 = int(start)
            h1 = h0 + self.seq_len
            f0 = h1
            f1 = f0 + self.pred_len
            if f1 > data.shape[0]:
                continue
            hist = data[h0:h1, :base_n]
            fut = data[f0:f1, :base_n]
            row = {"start": int(start)}
            for name, fn in metrics_map.items():
                try:
                    row[name] = fn(hist, fut, self.FEATURE_COLUMNS)
                except Exception:
                    row[name] = np.nan
            records.append(row)

        df = pd.DataFrame.from_records(records)
        
        if {"inter_cnt", "bus_cnt", "focus_cnt"}.issubset(df.columns) and not df.empty:
            for col in ("inter_cnt", "bus_cnt", "focus_cnt"):
                m = df[col].mean()
                s = df[col].std(ddof=0)
                df[col + "_z"] = 0.0 if s < 1e-8 else (df[col] - m) / s
            df["poi_density"] = df["inter_cnt_z"] + df["bus_cnt_z"] + df["focus_cnt_z"]
        return df

    def get_subsets(
        self,
        split,
        classifiers = None,
        thresholds  = None,
        top_k_quantile = 0.9,
        use_cache = True,
        save = True,
        overwrite = False,
    ):
        """
        通用子集划分：
        - classifiers: dict name->callable(df) 返回 boolean Series 或 start 列表
        - 如果 classifiers 为 None，则用一套基于分位数的默认规则（与常见场景匹配）
        - thresholds 可覆盖默认阈值
        返回 dict: name -> list of start indices
        """
        persist_path = f"./cache/s_{self.seq_len}_p_{self.pred_len}_subsets_v9.json"
        os.makedirs(Path(persist_path).parent, exist_ok=True)

        if use_cache and Path(persist_path).exists():
            self.load_subsets(persist_path)
            if self._subsets_cache.get(split) is not None:
                print("using cache")
                return dict(self._subsets_cache[split])
        df = self.compute_window_stats(split)
        if df.empty:
            return {}

        thr = thresholds or {}
        out = {}

        # default classifier rules expressed as lambdas on df
        if classifiers is None:
            q_up = df["ascent_fut"].quantile(top_k_quantile) if "ascent_fut" in df.columns else 0.0
            q_down = df["descent_fut"].quantile(top_k_quantile) if "descent_fut" in df.columns else 0.0
            q_stop = df["stop_ratio"].quantile(top_k_quantile) if "stop_ratio" in df.columns else 0.0
            q_poi = df["poi_density"].quantile(top_k_quantile) if "poi_density" in df.columns else 0.0
            q_power = df["power_mean"].quantile(top_k_quantile) if "power_mean" in df.columns else 0.0
            q_hvac = df["hvac_mean"].quantile(top_k_quantile) if "hvac_mean" in df.columns else 0.0
            if "OAT_mean" in df.columns:
                q_oat_high = df["OAT_mean"].quantile(top_k_quantile)
                q_oat_low = df["OAT_mean"].quantile(1.0 - top_k_quantile)
            else:
                q_oat_high = q_oat_low = 0.0

            classifiers = {
                "uphill": lambda d: d["ascent_fut"] >= thr.get("ascent_fut", q_up),
                "downhill": lambda d: d["descent_fut"] >= thr.get("descent_fut", q_down),
                "congested": lambda d: d["stop_ratio"] >= thr.get("stop_ratio", q_stop),
                "facility_dense": lambda d: d["poi_density"] >= thr.get("poi_density", q_poi),
                "speed_limit_switch": lambda d: d["limit_changes"] >= thr.get("limit_changes", 1),
                "high_hvac": lambda d: d["hvac_mean"] >= thr.get("hvac_mean", q_hvac),
                "high_load": lambda d: d["power_mean"] >= thr.get("power_mean", q_power),
                "low_load": lambda d: d["power_mean"] <= thr.get("power_mean", d["power_mean"].quantile(0.1) if "power_mean" in d else 0.0),
                "high_temp": lambda d: ("OAT_mean" in d.columns) and (d["OAT_mean"] >= thr.get("OAT_mean_high", q_oat_high)),
                "low_temp":  lambda d: ("OAT_mean" in d.columns) and (d["OAT_mean"] <= thr.get("OAT_mean_low", q_oat_low)),
            }

        # apply classifiers
        for name, clf in classifiers.items():
            try:
                res = clf(df)
                if isinstance(res, (list, np.ndarray, pd.Series)):
                    # boolean mask -> select starts
                    if isinstance(res, (np.ndarray, pd.Series)) and res.dtype == bool:
                        sel = df.loc[res, "start"].tolist()
                    else:
                        # assume list of starts or indices
                        sel = list(res)
                elif isinstance(res, (bool, np.bool_)):
                    sel = df.loc[res, "start"].tolist() if res else []
                else:
                    # unexpected: try interpret as boolean Series by applying to df
                    sel = []
                out[name] = sel
            except Exception:
                out[name] = []
        # cache results and metadata
        self._subsets_cache[split] = dict(out)
        self._subsets_meta[split] = {"thresholds": thr, "top_k": top_k_quantile}
        # optional persistence
        if save and persist_path is not None:
            print("saving subsets to", persist_path)
            self.save_subsets(persist_path, split=split, overwrite=overwrite)
        return out

    def save_subsets(self, path, split=None, overwrite=False):
        """
        Persist cached subsets to a JSON file.
        - path: file path to write (will create parent dirs).
        - split: if provided, only save that split; otherwise save all cached splits.
        - overwrite: if False and file exists, merge existing content with current cache.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        out_data = {}

        # load existing if we will merge
        if p.exists() and not overwrite:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    out_data = json.load(f)
            except Exception:
                out_data = {}

        if split is not None:
            if self._subsets_cache.get(split) is None:
                raise ValueError(f"No cached subsets for split={split}")
            out_data[split] = self._subsets_cache[split]
        else:
            # all splits: only include those that are cached
            for s in ("train", "val", "test"):
                if self._subsets_cache.get(s) is not None:
                    out_data[s] = self._subsets_cache[s]

        with open(p, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)

    def load_subsets(self, path, merge=True):
        """
        Load subsets from JSON file into in-memory cache.
        - merge: if True, merge with existing cache; otherwise replace.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not merge:
            self._subsets_cache = {"train": None, "val": None, "test": None}
        for s, v in data.items():
            if s in self._subsets_cache:
                self._subsets_cache[s] = v

    def clear_subsets_cache(self, split = None):
        """Clear cached subsets (in-memory)."""
        if split is None:
            self._subsets_cache = {"train": None, "val": None, "test": None}
            self._subsets_meta = {"train": {}, "val": {}, "test": {}}
        else:
            if split in self._subsets_cache:
                self._subsets_cache[split] = None
                self._subsets_meta[split] = {}

    def visualize_subsets(
        self,
        split,
        subset_names = None,
        features = None,
        max_samples = 6,
        window_len = None,
        save_dir = None,
        show = True,
    ):
        """
        Visualize windows belonging to one or more subsets.
        - split: "train"/"val"/"test"
        - subset_names: list of subset keys (if None, will compute and use all available subsets)
        - features: list of FEATURE_COLUMNS names to plot (default: ['Vehicle Speed[km/h]', 'OAT[DegC]', 'Energy_Consumption'])
        - max_samples: max windows to plot per subset
        - window_len: override window length (default = seq_len + pred_len)
        - save_dir: if provided, save figures under this directory as PNG files
        - show: whether to call plt.show()
        Returns dict: subset_name -> list of figure file paths (or matplotlib.Figure objects if not saved)
        """
        if plt is None:
            raise RuntimeError("matplotlib not available; install it (pip install matplotlib) to use visualize_subsets")
        assert split in ("train", "val", "test")
        wl = int(window_len) if window_len is not None else (self.seq_len + self.pred_len)
        data = getattr(self, split)
        stamp = getattr(self, f"{split}_stamp", None)
        # ensure subsets exist (compute without persisting)
        subsets = self.get_subsets(split, use_cache=False, save=False)
        if not subsets:
            return {}
        if subset_names is None:
            subset_names = list(subsets.keys())
        if features is None:
            features = ["Vehicle Speed[km/h]", "OAT[DegC]", "Energy_Consumption"]
        # filter to available features
        feat_names = [f for f in features if f in self.FEATURE_COLUMNS]
        if not feat_names:
            raise ValueError("no requested features found in FEATURE_COLUMNS")
        saved = {}
        save_dir = Path(save_dir) if save_dir is not None else None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
        for name in subset_names:
            starts = subsets.get(name, []) or []
            if len(starts) == 0:
                continue
            n_plot = min(len(starts), max_samples)
            paths_or_figs = []
            for i, st in enumerate(starts[:n_plot]):
                st = int(st)
                if st + wl > data.shape[0]:
                    continue
                window = data[st: st + wl, :len(self.FEATURE_COLUMNS)]
                # x axis: sample indices (seconds)
                x = np.arange(window.shape[0])
                nrows = len(feat_names)
                fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(6, 2.2 * nrows), sharex=True)
                if nrows == 1:
                    axes = [axes]
                for ax, fn in zip(axes, feat_names):
                    idx = self.FEATURE_COLUMNS.index(fn)
                    y = window[:, idx]
                    ax.plot(x, y, label=fn, linewidth=1.2)
                    ax.axvline(self.seq_len - 0.5, color="k", linestyle="--", alpha=0.6)  # history / future boundary
                    ax.set_ylabel(fn)
                    ax.grid(True, alpha=0.3)
                axes[-1].set_xlabel("sample (s, relative)")
                fig.suptitle(f"{name} | start={st} | window={wl} | idx={i+1}/{n_plot}", fontsize=10)
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                if save_dir is not None:
                    fname = f"{split}__{name}__start{st}__#{i+1}.png"
                    outpath = save_dir / fname
                    fig.savefig(str(outpath), dpi=150)
                    paths_or_figs.append(str(outpath))
                    plt.close(fig)
                else:
                    paths_or_figs.append(fig)
                    if show:
                        fig.show()
            saved[name] = paths_or_figs
        if show and save_dir is None:
            plt.show()
        return saved