import os
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter, LinearLocator, LogLocator

class TTAVisualizer:
    def __init__(self, save_dir, cfg):
        self.save_dir = save_dir
        self.cfg = cfg
        # 设置绘图风格
        plt.style.use('seaborn-v0_8-muted')
        
        # 统一调大全局字体，解决横纵坐标及标签看不清的问题
        plt.rcParams.update({
            'font.size': 20,           
            'axes.labelsize': 22,      
            'axes.titlesize': 24,      
            'xtick.labelsize': 20,     
            'ytick.labelsize': 20,     
            'legend.fontsize': 18,     
            'lines.linewidth': 2.5     
        })

        # --- 修改 1: 调整画幅比例为 1:4 (高:宽) ---
        self._fig_w = 16  # 设置宽度为 16
        self._fig_h = self._fig_w / 4  # 高度为宽度的 1/4 (即 4.0)

    def _figsize_custom(self, w: float | None = None):
        w = self._fig_w if w is None else float(w)
        return (w, w / 4)

    def _set_yaxis_ticks(self, ax, n_ticks: int = 3):
        """
        通用辅助函数：用于那些没有使用 _set_nice_step_yaxis 的轴。
        """
        if ax is None:
            return
        n_ticks = int(n_ticks)
        n_ticks = max(2, n_ticks)

        scale = ax.get_yscale()
        if scale == 'log':
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=n_ticks))
        else:
            ax.yaxis.set_major_locator(LinearLocator(n_ticks))

        # 强制保留 1 位小数
        def _fmt_one_decimal(y, _pos=None):
            if y is None or not np.isfinite(y):
                return ""
            return f"{float(y):.1f}"

        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_one_decimal))

    def _set_nice_step_yaxis(
        self,
        ax,
        y_data: np.ndarray,
        *,
        n_ticks: int = 3,
        pad_frac: float = 0.1,  
        force_zero_if_nonneg: bool = False, 
    ):
        """
        设置Y轴：
        1. 严格固定 3 个刻度（Min, Mid, Max）。
        2. 刻度标签强制保留 1 位小数。
        3. 上下留白，让曲线铺满，且不显示网格。
        """
        if ax is None or ax.get_yscale() == 'log':
            return

        y_arr = np.asarray(y_data)
        try:
            y_min = float(np.nanmin(y_arr))
            y_max = float(np.nanmax(y_arr))
        except ValueError:
            return
        if not (np.isfinite(y_min) and np.isfinite(y_max)):
            return

        # 处理数值完全相同的情况
        if y_max == y_min:
            span = max(abs(y_max), 1.0)
            y_min -= 0.1 * span
            y_max += 0.1 * span
        
        # 确定是否强制从 0 开始 (通常设为 False 以铺满画布)
        if force_zero_if_nonneg and y_min >= 0:
            if y_min < (y_max - y_min) * 0.5: 
                y_min = 0.0

        # 计算刻度 (Min, Mid, Max)，不强制对称
        tick_min = y_min
        tick_max = y_max
        tick_mid = (tick_min + tick_max) / 2.0
        
        ticks = np.array([tick_min, tick_mid, tick_max])

        # 设置 Y 轴显示范围 (Limits) - 加上留白
        span = tick_max - tick_min
        if span <= 1e-12:
            span = 1.0
        
        pad = span * pad_frac
        ylim_bottom = tick_min - pad
        ylim_top = tick_max + pad

        ax.set_ylim(ylim_bottom, ylim_top)
        ax.set_yticks(ticks)

        # 强制使用 1 位小数格式化
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{float(y):.1f}"))

    def _set_ylim_with_padding(
        self,
        ax,
        y_data: np.ndarray | None,
        *,
        bottom0_if_nonneg: bool = True,
        pad_frac: float = 0.06,
    ):
        """根据数据设置 y 轴范围，并加一点上下 padding。"""
        if ax is None or y_data is None:
            return

        if ax.get_yscale() == 'log':
            return

        y_arr = np.asarray(y_data)
        try:
            y_min = float(np.nanmin(y_arr))
            y_max = float(np.nanmax(y_arr))
        except ValueError:
            return
        if not (np.isfinite(y_min) and np.isfinite(y_max)):
            return

        span = y_max - y_min
        if span <= 0:
            span = max(abs(y_max), 1.0)

        pad = max(span * float(pad_frac), 1e-9)

        if bottom0_if_nonneg and y_min >= 0:
            y_low = 0.0
        else:
            y_low = y_min - pad
        y_high = y_max + pad

        if y_high <= y_low:
            y_high = y_low + 1.0

        ax.set_ylim(y_low, y_high)

    def _axis_start_from_zero(self, ax, *, x: bool = True, y: bool = True, y_data: np.ndarray | None = None):
        """让 X 轴从 0 开始。"""
        if ax is None:
            return
        if x:
            ax.set_xlim(left=0)
            ax.margins(x=0)

    def plot_all(self, data_dict):
        os.makedirs(self.save_dir, exist_ok=True)
        if data_dict is None:
            print("Visualizer: No data provided to plot.")
            return

        print(f"Visualizer: Generating plots in {self.save_dir}...")
        self.plot_best_samples_per_channel(data_dict)

    def _plot_gating_strategy(self, gating, mse=None):
        # 这里的画幅也相应调整为 1:4 左右 (16, 4)
        fig, ax1 = plt.subplots(figsize=(16, 4)) 

        # --- 1. 处理 Gating 曲线 (左轴) ---
        x_gating = np.arange(len(gating))
        if gating.ndim > 1 and gating.shape[1] > 1:
            mean_gating = np.mean(gating, axis=1)
            ax1.plot(x_gating, mean_gating, label='Mean Gating Weight (λ)', color='#27ae60', linewidth=2, zorder=3)
            ax1.fill_between(x_gating, gating.min(axis=1), gating.max(axis=1), alpha=0.1, color='#27ae60')
        else:
            ax1.plot(x_gating, gating.flatten(), label='Gating Weight (λ)', color='#27ae60', linewidth=2, zorder=3)

        # 0 参考线
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        ax1.set_xlabel("Test Samples / Steps", fontsize=20)
        ax1.set_ylabel("Gating Weight (λ)", color='#27ae60', fontsize=20)
        ax1.tick_params(axis='y', labelcolor='#27ae60', labelsize=18)
        ax1.tick_params(axis='x', labelsize=18)
        
        # 去掉网格
        # ax1.grid(True, alpha=0.2) 

        self._axis_start_from_zero(ax1, x=True, y=False)
        self._set_nice_step_yaxis(ax1, gating, n_ticks=3, pad_frac=0.1, force_zero_if_nonneg=False)

        # --- 2. 处理 Error 曲线 (右轴) ---
        if mse is not None:
            ax2 = ax1.twinx()
            
            if len(mse) != len(gating):
                x_mse = np.linspace(0, len(gating) - 1, num=len(mse))
                ax2.plot(x_mse, mse, color='#e74c3c', alpha=0.15, label='Raw MSE')
                
                window = max(1, len(mse) // len(gating)) if len(mse) > len(gating) else 5
                mse_smooth = np.convolve(mse, np.ones(window)/window, mode='valid')
                x_smooth = np.linspace(0, len(gating) - 1, num=len(mse_smooth))
                ax2.plot(x_smooth, mse_smooth, color='#e74c3c', linewidth=1.5, label='Smoothed MSE', zorder=2)
            else:
                ax2.plot(x_gating, mse, color='#e74c3c', linewidth=1.5, alpha=0.6, label='MSE')

            ax2.set_ylabel("Prediction Error (MSE)", color='#e74c3c', fontsize=20)
            ax2.tick_params(axis='y', labelcolor='#e74c3c', labelsize=18)
            ax2.set_yscale('log') 

            self._axis_start_from_zero(ax2, x=True, y=False)
            self._set_yaxis_ticks(ax2, n_ticks=3)
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, framealpha=0.9)
        else:
            ax1.legend(loc='upper left', fontsize=18)

        plt.title("Gating Evolution vs. Prediction Error", fontsize=24)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "gating_vs_error.pdf"), bbox_inches='tight', dpi=150)
        plt.close()

        if gating.ndim > 1 and gating.shape[1] > 1:
            plt.figure(figsize=(16, 4))
            sns.heatmap(gating.T, cmap='YlGnBu', cbar_kws={'label': 'λ'})
            plt.title("Gating Weights Heatmap")
            plt.xlabel("Test Samples")
            plt.ylabel("Channel Index")
            plt.savefig(os.path.join(self.save_dir, "gating_heatmap.pdf"), bbox_inches='tight')
            plt.close()

    def plot_best_samples_per_channel(self, data_dict):
        base = data_dict['preds_base']
        tta = data_dict['preds_tta']
        gt = data_dict['gts']
        
        mse_base = np.mean((base - gt)**2, axis=1) 
        mse_tta = np.mean((tta - gt)**2, axis=1)   
        
        improvement = mse_base - mse_tta

        num_samples = improvement.shape[0]
        num_channels = gt.shape[2]
        
        # --- 修改 2: 每个通道保留 6 张图 ---
        top_k = 10
        k = min(top_k, num_samples)
        print(f"Visualizer: Plotting TOP-{k} MOST IMPROVED samples for {num_channels} channels...")

        for c_idx in range(num_channels):
            imp_c = improvement[:, c_idx]
            topk_indices = np.argsort(-imp_c)[:k]
            for rank, s_idx in enumerate(topk_indices, start=1):
                self._draw_adapter_impact_plot(
                    base, tta, gt,
                    sample_idx=int(s_idx),
                    channel=c_idx,
                    title_suffix=f"(Top {rank} Improved for Var {c_idx})",
                    prefix=f"best_improvement_top{rank}",
                )

    def plot_worst_samples_per_channel(self, data_dict):
        base = data_dict['preds_base']
        tta = data_dict['preds_tta']
        gt = data_dict['gts']
        
        mse_base = np.mean((base - gt)**2, axis=1)
        mse_tta = np.mean((tta - gt)**2, axis=1)
        improvement = mse_base - mse_tta
        
        worst_improvement_indices = np.argmin(improvement, axis=0)
        
        num_channels = gt.shape[2]
        print(f"Visualizer: Plotting MOST DEGRADED samples for {num_channels} channels...")

        for c_idx in range(num_channels):
            s_idx = worst_improvement_indices[c_idx]
            self._draw_adapter_impact_plot(
                base, tta, gt, 
                sample_idx=s_idx, 
                channel=c_idx,
                title_suffix=f"(Most Degraded for Var {c_idx})",
                prefix="worst_degradation"
            )

    def plot_sample_predictions(self, data_dict, sample_idx):
        base = data_dict['preds_base']
        tta = data_dict['preds_tta']
        gt = data_dict['gts']
        
        for c_idx in range(gt.shape[2]):
            self._draw_adapter_impact_plot(
                base, tta, gt, 
                sample_idx=sample_idx, 
                channel=c_idx,
                title_suffix=f"",
                prefix="sample_prediction")

    def _draw_adapter_impact_plot(self, base, tta, gt, sample_idx, channel, title_suffix="", prefix="best"):
        def smooth_1d(y: np.ndarray, win: int = 7) -> np.ndarray:
            y = np.asarray(y)
            if y.ndim != 1: return y
            if win is None or win <= 1: return y
            win = int(win)
            win = min(win, y.shape[0])
            if win < 3: return y
            if win % 2 == 0: win -= 1
            if win < 3: return y
            pad = win // 2
            y_pad = np.pad(y, (pad, pad), mode='edge')
            kernel = np.ones(win, dtype=np.float32) / float(win)
            return np.convolve(y_pad, kernel, mode='valid')

        y_gt = gt[sample_idx, :, channel]
        y_base = base[sample_idx, :, channel]
        y_tta = tta[sample_idx, :, channel]

        is_best_improvement = str(prefix).startswith("best_improvement")

        if is_best_improvement:
            smooth_win = 7
            y_gt = smooth_1d(y_gt, smooth_win)
            y_base = smooth_1d(y_base, smooth_win)
            y_tta = smooth_1d(y_tta, smooth_win)

        y_delta = y_tta - y_base
        mse_base = np.mean((y_base - y_gt)**2)
        mse_tta = np.mean((y_tta - y_gt)**2)

        if is_best_improvement:
            fig, ax1 = plt.subplots(1, 1, figsize=self._figsize_custom())
            ax2 = None
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self._figsize_custom(), sharex=True, 
                                           gridspec_kw={'height_ratios': [1, 0.5]})

        lw = 1.5
        if is_best_improvement:
            base_color = "#0000FF"
            tta_color = "#FF0000"
        else:
            base_color = "#004CFE"
            tta_color = "#D00202"

        ax1.plot(y_gt, label='Ground Truth', color="#000000", linewidth=lw, linestyle='-')
        ax1.plot(y_base, label='Base Pred', color=base_color, linestyle='-', linewidth=lw)
        ax1.plot(y_tta, label='TTA Pred', color=tta_color, linewidth=lw, linestyle='-')

        y_stack = np.stack([y_gt, y_base, y_tta], axis=0)
        self._axis_start_from_zero(ax1, x=True, y=False)
        self._set_nice_step_yaxis(ax1, y_stack, n_ticks=3, pad_frac=0.1, force_zero_if_nonneg=False)

        if not is_best_improvement:
            ax1.set_title(f"Prediction Comparison (Sample {sample_idx}, Var {channel})\n{title_suffix}", fontsize=24, pad=15)
        
        # 去掉网格
        # ax1.grid(True, alpha=0.2) 

        if ax2 is not None:
            ax2.plot(y_delta, label='Adapter Adjustment (Delta)', color='#27ae60', linewidth=2)
            ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
            ax2.fill_between(range(len(y_delta)), y_delta, 0, color='#27ae60', alpha=0.1)
            
            ax2.set_title("Adapter Contribution (TTA_Pred - Base_Pred)", fontsize=22)
            ax2.set_ylabel("Adjustment Value", fontsize=20)
            ax2.set_xlabel("Time Step", fontsize=20)
            ax2.legend(loc='upper left', fontsize=18)
            
            # 去掉网格
            # ax2.grid(True, alpha=0.2)

            self._axis_start_from_zero(ax2, x=True, y=False)
            self._set_nice_step_yaxis(ax2, y_delta, n_ticks=3, pad_frac=0.1, force_zero_if_nonneg=False)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"{prefix}_impact_var{channel}_s{sample_idx}.pdf")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_input_and_predictions(self, data_dict, sample_idx=0, channel_idx=0, prefix="full_sequence"):
        def to_numpy(x): return x.cpu().numpy() if hasattr(x, 'cpu') else x

        try:
            inputs = to_numpy(data_dict['inputs'])      
            gts = to_numpy(data_dict['gts'])            
            base = to_numpy(data_dict['preds_base'])    
            tta = to_numpy(data_dict['preds_tta'])      
        except KeyError as e:
            print(f"Visualizer Error: Missing key {e} in data_dict.")
            return

        y_inp = inputs[sample_idx, :, channel_idx]
        y_gt = gts[sample_idx, :, channel_idx]
        y_base = base[sample_idx, :, channel_idx]
        y_tta = tta[sample_idx, :, channel_idx]
        y_delta = y_tta - y_base
        
        seq_len = len(y_inp)
        pred_len = len(y_gt)
        
        x_inp = np.arange(seq_len)
        x_pred = np.arange(seq_len, seq_len + pred_len)

        mse_base = np.mean((y_base - y_gt)**2)
        mse_tta = np.mean((y_tta - y_gt)**2)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self._figsize_custom(), sharex=True, 
                           gridspec_kw={'height_ratios': [1, 0.5]})

        # --- Subplot 1 ---
        ax1.plot(x_inp, y_inp, label='Input (Lookback)', color='#7f8c8d', linewidth=2, alpha=0.8)
        lw = 2
        ax1.plot(x_pred, y_gt, label='Ground Truth', color='black', linestyle='--', linewidth=lw)
        ax1.plot(x_pred, y_base, label='Base Model Pred', color="#26CE99", linestyle='-', alpha=0.6, marker='.', markersize=4, linewidth=lw)
        ax1.plot(x_pred, y_tta, label='TTA Adjusted Pred', color="#8B0000", linewidth=lw)

        ax1.axvline(x=seq_len - 1, color='#e74c3c', linestyle='-', linewidth=1.2, alpha=0.7)
        ax1.text(seq_len - 1, ax1.get_ylim()[1], ' Forecast Start ', color='#e74c3c', 
                 ha='right', va='top', fontweight='bold', fontsize=18)

        ax1.set_title(f"TTA Adaption Impact | Sample {sample_idx} | Channel {channel_idx}", fontsize=24)
        
        # 去掉网格
        # ax1.grid(True, alpha=0.2)

        # Use concatenation to handle different input/pred lengths.
        y_stack = np.concatenate([y_inp, y_gt, y_base, y_tta], axis=0)
        self._axis_start_from_zero(ax1, x=True, y=False)
        self._set_nice_step_yaxis(ax1, y_stack, n_ticks=3, pad_frac=0.1, force_zero_if_nonneg=False)

        # --- Subplot 2 ---
        ax2.plot(x_pred, y_delta, label='TTA Adjustment (Delta)', color='#3498db', linewidth=2)
        ax2.fill_between(x_pred, y_delta, 0, color='#3498db', alpha=0.2)
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
        ax2.axvline(x=seq_len - 1, color='#e74c3c', linestyle='-', alpha=0.2)
        
        ax2.set_ylabel("Delta", fontsize=20)
        ax2.set_xlabel("Time Steps", fontsize=20)
        ax2.legend(loc='upper left', fontsize=18)
        
        # 去掉网格
        # ax2.grid(True, alpha=0.2)

        self._axis_start_from_zero(ax2, x=True, y=False)
        self._set_nice_step_yaxis(ax2, y_delta, n_ticks=3, pad_frac=0.1, force_zero_if_nonneg=False)

        plt.tight_layout()
        save_name = f"{prefix}_s{sample_idx}_c{channel_idx}.pdf"
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
