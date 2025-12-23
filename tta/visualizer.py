import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class TTAVisualizer:
    def __init__(self, save_dir, cfg):
        self.save_dir = save_dir
        self.cfg = cfg
        # 设置绘图风格
        plt.style.use('seaborn-v0_8-muted')

    def plot_all(self, data_dict):
        os.makedirs(self.save_dir, exist_ok=True)
        if data_dict is None:
            print("Visualizer: No data provided to plot.")
            return

        print(f"Visualizer: Generating plots in {self.save_dir}...")

        self._plot_gating_strategy(data_dict['gating'], data_dict.get('mse', None))

        if data_dict['mse'] is not None:
            self._plot_error_analysis(data_dict['mse'])
            
        self.plot_best_samples_per_channel(data_dict)
        self.plot_worst_samples_per_channel(data_dict)
        self.plot_sample_predictions(data_dict, sample_idx=1560)

    def _plot_predictions(self, base, tta, gt, num_samples=3, start_idx=300,):
        """对比 Ground Truth, 原始预测 和 TTA 后的预测"""
        # base/tta/gt 形状: [Total_Samples, Pred_Len, Channels]
        n_samples, pred_len, n_vars = gt.shape
        samples_to_plot = min(num_samples + start_idx, n_samples)
        
        # 挑选典型的通道进行展示（例如第一个、中间一个、最后一个）
        channels_to_plot = [0, n_vars // 2, n_vars - 1] if n_vars > 2 else range(n_vars)

        for s_idx in range(start_idx, samples_to_plot):
            for c_idx in channels_to_plot:
                plt.figure(figsize=(10, 5))
                
                # 绘制三条线
                plt.plot(gt[s_idx, :, c_idx], label='Ground Truth', color='#2c3e50', linewidth=2, linestyle='--')
                plt.plot(base[s_idx, :, c_idx], label='Base Model', color='#e74c3c', alpha=0.7)
                plt.plot(tta[s_idx, :, c_idx], label='After TTA', color='#3498db', linewidth=1.5)

                plt.title(f"Forecast Comparison | Sample {s_idx} | Channel {c_idx}")
                plt.xlabel("Time Step (within Pred_Len)")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                save_path = os.path.join(self.save_dir, f"pred_s{s_idx}_c{c_idx}.png")
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()

    def _plot_gating_strategy(self, gating, mse=None):
        """
        可视化门控权重 (λ) 与 预测误差 (MSE) 的演化关系
        gating: [Total_Samples, n_vars]
        mse: [Total_Samples] (可能与 gating 长度不同，通过均值对齐)
        """
        fig, ax1 = plt.subplots(figsize=(14, 6))

        # --- 1. 处理 Gating 曲线 (左轴) ---
        x_gating = np.arange(len(gating))
        if gating.ndim > 1 and gating.shape[1] > 1:
            mean_gating = np.mean(gating, axis=1)
            ax1.plot(x_gating, mean_gating, label='Mean Gating Weight (λ)', color='#27ae60', linewidth=2, zorder=3)
            ax1.fill_between(x_gating, gating.min(axis=1), gating.max(axis=1), alpha=0.1, color='#27ae60')
        else:
            ax1.plot(x_gating, gating.flatten(), label='Gating Weight (λ)', color='#27ae60', linewidth=2, zorder=3)

        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        ax1.set_xlabel("Test Samples / Steps")
        ax1.set_ylabel("Gating Weight (λ)", color='#27ae60', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='#27ae60')
        ax1.grid(True, alpha=0.2)

        # --- 2. 处理 Error 曲线 (右轴) ---
        if mse is not None:
            ax2 = ax1.twinx()  # 创建共享 X 轴的右轴
            
            # 对齐逻辑：如果 MSE 和 Gating 长度不同（例如一个是 Sample 级，一个是 Batch 级）
            if len(mse) != len(gating):
                # 尝试将 MSE 聚合（如果是 Sample 级转 Batch 级）
                # 这里假设 gating 长度代表 Batch 数，通过插值或重采样对齐
                x_mse = np.linspace(0, len(gating) - 1, num=len(mse))
                # 为了平滑，我们画出原始误差的散点或浅色线，再画出滑动平均
                ax2.plot(x_mse, mse, color='#e74c3c', alpha=0.15, label='Raw MSE (Sample-wise)')
                
                # 计算滑动平均以对齐趋势
                window = max(1, len(mse) // len(gating)) if len(mse) > len(gating) else 5
                mse_smooth = np.convolve(mse, np.ones(window)/window, mode='valid')
                x_smooth = np.linspace(0, len(gating) - 1, num=len(mse_smooth))
                ax2.plot(x_smooth, mse_smooth, color='#e74c3c', linewidth=1.5, label='Smoothed MSE', zorder=2)
            else:
                ax2.plot(x_gating, mse, color='#e74c3c', linewidth=1.5, alpha=0.6, label='MSE')

            ax2.set_ylabel("Prediction Error (MSE)", color='#e74c3c', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='#e74c3c')
            ax2.set_yscale('log') # Error 通常跨度大，用对数轴更清晰
            
            # 合并两个轴的 Legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, framealpha=0.9)
        else:
            ax1.legend(loc='upper left')

        plt.title("Gating Evolution vs. Prediction Error", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "gating_vs_error.png"), bbox_inches='tight', dpi=150)
        plt.close()

        # 3. 保持原有的 Heatmap 逻辑
        if gating.ndim > 1 and gating.shape[1] > 1:
            plt.figure(figsize=(14, 6))
            sns.heatmap(gating.T, cmap='YlGnBu', cbar_kws={'label': 'λ'})
            plt.title("Gating Weights Heatmap (Channels vs Time)")
            plt.xlabel("Test Samples")
            plt.ylabel("Channel Index")
            plt.savefig(os.path.join(self.save_dir, "gating_heatmap.png"), bbox_inches='tight')
            plt.close()

    def _plot_error_analysis(self, mse_steps):
        plt.figure(figsize=(10, 5))
        
        window_size = min(20, len(mse_steps))
        if window_size > 0:
            smoothed_mse = np.convolve(mse_steps, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_mse, color='#8e44ad', label=f'MSE (Moving Avg {window_size})')
        
        plt.plot(mse_steps, color='#8e44ad', alpha=0.2, label='Batch MSE')
        
        plt.title("Prediction Error (MSE) over Test Stream")
        plt.xlabel("Test Step")
        plt.ylabel("MSE")
        plt.legend()
        plt.yscale('log') # 误差通常建议用对数坐标查看
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        plt.savefig(os.path.join(self.save_dir, "error_trend.png"), bbox_inches='tight')
        plt.close()
        
    def plot_best_samples_per_channel(self, data_dict):
        base = data_dict['preds_base']
        tta = data_dict['preds_tta']
        gt = data_dict['gts']
        
        mse_base = np.mean((base - gt)**2, axis=1) # [Samples, Channels]
        mse_tta = np.mean((tta - gt)**2, axis=1)   # [Samples, Channels]
        
        improvement = mse_base - mse_tta
        
        best_improvement_indices = np.argmax(improvement, axis=0)
        
        num_channels = gt.shape[2]
        print(f"Visualizer: Plotting MOST IMPROVED samples for {num_channels} channels...")

        for c_idx in range(num_channels):
            s_idx = best_improvement_indices[c_idx]
            self._draw_adapter_impact_plot(
                base, tta, gt, 
                sample_idx=s_idx, 
                channel=c_idx,
                title_suffix=f"(Most Improved for Var {c_idx})",
                prefix="best_improvement"
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
        delta = tta - base
        y_gt = gt[sample_idx, :, channel]
        y_base = base[sample_idx, :, channel]
        y_tta = tta[sample_idx, :, channel]
        y_delta = delta[sample_idx, :, channel]

        mse_base = np.mean((y_base - y_gt)**2)
        mse_tta = np.mean((y_tta - y_gt)**2)
        improvement = (mse_base - mse_tta) / (mse_base + 1e-9) * 100

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True, 
                                       gridspec_kw={'height_ratios': [1, 0.7]})

        ax1.plot(y_gt, label='Ground Truth', color='#333333', alpha=0.4, linewidth=1.2, linestyle='-')
        ax1.plot(y_base, label='Base Pred', color="#26CE99", linestyle='--', linewidth=1.8)
        ax1.plot(y_tta, label='TTA Pred', color="#EC591F", linewidth=2.2, linestyle='-')
        
        ax1.set_title(f"Prediction Comparison (Sample {sample_idx}, Var {channel})\n{title_suffix}", fontsize=13, pad=15)
        ax1.legend(loc='upper left', frameon=True)
        ax1.grid(True, alpha=0.2)

        info_color = 'green' if improvement > 0 else 'red'
        info_text = (
            f"Base MSE: {mse_base:.4f}\n"
            f"TTA MSE:  {mse_tta:.4f}\n"
            f"Change:   {improvement:+.2f}%"
        )
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=info_color)
        ax1.text(0.98, 0.95, info_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right', bbox=props, color=info_color, weight='bold')

        ax2.plot(y_delta, label='Adapter Adjustment (Delta)', color='#27ae60', linewidth=2)
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
        
        ax2.fill_between(range(len(y_delta)), y_delta, 0, color='#27ae60', alpha=0.1)
        
        ax2.set_title("Adapter Contribution (TTA_Pred - Base_Pred)", fontsize=11)
        ax2.set_ylabel("Adjustment Value")
        ax2.set_xlabel("Time Step")
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.2)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"{prefix}_impact_var{channel}_s{sample_idx}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()