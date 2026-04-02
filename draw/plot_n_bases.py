import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
# -------------------------- ICML Style Settings --------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.linewidth": 1.2
})

# -------------------------- Data --------------------------
x_labels = ['96', '192', '336', '720']
x = np.arange(len(x_labels))

# Throughput
y_petsa = [176.6784452, 148.1481481, 144.092219, 142.0454545]
y_tafas = [335.5704698, 324.6753247, 300.3003003, 270.2702703]
y_ours = [507.6142132, 490.1960784, 458.7155963, 381.6793893]

# Peak memory
y1_petsa = [58.39, 76.86, 103.98, 181.28]
y1_tafas = [92.33, 123.04, 171.47, 321.33]
y1_ours = [60.82, 76.9, 99.8, 152.7]

# Additional parameters
y2_petsa = [0.103736, 0.155576, 0.233336, 0.440696]
y2_tafas = [0.521528, 1.29836, 3.431288, 14.796152]
y2_ours = [0.148492, 0.252172, 0.407692, 0.822412]

# ==============================================================================
# 关键设置：定义固定的布局参数
# 这里的数值(0-1之间)代表相对画布的位置。
# 为了容纳图2右侧的Y轴标签，right必须留出足够的空间(例如0.85)。
# 虽然图1右侧不需要这么多空间，但为了保持两个图的"绘图框(Box)"大小一致，图1也必须用同样的参数。
# ==============================================================================
layout_params = {
    'left': 0.14,    # 左边留白 (给左Y轴标签)
    'right': 0.85,   # 右边留白 (给图2的右Y轴标签，图1虽然空着但也得留)
    'bottom': 0.14,  # 底部留白 (给X轴标签)
    'top': 0.95      # 顶部留白
}

# 颜色分配
colors = {
    'PETSA': 'tab:blue',    # PETSA用蓝色
    'TAFAS': 'tab:green',  # TAFAS用紫色
    'Ours': 'tab:red'       # Ours用红色
}

# -------------------------- Plot 1: Throughput --------------------------
fig1 = plt.figure(figsize=(4, 3)) # 固定画布大小
ax = fig1.add_subplot(111)

ax.plot(x, y_petsa, marker='o', linestyle='-', color=colors['PETSA'], label='PETSA')
ax.plot(x, y_tafas, marker='s', linestyle='--', color=colors['TAFAS'], label='TAFAS')
ax.plot(x, y_ours, marker='^', linestyle='-.', color=colors['Ours'], label='Ours')

ax.set_xlabel('Prediction Length')
ax.set_ylabel('Throughput (samples/s)')
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_ylim(130, 600)
ax.legend(loc='best')

# 应用固定布局
fig1.subplots_adjust(**layout_params)

# 保存时去掉 bbox_inches='tight'，直接保存固定尺寸的画布
fig1.savefig('throughput.pdf', dpi=300)
plt.close(fig1)

# -------------------------- Plot 2: Peak Memory + Params --------------------------

# -------------------------- Plot 2: Peak Memory + Params (优化版) --------------------------
fig2 = plt.figure(figsize=(4, 3))
ax1 = fig2.add_subplot(111)

# marker分配
mem_marker = 'o'
param_marker = 's'

# 左轴绘图 (Memory)
ax1.set_xlabel('Prediction Length')
ax1.set_ylabel('Peak Memory (MB)')
ax1.plot(x, y1_petsa, marker=mem_marker, linestyle='-', color=colors['PETSA'])
ax1.plot(x, y1_tafas, marker=mem_marker, linestyle='-', color=colors['TAFAS'])
ax1.plot(x, y1_ours, marker=mem_marker, linestyle='-', color=colors['Ours'])
ax1.tick_params(axis='y')
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)

# 右轴绘图 (Params)
ax2 = ax1.twinx()
ax2.set_ylabel('Add. Params (MB)')
ax2.plot(x, y2_petsa, marker=param_marker, linestyle='--', color=colors['PETSA'])
ax2.plot(x, y2_tafas, marker=param_marker, linestyle='--', color=colors['TAFAS'])
ax2.plot(x, y2_ours, marker=param_marker, linestyle='--', color=colors['Ours'])
ax2.tick_params(axis='y')

# ==============================================================================
# 核心修改：自定义图例 (Proxy Artists)
# ==============================================================================
# 1. 定义颜色的含义 (代表 Method)
legend_elements_methods = [
    Line2D([0], [0], color=colors['PETSA'], lw=1.8, label='PETSA'),
    Line2D([0], [0], color=colors['TAFAS'], lw=1.8, label='TAFAS'),
    Line2D([0], [0], color=colors['Ours'], lw=1.8, label='Ours')
]

# 2. 定义线型/Marker的含义 (代表 Metric)，统一用黑色显示
legend_elements_metrics = [
    Line2D([0], [0], color='k', marker=mem_marker, linestyle='-', lw=1.8, label='Memory'),
    Line2D([0], [0], color='k', marker=param_marker, linestyle='--', lw=1.8, label='Params')
]

# 3. 合并图例，或者分开放置
# 这里为了省空间，我们把它们放在一起，但是利用空行或者列数来区分
all_handles = legend_elements_methods + legend_elements_metrics

# 放置图例：使用 ncol=2 让图例变扁，少占垂直空间
# 或者放在左上角，根据你的数据，Memory左边较低，Params左边也较低，左上角相对空旷
ax1.legend(handles=all_handles, loc='upper left', ncol=2, fontsize=10, columnspacing=1.0)

# ==============================================================================

fig2.subplots_adjust(**layout_params)
fig2.savefig('memory_params.pdf', dpi=300)
plt.close(fig2)