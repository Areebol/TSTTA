# Baseline横线数据
# baseline_vals = [0.800514802, 0.655180797, 0.655005202]
# baseline_labels = ["Baseline1", "Baseline2", "Baseline3"]
# baseline_vals = [0.655180797, 0.655005202]
# baseline_labels = ["TAFAS", "PETSA"]
# baseline_colors = ["tab:gray", "tab:brown", "tab:cyan"]
import matplotlib.pyplot as plt
import numpy as np

# ICML 风格设置
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

# 数据
# n_bases = [2, 4, 8, 16, 32, 64, 128, 256]
n_bases = [2, 4, 8, 16, 32, 64]
pre_lens = [96, 192, 336, 720]
mse = np.array([
    [0.4907, 0.4907, 0.4936, 0.4944, 0.4935, 0.4912, 0.4873, 0.4908],
    [0.5770, 0.5757, 0.5751, 0.5765, 0.5766, 0.5722, 0.5773, 0.5764],
    [0.6054, 0.6034, 0.6033, 0.6033, 0.6024, 0.6026, 0.6103, 0.5973],
    [0.8133, 0.8159, 0.8089, 0.8011, 0.8021, 0.8145, 0.8104, 0.8221],
])
avg = [0.6216, 0.6214, 0.6202, 0.6188, 0.6187, 0.6201, 0.6213, 0.6216]
avg = avg[:-2]

layout_params = {
    'left': 0.14,
    'right': 0.85,
    'bottom': 0.14,
    'top': 0.95
}

# 颜色和 marker
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
markers = ['o', 's', '^', 'D']

# 1. 每个 pre_len 单独画图
# for i, (pl, color, marker) in enumerate(zip(pre_lens, colors, markers)):
#     fig = plt.figure(figsize=(4, 3))
#     ax = fig.add_subplot(111)
#     ax.plot(n_bases, mse[i], marker=marker, color=color, label=f'pre\\_len={pl}')
#     ax.set_xlabel('n$_{bases}$')
#     ax.set_ylabel('MSE')
#     ax.set_title(f'pre\\_len={pl}')
#     ax.set_xscale('log', base=2)
#     ax.set_xticks(n_bases)
#     ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
#     ax.legend(loc='best')
#     fig.subplots_adjust(**layout_params)
#     fig.savefig(f'patchtst_mse_prelen_{pl}.pdf', dpi=300)
#     plt.close(fig)

# 2. avg 行画图
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)
ax.plot(n_bases, avg, marker='o', color='tab:purple', label='avg')
# for val, label, color in zip(baseline_vals, baseline_labels, baseline_colors):
#     ax.axhline(y=val, linestyle='--', color=color, label=label)
ax.set_xlabel('N')
ax.set_ylabel('MSE')
# ax.set_title('Average MSE')
ax.set_xscale('log', base=2)
ax.set_xticks(n_bases)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.legend(loc='best')
fig.subplots_adjust(**layout_params)
fig.savefig('patchtst_mse_avg.pdf', dpi=300)
plt.close(fig)
