# TPA 在线知识更新与管理方案（V3）

## 1. 目标与约束

本方案在现有 `CoBA_TF_Adapter` 基础上增加在线模式原型的提取、筛选、替换和持久化能力，同时遵守以下约束：

- 离线 source 原型与在线原型默认各 16 个，总容量 32。
- source 原型及其 query 网络在离线预训练结束后冻结，在线阶段不可删除、覆盖或更新。
- frequency adapter 按现有延迟标签 TTA 流程更新，不因固定时间或原型写入而重置 gate、参数或 Adam 状态。
- 只有完整预测区间标签到达后才提出候选原型，部分标签更新不会生成原型。
- 原型不是按固定时间无条件写入；候选必须使 replay MSE 严格下降才会进入在线库。
- 新实现使用独立的 `TTA.METHOD=TPA`，原有 `COBA` 命令及 `scripts/coba_0408` 不变。

当前版本不实现扩容与原型合并，也不对已接纳原型做 EMA。在线原型一经写入即保持不变，直到被后续更优候选替换。

## 2. 模块结构

### 2.1 双区原型库

每个变量拥有独立的 key/value：

- source 区：`[V, 16, D]` 的 key 与 `[V, 16, H]` 的 value；来自训练集离线预训练并永久保护。
- online 区：`[V, 16, D]` 的 key 与 `[V, 16, H]` 的 value；初始为空，只接受通过 replay 验证的候选。

推理时将有效 source/online 原型拼成一个 codebook，使用现有检索温度 `temperature=10.0` 重新计算 softmax 权重。不会固定旧权重，也不会只在 online 区内部检索。

### 2.2 在线锚点池

锚点来自测试流中已经获得完整延迟标签的样本。每条记录保存：

- `sample_id`
- 归一化 query `q`
- 基础预测 `Y_base`
- 完整标签 `Y_gt`
- 当时 frequency adapter correction 快照
- 基础模型误差 `MSE(Y_base, Y_gt)`

不保存原始长输入序列。frequency correction 快照用于确保比较 D 与 D⁺ 时，两者使用完全相同的 adapter 贡献，避免候选评估混入 adapter 状态差异。

锚点池默认容量为 64。新锚点临时加入后，在展平的归一化 query 空间计算 L2/Frobenius 距离，找到全局最近的一对：

1. 删除基础模型误差较小、即更容易的样本；
2. 若误差相等，删除更旧的样本。

该机制不使用 novelty threshold，目标是在有限容量下保留困难且分散的在线模式。

## 3. 候选原型提取

候选只基于最新一个完整延迟标签批次。对样本 `i`：

```text
Delta_i = Y_final_i - Y_base_i
```

`Delta_i` 包含当前原型检索修正与当前 frequency adapter 修正。value 始终保留完整预测区间，形状为 `[V, H]`；任何提取方式都只沿样本维聚合，不沿 horizon 维求平均。

因此，原型不会因为反复测试而逐步变成所有时间步相同的均值。当前版本也不对已接纳 value 做 EMA/merge，进一步避免累计“压平”。如果同一批样本本身的时间曲线接近常数，候选可以接近常数，但这是数据内容而不是维度处理造成的。

### 3.1 简单平均（`mean`）

```text
k_c = normalize(mean_i(q_i))
v_c[v, h] = mean_i(Delta_i[h, v])
```

该模式作为低复杂度、低方差的基线。

### 3.2 查询加权蒸馏（`query_weighted`）

先得到候选 key，再对每个变量分别计算样本权重：

```text
k_c = normalize(mean_i(q_i))
s_i,v = cosine(q_i,v, k_c,v)
w_i,v = softmax_i(tau * s_i,v)
v_c[v, h] = sum_i(w_i,v * Delta_i[h, v])
```

这里的 `tau` 复用当前检索温度 10.0，不增加新的超参数。它只控制样本权重的尖锐程度：

- `tau` 较大时，更偏向 query 与候选 key 最一致的样本；
- `tau` 较小时，更接近简单平均。

`tau` 不是 energy 指标，也不直接决定候选是否写入。

## 4. coherence 指标的含义

实现记录以下诊断量：

```text
distill_coherence = ||v_c||² / mean_i(||Delta_i||²)
```

它描述批次修正中有多少能量在聚合后仍然保持一致：

- 较高：样本修正方向更一致，平均后保留得更多；
- 较低：样本修正互相抵消，批次内部模式可能不一致。

该指标仅写入日志，不设阈值，不参与原型接纳或淘汰。这样可以先通过实验观察它与性能的关系，避免和 PKA 中的 energy threshold 混用。

## 5. Replay 接纳与替换

Replay 集由“最新完整标签批次 + 在线锚点池”组成，并按 `sample_id` 去重。评价指标为所有 replay 元素上的 MSE；每个候选方案都重新计算完整 codebook 的检索权重。

### 5.1 在线槽位未满

比较：

- D：当前 source + online 原型库；
- D⁺：D + 候选原型。

仅当 `MSE(D⁺) < MSE(D)` 时写入候选，否则丢弃。

### 5.2 在线槽位已满

临时构造 `D + candidate`，依次尝试删除每一个旧 online 原型。删除候选本身等价于原始 D。选择 replay MSE 最小的方案，并且只有其 MSE 严格小于原始 D 时才替换旧槽位。

source 原型不参与枚举，因此永远不会被在线候选淘汰。

## 6. Frequency adapter 的作用与重置策略

Frequency adapter 继续负责快速、连续地拟合当前局部频域偏移；online prototypes 负责保存经过 replay 验证、可长期复用的模式知识。两者时间尺度不同但同时存在。

本方案不重置 frequency adapter。模式尚未切换时按固定时刻重置会破坏已学到的局部适配，而且“候选被接纳”本身也不等于环境发生切换。候选比较中 D 与 D⁺ 使用相同的 adapter correction，若候选与 adapter 重复修正导致过补偿，D⁺ 的 replay MSE 会变差并被拒绝。

## 7. 配置、运行与输出

主要配置：

```yaml
TTA:
  METHOD: TPA
  DUAL:
    CALI_NAME: TPAPrototypeAdapter
    CALI_INPUT_ENABLE: false
    CALI_OUTPUT_ENABLE: true
    COBA_ONLINE_ENABLED: true
  TPA:
    N_SOURCE: 16
    N_ONLINE: 16
    ANCHOR_CAPACITY: 64
    DISTILL_MODE: mean  # 或 query_weighted
    REPLAY_BATCH_SIZE: 128
    SAVE_STATE: true
```

运行脚本：

- `scripts/tpa_0727/run_regular_tpa.sh`
- `scripts/tpa_0727/run_transfer_tpa.sh`
- `scripts/tpa_0727/run_eved_tpa.sh`

三个脚本默认分别运行 `mean` 与 `query_weighted` 两种蒸馏方式。实验结束后，除原有 TTA 结果外，还会在 `RESULT_DIR` 保存：

- `*-prototype-updates.json`：每次候选的 coherence、接纳结果、前后 replay MSE、槽位和锚点统计；
- `*-prototype-state.pt`：source/online 原型、frequency adapter 参数、锚点池与更新历史，可用于复现实验或后续恢复。

## 8. 代码落点与兼容性

- `tta/tpa_memory.py`：原型 adapter、锚点池、两种候选蒸馏和 replay 接纳逻辑。
- `tta/tpa.py`：复用 CoBA 数据流，只覆盖完整延迟标签更新与 TPA 状态保存。
- `config.py`：新增独立 `TTA.TPA` 配置节点。
- `main.py`：新增独立 `TPA` dispatch。
- `tta/coba.py`：仅在 adapter factory 注册新类型；旧类型的参数和执行分支不变。
- `tests/test_tpa_memory.py`：覆盖多样性锚点、两种蒸馏、horizon 保留及 replay 接纳。

旧的 `TTA.METHOD=COBA` 和 `scripts/coba_0408` 不需要任何参数变化。
