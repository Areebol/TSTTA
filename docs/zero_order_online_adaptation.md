# Zero-order online adaptation

## Implementation

The online CoBA path can now select either ordinary backpropagation or a
forward-only, two-sided SPSA update. The implementation is in
`tta/zo_utils.py` and provides:

- parameter and gradient vector conversion without replacing Parameter storage;
- antithetic `+c/-c` SPSA with Rademacher or ZOA-style segmented-uniform noise;
- optional parameter-tensor blockwise estimation;
- gradient recording followed by the existing optimizer step;
- CUDA allocation and autograd saved-tensor profiling.

Only target-stream updates are switched to ZO. Source-domain CoBA codebook
pretraining remains BP because it is an offline cost. Fitness queries execute
under `torch.no_grad()`, and the unperturbed parameter vector is restored in a
`finally` block before the optimizer step.

## Reproduction

The end-to-end comparison is:

```bash
CONDA_ENV=tstta bash scripts/coba_0408/run_transfer_coba_zo_h2Toh1.sh all

# Override the ZO query/update allocation explicitly:
CONDA_ENV=tstta ZO_DIRECTIONS=25 ZO_STEPS=2 ZO_LR=0.015 \
  bash scripts/coba_0408/run_transfer_coba_zo_h2Toh1.sh zo
```

It uses physical GPU 2, PatchTST, ETTh2 -> ETTh1, horizon 96, seed 0. The script
defaults to the user-requested `tsftta` name. On this machine that environment
does not contain PyTorch; `tstta` is the existing runnable CUDA environment.
The tunable environment variables are `ZO_DIRECTIONS`, `ZO_STEPS`, `BP_STEPS`,
`ZO_LR`, `ZO_C`, `ZO_BLOCKWISE`, and `BP_LR`.

The full-model memory benchmark is:

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n tstta python \
  scripts/coba_0408/benchmark_full_model_zo_memory.py \
  --method bp --model PatchTST --pred-len 96 --batch-size 64 \
  --checkpoint-dir /linyuanping/dzs/data/Electric_vehicles_dataset/checkpoints/PatchTST/ETTh2_96

CUDA_VISIBLE_DEVICES=2 conda run -n tstta python \
  scripts/coba_0408/benchmark_full_model_zo_memory.py \
  --method zo --model PatchTST --pred-len 96 --batch-size 64 \
  --checkpoint-dir /linyuanping/dzs/data/Electric_vehicles_dataset/checkpoints/PatchTST/ETTh2_96
```

Both benchmark variants update all 6,927,456 PatchTST parameters with SGD on
the same real ETTh1 batch. SGD is used for both so optimizer state does not
confound the activation-memory comparison.

## Seed-0 results

### Forecasting performance

| Method | Online update | MSE | Change from Base |
|---|---:|---:|---:|
| Pretrained Base | none | 0.554353 | - |
| CoBA + BP | BP, steps=1, LR 0.03 | 0.475048 | -14.31% |
| CoBA + ZO | K=50, steps=10, LR 0.003, c=0.08 | 0.490624 | -11.50% |

The best observed compensated 10-step ZO run improves the pretrained model by
11.50%. Its MSE is 3.28% above the best one-step BP result. It also improves
the uncompensated 10-step ZO result by 5.29% and the three-step ZO result by
3.06%.

Direction/update ladder (`K` is the number of averaged SPSA directions per
optimizer step):

| K | Steps per batch | LR | MSE | Total update time |
|---:|---:|---:|---:|---:|
| 50 | 1 | 0.010 | 0.519681 | not recorded in initial run |
| 100 | 1 | 0.010 | 0.521945 | 179.80 s |
| 50 | 2 | 0.005 | 0.524030 | 180.31 s |
| 25 | 2 | 0.008 | 0.514446 | 89.96 s |
| 25 | 2 | 0.010 | 0.511336 | 90.28 s |
| 25 | 2 | 0.012 | 0.509774 | 90.00 s |
| 25 | 2 | 0.015 | **0.509218** | 90.09 s |
| 25 | 2 | 0.020 | 0.512464 | 90.14 s |
| 17 | 3 | 0.010 | 0.515946 | 91.92 s |

At approximately equal total query budgets, two optimizer steps outperform one
or three. Increasing K from 50 to 100 does not help, so directions should not
be increased without considering online overfitting and wall-clock cost.
The best short-query-budget setting in this table spends 90.09 s in online optimizer steps versus 1.88 s
for BP (47.93x), while retaining the same shallow-adapter peak memory.

Fixed-`K` iteration ablation (`K=50`, `c=0.01`, LR `0.015`; all other settings
are identical):

| Steps per batch | MSE | MAE | Online updates | Peak allocated | Saved for backward | Total update time | Mean/update |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | **0.506117** | **0.489769** | 789 | 244.25 MiB | 0 MiB | 269.81 s | 0.3420 s |
| 5 | 0.512404 | 0.493041 | 1,315 | 244.25 MiB | 0 MiB | 454.35 s | 0.3455 s |
| 10 | 0.518036 | 0.497168 | 2,630 | 244.25 MiB | 0 MiB | 904.54 s | 0.3439 s |

Three steps give the best result in this fixed-learning-rate ladder, improving
the pretrained Base MSE by 8.70%. Five and ten steps remain better than Base,
but progressively regress from the three-step result. The likely cause is
online over-adaptation: increasing steps also increases the effective update
budget per batch when LR is held fixed. Query directions and optimizer steps
are processed serially, so total time scales almost linearly while peak memory
does not change. Further high-step experiments should reduce LR with the step
count (for example, preserving the three-step effective budget with LR
`0.015 * 3 / steps`) instead of using LR `0.015` unchanged.

### Ten-step learning-rate and perturbation compensation

The compensated search fixes `K=50`, ten optimizer steps per online batch,
`c=0.01` for the learning-rate stage, and LR `0.003` for the perturbation
stage. All runs use PatchTST, ETTh2 -> ETTh1, horizon 96, and seed 0.

Learning-rate search:

| LR | c | MSE | MAE | Total update time |
|---:|---:|---:|---:|---:|
| **0.003** | 0.01 | **0.494969** | **0.481662** | 904.29 s |
| 0.0045 | 0.01 | 0.497765 | 0.483224 | 902.84 s |
| 0.006 | 0.01 | 0.502390 | 0.486637 | 905.78 s |

Perturbation-scale search at LR `0.003`:

| LR | c | MSE | MAE | Total update time |
|---:|---:|---:|---:|---:|
| 0.003 | 0.005 | 0.501320 | 0.486355 | 901.97 s |
| 0.003 | 0.01 | 0.494969 | 0.481662 | 904.29 s |
| 0.003 | 0.02 | 0.492399 | 0.479738 | 904.36 s |
| 0.003 | 0.04 | 0.491111 | 0.478821 | 904.38 s |
| 0.003 | **0.08** | **0.490624** | **0.478739** | 906.99 s |

The best observed configuration is LR `0.003`, `c=0.08`. It improves Base by
11.50%, leaves a 3.28% relative MSE gap to BP, and retains the same 244.25 MiB
peak allocation and 0 MiB backward saved-tensor storage. Learning-rate
compensation provides most of the gain. Increasing `c` then improves both MSE
and MAE, but the marginal MSE gain shrinks from 0.00257 (`0.01 -> 0.02`) to
0.00049 (`0.04 -> 0.08`). Because `c=0.08` is the upper search boundary, it
is the best observed point rather than a bracketed local optimum; multi-seed
validation should precede any further expansion.

Reproduce the best observed run with:

```bash
CONDA_ENV=tstta GPU_ID=2 ZO_STEPS=10 ZO_DIRECTIONS=50 \
  ZO_LR=0.003 ZO_C=0.08 ZO_BLOCKWISE=False \
  RESULT_ROOT=./results/zo_etth2_to_etth1/patchtst_96/k50_s10_comp_search/reproduce_best \
  bash scripts/coba_0408/run_transfer_coba_zo_h2Toh1.sh zo
```

### Full-model one-step memory

| Method | Peak allocated | Incremental peak | Saved for backward |
|---|---:|---:|---:|
| BP | 549.05 MiB | 504.09 MiB | 488.29 MiB |
| ZO | 202.75 MiB | 157.79 MiB | 0 MiB |

ZO reduces incremental update peak allocation by 68.70% and absolute peak
allocation by 63.07% in the full-model benchmark.

The roughly 346 MiB BP/ZO peak difference belongs to this full-model benchmark,
not the normal shallow-adapter run. Full-model BP retains activations across
the deep PatchTST stack (488.29 MiB of unique non-leaf saved-tensor storage).
ZO removes those tensors but adds parameter-vector buffers for the center,
perturbation, estimated gradient, and temporary perturbed vectors. Therefore
the net allocator reduction is about 346 MiB rather than the full 488 MiB.

For the normal CoBA experiment, PatchTST itself is frozen and only the shallow
2,723-parameter output adapter is updated. Its measured peaks are 244.22 MiB
for BP and 244.25 MiB for ZO, effectively identical. ZO removes the 14.87 MiB
backward saved-tensor set, but the resident PatchTST/test tensors, FFT forward
workspace, first-step Adam state, and allocator alignment dominate. The ZO
parameter vectors are tiny here, but the maximum transient forward allocation
is already close to the BP peak. K and steps are evaluated serially, so they
increase total time without increasing this peak.

### Inference-subtracted shallow-adapter memory

A matched one-step experiment now runs the same adapter/loss closure once under
`torch.no_grad()` immediately before each measured update. The BP adapter
forward was moved inside the saved-tensor profiling scope, while the frozen
base prediction remains detached. The resulting seed-0 measurements are:

| Metric | BP | ZO (K=50) |
|---|---:|---:|
| End-to-end update peak allocated | 244.218 MiB | 244.252 MiB |
| Matched no-grad peak allocated | 244.230 MiB | 244.230 MiB |
| Matched no-grad incremental peak | 119.945 MiB | 119.945 MiB |
| Peak excess over matched no-grad | 0 MiB | 0.034 MiB |
| **Adapter/loss tensors saved for backward** | **14.870 MiB** | **0 MiB** |
| Mean update time | 0.00902 s | 0.33885 s |

The adapter/loss saved-tensor row is the clean activation-memory comparison:
ZO eliminates 100% of the 14.870 MiB autograd-retained storage. The 0.034 MiB
ZO excess comes from parameter-vector, perturbation, and estimated-gradient
buffers. This is the defensible metric for a table labelled online-adaptation
activation memory. End-to-end peak allocation should still be reported beside
it for transparency.

Subtracting scalar CUDA peaks is not additive. In this run, the common FFT
forward workspace reaches roughly 119.945 MiB at a different time from the BP
backward activations, so `max(update) - max(no_grad)` is clipped to zero for BP
even though 14.870 MiB is retained for backward. The saved-tensor hook measures
unique saved storage other than optimizer-parameter storage at graph
construction time and therefore is not hidden
by a larger transient workspace. The matched-peak subtraction is retained only
as an auxiliary allocator diagnostic.

This experiment supports a 100% relative reduction in shallow-adapter saved
activations, but only a 14.870 MiB absolute reduction. A large end-to-end saving
requires adapting a deeper/larger module, using larger batches/horizons, or
updating the full model; subtracting inference cannot manufacture a large
absolute saving for a 2,723-parameter adapter. Raw CSV files are under
`results/zo_etth2_to_etth1/patchtst_96/matched_adapter_memory_v2`.

The full-model benchmark is therefore the appropriate evidence for model-update
memory scaling; claiming a 300+ MiB reduction for the tiny adapter would mix
two different experiments.

These are a single seed and one horizon, so they are an implementation check and
promising ablation rather than a paper-level statistical conclusion.
