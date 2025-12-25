import math
from yacs.config import CfgNode as CN


_C = CN()
# random seed number
_C.SEED = 0
# number of gpus per node
_C.NUM_GPUS = 8
_C.VISIBLE_DEVICES = 0
# directory to save result txt file
_C.RESULT_DIR = 'results/'
_C.NORMALIZE = 'NST'

_C.DATA_LOADER = CN()
_C.DATA_LOADER.NUM_WORKERS = 16
_C.DATA_LOADER.PIN_MEMORY = True
_C.DATA_LOADER.DROP_LAST = True

_C.DATA = CN()
_C.DATA.BASE_DIR = 'data/'
_C.DATA.NAME = 'weather'
_C.DATA.N_VAR = 21
_C.DATA.SEQ_LEN = 96
_C.DATA.LABEL_LEN = 48
_C.DATA.PRED_LEN = 96
_C.DATA.FEATURES = 'M'
_C.DATA.TIMEENC = 0
_C.DATA.FREQ = 'h'
_C.DATA.SCALE = "standard"  # standard, min-max
_C.DATA.TRAIN_RATIO = 0.7
_C.DATA.TEST_RATIO = 0.2
_C.DATA.DATE_IDX = 0
_C.DATA.TARGET_START_IDX = 0
_C.DATA.PERIOD_LEN = 24  # Used only when SAN is ENABLED
_C.DATA.STATION_TYPE = 'adaptive'  # Used only when SAN is ENABLED


_C.DATA.TYPE = None # 'noise, 'regime', 'season', 'trend', 'None'
_C.DATA.PERTURB_ROOT = '/lichenghao/dengzeshuai/codes/TSF/TSTTA/TTFBench/synthetic_series_single_mode'

_C.TRAIN = CN()
_C.TRAIN.ENABLE = False
_C.TRAIN.FINETUNE = False
_C.TRAIN.SPLIT = 'train'
_C.TRAIN.BATCH_SIZE = 256
_C.TRAIN.SHUFFLE = True
_C.TRAIN.DROP_LAST = True
# directory to save checkpoints
_C.TRAIN.CHECKPOINT_DIR = 'results/'
# path to checkpoint to resume training
_C.TRAIN.RESUME = ''
_C.TRAIN.RESUME_DIR = './'
# epoch period to evaluate on a validation set
_C.TRAIN.EVAL_PERIOD = 5
# iteration frequency to print progress meter
_C.TRAIN.PRINT_FREQ = 100
_C.TRAIN.BEST_METRIC_INITIAL = float("inf")
_C.TRAIN.BEST_LOWER = True


_C.VAL = CN()
_C.VAL.SPLIT = 'val'
_C.VAL.BATCH_SIZE = 256
_C.VAL.SHUFFLE = False
_C.VAL.DROP_LAST = False
_C.VAL.VIS = False

_C.TEST = CN()
_C.TEST.ENABLE = True
_C.TEST.SPLIT = 'test'
_C.TEST.BATCH_SIZE = 256
_C.TEST.SHUFFLE = False
_C.TEST.DROP_LAST = False

_C.TTA = CN()
_C.TTA.LOG_NAME = None
_C.TTA.ENABLE = False
_C.TTA.METHOD = 'TAFAS'
_C.TTA.MODULE_NAMES_TO_ADAPT = 'cali'  # all, norm, etc
_C.TTA.LOG = False
_C.TTA.RESET = False
_C.TTA.SOLVER = CN()
_C.TTA.SOLVER.OPTIMIZING_METHOD = 'adam'
_C.TTA.SOLVER.BASE_LR = 0.005
_C.TTA.SOLVER.WEIGHT_DECAY = 0.0001
_C.TTA.SOLVER.MOMENTUM = 0.9
_C.TTA.SOLVER.NESTEROV = True
_C.TTA.SOLVER.DAMPENING = 0.0

# TAFAS hyperparameters
_C.TTA.TAFAS = CN()
_C.TTA.TAFAS.PAAS = True
_C.TTA.TAFAS.PERIOD_N = 1
_C.TTA.TAFAS.BATCH_SIZE = 64
_C.TTA.TAFAS.STEPS = 1
_C.TTA.TAFAS.ADJUST_PRED = True
_C.TTA.TAFAS.CALI_MODULE = True
_C.TTA.TAFAS.GATING_INIT = 0.01
_C.TTA.TAFAS.HIDDEN_DIM = 128
_C.TTA.TAFAS.GCM_VAR_WISE = True

# ALPHA hyperparameters
_C.TTA.ALPHA = CN()
_C.TTA.ALPHA.LR = 1e-3
_C.TTA.ALPHA.ALPHA_LR_SCALE = 1
_C.TTA.ALPHA.PERIOD_N = 1
_C.TTA.ALPHA.STEPS = 1
_C.TTA.ALPHA.PAAS = True
_C.TTA.ALPHA.BATCH_SIZE = 64
_C.TTA.ALPHA.GATING_INIT = 0.01
_C.TTA.ALPHA.ADJUST_PRED = True
_C.TTA.ALPHA.SOFTMAX = True
_C.TTA.ALPHA.MIX_MODE = 'orig_plus_weighted_delta'  # or 'weighted_full'
_C.TTA.ALPHA.DELTA_NORM = 'none'  # 'none' or 'l2_unit'
_C.TTA.ALPHA.S_MAX = 1.0          # rho_eff = S_MAX * sigmoid(rho_raw)
_C.TTA.ALPHA.EPS = 1e-6
_C.TTA.ALPHA.TEST_TYPES = ''  # test domains
_C.TTA.ALPHA.KNOWLEDGE_TYPE = 'adapter'  # 'delta' or 'adapter'
_C.TTA.ALPHA.ADAPTER_MODE = 'pretrained'


## MatchTTA hyperparameters
_C.TTA.MATCH = CN()
_C.TTA.MATCH.LR = 1e-3
_C.TTA.MATCH.GATING_LR_SCALE = 1
_C.TTA.MATCH.GATING_INIT = 0.001
_C.TTA.MATCH.VAE_MODEL_PATH = 'checkpoints/vae_best.pth'
_C.TTA.MATCH.CLUSTERS_PATH = 'checkpoints/clusters.pt'
_C.TTA.MATCH.ADAPTER_BANK_DIR = 'checkpoints/adapters/'
_C.TTA.MATCH.N_CLUSTERS = 4
_C.TTA.MATCH.LATENT_DIM = 64
_C.TTA.MATCH.VAE_ENCODER_TYPE = 'mlp'
_C.TTA.MATCH.VAE_DECODER_TYPE = 'mlp'
_C.TTA.MATCH.GAMMA = 1.0
_C.TTA.MATCH.STEPS = 1
_C.TTA.MATCH.PAAS = True
_C.TTA.MATCH.BATCH_SIZE = 64
_C.TTA.MATCH.ADJUST_PRED = True
_C.TTA.MATCH.TEST_TYPES = 'original'


## PETSA hyperparameters
_C.TTA.PETSA = CN()
_C.TTA.PETSA.PAAS = True
_C.TTA.PETSA.PERIOD_N = 1
_C.TTA.PETSA.BATCH_SIZE = 64
_C.TTA.PETSA.STEPS = 1
_C.TTA.PETSA.ADJUST_PRED = True
_C.TTA.PETSA.CALI_MODULE = True
_C.TTA.PETSA.GATING_INIT = 0.01
_C.TTA.PETSA.HIDDEN_DIM = 128
_C.TTA.PETSA.GCM_VAR_WISE = True
_C.TTA.PETSA.GCM_VAR_WISE = True
_C.TTA.PETSA.RANK = 16
_C.TTA.PETSA.LOSS_ALPHA = 0.1


## COSA hyperparameters
_C.TTA.SIMPLE = CN()
_C.TTA.SIMPLE.BATCH_SIZE = 64
_C.TTA.SIMPLE.STEPS = 20
_C.TTA.SIMPLE.BUFFER_SIZE = 10
_C.TTA.SIMPLE.BUFFER_CONTEXT_SIZE = 5
_C.TTA.SIMPLE.ADAPT_FREQUENCY = 50                

# Fast Adaptation Optimization Settings
_C.TTA.SIMPLE.FAST_ADAPTATION = True              # Enable fast adaptation optimization
_C.TTA.SIMPLE.ADAPTIVE_LR = True                  # Enable adaptive learning rate adjustment
_C.TTA.SIMPLE.MAX_LR = 0.005                      # Maximum learning rate
_C.TTA.SIMPLE.MIN_LR = 0.0001                     # Minimum learning rate
_C.TTA.SIMPLE.MOMENTUM_FACTOR = 0.9               # Momentum factor for convergence acceleration
_C.TTA.SIMPLE.CONVERGENCE_THRESHOLD = 1e-4        # Early stopping convergence threshold
_C.TTA.SIMPLE.VAR_WISE_GATING = True              # Enable variable-wise gating
_C.TTA.SIMPLE.ADAPTER_LAYERS = 1                  # Number of FC layers in adapter (1 or 2)
_C.TTA.SIMPLE.HIDDEN_DIM = 64                     # Hidden dimension for 2-layer adapter
_C.TTA.SIMPLE.PER_BATCH_LR_RESET = True           # Reset learning rate at the start of each batch

# CSV Export Settings
_C.TTA.SIMPLE.SAVE_CSV = False                    # Enable detailed CSV export of predictions
_C.TTA.SIMPLE.SAVE_PAAS_CSV = False               # Enable CSV export of PAAS batch size information
_C.TTA.SIMPLE.PAAS = True
_C.TTA.SIMPLE.PERIOD_N = 1
_C.TTA.SIMPLE.ONLY_CONTEXT = False

## DynaTTA: Dynamic Test-Time Adaptation
_C.TTA.DYNATTA = CN()
_C.TTA.DYNATTA.MSE_BUFFER_SIZE = 256              # Size of MSE buffer for z-score computation
_C.TTA.DYNATTA.METRIC_HISTORY_SIZE = 256          # Size of metric history for normalization
_C.TTA.DYNATTA.ALPHA_MIN = 1e-4                   # Minimum adaptation rate
_C.TTA.DYNATTA.ALPHA_MAX = 1e-3                   # Maximum adaptation rate
_C.TTA.DYNATTA.KAPPA = 1.0                        # Sensitivity scale for adaptation rate
_C.TTA.DYNATTA.ETA = 0.1                          # Smoothing factor for adaptation rate
_C.TTA.DYNATTA.EPS = 1e-6                         # Numerical stability constant
_C.TTA.DYNATTA.WARMUP_FACTOR = 1                  # Warmup steps factor (multiplied by PRED_LEN)
_C.TTA.DYNATTA.UPDATE_BUFFERS_INTERVAL = 1        # Interval for updating buffers
_C.TTA.DYNATTA.UPDATE_METRICS_INTERVAL = 1        # Interval for updating metrics
_C.TTA.DYNATTA.RTAB_SIZE = 360                    # Recent Time-series Adaptation Buffer size
_C.TTA.DYNATTA.RDB_SIZE = 100                     # Representative Database size




_C.TTA.LINEAR = CN()
_C.TTA.LINEAR.LR = 1e-3
_C.TTA.LINEAR.ALPHA_LR_SCALE = 1
_C.TTA.LINEAR.GATING_LR_SCALE = 1
_C.TTA.LINEAR.GATING_INIT = 0.001
_C.TTA.LINEAR.ADAPTER_BANK_DIR = 'checkpoints/adapters/'
_C.TTA.LINEAR.N_CLUSTERS = 4
_C.TTA.LINEAR.LATENT_DIM = 64
_C.TTA.LINEAR.GAMMA = 1.0
_C.TTA.LINEAR.STEPS = 1
_C.TTA.LINEAR.PAAS = True
_C.TTA.LINEAR.BATCH_SIZE = 64
_C.TTA.LINEAR.ADJUST_PRED = True
_C.TTA.LINEAR.TEST_TYPES = 'original'
_C.TTA.LINEAR.BUFFER_SIZE = 10
_C.TTA.LINEAR.CONTEXT_SIZE = 5
_C.TTA.LINEAR.SOFTMAX = True
_C.TTA.LINEAR.S_MAX = 1.0
_C.TTA.LINEAR.EPS = 1e-6
_C.TTA.LINEAR.ADAPTER_MODE = 'learnable'  # 'pretrained' or 'learnable' or 'hybrid'

# Ours
_C.TTA.OURS = CN()
_C.TTA.OURS.LR = 1e-3
_C.TTA.OURS.PERIOD_N = 1
_C.TTA.OURS.STEPS_PER_BATCH = 1
_C.TTA.OURS.PAAS = True
_C.TTA.OURS.BATCH_SIZE = 64
_C.TTA.OURS.ADJUST_PRED = True
_C.TTA.OURS.S_MAX = 1.0          # rho_eff = S_MAX * sigmoid(rho_raw)
_C.TTA.OURS.EPS = 1e-6
_C.TTA.OURS.RESET = False

_C.TTA.OURS.LOSS = CN()
_C.TTA.OURS.LOSS.REG_COEFF = 0.2

_C.TTA.OURS.ADAPTER = CN()
_C.TTA.OURS.ADAPTER.NAME = 'linear'

_C.TTA.OURS.GATING = CN()
_C.TTA.OURS.GATING.NAME = 'tanh'
_C.TTA.OURS.GATING.INIT = 0.01
_C.TTA.OURS.GATING_LR_SCALE = 1
_C.TTA.OURS.GATING.WIN_SIZE = 24
_C.TTA.VISUALIZE = False


# DUAL adapter hyperparameters
_C.TTA.DUAL = CN()
_C.TTA.DUAL.PAAS = True
_C.TTA.DUAL.PERIOD_N = 1
_C.TTA.DUAL.BATCH_SIZE = 64
_C.TTA.DUAL.STEPS = 1
_C.TTA.DUAL.ADJUST_PRED = True
_C.TTA.DUAL.CALI_MODULE = True
_C.TTA.DUAL.GATING_INIT = 0.01
_C.TTA.DUAL.HIDDEN_DIM = 128
_C.TTA.DUAL.GCM_VAR_WISE = True
_C.TTA.DUAL.PETSA_LOWRANK = 16
_C.TTA.DUAL.PETSA_LOSS_ALPHA = 0.1
_C.TTA.DUAL.LOSS_NAME = "PETSA"
_C.TTA.DUAL.CALI_NAME = "tafas_GCM"
_C.TTA.DUAL.CALI_INPUT_ENABLE = True
_C.TTA.DUAL.CALI_OUTPUT_ENABLE = True

_C.MODEL = CN()
_C.MODEL.NAME = 'iTransformer'
_C.MODEL.task_name = 'long_term_forecast'
_C.MODEL.seq_len = _C.DATA.SEQ_LEN 
_C.MODEL.label_len = _C.DATA.LABEL_LEN # Not needed in iTransformer
_C.MODEL.pred_len = _C.DATA.PRED_LEN 
_C.MODEL.e_layers = 4
_C.MODEL.d_layers = 1 # Not needed in iTransformer
_C.MODEL.factor = 3 # Not used in iTransformer Full Attention. Used in Prob Attention (probabilistic attention) in informer
_C.MODEL.enc_in = _C.DATA.N_VAR # Used only in classification
_C.MODEL.dec_in = _C.DATA.N_VAR # Not needed in iTransformer
_C.MODEL.c_out = _C.DATA.N_VAR # Not needed in iTransformer
_C.MODEL.d_model = 512 # embedding dimension
_C.MODEL.d_ff = 512  # feedforward dimension d_model -> d_ff -> d_model
_C.MODEL.moving_avg = 25
_C.MODEL.output_attention = False # whether the attention weights are returned by the forward method of the attention class
_C.MODEL.dropout = 0.1
_C.MODEL.n_heads = 8
_C.MODEL.activation = 'gelu'
_C.MODEL.channel_independence = True
_C.MODEL.METRIC_NAMES = ('MAE',)
_C.MODEL.LOSS_NAMES = ('MSE',)
_C.MODEL.embed = 'timeF'
_C.MODEL.freq = 'h'
_C.MODEL.ignore_stamp = False
# OLS params
_C.MODEL.instance_norm = True
_C.MODEL.individual = False
_C.MODEL.alpha = 0.000001

# PatchTST specific defaults
_C.MODEL.patch_len = 16
_C.MODEL.stride = 8
_C.MODEL.num_class = 1

_C.NORM_MODULE = CN()
_C.NORM_MODULE.ENABLE = False  # NST
_C.NORM_MODULE.NAME = 'SAN'  # SAN, RevIN, DishTS

_C.SAN = CN()
_C.SAN.RESULT_DIR = 'results/station/'
_C.SAN.TRAIN = CN()
_C.SAN.TRAIN.CHECKPOINT_DIR = 'results/station/'
_C.SAN.SOLVER = CN()
_C.SAN.SOLVER.OPTIMIZING_METHOD = 'adam'
_C.SAN.SOLVER.START_EPOCH = 0
_C.SAN.SOLVER.MAX_EPOCH = 10
_C.SAN.SOLVER.BASE_LR = 0.001
_C.SAN.SOLVER.WEIGHT_DECAY = 0.0001
_C.SAN.SOLVER.MOMENTUM = 0.9
_C.SAN.SOLVER.NESTEROV = True
_C.SAN.SOLVER.DAMPENING = 0.0
_C.SAN.SOLVER.LR_POLICY = 'cosine'
_C.SAN.SOLVER.COSINE_END_LR = 0.0
_C.SAN.SOLVER.COSINE_AFTER_WARMUP = False
_C.SAN.SOLVER.WARMUP_EPOCHS = 0
_C.SAN.SOLVER.WARMUP_START_LR = 0.001

_C.REVIN = CN()
_C.REVIN.EPS = 1e-5
_C.REVIN.AFFINE = True
_C.REVIN.RESULT_DIR = 'results/revin/'
_C.REVIN.TRAIN = CN()
_C.REVIN.TRAIN.CHECKPOINT_DIR = 'results/revin/'

_C.DISHTS = CN()
_C.DISHTS.INIT = 'standard'  # standard, avg, uniform
_C.DISHTS.RESULT_DIR = 'results/dishts/'
_C.DISHTS.TRAIN = CN()
_C.DISHTS.TRAIN.CHECKPOINT_DIR = 'results/dishts/'

_C.SOLVER = CN()
_C.SOLVER.START_EPOCH = 0
_C.SOLVER.MAX_EPOCH = 30
_C.SOLVER.OPTIMIZING_METHOD = 'adam'
_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.NESTEROV = True
_C.SOLVER.DAMPENING = 0.0
_C.SOLVER.LR_POLICY = 'cosine'
_C.SOLVER.COSINE_END_LR = 0.0
_C.SOLVER.COSINE_AFTER_WARMUP = False
_C.SOLVER.WARMUP_EPOCHS = 0
_C.SOLVER.WARMUP_START_LR = 0.001

_C.WANDB = CN()
_C.WANDB.ENABLE = False
_C.WANDB.PROJECT = 'TAFAS'
_C.WANDB.NAME = ''
_C.WANDB.JOB_TYPE = ''
_C.WANDB.NOTES = ''
_C.WANDB.DIR = './'
_C.WANDB.SET_LOG_DIR = True


_C.FINETUNE = CN()
_C.FINETUNE.ENABLE = False
_C.FINETUNE.DOMAINS = 'noise,trend,regime,season'
_C.FINETUNE.MODE = 'linear_head'  # linear_head, norm, etc
_C.FINETUNE.EPOCHS = 1
_C.FINETUNE.LR = 0.0001
_C.FINETUNE.WEIGHT_DECAY = 0.0001
_C.FINETUNE.SAVE_DIR = 'results/finetune/'
_C.FINETUNE.EVAL_AFTER = True  # 微调完成后评估各域知识向量

_C.CLUSTER = CN()
_C.CLUSTER.ENABLE = True
_C.CLUSTER.RESUME = False
_C.CLUSTER.RESUME_DIR = './checkpoints/VAE/linear/ETTh1'
_C.CLUSTER.N_CLUSTERS = 10
_C.CLUSTER.LATENT_DIM = 64
_C.CLUSTER.KL_WEIGHT = 0
_C.CLUSTER.ENCODER_TYPE = "lstm"
_C.CLUSTER.DECODER_TYPE = "lstm"
_C.CLUSTER.CHECKPOINT_DIR = './tmp'



def get_cfg_defaults():

    return _C.clone()


def get_norm_module_cfg(cfg):
    return getattr(cfg, cfg.NORM_MODULE.NAME.upper())


def get_norm_method(cfg):
    assert cfg.NORM_MODULE.NAME in ('RevIN', 'SAN', 'DishTS')
    norm_method = cfg.NORM_MODULE.NAME if cfg.NORM_MODULE.ENABLE else 'NST'
    return norm_method
