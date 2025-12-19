import os
from yacs.config import CfgNode as CN

# import dataset classes from split modules
from .forecasting import (
    ForecastingDataset,
    Weather,
    Illness,
    Electricity,
    Traffic,
    Exchange,
    ETTh1,
    ETTh2,
    ETTm1,
    ETTm2,
)
# from .perturbations import (
#     PerturbationDataset,
#     ExchangePerturbation,
#     WeatherPerturbation,
#     IllnessPerturbation,
#     ETTh1Perturbation,
#     ETTh2Perturbation,
#     ETTm1Perturbation,
#     ETTm2Perturbation,
# )
from .eved import EVED

# ============== build dataset ==============
def build_dataset(cfg, split):
    data_name = cfg.DATA.NAME
    dataset_config = dict(
        data_dir=os.path.join(cfg.DATA.BASE_DIR, data_name),
        n_var=cfg.DATA.N_VAR,
        seq_len=cfg.DATA.SEQ_LEN,
        label_len=cfg.DATA.LABEL_LEN,
        pred_len=cfg.DATA.PRED_LEN,
        features=cfg.DATA.FEATURES,
        timeenc=cfg.DATA.TIMEENC,
        freq=cfg.DATA.FREQ,
        date_idx=cfg.DATA.DATE_IDX,
        target_start_idx=cfg.DATA.TARGET_START_IDX,
        scale=cfg.DATA.SCALE,
        split=split,
        train_ratio=cfg.DATA.TRAIN_RATIO,
        test_ratio=cfg.DATA.TEST_RATIO,
    )
        
    if data_name == "weather":
        dataset = Weather(**dataset_config)
    elif data_name == 'illness':
        dataset = Illness(**dataset_config)
    elif data_name == 'electricity':
        dataset = Electricity(**dataset_config)
    elif data_name == 'traffic':
        dataset = Traffic(**dataset_config)
    elif data_name == 'exchange_rate':
        dataset = Exchange(**dataset_config)
    elif data_name == 'ETTh1':
        dataset = ETTh1(**dataset_config)
    elif data_name == 'ETTh2':
        dataset = ETTh2(**dataset_config)
    elif data_name == 'ETTm1':
        dataset = ETTm1(**dataset_config)
    elif data_name == 'ETTm2':
        dataset = ETTm2(**dataset_config)
    elif data_name == 'eVED':
        dataset = EVED(**dataset_config)
    else:
        raise ValueError

    return dataset


def update_cfg_from_dataset(cfg: CN, dataset_name: str):
    cfg.DATA.NAME = dataset_name
    if dataset_name == 'weather':
        n_var = 21
        cfg.DATA.N_VAR = n_var
        cfg.DATA.FEATURES = 'M'
        cfg.DATA.TARGET_START_IDX = 0
        cfg.DATA.PERIOD_LEN = 12  #! for SAN
        cfg.DATA.TRAIN_RATIO = 0.7
        cfg.DATA.TEST_RATIO = 0.2
        
        cfg.MODEL.enc_in = n_var
        cfg.MODEL.dec_in = n_var
        cfg.MODEL.c_out = n_var
    elif dataset_name == 'illness':
        n_var = 7
        cfg.DATA.N_VAR = n_var
        cfg.DATA.FEATURES = 'M'
        cfg.DATA.TARGET_START_IDX = 0
        cfg.DATA.PERIOD_LEN = 6  #! for SAN
        cfg.DATA.TRAIN_RATIO = 0.7
        cfg.DATA.TEST_RATIO = 0.2
        
        cfg.MODEL.enc_in = n_var
        cfg.MODEL.dec_in = n_var
        cfg.MODEL.c_out = n_var
    elif dataset_name == 'electricity':
        n_var = 321
        cfg.DATA.N_VAR = n_var
        cfg.DATA.FEATURES = 'M'
        cfg.DATA.TARGET_START_IDX = 0
        cfg.DATA.PERIOD_LEN = 24  #! for SAN
        cfg.DATA.TRAIN_RATIO = 0.7
        cfg.DATA.TEST_RATIO = 0.2
        
        cfg.MODEL.enc_in = n_var
        cfg.MODEL.dec_in = n_var
        cfg.MODEL.c_out = n_var
    elif dataset_name == 'traffic':
        n_var = 862
        cfg.DATA.N_VAR = n_var
        cfg.DATA.FEATURES = 'M'
        cfg.DATA.TARGET_START_IDX = 0
        cfg.DATA.PERIOD_LEN = 24  #! for SAN
        cfg.DATA.TRAIN_RATIO = 0.7
        cfg.DATA.TEST_RATIO = 0.2
        
        cfg.MODEL.enc_in = n_var
        cfg.MODEL.dec_in = n_var
        cfg.MODEL.c_out = n_var
    elif dataset_name == 'exchange_rate':
        n_var = 8
        cfg.DATA.N_VAR = n_var
        cfg.DATA.FEATURES = 'M'
        cfg.DATA.TARGET_START_IDX = 0
        cfg.DATA.PERIOD_LEN = 6  #! for SAN
        cfg.DATA.TRAIN_RATIO = 0.7
        cfg.DATA.TEST_RATIO = 0.2
        
        cfg.MODEL.enc_in = n_var
        cfg.MODEL.dec_in = n_var
        cfg.MODEL.c_out = n_var
    elif dataset_name == 'ETTh1':
        n_var = 7
        cfg.DATA.N_VAR = n_var
        cfg.DATA.FEATURES = 'M'
        cfg.DATA.TARGET_START_IDX = 0
        cfg.DATA.PERIOD_LEN = 24  #! for SAN
        cfg.DATA.TRAIN_RATIO = 0.6
        cfg.DATA.TEST_RATIO = 0.2
        
        cfg.MODEL.enc_in = n_var
        cfg.MODEL.dec_in = n_var
        cfg.MODEL.c_out = n_var
    elif dataset_name == 'ETTh2':
        n_var = 7
        cfg.DATA.N_VAR = n_var
        cfg.DATA.FEATURES = 'M'
        cfg.DATA.TARGET_START_IDX = 0
        cfg.DATA.PERIOD_LEN = 24  #! for SAN
        cfg.DATA.TRAIN_RATIO = 0.6
        cfg.DATA.TEST_RATIO = 0.2
        
        cfg.MODEL.enc_in = n_var
        cfg.MODEL.dec_in = n_var
        cfg.MODEL.c_out = n_var
    elif dataset_name == 'ETTm1':
        n_var = 7
        cfg.DATA.N_VAR = n_var
        cfg.DATA.FEATURES = 'M'
        cfg.DATA.TARGET_START_IDX = 0
        cfg.DATA.PERIOD_LEN = 12  #! for SAN
        cfg.DATA.TRAIN_RATIO = 0.6
        cfg.DATA.TEST_RATIO = 0.2
        
        cfg.MODEL.enc_in = n_var
        cfg.MODEL.dec_in = n_var
        cfg.MODEL.c_out = n_var
    elif dataset_name == 'ETTm2':
        n_var = 7
        cfg.DATA.N_VAR = n_var
        cfg.DATA.FEATURES = 'M'
        cfg.DATA.TARGET_START_IDX = 0
        cfg.DATA.PERIOD_LEN = 12  #! for SAN
        cfg.DATA.TRAIN_RATIO = 0.6
        cfg.DATA.TEST_RATIO = 0.2
        
        cfg.MODEL.enc_in = n_var
        cfg.MODEL.dec_in = n_var
        cfg.MODEL.c_out = n_var
    elif dataset_name == 'eVED':
        # Base 14 vars + 6 future-known vars for next PRED_LEN seconds
        n_var = 14 + 6
        cfg.DATA.N_VAR = n_var
        cfg.DATA.FEATURES = 'M'  # single target
        # vehicle speed index is 12,
        # Energy_Consumption index is 13
        cfg.DATA.TARGET_START_IDX = 12
        cfg.DATA.PERIOD_LEN = 60
        cfg.DATA.TRAIN_RATIO = 0.7
        cfg.DATA.TEST_RATIO = 0.2

        cfg.MODEL.enc_in = n_var
        cfg.MODEL.dec_in = n_var
        # cfg.MODEL.c_out = 1   # predict energy consumption
        cfg.MODEL.c_out = 2   # predict vehicle speed and energy consumption
    else:
        raise ValueError
