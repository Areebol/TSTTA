import os
from models.build import build_model, load_best_model, build_norm_module
from utils.parser import parse_args, load_config
from utils.log import init_wandb
from datasets.build import update_cfg_from_dataset, build_dataset
from trainer import build_trainer
from predictor import Predictor
from utils.misc import set_seeds, set_devices, prepare_inputs
from tta.tafas import build_adapter
import tta.cosa as cosa
import tta.petsa as petsa
import tta.dynatta as dynatta
from tta.ours import build_tta_runner
from config import get_norm_module_cfg

def main():
    args = parse_args()
    cfg = load_config(args)
    if isinstance(cfg.DATA.TYPE, str) and cfg.DATA.TYPE.lower() in ('none', 'null', 'clean', ''):
        cfg.DATA.TYPE = None
    update_cfg_from_dataset(cfg, cfg.DATA.NAME)
    
    # select cuda devices
    set_devices(cfg.VISIBLE_DEVICES)

    # set wandb logger
    if cfg.WANDB.ENABLE:
        init_wandb(cfg)

    if not os.path.exists(cfg.RESULT_DIR):
        os.makedirs(cfg.RESULT_DIR, exist_ok=True)

    with open(os.path.join(cfg.RESULT_DIR, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())
    
    # set random seed
    set_seeds(cfg.SEED)

    # build model
    model = build_model(cfg)
    norm_module = build_norm_module(cfg) if cfg.NORM_MODULE.ENABLE else None

    if cfg.TRAIN.ENABLE:
        # build trainer
        trainer = build_trainer(cfg, model, norm_module=norm_module)
        trainer.train()

    if cfg.TTA.ENABLE:
        model = load_best_model(cfg, model)
        if cfg.NORM_MODULE.ENABLE:
            norm_module = load_best_model(get_norm_module_cfg(cfg), norm_module)
        if cfg.TTA.METHOD in ['TAFAS']:
            adapter = build_adapter(cfg, model, norm_module=norm_module)
        elif cfg.TTA.METHOD in ['PETSA']:
            adapter = petsa.build_adapter(cfg, model)
        elif cfg.TTA.METHOD in ['DynaTTA']:
            adapter = dynatta.build_adapter(cfg, model)
        elif cfg.TTA.METHOD in ['COSA']:
            adapter = cosa.build_adapter(cfg, model)
        elif cfg.TTA.METHOD == "Ours":
            adapter = build_tta_runner(cfg, model)
        else:
            print(f"Unknown TTA method: {cfg.TTA.METHOD}")
        
        adapter.adapt()
    
    if cfg.TEST.ENABLE:
        model = load_best_model(cfg, model)
        if cfg.NORM_MODULE.ENABLE:
            norm_module = load_best_model(get_norm_module_cfg(cfg), norm_module)
        predictor = Predictor(cfg, model, norm_module=norm_module)
        predictor.predict()


if __name__ == '__main__':
    main()
