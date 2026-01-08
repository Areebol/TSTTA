import sys
import subprocess
import argparse

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcm_n_bases', type=int, default=8)
    parser.add_argument('--base_lr', type=float, default=0.009018)
    parser.add_argument('--online_lr', type=float, default=1e-3)
    parser.add_argument('--pretrain_epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=512)
    
    args, unknown = parser.parse_known_args()

    cmd = [
        sys.executable, 'main.py',
        'SEED', '1',
        'DATA.NAME', 'ETTh1',
        'DATA.PRED_LEN', '192',
        'MODEL.NAME', 'DLinear',
        'MODEL.pred_len', '192',
        'TRAIN.ENABLE', 'False',
        'TEST.ENABLE', 'False',
        'TTA.ENABLE', 'True',
        'TTA.METHOD', 'Ours-tta',
        'TTA.DUAL.LOSS_NAME', 'COBA',
        'TTA.DUAL.CALI_NAME', 'coba-GCM',
        'TTA.DUAL.COBA_ONLINE_ENABLED', 'True',
        'TRAIN.CHECKPOINT_DIR', 'checkpoints/DLinear/ETTh1_192/',
        'RESULT_DIR', './results/sweep_logs/',
        'WANDB.ENABLE', 'True',
    ]

    cmd.extend(['TTA.DUAL.GCM_N_BASES', str(args.gcm_n_bases)])
    cmd.extend(['TTA.SOLVER.BASE_LR', str(args.base_lr)])
    cmd.extend(['TTA.DUAL.PRETRAIN_EPOCHS', str(args.pretrain_epochs)])
    cmd.extend(['TRAIN.BATCH_SIZE', str(args.batch_size)])
    cmd.extend(['TTA.DUAL.COBA_ONLINE_LR', str(args.online_lr)])

    print(f"====== Wrapper is executing command: ======")
    print(" ".join(cmd))
    print(f"===========================================")

    subprocess.check_call(cmd)

if __name__ == '__main__':
    run()