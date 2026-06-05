#!/usr/bin/env python3
import os
import subprocess
import argparse
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(ROOT, 'configs')


parser = argparse.ArgumentParser(description='Run all experiments')
parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
parser.add_argument('--model', type=str, help='dataset name, e.g. ecl, etth1, ...')
parser.add_argument('--dataset', type=str, help='dataset name, e.g. ecl, etth1, ...')
parser.add_argument('--gpu', type=int, default=0, help='gpu id (default 0)')
args = parser.parse_args()


def parse_yml(yml_file):
    with open(yml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data


def run_one(cfg, seed, loss, pred_len, gpu=0):
    cmd = ['python', '-u', 'run.py']
    # set required
    cmd.extend(['--is_training', '1'])
    cmd.extend(['--loss', loss])
    cmd.extend(['--random_seed', str(seed)])
    cmd.extend(['--pred_len', str(pred_len)])
    cmd.extend(['--gpu', str(gpu)])
    cmd.extend(['--use_gpu', 'True'])
    # add other from cfg
    for key, value in cfg.items():
        if key in ['seed','loss', 'pred_len', 'gpu', 'is_training']:
            continue
        if value is None:
            continue  # Skip None values
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        elif isinstance(value, list):
            # Handle list arguments by extending with each element
            if len(value) > 0:
                cmd.append(f'--{key}')
                cmd.extend([str(v) for v in value])
        else:
            cmd.extend([f'--{key}', str(value)])
    print('=> Running:', ' '.join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    config_file = os.path.join(ROOT, 'configs', f'boost_performence/{args.model}/{args.dataset.lower()}.yml')
    if not os.path.exists(config_file):
        print(f'Config file {config_file} not found')
        raise SystemExit(1)

    config_data = yaml.safe_load(open(config_file, 'r'))
    fixed = config_data.get('fixed', {})
    pred_len_configs = config_data.get('pred_len_configs', {})

    # for all models
    SEED_LIST = [2026]
    PRED_LEN_LIST = [96, 192, 336, 720]
    LOSS_LIST = ['mse', 'mae']
    D_MODEL_LIST = [[128, 128]]
    N_LIST = [0]
    LR_LIST = [0.001]
    TYPE_LIST = ["sine"]
    LONG_LEN_LIST = [128]

    for pred_len in PRED_LEN_LIST:
        for seed in SEED_LIST:
            for loss in LOSS_LIST:
                for lr in LR_LIST:
                    for d_model in D_MODEL_LIST:
                        for N in N_LIST:
                            for long_len in LONG_LEN_LIST:
                                for wave_type in TYPE_LIST:
                                    if pred_len not in pred_len_configs:
                                        continue
                                    cfg = fixed.copy()
                                    cfg.update(pred_len_configs[pred_len])
                                    cfg.update({
                                        'pred_len': pred_len,
                                        # 'alpha': alpha,
                                        # 'learning_rate': lr,
                                        # 'd_model': d_model[0],
                                        # 'd_ff': d_model[1],
                                        # 'N': N,
                                        # 'use_pos': N[0],
                                        # 'use_norm': N[1],
                                        # 'long_len': long_len,
                                        # 'long_type': wave_type,
                                        # 'short_type': wave_type,
                                    })
                                    cfg['model'] = args.model
                                    cfg['root_path'] = args.root_path

                                    run_one(cfg, seed, loss, pred_len, gpu=args.gpu)

    print('Single experiments completed.')
