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


def run_one(cfg, seed, loss, seq_len, gpu=0):
    cmd = ['python', '-u', 'run.py']
    # set required
    cmd.extend(['--is_training', '1'])
    cmd.extend(['--loss', loss])
    cmd.extend(['--random_seed', str(seed)])
    cmd.extend(['--seq_len', str(seq_len)])
    cmd.extend(['--gpu', str(gpu)])
    cmd.extend(['--use_gpu', 'True'])
    # add other from cfg
    for key, value in cfg.items():
        if key in ['seed','loss', 'seq_len', 'gpu', 'is_training']:
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
    config_file = os.path.join(ROOT, 'configs', f'increasing_lookback/{args.model}/{args.dataset.lower()}.yml')
    if not os.path.exists(config_file):
        print(f'Config file {config_file} not found')
        raise SystemExit(1)

    config_data = yaml.safe_load(open(config_file, 'r'))
    fixed = config_data.get('fixed', {})
    seq_len_configs = config_data.get('seq_len_configs', {})

    # for all models
    seed = 2026
    SEQ_LEN_LIST = [48, 96, 192, 336, 720]
    LOSS_LIST = ['mse', 'mae']
    E_LIST = [3]
    LR_LIST = [0.001]

    for seq_len in SEQ_LEN_LIST:
            for loss in LOSS_LIST:
                for lr in LR_LIST:
                    for e_layers in E_LIST:
                        if seq_len not in seq_len_configs:
                            continue
                        cfg = fixed.copy()
                        cfg.update(seq_len_configs[seq_len])
                        cfg.update({
                            'seq_len': seq_len,
                            # 'alpha': alpha,
                            # 'learning_rate': lr,
                            # 'd_model': 512,
                            # 'd_ff': 512,
                            # 'e_layers': e_layers
                            })
                        cfg['model'] = args.model
                        cfg['root_path'] = args.root_path

                        run_one(cfg, seed, loss, seq_len, gpu=args.gpu)

    print('Single experiments completed.')
