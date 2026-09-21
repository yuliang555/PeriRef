#!/usr/bin/env python3
import os
import subprocess
import argparse
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(ROOT, 'configs')


parser = argparse.ArgumentParser(description='Run all experiments')
parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
parser.add_argument('--gpu', type=int, default=0, help='gpu id (default 0)')
args = parser.parse_args()


def parse_yml(yml_file):
    with open(yml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data


def run_one(cfg, seed, pred_len, gpu=0):
    cmd = ['python', '-u', 'run.py']
    # set required
    cmd.extend(['--is_training', '1'])
    cmd.extend(['--random_seed', str(seed)])
    cmd.extend(['--pred_len', str(pred_len)])
    cmd.extend(['--gpu', str(gpu)])
    cmd.extend(['--use_gpu', 'True'])
    # add other from cfg
    for key, value in cfg.items():
        if key in ['seed', 'pred_len', 'gpu', 'is_training']:
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
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    DATASET_LIST = ['ecl', 'traffic', 'weather2k', 'soil']
    PRED_LEN_LIST = [96, 192, 336, 720]
    SEED_LIST = [2025, 2026, 2027]

    for dataset in DATASET_LIST:
        config_file = os.path.join(ROOT, 'configs', f'boost_performence/PeriRef/{dataset.lower()}.yml')
        if not os.path.exists(config_file):
            print(f'Config file {config_file} not found')
            raise SystemExit(1)

        config_data = yaml.safe_load(open(config_file, 'r'))
        fixed = config_data.get('fixed', {})
        pred_len_configs = config_data.get('pred_len_configs', {})

        for pred_len in PRED_LEN_LIST:
            for seed in SEED_LIST:
                if pred_len not in pred_len_configs:
                    continue
                cfg = fixed.copy()
                cfg.update(pred_len_configs[pred_len])
                cfg.update({
                    'pred_len': pred_len,
                })
                cfg['model'] = 'PeriRef'
                cfg['root_path'] = args.root_path

                run_one(cfg, seed, pred_len, gpu=args.gpu)

    print('Single experiments completed.')
