#!/usr/bin/env python3
import os
import subprocess
import argparse
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(ROOT, 'configs')


def parse_yml(yml_file):
    with open(yml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data


def run_one(cfg, pred_len):
    cmd = ['python', '-u', 'run.py']
    # set required
    cmd.extend(['--is_training', '2'])
    cmd.extend(['--pred_len', str(pred_len)])
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
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run harmonic regression experiments')
    parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
    args = parser.parse_args()

    for dataset in ['ecl', 'traffic', 'weather2k', 'soil']:
        config_file = os.path.join(ROOT, 'configs', f'boost_performence/PeriRef/{dataset.lower()}.yml')
        if not os.path.exists(config_file):
            print(f'Config file {config_file} not found')
            raise SystemExit(1)

        config_data = yaml.safe_load(open(config_file, 'r'))
        fixed = config_data.get('fixed', {})
        pred_len_configs = config_data.get('pred_len_configs', {})

        pred_len = 720
        cfg = fixed.copy()
        cfg.update(pred_len_configs[pred_len])
        cfg.update({
            'pred_len': pred_len,
        })
        cfg['root_path'] = args.root_path
        cfg['ref_len'] = cfg.get('seq_len') + cfg.get('pred_len')

        run_one(cfg, pred_len)

    print('Harmonic regression experiments completed.')
