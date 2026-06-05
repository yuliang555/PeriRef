
# PeriRef Project README

This repository contains the PeriRef models, training and evaluation code, configuration files, and utility scripts. This README describes two common workflows:

- Reproducing PeriRef experiments in bulk using `run_boost.py`.
- Performing dataset periodicity diagnostics (short and long periods) and exporting ACF plots using `run_statistics.py`.

Requirements and setup
- Python 3.8 or newer (adjust as needed for your environment).
- Install dependencies:

```bash
pip install -r requirements.txt
```

- Datasets: Place your datasets under a root directory (for example `./datasets`). `run_statistics.py` uses a default `ROOT_PATH` set in the script (`/home/yl/datasets`) which can be overridden from the command line.

Reproducing PeriRef experiments (bulk runs)

Script: `run_boost.py` (located at the repository root). This script reads configuration files from `configs/boost_performence/{model}/{dataset}.yml` and launches `run.py` for each experiment configuration.

Main options:
- `--root_path`: root directory of the datasets (e.g. `./datasets`).
- `--model`: model/experiment folder (e.g. `PeriRef`, `PeriRef_Concat`, `iTransformer`, etc.).
- `--dataset`: dataset name (the script expects lower-case dataset filenames such as `ecl`, `traffic`, `soil`, `weather2k`).
- `--gpu`: GPU id (default: `0`).

Example config path:
`configs/boost_performence/PeriRef/ecl.yml`

Example command:

```bash
# From the repository root (example: run PeriRef on the ECL dataset)
python run_boost.py --root_path ./datasets --model PeriRef --dataset ecl --gpu 0
```

Notes: `run_boost.py` iterates over a set of `pred_len`, `seed`, `loss`, and other combinations defined in the script and calls `run.py` for each combination. Each call prints the full command and executes `run.py` from the repository root; the produced outputs (logs, models, evaluation results) are controlled by `run.py` and the YAML configuration.

Dataset periodicity diagnostics (short and long periods)

Script: `run_statistics.py`. This script reads dataset definitions from `utils/periodicity.py`, computes short- and long-period periodicity metrics, and generates Top-K long-period ACF PDF plots saved under `ACF_Long/`.

Usage examples:

```bash
# Use the default ROOT_PATH defined in the script
python run_statistics.py

# Override the dataset root directory
python run_statistics.py --root-path ./datasets
# Or pass the dataset root as a positional argument
python run_statistics.py ./datasets
```

Notes:
- The script validates that the dataset files listed in `utils/periodicity.py` (the `DATASET_CONFIGS`) exist under the provided `root_path`.
- Output: The script prints a brief summary for each dataset and saves the Top-3 long-period ACF PDF plots to `ACF_Long/` under the provided `root_path`.

Troubleshooting
- Missing configuration file: verify that `--model` and `--dataset` map to an existing `configs/boost_performence/{model}/{dataset}.yml` (dataset names are lower-case for the YAML files).
- Missing dataset files: confirm dataset CSV locations listed in `utils/periodicity.py` exist under your `root_path`.
- GPU or CUDA errors: ensure your installed PyTorch/CUDA versions and drivers match your environment and available hardware.

Next steps
- If you want, I can add quick helper scripts for running common experiments, or expose more fine-grained command-line options in `run_boost.py`.

