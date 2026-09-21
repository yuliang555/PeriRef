# PeriRef

Code for **PeriRef: Semantically Aligned Periodic References for Look-Back-Limited Time-Series Forecasting**.

PeriRef combines known periodic references with observations through cross-attention for time-series forecasting with limited look-back windows.

## Installation and execution

Install the dependencies from the project root:

```bash
pip install -r requirements.txt
```

The dataset CSV files are stored using Git LFS; after cloning, install Git LFS and download them:

```bash
git lfs install
git lfs pull
```

Ensure `ECL.csv`, `Traffic.csv`, `Weather2K.csv`, and `DeepSoil.csv` are in `./datasets/`, or update `root_path` in `run.sh` to your dataset directory; set `gpu` to the desired GPU index (default: `0`).

Run the experiments in a Bash environment (Linux or WSL):

```bash
mkdir -p logs
bash run.sh
```

By default, `run.sh` enables only `run_periref.py`; uncomment the corresponding command block to enable another experiment, and comment out any blocks you do not want to run. Enabled commands run in the background, with standard output saved to the corresponding files under `logs/`.

## Experiments in `run.sh`

The forecasting experiments use Electricity, Traffic, Weather2K, and DeepSoil; the ablation and controlled comparison experiments use a prediction horizon of 720 steps.

| Program | Paper reference | Experiment |
| --- | --- | --- |
| `run_periref.py` | Table 2 | Runs the main PeriRef forecasting experiments with prediction horizons of 96, 192, 336, and 720 steps using seeds 2025, 2026, and 2027. |
| `run_periref_generic_random.py` | Table 3, RF-Ref | Replaces the known periods with randomly sampled periods to test the value of semantically aligned references. |
| `run_periref_generic_learnable.py` | Table 3, LF-Ref | Uses learnable Fourier frequencies to compare generic periodic references with semantically aligned references. |
| `run_periref_scale.py` | Table 3, S-Ref / L-Ref | Evaluates short-period-only and long-period-only references to assess their contributions relative to the full multi-scale model. |
| `run_periref_embedding.py` | Table 3, Share-Embed | Shares the temporal embedding across short- and long-period reference groups to assess the benefit of separate group embeddings. |
| `run_harmonic_regression.py` | Table 4, HR | Evaluates harmonic regression as a fixed periodic prior by fitting a regularized linear combination of the same periodic references as PeriRef. |
| `run_seasonal_climatology.py` | Table 4, SC | Evaluates seasonal climatology by predicting the mean training observation at the same calendar phase. |
| `run_periref_mlp_x.py` | Table 4, X-MLP | Infers reference composition weights from observations alone using an MLP to assess the benefit of observation-reference matching. |
| `run_periref_fourier_phaseonly.py` | Table 4, PhF | Uses phase-only Fourier references without trajectories to assess the contribution of trajectory information. |
| `run_periref_fourier_pointwise.py` | Table 4, PtF | Embeds Fourier features at each timestamp as a token to compare pointwise representations with PeriRef's trajectory tokens. |
| `run_periref_perturb.py` | Table 5 | Perturbs the configured long periods by -3%, -1%, +1%, and +3% to evaluate sensitivity to period specification. |
| `run_acf_diagnostics.py` | Section 2.4 and Figure 1(b) | Computes training-set short- and long-period diagnostics and plots long-period autocorrelation functions (ACFs). |
