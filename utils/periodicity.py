from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRAIN_RATIO = 0.70
TOP_N = 5

DATASET_CONFIGS = {
    "ECL": {"file": "ECL.csv", "short_period": 168, "annual_period": 8760},
    "Traffic": {"file": "Traffic.csv", "short_period": 168, "annual_period": 8760},
    "Weather2K": {"file": "Weather2K.csv", "short_period": 8, "annual_period": 2920},
    "DeepSoil": {"file": "DeepSoil.csv", "short_period": 24, "annual_period": 8760},
}



def fmt_float(value: float, digits: int = 6) -> str:
    if pd.isna(value):
        return "NaN"
    return f"{float(value):.{digits}f}"


def safe_filename(name: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name))
    safe = safe.strip().strip(".")
    return safe or "channel"


def next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def moving_average_1d(values: np.ndarray, period: int) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(values.shape, np.nan, dtype=float)

    filled = pd.Series(values, dtype="float64").interpolate(limit_direction="both")
    filled = filled.fillna(filled.mean()).to_numpy(dtype=float)

    kernel = np.ones(period, dtype=float)
    sums = np.convolve(filled, kernel, mode="same")
    counts = np.convolve(np.ones_like(filled), kernel, mode="same")
    return sums / counts


def periodic_strength(values: np.ndarray, period: int) -> tuple[float, float, float]:
    finite_count = np.isfinite(values).sum()
    if finite_count < period * 2:
        return np.nan, np.nan, np.nan

    filled = pd.Series(values, dtype="float64").interpolate(limit_direction="both")
    filled = filled.fillna(filled.mean()).to_numpy(dtype=float)
    trend = moving_average_1d(values, period)
    detrended = filled - trend

    positions = np.arange(detrended.size) % period
    profile = np.empty(period, dtype=float)
    for pos in range(period):
        profile[pos] = np.mean(detrended[positions == pos])
    seasonal = profile[positions]
    remainder = detrended - seasonal

    seasonal_remainder_var = float(np.var(seasonal + remainder))
    remainder_var = float(np.var(remainder))
    periodic_var = float(np.var(seasonal))
    if seasonal_remainder_var <= 0 or not np.isfinite(seasonal_remainder_var):
        return np.nan, periodic_var, remainder_var

    strength = max(0.0, 1.0 - remainder_var / seasonal_remainder_var)
    return float(strength), periodic_var, remainder_var


def moving_average_frame(values: pd.DataFrame, window: int) -> pd.DataFrame:
    return values.rolling(window=window, center=True, min_periods=1).mean()


def acf_fft(values: np.ndarray) -> np.ndarray | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None

    values = values - values.mean()
    variance_sum = float(np.dot(values, values))
    if variance_sum <= 0 or not math.isfinite(variance_sum):
        return None

    fft_len = next_power_of_two(values.size * 2 - 1)
    spectrum = np.fft.fft(values, n=fft_len)
    acf = np.fft.ifft(spectrum * np.conjugate(spectrum)).real[: values.size]
    acf /= acf[0]
    return acf


def ordinal_label(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return rf"${value}^{{{suffix}}}$ year"


def annual_lags(acf_length: int, annual_period: int) -> list[int]:
    return [
        lag
        for lag in range(annual_period, acf_length, annual_period)
        if lag < acf_length
    ]


def load_full_values(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")


def load_train_values(csv_path: Path, train_ratio: float = TRAIN_RATIO) -> pd.DataFrame:
    with csv_path.open("rb") as handle:
        row_count = sum(1 for _ in handle) - 1

    train_rows = int(row_count * train_ratio)
    if train_rows < 1:
        raise ValueError(f"{csv_path} has no train rows with train_ratio={train_ratio}")

    df = pd.read_csv(csv_path, nrows=train_rows)
    values = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    values = values.interpolate(axis=0, limit_direction="both")
    return values.fillna(values.mean(numeric_only=True))


@dataclass(frozen=True)
class PeriodicitySummary:
    dataset: str
    metric: str
    period: int
    mean_strength: float
    max_strength: float
    max_channel: str

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "metric": self.metric,
            "period": self.period,
            "mean_strength": self.mean_strength,
            "max_strength": self.max_strength,
            "max_channel": self.max_channel,
        }

    def format(self) -> str:
        return (
            f"{self.dataset:<10} | {self.metric:<5} | period={self.period:<5} | "
            f"mean={fmt_float(self.mean_strength)} | "
            f"max={fmt_float(self.max_strength)} | channel={self.max_channel}"
        )


class Short_Periodicity:
    """Compute short-period variance strength for every channel."""

    metric_name = "short"

    def __init__(
        self,
        root_path: str | Path,
        dataset: str,
        csv_file: str,
        short_period: int,
    ) -> None:
        self.root_path = Path(root_path)
        self.dataset = dataset
        self.csv_file = csv_file
        self.short_period = int(short_period)
        self.result: pd.DataFrame | None = None

    @property
    def csv_path(self) -> Path:
        return self.root_path / self.csv_file

    def run(self) -> pd.DataFrame:
        values = load_full_values(self.csv_path)
        rows = []
        for channel in values.columns:
            series = values[channel].to_numpy(dtype=float)
            strength, periodic_var, remainder_var = periodic_strength(
                series,
                self.short_period,
            )
            rows.append(
                {
                    "dataset": self.dataset,
                    "channel": str(channel),
                    "period": self.short_period,
                    "strength": strength,
                    "periodic_variance": periodic_var,
                    "remainder_variance": remainder_var,
                }
            )

        result = pd.DataFrame(rows).sort_values(
            "strength",
            ascending=False,
            na_position="last",
        )
        self.result = result.reset_index(drop=True)
        return self.result

    def summary(self) -> PeriodicitySummary:
        if self.result is None:
            self.run()
        assert self.result is not None

        valid = self.result.dropna(subset=["strength"])
        if valid.empty:
            mean_strength = np.nan
            max_strength = np.nan
            max_channel = ""
        else:
            mean_strength = float(valid["strength"].mean())
            max_row = valid.iloc[valid["strength"].to_numpy().argmax()]
            max_strength = float(max_row["strength"])
            max_channel = str(max_row["channel"])

        return PeriodicitySummary(
            dataset=self.dataset,
            metric=self.metric_name,
            period=self.short_period,
            mean_strength=mean_strength,
            max_strength=max_strength,
            max_channel=max_channel,
        )

    def format_summary(self) -> str:
        return self.summary().format()


class Long_Periodicity:
    """Compute annual-period ACF strength for every channel."""

    metric_name = "long"

    def __init__(
        self,
        root_path: str | Path,
        dataset: str,
        csv_file: str,
        short_period: int,
        annual_period: int,
        train_ratio: float = TRAIN_RATIO,
        cache_acf: bool = True,
    ) -> None:
        self.root_path = Path(root_path)
        self.dataset = dataset
        self.csv_file = csv_file
        self.short_period = int(short_period)
        self.annual_period = int(annual_period)
        self.train_ratio = float(train_ratio)
        self.cache_acf = cache_acf
        self.acf_by_channel: dict[str, np.ndarray] = {}
        self.result: pd.DataFrame | None = None

    @property
    def csv_path(self) -> Path:
        return self.root_path / self.csv_file

    def run(self) -> pd.DataFrame:
        values = load_train_values(self.csv_path, self.train_ratio)
        smoothed = moving_average_frame(values, self.short_period)

        rows = []
        acf_cache: dict[str, np.ndarray] = {}
        for channel in smoothed.columns:
            channel_name = str(channel)
            acf = acf_fft(smoothed[channel].to_numpy(dtype=float))
            if acf is None or self.annual_period >= acf.size:
                strength = np.nan
                acf_length = 0 if acf is None else int(acf.size)
            else:
                strength = float(acf[self.annual_period])
                acf_length = int(acf.size)
                if self.cache_acf:
                    acf_cache[channel_name] = acf

            rows.append(
                {
                    "dataset": self.dataset,
                    "channel": channel_name,
                    "short_period": self.short_period,
                    "annual_period": self.annual_period,
                    "train_ratio": self.train_ratio,
                    "train_rows": len(smoothed),
                    "acf_length": acf_length,
                    "strength": strength,
                }
            )

        self.acf_by_channel = acf_cache
        result = pd.DataFrame(rows).sort_values(
            "strength",
            ascending=False,
            na_position="last",
        )
        self.result = result.reset_index(drop=True)
        return self.result

    def summary(self) -> PeriodicitySummary:
        if self.result is None:
            self.run()
        assert self.result is not None

        valid = self.result.dropna(subset=["strength"])
        if valid.empty:
            mean_strength = np.nan
            max_strength = np.nan
            max_channel = ""
        else:
            mean_strength = float(valid["strength"].mean())
            max_row = valid.iloc[valid["strength"].to_numpy().argmax()]
            max_strength = float(max_row["strength"])
            max_channel = str(max_row["channel"])

        return PeriodicitySummary(
            dataset=self.dataset,
            metric=self.metric_name,
            period=self.annual_period,
            mean_strength=mean_strength,
            max_strength=max_strength,
            max_channel=max_channel,
        )

    def format_summary(self) -> str:
        return self.summary().format()


class Top_Draw:
    """Draw ACF PDFs for the Top-3 long-period strength channels."""

    def __init__(
        self,
        root_path: str | Path,
        dataset: str,
        csv_file: str,
        short_period: int,
        annual_period: int,
        out_dir: str | Path,
        top_n: int = TOP_N,
        train_ratio: float = TRAIN_RATIO,
    ) -> None:
        self.root_path = Path(root_path)
        self.dataset = dataset
        self.csv_file = csv_file
        self.short_period = int(short_period)
        self.annual_period = int(annual_period)
        self.out_dir = Path(out_dir)
        self.top_n = int(top_n)
        self.train_ratio = float(train_ratio)

    def run(
        self,
        long_result: pd.DataFrame | None = None,
        acf_by_channel: dict[str, np.ndarray] | None = None,
    ) -> pd.DataFrame:
        if long_result is None or acf_by_channel is None:
            calculator = Long_Periodicity(
                self.root_path,
                self.dataset,
                self.csv_file,
                self.short_period,
                self.annual_period,
                train_ratio=self.train_ratio,
                cache_acf=True,
            )
            long_result = calculator.run()
            acf_by_channel = calculator.acf_by_channel

        valid = long_result.dropna(subset=["strength"]).head(self.top_n)
        rows = []
        for _, row in valid.iterrows():
            channel = str(row["channel"])
            acf = acf_by_channel.get(channel)
            if acf is None:
                continue

            out_path = self.plot_channel(channel, acf)
            rows.append(
                {
                    "dataset": self.dataset,
                    "channel": channel,
                    "annual_period": self.annual_period,
                    "strength": float(row["strength"]),
                    "path": str(out_path),
                }
            )

        return pd.DataFrame(rows)

    def plot_channel(self, channel: str, acf: np.ndarray) -> Path:
        lags = np.arange(acf.size)
        year_lags = annual_lags(acf.size, self.annual_period)
        year_values = acf[year_lags] if year_lags else np.array([])

        fig, ax = plt.subplots(figsize=(9, 3), dpi=150)
        ax.vlines(
            lags,
            0.0,
            acf,
            color="#1f77b4",
            linewidth=0.35,
            alpha=0.85,
            rasterized=True,
        )
        ax.axhline(0.0, color="#777777", linewidth=0.65, alpha=0.8)

        if year_lags:
            ax.vlines(
                year_lags,
                0.0,
                year_values,
                color="#d62728",
                linewidth=2.0,
                zorder=3,
            )
            ax.scatter(year_lags, year_values, color="#d62728", s=18, zorder=4)

        right_limit = acf.size - 0.5
        if year_lags:
            right_limit = max(right_limit, year_lags[-1] + self.annual_period * 0.18)
        ax.set_xlim(-0.5, right_limit)
        ax.set_xlabel("")
        ax.set_ylabel("ACF", labelpad=0, fontsize=15)
        ax.set_xticks(year_lags)
        ax.set_xticklabels(
            [ordinal_label(index) for index in range(1, len(year_lags) + 1)],
            fontsize=14,
        )
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.tick_params(axis="both", which="major", pad=2)
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.55)
        ax.grid(False, axis="x")
        ax.margins(x=0.0, y=0.035)
        fig.subplots_adjust(left=0.105, right=0.995, bottom=0.125, top=0.995)

        out_dir = self.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"ACF_{self.dataset}_{safe_filename(channel)}.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
