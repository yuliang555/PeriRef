from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.periodicity import (
    DATASET_CONFIGS,
    Long_Periodicity,
    Short_Periodicity,
    Top_Draw,
)


# Edit this value when you want to hard-code the dataset root directory.
ROOT_PATH = Path("/home/yl/datasets")
ACF_LONG_DIR = Path("ACF_Long")


def clean_input_path(value: str) -> Path:
    value = value.strip().strip('"').strip("'")
    return Path(value or ".").expanduser().resolve()


def ensure_dataset_files(root_path: Path) -> None:
    missing = []
    for name, config in DATASET_CONFIGS.items():
        csv_path = root_path / str(config["file"])
        if not csv_path.exists():
            missing.append(f"{name}: {csv_path}")

    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(f"Missing dataset files:\n{missing_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute short/long periodicity and draw Top3 long-period ACF PDFs."
    )
    parser.add_argument(
        "root_path",
        nargs="?",
        default=None,
        help="Dataset root directory. Defaults to ROOT_PATH in this file.",
    )
    parser.add_argument(
        "--root-path",
        "--root_path",
        dest="root_path_option",
        default=None,
        help="Dataset root directory. Overrides the positional argument.",
    )
    return parser.parse_args()


def resolve_root_path(args: argparse.Namespace) -> Path:
    value = args.root_path_option or args.root_path
    if value is None:
        return ROOT_PATH.expanduser().resolve()
    return clean_input_path(value)


def main() -> None:
    args = parse_args()
    root_path = resolve_root_path(args)
    ensure_dataset_files(root_path)

    summary_rows = []

    print(f"Root path: {root_path}")
    print(f"ACF output path: {ACF_LONG_DIR}")
    print()

    for dataset, config in DATASET_CONFIGS.items():
        csv_file = str(config["file"])
        short_period = int(config["short_period"])
        annual_period = int(config["annual_period"])

        print(f"Processing {dataset}...")

        short_calc = Short_Periodicity(
            root_path=root_path,
            dataset=dataset,
            csv_file=csv_file,
            short_period=short_period,
        )
        short_calc.run()
        short_summary = short_calc.summary()
        summary_rows.append(short_summary.as_dict())
        print(f"  {short_calc.format_summary()}")

        long_calc = Long_Periodicity(
            root_path=root_path,
            dataset=dataset,
            csv_file=csv_file,
            short_period=short_period,
            annual_period=annual_period,
            cache_acf=True,
        )
        long_df = long_calc.run()
        long_summary = long_calc.summary()
        summary_rows.append(long_summary.as_dict())
        print(f"  {long_calc.format_summary()}")

        top_draw = Top_Draw(
            root_path=root_path,
            dataset=dataset,
            csv_file=csv_file,
            short_period=short_period,
            annual_period=annual_period,
            out_dir=ACF_LONG_DIR,
        )
        top_df = top_draw.run(long_df, long_calc.acf_by_channel)
        for _, row in top_df.iterrows():
            print(
                "  Top-K "
                f"{row['channel']}: strength={row['strength']:.6f} -> {row['path']}"
            )
        print()

    summary = pd.DataFrame(summary_rows)

    print("Summary:")
    print(summary.to_string(index=False))
    print()
    print(f"Saved Top3 ACF PDFs under: {root_path / ACF_LONG_DIR}")


if __name__ == "__main__":
    main()
