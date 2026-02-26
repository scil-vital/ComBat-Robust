#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add an outlier score column to CSV data.

Supports:
- Single file
- Parametric path with '*' to load multiple metric files

If '*' is used:
    - All matching files are loaded
    - Concatenated into one DataFrame
    - Scores computed globally

Examples
--------
add_outlier_column data/*.raw.csv --method MAD

add_outlier_column site_FA.raw.csv --method MLP2_ALL
"""

import argparse
import glob
import logging
import os

import pandas as pd

from clinical_combat.utils.scilpy_utils import (
    add_overwrite_arg,
    add_verbose_arg,
    assert_outputs_exist,
)
from clinical_combat.robust.robust import add_outlier_column

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument(
        "data",
        nargs="+",
        help="CSV file path or parametric path with '*' (one file per metric).",
    )

    p.add_argument(
        "-m",
        "--method",
        default="MAD",
        help="Outlier detection method (e.g., MAD, IQR, MLP2_ALL, G_ZS, VS, etc.).",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["ad", "adt", "afd", "fa", "fat", "fw", "md", "mdt", "rd", "rdt"],
        help="Metrics to process (e.g., ad, adt, afd, fa, fat, fw, md, mdt, rd, rdt).",
    )

    p.add_argument(
        "--score_col",
        default=None,
        help="Name of the output column. Default: OUT_{method}",
    )

    add_verbose_arg(p)

    return p

def load_files(file_list: list, allowed_metrics: list):
    """
    Load multiple files.

    If a file contains ANY metric not in allowed_metrics,
    raise an error and reject the file.
    """

    kept_files = []
    dfs = []

    allowed_set = set(allowed_metrics)

    for path in file_list:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        if "metric" not in df.columns:
            raise ValueError(f"No 'metric' column in file: {path}")

        file_metrics = set(df["metric"].unique())
        invalid_metrics = file_metrics - allowed_set

        if invalid_metrics:
            raise ValueError(
                f"File rejected: {path} | Invalid metrics found: {sorted(invalid_metrics)}"
            )

        kept_files.append(path)
        dfs.append(df)

    if not dfs:
        raise ValueError("No valid files loaded.")

    big_df = pd.concat(dfs, ignore_index=True)

    return kept_files, big_df

def save_back(file_list, big_df):
    """
    Write updated data back to each original file.

    Assumes each file corresponds to one metric
    and all share the same patient SIDs.
    """
    for f in file_list:
        logging.info("Writing file: %s", f)

        # Reload original file to detect metric
        orig = pd.read_csv(f)
        metric = orig["metric"].iloc[0]

        # Extract rows corresponding to that metric
        subset = big_df[big_df["metric"] == metric]

        subset.to_csv(f, index=False)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    file_list, df = load_files(args.data, args.metrics)

    logging.info("Total rows loaded: %d", len(df))

    logging.info("Applying method: %s", args.method)

    df = add_outlier_column(
        df,
        method=args.method,
        score_col=args.score_col,
    )

    save_back(file_list, df)


if __name__ == "__main__":
    main()