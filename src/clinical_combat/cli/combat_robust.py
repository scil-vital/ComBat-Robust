#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add an outlier score column to the CSV files of a moving site.

Takes one CSV file per metric, all for the same site (for example
site.fa.csv site.md.csv, or a shell wildcard such as data/*.csv). The files
are concatenated and scored jointly, then each file is rewritten in place
with a new column named after the method (e.g., MAD, G_ZS, MLP2_ALL).
The rewritten rows are sorted by site, sid and bundle.
Global methods (G_*, PCA_*, MLP*) score each patient across all the files
given, and the MLP models expect the ten metrics.

IQR, HC and NO add no column: they are applied at fit time with
combat_fit --robust, so the files are left untouched.

Examples
--------
combat_robust data/*.raw.csv --method MAD

combat_robust site.fa.csv site.md.csv --method G_MAD

combat_robust data/site.*.csv --method MLP_TEST
"""

import argparse
import logging
import os

import pandas as pd

from clinical_combat.robust.robust import add_outlier_column
from clinical_combat.robust.robust_utils import METRICS
from clinical_combat.utils.scilpy_utils import add_verbose_arg


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument(
        "data",
        nargs="+",
        help="CSV file paths of the same site, one file per metric.",
    )

    p.add_argument(
        "-m",
        "--method",
        default="MAD",
        help="Outlier detection method (e.g., MAD, MLP2_ALL, G_ZS, VS, etc.). "
        "IQR, HC and NO add no column. [%(default)s]",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=list(METRICS),
        help="Metrics accepted in the input files. [%(default)s]",
    )
    add_verbose_arg(p)

    return p

def load_files(file_list: list, allowed_metrics: list, return_metrics=False):
    """
    Load multiple files, one metric per file.

    If a file contains ANY metric not in allowed_metrics, or more than one
    metric, or a metric already found in a previous file, raise an error.

    Returns
    -------
    kept_files: list
        The loaded file paths.
    big_df: DataFrame
        The concatenated data.
    metric_of_file: dict
        Metric of each file, indexed by path. Only returned when
        return_metrics is True.
    """

    kept_files = []
    dfs = []
    metric_of_file = {}
    file_of_metric = {}

    allowed_set = set(allowed_metrics)

    for path in file_list:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        df = pd.read_csv(path, low_memory=False)

        if "metric" not in df.columns:
            raise ValueError(f"No 'metric' column in file: {path}")

        file_metrics = set(df["metric"].unique())
        invalid_metrics = file_metrics - allowed_set

        if invalid_metrics:
            raise ValueError(
                f"File rejected: {path} | Invalid metrics found: {sorted(invalid_metrics)}"
            )

        if len(file_metrics) != 1:
            raise ValueError(
                f"File rejected: {path} | Expected exactly one metric per "
                f"file, found: {sorted(file_metrics)}"
            )

        metric = next(iter(file_metrics))
        if metric in file_of_metric:
            raise ValueError(
                f"Metric '{metric}' found in both {file_of_metric[metric]} "
                f"and {path}: expected one file per metric."
            )

        file_of_metric[metric] = path
        metric_of_file[path] = metric
        kept_files.append(path)
        dfs.append(df)

    if not dfs:
        raise ValueError("No valid files loaded.")

    big_df = pd.concat(dfs, ignore_index=True)

    if return_metrics:
        return kept_files, big_df, metric_of_file
    return kept_files, big_df

def save_back(file_list, big_df, metric_of_file=None):
    """
    Write updated data back to each original file.

    Each file corresponds to one metric (see load_files) and receives the
    rows of big_df for that metric. When metric_of_file is None, the metric
    of each file is read from the file itself before it is overwritten.
    """
    for f in file_list:
        logging.info("Writing file: %s", f)

        if metric_of_file is None:
            metric = pd.read_csv(f)["metric"].iloc[0]
        else:
            metric = metric_of_file[f]

        # Extract rows corresponding to the metric of that file
        subset = big_df[big_df["metric"] == metric]

        subset.to_csv(f, index=False)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # combat_fit upper-cases --robust, so the column must be upper-case too.
    args.method = args.method.upper()

    if args.method in ("IQR", "HC", "NO"):
        logging.warning(
            "%s is applied at fit time (combat_fit --robust %s): "
            "nothing to score, files left untouched.",
            args.method,
            args.method,
        )
        return

    file_list, df, metric_of_file = load_files(
        args.data, args.metrics, return_metrics=True
    )

    logging.info("Total rows loaded: %d", len(df))

    logging.info("Applying method: %s", args.method)

    df = add_outlier_column(
        df,
        method=args.method,
    )

    save_back(file_list, df, metric_of_file)


if __name__ == "__main__":
    main()
