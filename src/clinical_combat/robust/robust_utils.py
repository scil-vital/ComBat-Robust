import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from clinical_combat.cli import combat_info
from clinical_combat.harmonization.Combat import Combat


def get_metrics():
    """
    Return the list of diffusion metrics.

    Returns
    -------
    metrics: list
        List of metric names.
    """
    return ["ad", "adt", "afd", "fa", "fat", "fw", "md", "mdt", "rd", "rdt"]


def get_site(mov_data_file):
    """
    Read a CSV file and return its unique site label.

    mov_data_file: str
        Path to a moving site CSV file.

    Returns
    -------
    site: str
        The site label.
    """
    mov_data = pd.read_csv(mov_data_file)
    return str(mov_data.site.unique()[0])


def add_nb_patients_and_diseased(df):
    """
    Add patient count and disease ratio information extracted from the site name.

    The function expects the site name to include patterns like:
    '<N>_patients' and '<P>_percent'.

    df: dataframe
        Input DataFrame containing a 'site' column.

    Returns
    -------
    df: dataframe
        DataFrame with added columns:
        'num_patients', 'disease_ratio', 'num_diseased'.
    """
    df = df.copy()
    df["num_patients"] = df["site"].str.extract(r"(\d+)_patients")[0].astype(int)
    df["disease_ratio"] = df["site"].str.extract(r"(\d+)_percent")[0].astype(int)
    df["num_diseased"] = (df["num_patients"] * df["disease_ratio"] / 100).astype(
        int
    )
    return df


def get_design_matrices(df, ignore_handedness=False):
    """
    Build per-bundle design matrices and response vectors.

    The model includes:
    intercept, sex (categorical), handedness (categorical, optional), age.

    df: dataframe
        Input long-format DataFrame with columns including
        'sid', 'bundle', 'sex', 'handedness', 'age', and 'mean'.

    ignore_handedness: bool
        If True, do not include handedness in the design matrix.

    Returns
    -------
    design: list
        List of design matrices (as stacked arrays) per bundle.

    y: list
        List of response vectors (mean values) per bundle.
    """
    design = []
    y = []

    for bundle in list(np.unique(df["bundle"])):
        data = df.query("bundle == @bundle")

        parts = [np.ones(len(data["sid"]))]  # intercept
        parts.append(Combat.to_category(data["sex"]))

        if not ignore_handedness:
            parts.append(Combat.to_category(data["handedness"]))

        parts.append(data["age"].to_numpy())

        design.append(np.array(parts))
        y.append(data["mean"].to_numpy())

    return design, y


def remove_covariates_effects(df):
    """
    Remove covariate effects from 'mean' using a per-bundle linear model.

    The fitted covariate effects are subtracted from the original values and
    stored in 'mean_no_cov'. If handedness has a single unique value, it is
    excluded from the model.

    df: dataframe
        Input long-format DataFrame.

    Returns
    -------
    df: dataframe
        DataFrame with an additional column 'mean_no_cov'.
    """
    df = df.sort_values(by=["site", "sid", "bundle"]).copy()

    ignore_handedness = df["handedness"].nunique() == 1
    design, y = get_design_matrices(df, ignore_handedness=ignore_handedness)
    _alpha, beta = Combat.get_alpha_beta(design, y)

    df["mean_no_cov"] = df["mean"]

    bundles = list(np.unique(df["bundle"]))
    for i, bundle in enumerate(bundles):
        covariate_effect = np.dot(design[i][1:, :].T, beta[i])
        df.loc[df["bundle"] == bundle, "mean_no_cov"] = y[i] - covariate_effect

    return df


def get_camcan_file(metric, cleaned=False):
    """
    Return path to CamCAN file for a metric.

    metric: str
        CamCAN metric name (e.g., 'fa', 'mdt').

    cleaned: bool
        If True, return path to cleaned CamCAN export; otherwise raw.

    Returns
    -------
    path: str
        Full path to the CamCAN file.
    """
    base_dir = (
        os.path.join("DATA", "processed", "CamCAN_clean")
        if cleaned
        else os.path.join("DATA", "raw_camcan")
    )
    suffix = "clean" if cleaned else "raw"
    return os.path.join(base_dir, f"CamCAN.{metric}.{suffix}.csv.gz")


def remove_covariates_effects_metrics(df):
    """
    Apply covariate effect removal independently for each metric.

    If 'mean_no_cov' already exists, it is dropped and recomputed.

    df: dataframe
        Input long-format DataFrame.

    Returns
    -------
    total: dataframe
        Concatenated DataFrame with 'mean_no_cov' computed per metric.
    """
    df = df.copy()
    if "mean_no_cov" in df.columns:
        df = df.drop(columns=["mean_no_cov"])

    total = []
    for metric in df["metric"].unique():
        df_metric = df[df["metric"] == metric]
        total.append(remove_covariates_effects(df_metric))

    return pd.concat(total, ignore_index=True)