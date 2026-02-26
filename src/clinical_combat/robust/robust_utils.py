import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from clinical_combat.cli import combat_info
from clinical_combat.harmonization.Combat import Combat


def get_diseases(include_syn=False):
    """
    Return the list of diseases used in the datasets.

    include_syn: bool
        If True, include synthetic disease labels.

    Returns
    -------
    diseases: list
        List of disease labels.
    """
    if include_syn:
        return [
            "AD",
            "ADHD",
            "BIP",
            "MCI",
            "SCHZ",
            "TBI",
            "SYN_0.5",
            "SYN_1",
            "SYN_2",
            "SYN_-1",
        ]
    return ["AD", "ADHD", "BIP", "MCI", "SCHZ", "TBI"]


def get_metrics():
    """
    Return the list of diffusion metrics.

    Returns
    -------
    metrics: list
        List of metric names.
    """
    return ["ad", "adt", "afd", "fa", "fat", "fw", "md", "mdt", "rd", "rdt"]


def compare_sid(df1, df2):
    """
    Compare whether two DataFrames contain the same set of subject IDs.

    df1: dataframe
        First input DataFrame.

    df2: dataframe
        Second input DataFrame.

    Returns
    -------
    same: bool
        True if both DataFrames share the same unique 'sid' values.
    """
    sid_df1 = set(df1["sid"])
    sid_df2 = set(df2["sid"])
    return sid_df1 == sid_df2


def get_bundles(mov_data_file):
    """
    Extract bundle names from a moving site file.

    mov_data_file: str
        Path to a moving site CSV file.

    Returns
    -------
    bundles: list
        List of bundle names.
    """
    return combat_info.get_bundles(mov_data_file)


def get_info(mov_data_file):
    """
    Extract basic counts from a moving site file.

    mov_data_file: str
        Path to a moving site CSV file.

    Returns
    -------
    info: list
        [nb_total, nb_hc, nb_sick]
    """
    df, _bundles = combat_info.info(mov_data_file)

    match = re.findall(r"HC\(n=(\d+)", df["DetailInfos"]["Disease"])
    nb_hc = int(match[0]) if match else 0

    nb_total = df["DetailInfos"]["Number of Subject"]
    nb_sick = nb_total - nb_hc
    return [nb_total, nb_hc, nb_sick]


def robust_text(value):
    """
    Convert a robust flag into a standardized label.

    value: str
        Robust setting label.

    Returns
    -------
    text: str
        Standardized text.
    """
    return "NoRobust" if value == "No" else value


def rwp_text(value):
    """
    Convert a boolean flag into an RWP label.

    value: bool
        Whether RWP is enabled.

    Returns
    -------
    text: str
        'RWP' if True, else 'NoRWP'.
    """
    return "RWP" if value else "NoRWP"


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


def get_metric(mov_data_file):
    """
    Read a CSV file and return its unique metric label.

    mov_data_file: str
        Path to a moving site CSV file.

    Returns
    -------
    metric: str
        The metric label.
    """
    mov_data = pd.read_csv(mov_data_file)
    return str(mov_data.metric.unique()[0])


def get_disease(mov_data_file):
    """
    Infer the disease label for a site file.

    If the only label is 'HC', returns 'HC'. Otherwise returns the first
    non-HC disease label found.

    mov_data_file: str
        Path to a moving site CSV file.

    Returns
    -------
    disease: str
        The inferred disease label.
    """
    mov_data = pd.read_csv(mov_data_file)
    unique_diseases = mov_data["disease"].unique()

    if len(unique_diseases) == 1 and unique_diseases[0] == "HC":
        return "HC"

    return next(d for d in unique_diseases if d != "HC")


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


def scatter(df_train, df_test, title, bundle="mni_MCP"):
    """
    Plot a scatter of age vs mean for a given bundle, comparing two datasets.

    df_train: dataframe
        Training DataFrame.

    df_test: dataframe
        Test DataFrame.

    title: str
        Plot title.

    bundle: str
        Bundle name to filter on.

    Returns
    -------
    None
    """
    train_bundle = df_train[df_train["bundle"] == bundle]
    test_bundle = df_test[df_test["bundle"] == bundle]

    plt.figure(figsize=(10, 5))
    plt.scatter(
        train_bundle["age"],
        train_bundle["mean"],
        label="Train",
        alpha=0.5,
        color="green",
    )
    plt.scatter(
        test_bundle["age"],
        test_bundle["mean"],
        label="Test",
        alpha=0.5,
        color="red",
    )
    plt.xlabel("Age")
    plt.ylabel("Mean")
    plt.title(title)
    plt.legend()
    plt.show()


def get_complete_combination(
    folder_path,
    file_pattern="adni_compilation*.csv.gz",
    include_camcan=False,
    is_camcan=False,
):
    """
    Load multiple compilation files and optionally append CamCAN, then remove
    covariate effects.

    folder_path: str
        Folder containing compilation files.

    file_pattern: str
        Glob pattern to match compilation files.

    include_camcan: bool
        If True, append CamCAN raw data for the same metric.

    is_camcan: bool
        If True, use CamCAN naming convention for the site label.

    Returns
    -------
    df_combined: dataframe
        Concatenated and processed DataFrame.
    """
    file_pattern_path = os.path.join(folder_path, file_pattern)
    file_list = glob.glob(file_pattern_path)

    df_list = []
    for file_path in file_list:
        df = pd.read_csv(file_path)

        if include_camcan:
            metric = df["metric"].unique()[0]
            camcan_file = os.path.join(
                "DONNES_F", "CamCAN", f"CamCAN.{metric}.raw.csv.gz"
            )
            df = pd.concat([df, pd.read_csv(camcan_file)], ignore_index=True)

        if is_camcan:
            df["site"] = "CamCAN_compilation"
        else:
            df["old_site"] = df["site"]
            diseases = df["disease"].unique()
            disease = diseases[diseases != "HC"][0]
            df["site"] = f"{disease}_compilation"

        df = remove_covariates_effects(df)
        df_list.append(df)

    return pd.concat(df_list, ignore_index=True)


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


def transform_into_matrix(df):
    """
    Pivot a long-format DataFrame into a wide-format subject matrix.

    A combined column name 'metric_bundle' is created as '{bundle}_{metric}',
    then the table is pivoted to one row per subject.

    df: dataframe
        Input long-format DataFrame.

    Returns
    -------
    df_pivot: dataframe
        Wide-format DataFrame with one row per 'sid' and one column per
        '{bundle}_{metric}'.
    """
    df = df.copy()
    if "metric_bundle" not in df.columns:
        df["metric_bundle"] = df["bundle"] + "_" + df["metric"]

    df_pivot = df.pivot(index="sid", columns="metric_bundle", values="mean")
    df_pivot = df_pivot.reset_index()

    df_pivot.columns.name = None
    df_pivot = df_pivot.rename(
        columns=lambda x: x.replace(" ", "_") if isinstance(x, str) else x
    )

    return df_pivot


def show_scatter_plot(df, column, bundle):
    """
    Display a scatter plot of age vs a chosen column for a given bundle.

    df: dataframe
        Input long-format DataFrame.

    column: str
        Column to plot on the y-axis (e.g., 'mean' or 'mean_no_cov').

    bundle: str
        Bundle name to filter on.

    Returns
    -------
    None
    """
    actual = df[df["bundle"] == bundle]
    diseases = df["disease"].unique()
    metric = df["metric"].unique()[0]
    disease = diseases[diseases != "HC"][0] if np.any(diseases != "HC") else "HC"

    plt.clf()
    sns.scatterplot(
        data=actual,
        x="age",
        y=column,
        hue="disease",
        palette={"HC": "blue", disease: "red"},
    )
    plt.title(
        f"Scatter plot of {disease} in metric {metric} for bundle: {bundle}"
    )
    plt.xlabel("age")
    plt.ylabel(column)
    plt.legend()
    plt.show()


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
        os.path.join("DONNES", "processed", "CamCAN_clean")
        if cleaned
        else os.path.join("DONNES", "raw_CAMCAN")
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