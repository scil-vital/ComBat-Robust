import os
import subprocess

import pandas as pd

from src.clinical_combat.robust.robust_utils import get_site


def get_output_model_filename(
    mov_data_file, metric, harmonizartion_method, robust
):
    """
    Build the output model filename for a moving dataset.

    mov_data_file: str
        Path to the moving dataset file.
    metric: str
        Metric name used for harmonization.
    harmonizartion_method: str
        Harmonization method used.
    robust: str
        Robust harmonization method used.

    Returns
    -------
    str
        Model filename ending with `.model.csv`.
    """
    return (
        f"{get_site(mov_data_file)}."
        f"{metric}."
        f"{harmonizartion_method}."
        f"{robust}.model.csv"
    )


def get_output_filename(mov_data_file, metric, harmonizartion_method, robust):
    """
    Build the output filename for harmonized results.

    mov_data_file: str
        Path to the moving dataset file.
    metric: str
        Metric name used for harmonization.
    harmonizartion_method: str
        Harmonization method used.
    robust: str
        Robust harmonization method used.

    Returns
    -------
    str
        Results filename ending with `.csv`.
    """
    site = get_site(mov_data_file)
    if "test" in mov_data_file:
        site += "_test"

    return f"{site}.{metric}.{harmonizartion_method}.{robust}.csv"


def robust_script(mov_data_files, robust_method):
    """
    Run `combat_robust` for a set of moving dataset files.

    mov_data_files: str
        Space-separated paths to moving dataset files.
    robust_method: str
        Robust harmonization strategy identifier.
    """
    cmd = f"combat_robust {mov_data_files} --method {robust_method}"
    subprocess.call(cmd, shell=True)


def fit_script(
    mov_data_file,
    ref_data_file,
    metric,
    harmonizartion_method,
    robust,
    directory,
    robust_threshold=None,
    eb=False,
):
    """
    Fit a harmonization model for a moving dataset.

    mov_data_file: str
        Path to the moving dataset file.
    ref_data_file: str
        Path to the reference dataset file.
    metric: str
        Metric name used for harmonization.
    harmonizartion_method: str
        Harmonization method used.
    robust: str
        Robust harmonization method used.
    directory: str
        Output directory for the fitted model.
    robust_threshold: float | None
        Optional threshold passed to the robust method.
    eb: bool
        If True, enable empirical Bayes; otherwise disable it.

    Returns
    -------
    str
        Path to the fitted model file.
    """
    output_model_filename = get_output_model_filename(
        mov_data_file, metric, harmonizartion_method, robust
    )
    output_model_path = os.path.join(directory, output_model_filename)
    if os.path.exists(output_model_path):
        return output_model_path

    cmd_parts = [
        "combat_fit",
        ref_data_file,
        mov_data_file,
        "--out_dir",
        directory,
        "--output_model_filename",
        output_model_filename,
        "--method",
        harmonizartion_method,
        "--robust",
        robust,
        "-f",
    ]
    if robust_threshold is not None:
        cmd_parts.extend(["--robust-threshold", str(robust_threshold)])
    if not eb:
        cmd_parts.append("--no_empirical_bayes")

    subprocess.call(" ".join(cmd_parts), shell=True)
    return output_model_path


def apply_script(
    mov_data_file,
    model_filename,
    metric,
    harmonizartion_method,
    robust,
    directory,
):
    """
    Apply a fitted harmonization model to a moving dataset.

    mov_data_file: str
        Path to the moving dataset file.
    model_filename: str
        Filename of the fitted model to load.
    metric: str
        Metric name used for harmonization.
    harmonizartion_method: str
        Harmonization method used.
    robust: str
        Robust harmonization method used.
    directory: str
        Output directory for the harmonized results.

    Returns
    -------
    str
        Path to the harmonized results file.
    """
    output_filename = get_output_filename(
        mov_data_file, metric, harmonizartion_method, robust
    )
    output_file_path = os.path.join(directory, output_filename)
    if os.path.exists(output_file_path):
        return output_file_path

    cmd_parts = [
        "combat_apply",
        mov_data_file,
        model_filename,
        "--out_dir",
        directory,
        "--output_results_filename",
        output_filename,
    ]
    subprocess.call(" ".join(cmd_parts), shell=True)
    return output_file_path


def visualize_harmonization(
    f, new_f, ref_data_file, directory, bundles="", title=""
):
    """
    Visualize harmonization results against a reference dataset.

    f: str
        Path to the original moving dataset file.
    new_f: str
        Path to the harmonized dataset file.
    ref_data_file: str
        Path to the reference dataset file.
    directory: str
        Output directory for visualization artifacts.
    bundles: str
        Optional bundle filter passed to the visualization script.
    title: str
        Optional title suffix for output files.
    """
    cmd_parts = [
        "scripts/combat_visualize_harmonization.py",
        ref_data_file,
        f,
        new_f,
        "--out_dir",
        directory,
        "-f",
    ]
    if bundles:
        cmd_parts.extend(["--bundles", bundles])
    if title:
        cmd_parts.extend(["--outname", title, "--add_suffix", title])

    subprocess.call(" ".join(cmd_parts), shell=True)


def calculate_mae_std(df, compilation_df):
    """
    Compute mean absolute error normalized by standard deviation per bundle.

    df: pandas.DataFrame
        Harmonized dataset.
    compilation_df: pandas.DataFrame
        Ground truth to compare with.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing the mean normalized absolute error per
        bundle.
    """
    common_sids = df["sid"].unique()
    filtered_compilation_df = compilation_df[
        compilation_df["sid"].isin(common_sids)
    ]

    if len(filtered_compilation_df) != len(df):
        raise ValueError(
            "Mismatched row counts between df "
            f"({len(df)}) and filtered_compilation_df "
            f"({len(filtered_compilation_df)})"
        )

    comparison_df = pd.DataFrame()

    for bundle in df["bundle"].unique():
        df_bundle = df[df["bundle"] == bundle]
        compilation_bundle = filtered_compilation_df[
            filtered_compilation_df["bundle"] == bundle
        ]
        std_val = compilation_bundle["mean_no_cov"].std()

        merged_df = pd.merge(
            df_bundle,
            compilation_bundle,
            on=["sid", "bundle"],
            suffixes=("_df", "_compilation"),
        )

        merged_df["abs_diff_mean"] = (
            merged_df["mean_df"] - merged_df["mean_compilation"]
        ).abs() / std_val
        comparison_df[bundle] = merged_df["abs_diff_mean"]

    mean_df = pd.DataFrame(comparison_df.mean()).transpose()
    return mean_df
