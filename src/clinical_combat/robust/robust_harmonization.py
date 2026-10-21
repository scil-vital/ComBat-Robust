import glob
import os
import subprocess

import pandas as pd

from .robust_utils import get_site


def get_output_model_filename(
    mov_data_file, metric, harmonization_method, robust
):
    """
    Build the output model filename for a moving dataset.

    mov_data_file: str
        Path to the moving dataset file.
    metric: str
        Metric name used for harmonization.
    harmonization_method: str
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
        f"{harmonization_method}."
        f"{robust}.model.csv"
    )


def get_output_filename(mov_data_file, metric, harmonization_method, robust):
    """
    Build the output filename for harmonized results.

    mov_data_file: str
        Path to the moving dataset file.
    metric: str
        Metric name used for harmonization.
    harmonization_method: str
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

    return f"{site}.{metric}.{harmonization_method}.{robust}.csv"


def _expand_data_files(mov_data_files):
    """
    Expand moving dataset files given as a string, a path or a list of paths.

    mov_data_files: str, os.PathLike or list
        A path-like object, or a string that is an existing file path, is
        one path used as is (no split, no glob). Any other string is split
        on whitespace, and each piece may contain glob wildcards (expanded
        and sorted). A list of paths is used as is.

    Returns
    -------
    files: list
        List of file paths.
    """
    if isinstance(mov_data_files, os.PathLike):
        files = [os.fspath(mov_data_files)]
    elif isinstance(mov_data_files, str):
        if os.path.isfile(mov_data_files):
            files = [mov_data_files]
        else:
            files = []
            for piece in mov_data_files.split():
                if glob.has_magic(piece):
                    matches = sorted(glob.glob(piece))
                    if not matches:
                        raise FileNotFoundError(f"No file matches: {piece}")
                    files.extend(matches)
                else:
                    files.append(piece)
    else:
        files = [os.fspath(path) for path in mov_data_files]

    if not files:
        raise FileNotFoundError("No moving dataset file given.")
    return files


def robust_script(mov_data_files, robust_method):
    """
    Run `combat_robust` for a set of moving dataset files.

    mov_data_files: str, os.PathLike or list
        Paths to moving dataset files: a single path (os.PathLike, or a
        string that is an existing file), a string of whitespace-separated
        paths or glob patterns (for example 'dir/train_100_70_3_*.csv'),
        or a list of paths.
    robust_method: str
        Robust harmonization strategy identifier.
    """
    files = _expand_data_files(mov_data_files)
    subprocess.run(
        ["combat_robust", *files, "--method", robust_method], check=True
    )


def fit_script(
    mov_data_file,
    ref_data_file,
    metric,
    harmonization_method,
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
    harmonization_method: str
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
        mov_data_file, metric, harmonization_method, robust
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
        harmonization_method,
        "--robust",
        robust,
        "-f",
    ]
    if robust_threshold is not None:
        cmd_parts.extend(["--robust_threshold", str(robust_threshold)])
    if not eb:
        cmd_parts.append("--no_empirical_bayes")

    subprocess.run(cmd_parts, check=True)
    return output_model_path


def apply_script(
    mov_data_file,
    model_filename,
    metric,
    harmonization_method,
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
    harmonization_method: str
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
        mov_data_file, metric, harmonization_method, robust
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
    subprocess.run(cmd_parts, check=True)
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
    bundles: str or list
        Optional bundle filter passed to the visualization script, as a
        string of whitespace-separated bundle names or a list of names.
    title: str
        Optional title suffix for output files.
    """
    cmd_parts = [
        "combat_visualize_harmonization",
        ref_data_file,
        f,
        new_f,
        "--out_dir",
        directory,
        "-f",
    ]
    if bundles:
        bundle_names = bundles.split() if isinstance(bundles, str) else bundles
        cmd_parts.extend(["--bundles", *bundle_names])
    if title:
        cmd_parts.extend(["--outname", title, "--add_suffix", title])

    subprocess.run(cmd_parts, check=True)


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
