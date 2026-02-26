# -*- coding: utf-8 -*-
"""
Outlier scoring module.

Goal
Add a score/flag column to a DataFrame instead of dropping rows.
A separate step can apply a threshold on that column.

Rules
Most methods return a continuous score (higher means more outlier-like).
VS and MMS return a binary flag (0/1).
IQR stores the IQR value (Q3 - Q1) as a constant per bundle group.
MLP, G_ZS, and G_MAD are global scores per patient (sid), not per bundle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from clinical_combat.robust.robust_utils import remove_covariates_effects_metrics
from clinical_combat.robust.robust_MLP import predict_malades_MLP

METRICS_HIGH = {"md", "mdt", "rd", "rdt", "fw", "ad", "adt"}  # pathology ↑
METRICS_LOW = {"fa", "fat", "afd"}  # pathology ↓

GLOBAL_METHODS = {
    "G_ZS",
    "G_MAD",
}
DEFAULT_THRESHOLDS = {
    "ZS": 3.0,
    "IQR": 1.5,
    "MAD": 3.0,
    "SN": 3.0,
    "QN": 3.0,
    "VS": 1.0,
    "MMS": 0.001,
    "G_ZS":1.0,
    "G_MAD" :1.0,
    "MLP": 0.6,  # applies to any method starting with "MLP"
}


def score_from_sid_table(df, sid_table):
    """
    Map patient-level scores to all rows via sid.

    df: DataFrame
        Input DataFrame with a 'sid' column.
    sid_table: DataFrame
        Must contain columns: 'sid' and 'prob_outlier'.

    Returns
    -------
    scores: Series
        Score for each row of df (mapped by sid). Missing sids get 0.
    """
    if sid_table is None or sid_table.empty:
        return pd.Series(0.0, index=df.index, dtype=float)

    sid2score = dict(
        zip(
            sid_table["sid"].astype(str),
            sid_table["prob_outlier"].astype(float),
        )
    )
    scores = df["sid"].astype(str).map(sid2score).fillna(0.0).astype(float)
    return pd.Series(scores.to_numpy(), index=df.index, dtype=float)


def flag_from_indices(df, outlier_idx):
    """
    Build a binary flag Series from a list of indices.

    df: DataFrame
        Input DataFrame.
    outlier_idx: list
        List of row indices to flag.

    Returns
    -------
    flags: Series
        0/1 flag aligned with df.index.
    """
    flags = pd.Series(0, index=df.index, dtype=int)
    if outlier_idx:
        flags.loc[outlier_idx] = 1
    return flags


def vs_outlier_indices(data, column="mean_no_cov"):
    """
    Balance around the median by removing extreme values on the suspicious side.

    The method iteratively removes the most extreme value on the side expected
    to contain pathology-related deviations until left/right average deviations
    around the median are balanced.

    data: DataFrame
        Group DataFrame containing one metric (data['metric'].iloc[0]).
    column: str
        Column used for scoring.

    Returns
    -------
    outliers_idx: list
        Indices (from the input DataFrame) to flag as outliers.
    """
    metric = str(data["metric"].iloc[0])

    if metric in METRICS_HIGH:
        side = "right"
    elif metric in METRICS_LOW:
        side = "left"
    else:
        return []

    outliers_idx = []
    work = data[[column]].copy()
    median = float(work[column].median())

    while len(work) > 2:
        left = work.loc[work[column] < median, column]
        right = work.loc[work[column] > median, column]

        left_mean = float((left - median).abs().mean()) if len(left) else 0.0
        right_mean = float((right - median).abs().mean()) if len(right) else 0.0

        if abs(left_mean - right_mean) <= 1e-6:
            break

        if side == "right":
            if right_mean <= left_mean:
                break
            target_idx = work[column].idxmax()
        else:
            if left_mean <= right_mean:
                break
            target_idx = work[column].idxmin()

        outliers_idx.append(target_idx)
        work = work.drop(target_idx)

    return outliers_idx


def mms_outlier_indices(data, threshold=0.001):
    """
    Flag outliers until mean and median converge.

    At each step, the most extreme value on the suspicious side is removed
    until the relative mean/median difference is below `threshold`.

    data: DataFrame
        Group DataFrame containing one metric (data['metric'].iloc[0]).
    threshold: float
        Relative convergence criterion: |median - mean| / |median|.

    Returns
    -------
    outliers_idx: list
        Indices (from the input DataFrame) to flag as outliers.
    """
    column = "mean_no_cov"
    metric = str(data["metric"].iloc[0])

    if metric in METRICS_HIGH:
        pick_idx = lambda s: s.idxmax()
        direction_ok = lambda mean, median: mean > median
    elif metric in METRICS_LOW:
        pick_idx = lambda s: s.idxmin()
        direction_ok = lambda mean, median: mean < median
    else:
        return []

    outliers_idx = []
    work = data[[column]].copy()

    while len(work) > 2:
        median = float(work[column].median())
        mean = float(work[column].mean())

        if median == 0.0:
            break
        if not direction_ok(mean, median):
            break
        if abs(median - mean) / abs(median) < float(threshold):
            break

        target_idx = pick_idx(work[column])
        outliers_idx.append(target_idx)
        work = work.drop(target_idx)

    return outliers_idx


def z_score_detection(df, mean_col="mean_no_cov"):
    """
    Compute a global z-score outlier score per patient (sid).

    For each metric_bundle, compute mean/std, then compute |z| for each row.
    The final sid score is mean(|z|) across all rows of that sid.

    df: DataFrame
        Input DataFrame containing: sid, metric_bundle, mean_col.
    mean_col: str
        Column holding covariate-corrected values.

    Returns
    -------
    sid_scores: DataFrame
        Columns: sid, prob_outlier.
    """
    stats = (
        df.groupby("metric_bundle")[mean_col]
        .agg(["mean", "std"])
        .rename(columns={"mean": "global_mean", "std": "global_std"})
    )
    stats["global_std"] = stats["global_std"].replace(0, 1e-6)

    df_z = df.merge(stats, on="metric_bundle", how="left")
    df_z["abs_zscore"] = ((df_z[mean_col] - df_z["global_mean"]) / df_z["global_std"]).abs()

    return (
        df_z.groupby("sid", as_index=False)
        .agg(prob_outlier=("abs_zscore", "mean"))
        .astype({"prob_outlier": float})
    )


def mad_detection(df, mean_col="mean_no_cov"):
    """
    Compute a global MAD-score outlier score per patient (sid).

    For each metric_bundle, compute median and MAD (scaled by 1.4826),
    then compute |x - median| / MAD for each row.
    The final sid score is mean(score) across all rows of that sid.

    df: DataFrame
        Input DataFrame containing: sid, metric_bundle, mean_col.
    mean_col: str
        Column holding covariate-corrected values.

    Returns
    -------
    sid_scores: DataFrame
        Columns: sid, prob_outlier.
    """
    global_median = df.groupby("metric_bundle")[mean_col].median().rename("global_median")
    mad_raw = df.groupby("metric_bundle")[mean_col].apply(
        lambda x: (x - x.median()).abs().median()
    )
    stats = pd.concat([global_median, mad_raw.rename("mad_raw")], axis=1)

    stats["mad"] = (stats["mad_raw"] * 1.4826).replace(0, 1e-6)

    df_m = df.merge(stats[["global_median", "mad"]], on="metric_bundle", how="left")
    df_m["abs_madscore"] = (df_m[mean_col] - df_m["global_median"]).abs() / df_m["mad"]

    return (
        df_m.groupby("sid", as_index=False)
        .agg(prob_outlier=("abs_madscore", "mean"))
        .astype({"prob_outlier": float})
    )


def score_g_zs(df, mean_col="mean_no_cov"):
    """
    Global z-score score per sid, mapped back to all rows.

    df: DataFrame
        Input DataFrame.
    mean_col: str
        Column holding covariate-corrected values.

    Returns
    -------
    scores: Series
        Score for each row of df.
    """
    sid_scores = z_score_detection(df, mean_col=mean_col)
    return score_from_sid_table(df, sid_scores)


def score_g_mad(df, mean_col="mean_no_cov"):
    """
    Global MAD score per sid, mapped back to all rows.

    df: DataFrame
        Input DataFrame.
    mean_col: str
        Column holding covariate-corrected values.

    Returns
    -------
    scores: Series
        Score for each row of df.
    """
    sid_scores = mad_detection(df, mean_col=mean_col)
    return score_from_sid_table(df, sid_scores)


def remove_outliers_iqr(data, k):
    """
    Remove rows considered outliers based on the IQR method
    applied to the column 'mean_no_cov'.

    An observation is removed if its value lies outside
    [Q1 - k·IQR, Q3 + k·IQR].

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.

    k : float
        IQR multiplier controlling the aggressiveness of
        outlier removal (1.5 = standard, 3 = extreme).

    Returns
    -------
    cleaned_df : DataFrame
        DataFrame with outlier rows removed.
    """

    col = "mean_no_cov"
    group_cols = ["metric", "bundle"]

    # Q1/Q3/IQR per group, aligned to rows via transform
    q1 = data.groupby(group_cols)[col].transform(lambda s: s.quantile(0.25))
    q3 = data.groupby(group_cols)[col].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    mask = (data[col] >= lower) & (data[col] <= upper)

    return data[mask].reset_index(drop=True)


def score_zscore_bundle(data):
    """
    Compute |z| within the group. If std=0, return 0.

    data: DataFrame
        Group DataFrame.

    Returns
    -------
    scores: Series
        Absolute z-score per row in the group.
    """
    if data.empty:
        return pd.Series(dtype=float, index=data.index)

    x = data["mean_no_cov"].astype(float)
    mu = float(x.mean())
    sigma = float(x.std(ddof=0))

    if sigma == 0.0 or np.isnan(sigma):
        return pd.Series(0.0, index=data.index, dtype=float)

    return ((x - mu) / sigma).abs().astype(float)


def score_mad_bundle(data):
    """
    Compute |modified_z| based on MAD within the group.

    data: DataFrame
        Group DataFrame.

    Returns
    -------
    scores: Series
        Absolute modified z-score per row in the group.
    """
    if data.empty:
        return pd.Series(dtype=float, index=data.index)

    x = data["mean_no_cov"].astype(float)
    med = float(x.median())
    mad = float(np.median(np.abs(x.to_numpy() - med)))

    if mad == 0.0 or np.isnan(mad):
        return pd.Series(0.0, index=data.index, dtype=float)

    modified_z = 0.6745 * (x - med) / mad
    return modified_z.abs().astype(float)


def score_sn(data):
    """
    Compute |x - median| / Sn within the group.

    data: DataFrame
        Group DataFrame.

    Returns
    -------
    scores: Series
        Sn-based score per row in the group.
    """
    if data.empty:
        return pd.Series(dtype=float, index=data.index)

    x = data["mean_no_cov"].astype(float).to_numpy()
    med = float(np.median(x))
    diffs = np.abs(x[:, None] - x[None, :])
    sn = 1.1926 * float(np.median(np.median(diffs, axis=1)))

    if sn == 0.0 or np.isnan(sn):
        return pd.Series(0.0, index=data.index, dtype=float)

    return pd.Series((np.abs(x - med) / sn).astype(float), index=data.index)


def score_qn(data):
    """
    Compute |x - median| / Qn within the group.

    data: DataFrame
        Group DataFrame.

    Returns
    -------
    scores: Series
        Qn-based score per row in the group.
    """
    if data.empty:
        return pd.Series(dtype=float, index=data.index)

    x = data["mean_no_cov"].astype(float).to_numpy()
    if len(x) < 2:
        return pd.Series(0.0, index=data.index, dtype=float)

    med = float(np.median(x))
    diffs = np.abs(x[:, None] - x[None, :])
    pairwise = diffs[np.triu_indices(len(x), k=1)]
    qn = 2.2219 * float(np.percentile(pairwise, 25))

    if qn == 0.0 or np.isnan(qn):
        return pd.Series(0.0, index=data.index, dtype=float)

    return pd.Series((np.abs(x - med) / qn).astype(float), index=data.index)


def score_mlp_from_model(df, run_name):
    """
    MLP global score per sid, mapped back to all rows.

    df: DataFrame
        Input DataFrame.
    run_name: str
        Name of the MLP run (e.g., 'MLP1_ALL').

    Returns
    -------
    scores: Series
        Score for each row of df.
    """
    pred = predict_malades_MLP(df, run_name=run_name)  # columns: sid, prob_outlier
    return score_from_sid_table(df, pred)


def score_vs_flag(data):
    """
    VS method binary flag within the group.

    data: DataFrame
        Group DataFrame.

    Returns
    -------
    flags: Series
        0/1 flag per row in the group.
    """
    idx = vs_outlier_indices(data, column="mean_no_cov")
    return flag_from_indices(data, idx)


def score_mms_flag(data):
    """
    MMS method binary flag within the group.

    data: DataFrame
        Group DataFrame.
    threshold: float
        Convergence threshold.

    Returns
    -------
    flags: Series
        0/1 flag per row in the group.
    """
    idx = mms_outlier_indices(data, threshold=0.001)
    return flag_from_indices(data, idx)


SCORE_METHODS = {
    "IQR": lambda d: score_iqr_value_only(d),
    "ZS": lambda d: score_zscore_bundle(d),
    "MAD": lambda d: score_mad_bundle(d),
    "SN": lambda d: score_sn(d),
    "QN": lambda d: score_qn(d),
    "VS": lambda d: score_vs_flag(d),
    "MMS": lambda d: score_mms_flag(d),
    "G_ZS": lambda d: score_g_zs(d, mean_col="mean_no_cov"),
    "G_MAD": lambda d: score_g_mad(d, mean_col="mean_no_cov"),
}


def get_scorer(method):
    """
    Resolve a scoring function.

    method: str
        Method name. Must be in SCORE_METHODS or start with 'MLP'.

    Returns
    -------
    scorer: callable
        Function (df, threshold) -> Series.
    """
    if method in SCORE_METHODS:
        return SCORE_METHODS[method]
    if method.upper().startswith("MLP"):
        return lambda d, t: score_mlp_from_model(d, run_name=method)

    supported = sorted(SCORE_METHODS.keys())
    raise ValueError(
        f"Unsupported method '{method}'. Supported: {supported} or any name starting with 'MLP'."
    )


def add_outlier_column(df, method, by_bundle=True):
    """
    Add an outlier score/flag column to df.

    df: DataFrame
        Must include: sid, bundle, metric.
        remove_covariates_effects_metrics(.) must create 'mean_no_cov'.
        If 'metric_bundle' is missing, it will be created as 'metric' + '_' + 'bundle'.
    method: str
        One of SCORE_METHODS keys, or any name starting with 'MLP'.
    threshold: float
        Optional parameter used by some methods (mainly MMS).
    score_col: str
        Name of the created score column. Default is `method`.
    by_bundle: bool
        If True, compute per metric_bundle for bundle-level methods.
        Forced to False for global methods (G_ZS, G_MAD, MLP*).

    Returns
    -------
    out: DataFrame
        Copy of df with the score column added.
        Drops 'mean_no_cov' and 'metric_bundle' before returning.
    """
    if method == "IQR":
        print("IQR outliers are removed directly during fit(); no additional filtering step will be applied.")
        return df
    scorer = get_scorer(method)

    cleaned = remove_covariates_effects_metrics(df)
    if "metric_bundle" not in cleaned.columns:
        cleaned = cleaned.copy()
        cleaned["metric_bundle"] = (
            cleaned["metric"].astype(str) + "_" + cleaned["bundle"].astype(str)
        )

    out = cleaned.copy()
    col_name = method

    is_global = method in GLOBAL_METHODS or method.upper().startswith("MLP")
    if is_global:
        out[col_name] = scorer(out).astype(float)
        return out.drop(columns=["mean_no_cov", "metric_bundle"], errors="ignore")

    if by_bundle and "bundle" in out.columns:
        out[col_name] = 0.0
        for _, sub in out.groupby("metric_bundle", sort=False):
            out.loc[sub.index, col_name] = scorer(sub).astype(float)
        return out.drop(columns=["mean_no_cov", "metric_bundle"], errors="ignore")

    out[col_name] = scorer(out).astype(float)
    return out.drop(columns=["mean_no_cov", "metric_bundle"], errors="ignore")


def filter_outliers_by_threshold(df, method, threshold):
    """
    Return indices where df[score_col] >= threshold.

    df: DataFrame
        Input DataFrame containing the score_col.
    score_col: str
        Name of the score column.
    threshold: float
        Threshold applied to the score column.

    Returns
    -------
    outliers_idx: list
        List of DataFrame indices where score >= threshold.
    """
    if threshold is None:
        threshold = DEFAULT_THRESHOLDS["MLP"] if method.startswith("MLP") else DEFAULT_THRESHOLDS[method]

    if method == "NO":
        return df
    if method == "HC":
        return df[df["disease"] == "HC"]
    if method not in df.columns:
        raise KeyError(f"Missing column: {method}")  
    if method == "IQR":
        return remove_outliers_iqr(df, k=threshold)
    
    return df[pd.to_numeric(df[method], errors="coerce").fillna(0.0) < threshold]