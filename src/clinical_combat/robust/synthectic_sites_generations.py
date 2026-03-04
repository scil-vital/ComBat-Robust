import json
import os
import subprocess

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from clinical_combat.harmonization.Combat import Combat
from clinical_combat.robust.robust_utils import remove_covariates_effects


def augment_df(df, new_copies=2):
    """
    Create augmented versions of a dataset by adjusting age, subject id, and
    mean values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing at least `sid`, `age`, and `mean` columns.
    new_copies : int, optional
        Number of augmented copies to generate (includes the original).

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with the original and augmented samples.
    """
    augmented_dfs = [df]

    for copy_index in range(1, new_copies):
        temp_df = df.copy()
        sid_modifications = {
            sid_val: np.random.choice([-1, 1])
            for sid_val in temp_df["sid"].unique()
        }

        temp_df["age"] = temp_df.apply(
            lambda row: row["age"] + sid_modifications[row["sid"]], axis=1
        )
        temp_df["sid"] = temp_df["sid"].astype(str) + f"_aug{copy_index}"
        temp_df["mean"] = temp_df["mean"] * (
            1 + np.random.choice([-0.02, -0.01, 0.01, 0.02], size=len(temp_df))
        )

        augmented_dfs.append(temp_df)

    return pd.concat(augmented_dfs, ignore_index=True)


def get_design_matrices(df, ignore_handedness=False, ignore_sex=False):
    """
    Build design matrices and targets for ComBat regression per bundle.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing `bundle`, `sid`, `age`, `mean`, `sex`,
        and `handedness`.
    ignore_handedness : bool, optional
        If True, drop handedness from the design matrix.
    ignore_sex : bool, optional
        If True, drop sex from the design matrix.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        Design matrices and target vectors for each bundle.
    """
    design = []
    targets = []
    for bundle in np.unique(df["bundle"]):
        data = df.query("bundle == @bundle")
        hstack_list = [np.ones(len(data["sid"]))]  # intercept
        if not ignore_sex:
            hstack_list.append(Combat.to_category(data["sex"]))
        if not ignore_handedness:
            hstack_list.append(Combat.to_category(data["handedness"]))
        ages = data["age"].to_numpy()
        hstack_list.append(ages)
        design.append(np.array(hstack_list))
        targets.append(data["mean"].to_numpy())
    return design, targets


def split_train_test(df, test_size=0.2, random_state=None):
    """
    Split the dataset while keeping each subject (`sid`) in a single split and
    preserving disease proportions.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to split.
    test_size : float, optional
        Proportion of the dataset to include in the test split.
    random_state : int or None, optional
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Training subset.
    pd.DataFrame
        Testing subset.
    """
    unique_sids = df.groupby("sid").first().reset_index()

    train_sids, test_sids = train_test_split(
        unique_sids,
        test_size=test_size,
        random_state=random_state,
        stratify=unique_sids["disease"],
    )

    train_df = df[df["sid"].isin(train_sids["sid"])]
    test_df = df[df["sid"].isin(test_sids["sid"])]

    return train_df, test_df


def sample_patients(df, num_patients, disease_ratio, index):
    """
    Sample a balanced subset of patients with a requested disease ratio.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing `sid`, `disease`, and `site` columns.
    num_patients : int
        Total number of patients to sample.
    disease_ratio : float
        Fraction of patients with a disease (non-HC).
    index : int
        Identifier used to suffix the generated site label.

    Returns
    -------
    pd.DataFrame
        Sampled patients with an updated `site` label.
    """
    num_diseased = int(num_patients * disease_ratio)
    num_healthy = num_patients - num_diseased

    healthy_patients = df[df["disease"] == "HC"]
    diseased_patients = df[df["disease"] != "HC"]

    if (
        len(healthy_patients["sid"].unique()) < num_healthy
        or len(diseased_patients["sid"].unique()) < num_diseased
    ):
        raise ValueError(
            "Not enough healthy or diseased patients for the requested sample."
        )

    sampled_healthy_sid = healthy_patients["sid"].drop_duplicates().sample(
        n=num_healthy
    )
    sampled_diseased_sid = diseased_patients["sid"].drop_duplicates().sample(
        n=num_diseased
    )

    sampled_healthy = healthy_patients[
        healthy_patients["sid"].isin(sampled_healthy_sid)
    ]
    sampled_diseased = diseased_patients[
        diseased_patients["sid"].isin(sampled_diseased_sid)
    ]

    sampled_df = pd.concat([sampled_healthy, sampled_diseased])
    sampled_df["site"] = (
        f"{num_patients}_patients_{int(disease_ratio * 100)}_percent_{index}"
    )

    return sampled_df


def generate_biaised_data(
    df1,
    df2,
    fixed_bias=False,
    centered_bias=False,
    additive_uniform_low=-3,
    additive_uniform_high=3,
    multiplicative_uniform_low=0.5,
    multiplicative_uniform_high=2,
    additive_std_low=0.01,
    additive_std_high=0.1,
    multiplicative_std_low=0.01,
    multiplicative_std_high=0.1,
):
    """
    Create additive and multiplicative biases per bundle and apply them to two
    datasets independently.

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataset used to sample bias parameters and receive bias.
    df2 : pd.DataFrame
        Second dataset that receives the same set of biases.
    fixed_bias : bool, optional
        If True, use fixed bias values instead of random draws.
    additive_uniform_low : float, optional
        Lower bound for the additive bias mean sampling.
    additive_uniform_high : float, optional
        Upper bound for the additive bias mean sampling.
    multiplicative_uniform_low : float, optional
        Lower bound for the multiplicative bias mean sampling.
    multiplicative_uniform_high : float, optional
        Upper bound for the multiplicative bias mean sampling.
    additive_std_low : float, optional
        Lower bound for additive bias standard deviation sampling.
    additive_std_high : float, optional
        Upper bound for additive bias standard deviation sampling.
    multiplicative_std_low : float, optional
        Lower bound for multiplicative bias standard deviation sampling.
    multiplicative_std_high : float, optional
        Upper bound for multiplicative bias standard deviation sampling.

    Returns
    -------
    tuple
        Biased versions of `df1` and `df2`, dictionaries of additive and
        multiplicative biases per bundle, and the bias parameters drawn.
    """
    additive_bias_per_bundle = {}
    multiplicative_bias_per_bundle = {}

    additive_mean = np.random.uniform(
        low=additive_uniform_low, high=additive_uniform_high
    )
    multiplicative_mean = np.random.uniform(
        low=multiplicative_uniform_low, high=multiplicative_uniform_high
    )

    additive_std = np.random.uniform(low=additive_std_low, high=additive_std_high)
    multiplicative_std = np.random.uniform(
        low=multiplicative_std_low, high=multiplicative_std_high
    )

    bundle_column = "metric_bundle" if "metric_bundle" in df1.columns else "bundle"

    for bundle in df1[bundle_column].unique():
        if fixed_bias:
            additive_bias_per_bundle[bundle] = 2
            multiplicative_bias_per_bundle[bundle] = 1.25
        else:
            additive_bias_per_bundle[bundle] = np.random.normal(
                loc=additive_mean, scale=additive_std
            )
            multiplicative_bias_per_bundle[bundle] = np.random.normal(
                loc=multiplicative_mean, scale=multiplicative_std
            )

    biased_df = apply_bias(
        df1, df2, additive_bias_per_bundle, multiplicative_bias_per_bundle, centered_bias=centered_bias
    )

    biased_df1 = biased_df[biased_df["sid"].isin(df1["sid"])]
    biased_df2 = biased_df[biased_df["sid"].isin(df2["sid"])]
    bias_parameters = {
        "additive_mean": additive_mean,
        "multiplicative_mean": multiplicative_mean,
        "additive_std": additive_std,
        "multiplicative_std": multiplicative_std,
    }

    return (
        biased_df1,
        biased_df2,
        additive_bias_per_bundle,
        multiplicative_bias_per_bundle,
        bias_parameters,
    )


def apply_bias(df1, df2, additive_bias_per_bundle, multiplicative_bias_per_bundle, centered_bias=False):
    """
    Apply provided additive and multiplicative biases to two datasets.

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataset.
    df2 : pd.DataFrame
        Second dataset.
    additive_bias_per_bundle : dict
        Additive bias per bundle key.
    multiplicative_bias_per_bundle : dict
        Multiplicative bias per bundle key.

    Returns
    -------
    pd.DataFrame
        Concatenated biased dataset.
    """
    biased_df_all = pd.concat([df1, df2], ignore_index=True)
    new_biased_df = pd.DataFrame()
    for metric in biased_df_all["metric"].unique():
        biased_df = biased_df_all[biased_df_all["metric"] == metric]
        biased_df = biased_df.sort_values(by=["site", "sid", "bundle"])
        ignore_handedness = True
        ignore_sex = False
        if biased_df["sex"].nunique() == 1:
            ignore_sex = True
        if biased_df["handedness"].nunique() == 1:
            ignore_handedness = True
        design, target = get_design_matrices(biased_df, ignore_handedness, ignore_sex)
        design_hc, y_hc = get_design_matrices(
            biased_df[biased_df["disease"] == "HC"], ignore_handedness, ignore_sex
        )
        alpha, beta = Combat.get_alpha_beta(design_hc, y_hc)

        for i, bundle in enumerate(np.unique(biased_df["bundle"])):
            additive_bias = additive_bias_per_bundle[f"{metric}_{bundle}"]
            multiplicative_bias = multiplicative_bias_per_bundle[f"{metric}_{bundle}"]
            covariate_effect = np.dot(design[i][1:, :].transpose(), beta[i])
            if centered_bias:
                biased_df.loc[biased_df["bundle"] == bundle, "mean"] = (
                     (y[i] - covariate_effect - alpha[i]) * multiplicative_bias + additive_bias * np.std(y[i]) + (covariate_effect + alpha[i])
                )
            else:
                biased_df.loc[biased_df["bundle"] == bundle, "mean"] = (
                    target[i] * multiplicative_bias + additive_bias * np.std(target[i])
                )

        new_biased_df = pd.concat([new_biased_df, biased_df], ignore_index=True)

    return new_biased_df


def process_test(
    sample_size,
    disease_ratio,
    i,
    train_df,
    test_df,
    directory,
    data_path,
    fixed_bias=False,
    centered_bias=False
):
    """
    Generate one synthetic site split, save outputs, and visualize when needed.

    Parameters
    ----------
    sample_size : int
        Number of patients to include in the synthetic training site.
    disease_ratio : float
        Fraction of diseased patients to sample.
    i : int
        Replicate index for naming outputs.
    train_df : pd.DataFrame
        Training portion of the original dataset.
    test_df : pd.DataFrame
        Testing portion of the original dataset.
    directory : str
        Root directory where outputs are written.
    data_path : str
        Path to the original CSV dataset.
    fixed_biais : bool, optional
        Whether to use fixed bias parameters.

    Returns
    -------
    None
    """
    size_dir = os.path.join(directory, f"{sample_size}_{int(disease_ratio * 100)}")
    temp_dir = os.path.join(size_dir, f"{i}")
    os.makedirs(temp_dir, exist_ok=True)
    (
        train_df_biaised,
        test_df_biaised,
        gammas,
        deltas,
        parameters,
    ) = generate_biaised_data(train_df, test_df, fixed_bias, centered_bias)
    sampled_df_biaied = sample_patients(
        train_df_biaised, sample_size, disease_ratio, i
    )

    train_sids = sampled_df_biaied["sid"].unique()
    ground_truth_train = train_df[train_df["sid"].isin(train_sids)]
    ground_truth_test = test_df

    if "metric_bundle" in sampled_df_biaied.columns:

        ground_truth_train_file = os.path.join(
            temp_dir, f"gt_train_{sample_size}_{int(disease_ratio * 100)}_{i}.csv"
        )
        ground_truth_train.to_csv(ground_truth_train_file, index=False)

        ground_truth_test_file = os.path.join(
            temp_dir, f"gt_test_{sample_size}_{int(disease_ratio * 100)}_{i}.csv"
        )
        ground_truth_test.to_csv(ground_truth_test_file, index=False)

        for metric in sampled_df_biaied["metric"].unique():
            metric_df = sampled_df_biaied[sampled_df_biaied["metric"] == metric]
            metric_test_df = test_df_biaised[test_df_biaised["metric"] == metric]
            gt_metric_df = ground_truth_train[
                ground_truth_train["metric"] == metric
            ]
            gt_metric_test_df = ground_truth_test[
                ground_truth_test["metric"] == metric
            ]

            gt_metric_df = remove_covariates_effects(gt_metric_df)
            gt_metric_test_df = remove_covariates_effects(gt_metric_test_df)

            metric_train_file = os.path.join(
                temp_dir,
                f"train_{sample_size}_{int(disease_ratio * 100)}_{i}_{metric}.csv",
            )
            metric_df.to_csv(metric_train_file, index=False)

            metric_test_file = os.path.join(
                temp_dir,
                f"test_{sample_size}_{int(disease_ratio * 100)}_{i}_{metric}.csv",
            )
            metric_test_df.to_csv(metric_test_file, index=False)

            gt_metric_train_file = os.path.join(
                temp_dir,
                f"gt_train_{sample_size}_{int(disease_ratio * 100)}_{i}_{metric}.csv",
            )
            gt_metric_df.to_csv(gt_metric_train_file, index=False)

            gt_metric_test_file = os.path.join(
                temp_dir,
                f"gt_test_{sample_size}_{int(disease_ratio * 100)}_{i}_{metric}.csv",
            )
            gt_metric_test_df.to_csv(gt_metric_test_file, index=False)
    else:
        temp_train_file = os.path.join(
            temp_dir, f"train_{sample_size}_{int(disease_ratio * 100)}_{i}.csv"
        )
        sampled_df_biaied.to_csv(temp_train_file, index=False)

        temp_test_file = os.path.join(
            temp_dir, f"test_{sample_size}_{int(disease_ratio * 100)}_{i}.csv"
        )
        test_df_biaised.to_csv(temp_test_file, index=False)

        ground_truth_train_file = os.path.join(
            temp_dir, f"gt_train_{sample_size}_{int(disease_ratio * 100)}_{i}.csv"
        )
        ground_truth_train.to_csv(ground_truth_train_file, index=False)

        ground_truth_test_file = os.path.join(
            temp_dir, f"gt_test_{sample_size}_{int(disease_ratio * 100)}_{i}.csv"
        )
        ground_truth_test.to_csv(ground_truth_test_file, index=False)


def generate_sites(
    sample_sizes,
    disease_ratios,
    num_tests,
    directory,
    data_path,
    camcan_hc_only=True,
    disease=None,
    fixed_bias=False,
    centered_bias=False,
    n_jobs=-1,
):
    """
    Generate synthetic sites for multiple configurations and persist outputs.

    Parameters
    ----------
    sample_sizes : list[int]
        Patient counts to sample for each synthetic site.
    disease_ratios : list[float]
        Fractions of diseased patients to include.
    num_tests : int
        Number of replicates per configuration.
    directory : str
        Root directory where synthetic datasets are saved.
    data_path : str
        Path to the CSV dataset used as a source.

    camcan_hc_only : bool, optional
        If True, keep only CamCAN healthy controls.
    disease : str or None, optional
        Disease filter; keeps HC plus the specified disease.
    fixed_biais : bool, optional
        Whether to use fixed bias parameters.
    n_jobs : int, optional
        Number of parallel jobs.

    Returns
    -------
    None
    """
    df = pd.read_csv(data_path)
    df = df[~df["bundle"].isin(["left_ventricle", "right_ventricle"])]
    if camcan_hc_only:
        df = df[~((df["disease"] == "HC") & (df["source_site"] != "CamCAN"))]
    if disease == "ASTMIX":
        df = df[df["disease"].isin(["AD", "SCHZ", "TBI", "HC"])]
    elif disease is not None and disease != "ALL":
        df = df[(df["disease"] == disease) | (df["disease"] == "HC")]

    train_df, test_df = split_train_test(df, test_size=0.05, random_state=43)

    Parallel(n_jobs=n_jobs)(
        delayed(process_test)(
            sample_size,
            disease_ratio,
            i,
            train_df,
            test_df,
            directory,
            data_path,
            fixed_bias,
            centered_bias
        )
        for sample_size in sample_sizes
        for disease_ratio in disease_ratios
        for i in range(num_tests)
    )


def generate_sites_no_file(
    sample_sizes, disease_ratios, num_tests, df, disease=None, n_jobs=-1
):
    """
    Generate synthetic site samples directly from a provided DataFrame.

    Parameters
    ----------
    sample_sizes : list[int]
        Patient counts to sample for each synthetic site.
    disease_ratios : list[float]
        Fractions of diseased patients to include.
    num_tests : int
        Number of replicates per configuration.
    df : pd.DataFrame
        Dataset used to sample synthetic sites.
    disease : str or None, optional
        Disease filter; keeps HC plus the specified disease.
    n_jobs : int, optional
        Number of parallel jobs.

    Returns
    -------
    list[pd.DataFrame]
        List of sampled synthetic site datasets.
    """
    df = df[~df["bundle"].isin(["left_ventricle", "right_ventricle"])]
    df = df[~((df["disease"] == "HC") & (df["source_site"] != "CamCAN"))]
    if disease == "ASTMIX":
        df = df[df["disease"].isin(["AD", "SCHZ", "TBI", "HC"])]
    elif disease is not None:
        df = df[(df["disease"] == disease) | (df["disease"] == "HC")]
    dfs = Parallel(n_jobs=n_jobs)(
        delayed(sample_patients)(df, sample_size, disease_ratio, i)
        for sample_size in sample_sizes
        for disease_ratio in disease_ratios
        for i in range(num_tests)
    )
    return dfs


def mlp_syn_sites_biais(df, sample_size, disease_ratio, i):
    """
    Create a biased synthetic site sample for MLP experiments.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset used to sample patients.
    sample_size : int
        Total number of patients to sample.
    disease_ratio : float
        Fraction of diseased patients to include.
    i : int
        Replicate index for naming outputs.

    Returns
    -------
    pd.DataFrame
        Sampled and biased synthetic site dataset.
    """
    sampled_df = sample_patients(df, sample_size, disease_ratio, i)
    sampled_df_biaied, *_ = generate_biaised_data(
        sampled_df, pd.DataFrame()
    )
    return sampled_df_biaied
