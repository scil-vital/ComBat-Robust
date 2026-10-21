#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests of the robust (outlier scoring) modules, on the example site files."""
import os
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from clinical_combat import COMBAT_ROOT
from clinical_combat.cli import combat_robust
from clinical_combat.robust import robust as R
from clinical_combat.robust import robust_harmonization as RH
from clinical_combat.robust import robust_utils as U
from clinical_combat.robust.robust_MLP import (
    DEFAULT_MODEL_CONFIG,
    build_mlp_from_config,
)
from clinical_combat.robust.synthetic_sites_generation import (
    apply_bias,
    generate_biased_data,
    generate_sites_no_file,
    split_train_test,
)

DATA_DIR = os.path.join(COMBAT_ROOT, "src/clinical_combat/data")
GLOBAL_PREFIXES = ("G_", "PCA_", "MLP")
SCORED_METHODS = [*R.SCORE_METHODS, "MLP_TEST"]
# Metrics used where the full set of ten is not needed, to keep the tests fast.
# One "pathology goes up" metric (md) and one "pathology goes down" one (fa).
SMALL_METRICS = ("md", "fa")


def _example_file(metric):
    return os.path.join(DATA_DIR, f"70_pct_example_{metric}.csv")


def _read_example(metric):
    """
    Read an example file without any score column.

    The README example is run on these files in place, which adds a score
    column (MAD, MLP_TEST, ...). The tests start from the files as shipped.
    """
    df = pd.read_csv(_example_file(metric))
    scores = [
        c for c in df.columns
        if c.upper() in R.SCORE_METHODS or c.upper().startswith("MLP")
    ]
    return df.drop(columns=scores)


@pytest.fixture(scope="module")
def example_df():
    """The 10 example metric files of one site (100 subjects, 43 bundles)."""
    return pd.concat(
        [_read_example(m) for m in U.get_metrics()], ignore_index=True
    )


@pytest.fixture(scope="module")
def scored_cache(example_df):
    """
    Score columns computed once per method, shared by the tests.

    Maps each method to (input frame, scored frame). The MLP needs the ten
    metrics, the other methods are run on two metrics only.
    """
    small = example_df[example_df["metric"].isin(SMALL_METRICS)]
    small = small.reset_index(drop=True)
    cache = {}
    previous = os.getcwd()
    os.chdir(COMBAT_ROOT)  # the MLP weights are loaded from Pytorch_models/
    try:
        for method in SCORED_METHODS:
            data = example_df if method.startswith("MLP") else small
            cache[method] = (data, R.add_outlier_column(data, method))
    finally:
        os.chdir(previous)
    return cache


@pytest.fixture(scope="module")
def md_features(example_df):
    """Covariate-corrected 'md' data with the metric_bundle feature name."""
    md = U.remove_covariates_effects_metrics(
        example_df[example_df["metric"] == "md"]
    )
    md["metric_bundle"] = md["metric"] + "_" + md["bundle"]
    return md


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", SCORED_METHODS)
def test_scorer_adds_one_finite_column(method, scored_cache):
    data, out = scored_cache[method]

    assert list(out.columns) == [*data.columns, method]
    assert len(out) == len(data)
    assert "mean_no_cov" not in out.columns
    assert "metric_bundle" not in out.columns

    score = out[method]
    assert score.dtype == float
    assert np.isfinite(score).all()
    assert (score >= 0).all()

    if method.startswith(GLOBAL_PREFIXES):
        # One score per patient, repeated on all of its rows.
        assert (out.groupby("sid")[method].nunique() == 1).all()
    if method in ("VS", "MMS"):
        assert set(score.unique()) <= {0.0, 1.0}


def test_scorer_rows_are_sorted_per_metric(scored_cache):
    data, out = scored_cache["MAD"]
    assert list(out.index) == list(range(len(out)))
    assert list(out["metric"].unique()) == list(data["metric"].unique())
    for _, rows in out.groupby("metric", sort=False):
        keys = list(zip(rows["site"], rows["sid"], rows["bundle"]))
        assert keys == sorted(keys)


def test_mlp_scorer_is_case_insensitive(example_df, scored_cache, monkeypatch):
    monkeypatch.chdir(COMBAT_ROOT)
    lower = R.add_outlier_column(example_df, "mlp_test")
    assert_frame_equal(
        lower.rename(columns={"mlp_test": "MLP_TEST"}),
        scored_cache["MLP_TEST"][1],
    )


def test_mlp_scorer_rejects_unknown_metric(example_df):
    bad = example_df.assign(metric="not_a_metric")
    with pytest.raises(ValueError, match="Invalid metrics"):
        R.score_mlp_from_model(bad, "MLP_TEST")


@pytest.mark.parametrize("method", ["IQR", "HC", "NO"])
def test_add_outlier_column_leaves_df_unchanged(method, example_df):
    assert R.add_outlier_column(example_df, method) is example_df


@pytest.mark.parametrize("sex", [1, 2])
def test_scorers_run_when_sex_is_constant(sex, example_df):
    # The README tells users to fill an unknown sex with 1: the covariates
    # are then fitted without sex instead of on a singular design matrix.
    data = example_df[example_df["metric"].isin(SMALL_METRICS)]
    data = data.assign(sex=sex).reset_index(drop=True)

    cleaned = U.remove_covariates_effects_metrics(data)
    assert np.isfinite(cleaned["mean_no_cov"]).all()

    for method in ("MAD", "G_MAD"):
        out = R.add_outlier_column(data, method)
        assert len(out) == len(data)
        assert np.isfinite(out[method]).all()
        assert (out[method] >= 0).all()


def test_get_scorer_unknown_method():
    with pytest.raises(ValueError, match="Unsupported method"):
        R.get_scorer("FOO")
    assert callable(R.get_scorer("MAD"))
    assert callable(R.get_scorer("mlp1_all"))


def test_zscore_and_mad_hand_checked():
    # Values 1, 2, 3, 4, 10: mean 4, population std sqrt(10),
    # median 3 and MAD 1.
    data = pd.DataFrame({"mean_no_cov": [1.0, 2.0, 3.0, 4.0, 10.0]})

    zs = R.score_zscore_bundle(data)
    expected_zs = np.abs(np.array([-3.0, -2.0, -1.0, 0.0, 6.0])) / np.sqrt(10)
    np.testing.assert_allclose(zs.to_numpy(), expected_zs, rtol=1e-12)

    mad = R.score_mad_bundle(data)
    expected_mad = 0.6745 * np.array([2.0, 1.0, 0.0, 1.0, 7.0])
    np.testing.assert_allclose(mad.to_numpy(), expected_mad, rtol=1e-12)

    constant = pd.DataFrame({"mean_no_cov": [2.0, 2.0, 2.0]})
    assert (R.score_zscore_bundle(constant) == 0.0).all()
    assert (R.score_mad_bundle(constant) == 0.0).all()


def test_zscore_and_mad_keep_the_group_index():
    data = pd.DataFrame(
        {"mean_no_cov": [1.0, 2.0, 3.0, 4.0, 10.0]}, index=[10, 11, 12, 13, 14]
    )
    assert list(R.score_zscore_bundle(data).index) == [10, 11, 12, 13, 14]
    assert list(R.score_mad_bundle(data).index) == [10, 11, 12, 13, 14]


@pytest.mark.parametrize(
    "metric, values, expected",
    [
        # Pathology goes up for md: the large value is removed.
        ("md", [1.0, 1.0, 1.0, 1.0, 10.0], [4]),
        # Pathology goes down for fa: the small value is removed.
        ("fa", [1.0, 1.0, 1.0, 1.0, -8.0], [4]),
        # Wrong side: mean below the median for md, mean above for fa.
        ("md", [1.0, 1.0, 1.0, 1.0, -8.0], []),
        ("fa", [1.0, 1.0, 1.0, 1.0, 10.0], []),
        # Metric without a known direction.
        ("unknown", [1.0, 1.0, 1.0, 1.0, 10.0], []),
    ],
)
def test_mms_outlier_indices_hand_checked(metric, values, expected):
    data = pd.DataFrame({"metric": metric, "mean_no_cov": values})
    assert R.mms_outlier_indices(data) == expected


def test_vs_outlier_indices_hand_checked():
    # Median 1, one extreme value on the right: removed for a "high" metric.
    values = [0.9, 1.0, 1.1, 1.0, 5.0]
    high = pd.DataFrame({"metric": "md", "mean_no_cov": values})
    assert R.vs_outlier_indices(high) == [4]
    # Nothing to remove on the right side for a "low" metric.
    low = pd.DataFrame({"metric": "fa", "mean_no_cov": values})
    assert R.vs_outlier_indices(low) == []


def test_score_from_sid_table_maps_scores_and_defaults_to_zero():
    df = pd.DataFrame({"sid": ["a", "a", "b", "c"]}, index=[5, 6, 7, 8])
    table = pd.DataFrame({"sid": ["a", "b"], "prob_outlier": [0.25, 0.75]})
    scores = R.score_from_sid_table(df, table)
    assert list(scores.index) == [5, 6, 7, 8]
    assert list(scores) == [0.25, 0.25, 0.75, 0.0]


# ---------------------------------------------------------------------------
# PCA scorers
# ---------------------------------------------------------------------------
def test_pca_scorers_match_the_registered_methods(md_features):
    # The registered scorers are the plain functions with their defaults.
    assert R.SCORE_METHODS["PCA_ZS"] is R.score_pca_zs
    assert R.SCORE_METHODS["PCA_MAD"] is R.score_pca_mad
    scores = R.score_pca_zs(md_features)
    assert len(scores) == len(md_features)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("n_subjects", [3, 9])
def test_pca_scorers_zero_scores_with_few_subjects(md_features, n_subjects):
    sids = md_features["sid"].unique()[:n_subjects]
    few = md_features[md_features["sid"].isin(sids)]

    with pytest.warns(UserWarning, match="only %d subject" % n_subjects):
        zs = R.pca_global_detection(few, scaling="zs")
    with pytest.warns(UserWarning, match="only %d subject" % n_subjects):
        mad = R.pca_global_detection(few, scaling="mad")

    for sid_scores in (zs, mad):
        assert sorted(sid_scores["sid"]) == sorted(sids)
        assert (sid_scores["prob_outlier"] == 0.0).all()

    for scorer in (R.score_pca_zs, R.score_pca_mad):
        with pytest.warns(UserWarning):
            row_scores = scorer(few)
        assert len(row_scores) == len(few)
        assert (row_scores == 0.0).all()


def test_pca_scorers_work_with_ten_subjects(md_features):
    sids = md_features["sid"].unique()[:10]
    ten = md_features[md_features["sid"].isin(sids)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for scaling in ("zs", "mad"):
            sid_scores = R.pca_global_detection(ten, scaling=scaling)
            assert len(sid_scores) == 10
            assert np.isfinite(sid_scores["prob_outlier"]).all()


@pytest.mark.parametrize(
    "detection",
    [
        lambda df: R.pca_global_detection(df, scaling="zs"),
        lambda df: R.pca_global_detection(df, scaling="mad", agg="rms"),
    ],
)
def test_pca_scorers_zero_scores_without_usable_feature(md_features, detection):
    constant = md_features.assign(mean_no_cov=1.0)
    with pytest.warns(UserWarning, match="no usable"):
        sid_scores = detection(constant)
    assert len(sid_scores) == constant["sid"].nunique()
    assert (sid_scores["prob_outlier"] == 0.0).all()


def test_pca_scorers_handle_missing_and_constant_features(md_features):
    messy = md_features.copy()
    rng = np.random.RandomState(0)
    messy.loc[rng.rand(len(messy)) < 0.03, "mean_no_cov"] = np.nan
    messy.loc[messy["metric_bundle"] == "md_mni_AC", "mean_no_cov"] = np.nan
    messy.loc[messy["metric_bundle"] == "md_mni_AF_L", "mean_no_cov"] = 0.5
    for scaling in ("zs", "mad"):
        sid_scores = R.pca_global_detection(messy, scaling=scaling)
        assert len(sid_scores) == messy["sid"].nunique()
        assert np.isfinite(sid_scores["prob_outlier"]).all()


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
@pytest.fixture
def scored_small():
    return pd.DataFrame(
        {
            "sid": list("abcdef"),
            "disease": ["HC", "HC", "AD", "AD", "HC", "TBI"],
            "MAD": [0.5, 2.9, 3.0, 3.1, np.nan, 10.0],
            "MLP_TEST": [0.1, 0.59, 0.6, 0.61, 0.0, 1.0],
            "mlp_test": [0.1, 0.59, 0.6, 0.61, 0.0, 1.0],
        }
    )


def test_filter_keeps_scores_below_threshold(scored_small):
    kept = R.filter_outliers_by_threshold(scored_small, "MAD", 3.0)
    # Strictly below the threshold, a missing score counts as 0.
    assert list(kept["sid"]) == ["a", "b", "e"]
    kept = R.filter_outliers_by_threshold(scored_small, "MAD", 1.0)
    assert list(kept["sid"]) == ["a", "e"]


def test_filter_uses_the_default_threshold(scored_small):
    kept = R.filter_outliers_by_threshold(scored_small, "MAD", None)
    assert list(kept["sid"]) == ["a", "b", "e"]
    assert R.DEFAULT_THRESHOLDS["MAD"] == 3.0


@pytest.mark.parametrize("method", ["MLP_TEST", "mlp_test"])
def test_filter_mlp_names_use_the_mlp_default(scored_small, method):
    kept = R.filter_outliers_by_threshold(scored_small, method, None)
    assert list(kept["sid"]) == ["a", "b", "e"]
    kept = R.filter_outliers_by_threshold(scored_small, method, 0.05)
    assert list(kept["sid"]) == ["e"]


def test_filter_hc_keeps_only_healthy_controls(scored_small):
    kept = R.filter_outliers_by_threshold(scored_small, "HC", None)
    assert list(kept["sid"]) == ["a", "b", "e"]
    assert (kept["disease"] == "HC").all()


def test_filter_no_returns_all_rows(scored_small):
    kept = R.filter_outliers_by_threshold(scored_small, "NO", None)
    assert_frame_equal(kept, scored_small)


def test_filter_unknown_method_without_threshold(scored_small):
    with pytest.raises(ValueError, match="No default threshold.*FOO"):
        R.filter_outliers_by_threshold(scored_small, "FOO", None)
    # With a threshold, a missing score column is still reported as such.
    with pytest.raises(KeyError, match="Missing column"):
        R.filter_outliers_by_threshold(scored_small, "FOO", 1.0)


def test_filter_iqr(example_df):
    md = example_df[example_df["metric"] == "md"]
    kept = R.filter_outliers_by_threshold(md, "IQR", None)
    assert "mean_no_cov" in kept.columns
    assert 0 < len(kept) <= len(md)
    assert list(kept.index) == list(range(len(kept)))

    # A huge multiplier keeps everything, a null one keeps the inner values.
    assert len(R.filter_outliers_by_threshold(md, "IQR", 1e6)) == len(md)
    assert len(R.filter_outliers_by_threshold(md, "IQR", 0.0)) < len(kept)


# ---------------------------------------------------------------------------
# robust_utils
# ---------------------------------------------------------------------------
def test_get_metrics():
    assert U.get_metrics() == list(U.METRICS)
    assert U.get_metrics() is not U.get_metrics()
    assert len(U.get_metrics()) == 10


def test_get_design_matrices_options(example_df):
    md = example_df[example_df["metric"] == "md"]
    n_bundles = md["bundle"].nunique()

    design, y = U.get_design_matrices(md)
    assert len(design) == len(y) == n_bundles
    # intercept, sex, handedness, age
    assert design[0].shape == (4, 100)
    assert y[0].shape == (100,)

    design_h, _ = U.get_design_matrices(md, ignore_handedness=True)
    assert design_h[0].shape == (3, 100)
    design_s, _ = U.get_design_matrices(md, ignore_sex=True)
    assert design_s[0].shape == (3, 100)
    design_hs, _ = U.get_design_matrices(md, True, True)
    assert design_hs[0].shape == (2, 100)

    # Keeping the sex column is the default, and ignoring it drops that row.
    design_default, _ = U.get_design_matrices(md, False, False)
    np.testing.assert_array_equal(design_default[0], design[0])
    np.testing.assert_array_equal(
        np.delete(design[0], 1, axis=0), design_s[0]
    )


# ---------------------------------------------------------------------------
# Synthetic sites
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def train_test(example_df):
    df = example_df[example_df["metric"].isin(SMALL_METRICS)]
    df = df.assign(metric_bundle=df["metric"] + "_" + df["bundle"])
    return split_train_test(df, test_size=0.2, random_state=1)


def _bias_dicts(df, add, mult):
    keys = df["metric_bundle"].unique()
    return {k: add for k in keys}, {k: mult for k in keys}


def _sorted(df):
    return df.sort_values(["metric", "sid", "bundle"]).reset_index(drop=True)


@pytest.mark.parametrize("centered_bias", [False, True])
def test_apply_bias_without_bias_keeps_the_values(train_test, centered_bias):
    train, test = train_test
    add, mult = _bias_dicts(train, 0.0, 1.0)

    biased = apply_bias(train, test, add, mult, centered_bias=centered_bias)

    expected = _sorted(pd.concat([train, test], ignore_index=True))
    biased = _sorted(biased)
    assert list(biased.columns) == list(expected.columns)
    np.testing.assert_allclose(
        biased["mean"].to_numpy(), expected["mean"].to_numpy(), rtol=1e-9
    )
    assert_frame_equal(biased.drop(columns="mean"), expected.drop(columns="mean"))


def test_apply_bias_centered_runs_with_a_real_bias(train_test):
    train, test = train_test
    add, mult = _bias_dicts(train, 0.5, 1.5)

    centered = _sorted(apply_bias(train, test, add, mult, centered_bias=True))
    plain = _sorted(apply_bias(train, test, add, mult, centered_bias=False))
    original = _sorted(pd.concat([train, test], ignore_index=True))

    assert np.isfinite(centered["mean"]).all()
    assert len(centered) == len(original)
    assert not np.allclose(centered["mean"], original["mean"])
    assert not np.allclose(centered["mean"], plain["mean"])


@pytest.mark.parametrize("centered_bias", [False, True])
@pytest.mark.parametrize("fixed_bias", [False, True])
def test_generate_biased_data_keeps_the_rows(train_test, fixed_bias, centered_bias):
    train, test = train_test
    np.random.seed(2)

    biased_train, biased_test, gammas, deltas, parameters = generate_biased_data(
        train, test, fixed_bias=fixed_bias, centered_bias=centered_bias
    )

    for biased, original in ((biased_train, train), (biased_test, test)):
        assert len(biased) == len(original)
        assert set(biased["sid"]) == set(original["sid"])
        assert list(biased.columns) == list(original.columns)
        assert np.isfinite(biased["mean"]).all()
        assert_frame_equal(
            _sorted(biased).drop(columns="mean"),
            _sorted(original).drop(columns="mean"),
        )
        assert not np.allclose(_sorted(biased)["mean"], _sorted(original)["mean"])

    assert set(gammas) == set(deltas) == set(train["metric_bundle"])
    assert set(parameters) == {
        "additive_mean",
        "multiplicative_mean",
        "additive_std",
        "multiplicative_std",
    }
    if fixed_bias:
        assert set(gammas.values()) == {2}
        assert set(deltas.values()) == {1.25}


def test_generate_biased_data_is_reproducible(train_test):
    train, test = train_test
    np.random.seed(5)
    first = generate_biased_data(train, test)
    np.random.seed(5)
    second = generate_biased_data(train, test)
    assert_frame_equal(first[0], second[0])
    assert_frame_equal(first[1], second[1])
    assert first[2] == second[2]


def test_generate_sites_no_file_all_keeps_every_disease(example_df):
    kwargs = dict(sample_sizes=[10], disease_ratios=[0.3], num_tests=1, n_jobs=1)

    np.random.seed(11)
    sites_all = generate_sites_no_file(df=example_df, disease="ALL", **kwargs)
    np.random.seed(11)
    sites_none = generate_sites_no_file(df=example_df, disease=None, **kwargs)

    assert len(sites_all) == len(sites_none) == 1
    assert_frame_equal(sites_all[0], sites_none[0])
    site = sites_all[0]
    assert site["sid"].nunique() == 10
    assert site.drop_duplicates("sid")["disease"].ne("HC").sum() == 3


# ---------------------------------------------------------------------------
# robust_harmonization
# ---------------------------------------------------------------------------
def test_output_filenames(tmp_path):
    # The check for a test set looks at the file name only, so these paths
    # give the same names wherever the repository lives, also under folders
    # whose name contains "test".
    folder = tmp_path / "latest_tests"
    folder.mkdir()
    plain = folder / "70_pct_example_md.csv"
    test_set = folder / "test_70_pct_example_md.csv"
    _read_example("md").to_csv(plain, index=False)
    shutil.copy(plain, test_set)
    site = U.get_site(plain)

    for mov in (_example_file("md"), str(plain)):
        assert RH.get_output_model_filename(
            mov, "md", harmonization_method="pairwise", robust="MAD"
        ) == f"{site}.md.pairwise.MAD.model.csv"
        assert RH.get_output_filename(
            mov, "md", harmonization_method="pairwise", robust="MAD"
        ) == f"{site}.md.pairwise.MAD.csv"

    # A test set is told apart by its file name, and has its own output.
    assert RH.get_output_model_filename(
        str(test_set), "md", harmonization_method="pairwise", robust="MAD"
    ) == f"{site}.md.pairwise.MAD.model.csv"
    assert RH.get_output_filename(
        str(test_set), "md", harmonization_method="pairwise", robust="MAD"
    ) == f"{site}_test.md.pairwise.MAD.csv"


def test_expand_data_files(tmp_path):
    for name in ("b_3.csv", "a_3.csv", "a_4.csv"):
        (tmp_path / name).write_text("x\n")
    pattern = str(tmp_path / "*_3.csv")
    single = str(tmp_path / "a_4.csv")

    assert RH._expand_data_files(pattern) == [
        str(tmp_path / "a_3.csv"),
        str(tmp_path / "b_3.csv"),
    ]
    assert RH._expand_data_files(f"{single} {pattern}") == [
        single,
        str(tmp_path / "a_3.csv"),
        str(tmp_path / "b_3.csv"),
    ]
    assert RH._expand_data_files([single, "other.csv"]) == [single, "other.csv"]

    with pytest.raises(FileNotFoundError):
        RH._expand_data_files(str(tmp_path / "*_9.csv"))
    with pytest.raises(FileNotFoundError):
        RH._expand_data_files([])


def test_robust_script_runs_combat_robust_without_a_shell(tmp_path, monkeypatch):
    for name in ("train_2.csv", "train_1.csv"):
        (tmp_path / name).write_text("x\n")
    calls = []
    monkeypatch.setattr(
        RH.subprocess, "run", lambda cmd, **kwargs: calls.append((cmd, kwargs))
    )

    RH.robust_script(str(tmp_path / "train_*.csv"), "MAD")

    assert calls == [
        (
            [
                "combat_robust",
                str(tmp_path / "train_1.csv"),
                str(tmp_path / "train_2.csv"),
                "--method",
                "MAD",
            ],
            {"check": True},
        )
    ]


def test_visualize_harmonization_uses_the_cli_options(tmp_path, monkeypatch):
    from clinical_combat.cli import combat_visualize_harmonization as viz

    calls = []
    monkeypatch.setattr(
        RH.subprocess, "run", lambda cmd, **kwargs: calls.append((cmd, kwargs))
    )

    RH.visualize_harmonization(
        "mov.csv", "harm.csv", "ref.csv", str(tmp_path), bundles="AF_L CST_L", title="t"
    )

    (cmd, kwargs), = calls
    assert cmd[0] == "combat_visualize_harmonization"
    assert kwargs == {"check": True}
    args = viz._build_arg_parser().parse_args(cmd[1:])
    assert args.in_reference == "ref.csv"
    assert args.in_movings == ["mov.csv", "harm.csv"]
    assert args.bundles == ["AF_L", "CST_L"]
    assert args.outname == args.add_suffix == "t"
    assert args.overwrite


# ---------------------------------------------------------------------------
# robust_MLP
# ---------------------------------------------------------------------------
def test_build_mlp_from_config_defaults():
    import torch.nn as nn

    model = build_mlp_from_config()
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    assert [m.out_features for m in linears] == DEFAULT_MODEL_CONFIG["hidden_dims"] + [1]
    assert linears[0].in_features == DEFAULT_MODEL_CONFIG["in_features"]
    dropouts = [m.p for m in model.net if isinstance(m, nn.Dropout)]
    assert dropouts == [0.5, 0.5, 0.5]

    custom = build_mlp_from_config(
        {"hidden_dims": [8], "dropout": 0.1, "activation": "gelu", "batch_norm": False}
    )
    kinds = [type(m) for m in custom.net]
    assert kinds == [nn.Linear, nn.GELU, nn.Dropout, nn.Linear]
    assert custom.net[2].p == 0.1


# ---------------------------------------------------------------------------
# combat_robust command line
# ---------------------------------------------------------------------------
def _run_combat_robust(files, *args, cwd=None):
    cmd = [sys.executable, "-m", "clinical_combat.cli.combat_robust", *files, *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)


@pytest.fixture
def site_files(tmp_path):
    """Writable copies of the 10 example files, without score columns."""
    files = []
    for metric in U.get_metrics():
        dst = tmp_path / f"site.{metric}.csv"
        _read_example(metric).to_csv(dst, index=False)
        files.append(str(dst))
    return files


def test_combat_robust_adds_the_column_to_every_file(site_files, example_df):
    originals = {f: pd.read_csv(f) for f in site_files}

    # A lower-case method name gives the upper-case column that combat_fit
    # looks for.
    result = _run_combat_robust(site_files, "--method", "mad")
    assert result.returncode == 0, result.stderr

    expected = R.add_outlier_column(example_df, "MAD").reset_index(drop=True)
    for path, original in originals.items():
        out = pd.read_csv(path)
        assert list(out.columns) == [*original.columns, "MAD"]
        assert len(out) == len(original)
        assert out["metric"].nunique() == 1
        assert out["metric"].iloc[0] == original["metric"].iloc[0]
        assert np.isfinite(out["MAD"]).all()
        # Same rows and scores as the in-process computation.
        rows = expected[expected["metric"] == out["metric"].iloc[0]]
        assert_frame_equal(out, rows.reset_index(drop=True))
        # The original columns are untouched, up to the row order.
        keys = ["sid", "bundle"]
        assert_frame_equal(
            out.drop(columns="MAD").sort_values(keys).reset_index(drop=True),
            original.sort_values(keys).reset_index(drop=True),
        )


def test_combat_robust_mlp(site_files):
    result = _run_combat_robust(site_files, "-m", "MLP_TEST", cwd=COMBAT_ROOT)
    assert result.returncode == 0, result.stderr
    out = pd.read_csv(site_files[0])
    assert "MLP_TEST" in out.columns
    assert ((out["MLP_TEST"] >= 0) & (out["MLP_TEST"] <= 1)).all()


def test_combat_robust_hc_leaves_files_untouched(site_files):
    before = {f: open(f, "rb").read() for f in site_files}

    result = _run_combat_robust(site_files, "--method", "hc")

    assert result.returncode == 0, result.stderr
    assert "nothing to score" in result.stderr
    assert {f: open(f, "rb").read() for f in site_files} == before


@pytest.mark.parametrize("method", ["IQR", "NO", "HC"])
def test_combat_robust_main_skips_fit_time_methods(
    site_files, method, monkeypatch, caplog
):
    before = {f: open(f, "rb").read() for f in site_files}
    monkeypatch.setattr(
        sys, "argv", ["combat_robust", *site_files, "--method", method]
    )

    combat_robust.main()

    assert "nothing to score" in caplog.text
    assert {f: open(f, "rb").read() for f in site_files} == before


def test_combat_robust_rejects_a_file_with_two_metrics(site_files, tmp_path):
    mixed = tmp_path / "mixed.csv"
    pd.concat(
        [pd.read_csv(site_files[0]), pd.read_csv(site_files[1])], ignore_index=True
    ).to_csv(mixed, index=False)
    others = site_files[2:]
    before = {f: open(f, "rb").read() for f in [str(mixed), *others]}

    result = _run_combat_robust([str(mixed), *others], "--method", "MAD")

    assert result.returncode != 0
    assert "exactly one metric" in result.stderr
    assert {f: open(f, "rb").read() for f in before} == before


def test_load_files_rejects_two_files_with_the_same_metric(site_files, tmp_path):
    duplicate = tmp_path / "duplicate.csv"
    shutil.copy(site_files[0], duplicate)

    with pytest.raises(ValueError, match="expected one file per metric"):
        combat_robust.load_files([*site_files, str(duplicate)], U.get_metrics())


def test_load_files_rejects_unknown_metrics(site_files):
    with pytest.raises(ValueError, match="Invalid metrics"):
        combat_robust.load_files(site_files, ["fa", "md"])


def test_load_files_returns_the_metric_of_each_file(site_files):
    files, df, metric_of_file = combat_robust.load_files(
        site_files, U.get_metrics(), return_metrics=True
    )
    assert files == site_files
    assert list(metric_of_file.values()) == U.get_metrics()
    assert len(df) == 10 * 4300


def test_load_files_default_returns_two_values(site_files):
    files, df = combat_robust.load_files(site_files, U.get_metrics())
    assert files == site_files
    assert len(df) == 10 * 4300


def test_save_back_without_metric_of_file_reads_the_metric_from_each_file(
    site_files,
):
    files, df, metric_of_file = combat_robust.load_files(
        site_files, U.get_metrics(), return_metrics=True
    )
    df = df.assign(NEW=1.0)

    combat_robust.save_back(files, df)

    for f in files:
        out = pd.read_csv(f)
        assert set(out["metric"]) == {metric_of_file[f]}
        assert "NEW" in out.columns


def test_robust_script_accepts_a_pathlib_path(tmp_path, monkeypatch):
    path = tmp_path / "site.fa.csv"
    path.write_text("x\n")
    calls = []
    monkeypatch.setattr(
        RH.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd)
    )

    RH.robust_script(path, "MAD")

    assert RH._expand_data_files(path) == [str(path)]
    assert calls == [["combat_robust", str(path), "--method", "MAD"]]


def test_robust_script_keeps_an_existing_path_with_a_space(tmp_path, monkeypatch):
    folder = tmp_path / "my data"
    folder.mkdir()
    path = str(folder / "site [1].fa.csv")
    open(path, "w").write("x\n")
    calls = []
    monkeypatch.setattr(
        RH.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd)
    )

    RH.robust_script(path, "MAD")

    assert RH._expand_data_files(path) == [path]
    assert calls == [["combat_robust", path, "--method", "MAD"]]


def test_combat_fit_ignores_the_mlp_score_column(scored_cache, tmp_path):
    # The MLP score column must not reach the model: fitting a file that has
    # the MLP_TEST column gives the same model as the same file without it.
    scored = scored_cache["MLP_TEST"][1]
    with_score = tmp_path / "with_score.csv"
    without_score = tmp_path / "without_score.csv"
    md = scored[scored["metric"] == "md"]
    md.to_csv(with_score, index=False)
    md.drop(columns="MLP_TEST").to_csv(without_score, index=False)

    ref = os.path.join(DATA_DIR, "CamCAN.md.raw.csv.gz")
    models = []
    for name, mov in (("with", with_score), ("without", without_score)):
        out_dir = tmp_path / f"fit_{name}"
        cmd = [
            sys.executable, "-m", "clinical_combat.cli.combat_fit",
            ref, str(mov), "--out_dir", str(out_dir), "-o", "model.csv",
            "--method", "pairwise", "--robust", "NO",
        ]
        fit = subprocess.run(cmd, capture_output=True, text=True)
        assert fit.returncode == 0, fit.stderr
        models.append((out_dir / "model.csv").read_bytes())

    assert models[0] == models[1]
