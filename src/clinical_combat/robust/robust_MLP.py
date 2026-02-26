import json
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


MODEL_DIR = Path("Pytorch_models")


def _compute_z(df, value_col="mean_no_cov"):
    """
    Compute z-scores per feature group (metric_bundle).

    df: dataframe
        Input long-format DataFrame containing at least:
        'metric_bundle' and the value column.

    value_col: str
        Column name containing the values to standardize.

    Returns
    -------
    df: dataframe
        Same DataFrame with an added 'zscore' column.
        Intermediate columns are removed.
    """
    stats = (
        df.groupby("metric_bundle")[value_col]
        .agg(["mean", "std"])
        .rename(columns={"mean": "gmean", "std": "gstd"})
    )
    stats["gstd"] = stats["gstd"].replace(0, 1e-6)

    df = df.merge(stats, on="metric_bundle", how="left")
    df["zscore"] = (df[value_col] - df["gmean"]) / df["gstd"]
    return df.drop(columns=["gmean", "gstd"])


def _pivot_features(df, value_col="zscore", bundle_col="metric_bundle"):
    """
    Pivot a long-format DataFrame into a feature matrix (sid x features).

    df: dataframe
        Input long-format DataFrame.

    value_col: str
        Column name to use as values in the pivot (default: 'zscore').

    bundle_col: str
        Column name defining the feature keys (default: 'metric_bundle').

    Returns
    -------
    mat: dataframe
        Wide-format DataFrame indexed by 'sid' with one column per feature.
    """
    return df.pivot(index="sid", columns=bundle_col, values=value_col)


def predict_outliers_mlp(df, run_name):
    """
    Predict outlier probabilities with a saved PatientMLP run.

    This function:
    1) Loads the model weights and hyperparameters from MODEL_DIR
    2) Removes ventricle bundles
    3) Computes per-metric_bundle z-scores using 'mean_no_cov'
    4) Pivots into a (sid x features) matrix
    5) Runs inference and returns per-subject probabilities

    df: dataframe
        Input long-format DataFrame containing at least:
        'sid', 'bundle', 'metric_bundle', and 'mean_no_cov'.

    run_name: str
        Name of the saved run (case-insensitive). Expects:
        '{run_name}_weights.pt' and '{run_name}_params.json' in MODEL_DIR.

    Returns
    -------
    out: dataframe
        DataFrame with columns:
        'sid' and 'prob_outlier'.
    """
    run_name = str(run_name).lower()

    state_dict = torch.load(
        MODEL_DIR / f"{run_name}_weights.pt",
        map_location="cpu",
    )

    with open(MODEL_DIR / f"{run_name}_params.json", encoding="utf-8") as fp:
        hp = json.load(fp)

    if "hidden_dims" in hp:
        hidden_dims = tuple(int(x) for x in hp["hidden_dims"])
    else:
        hidden_dims = (int(hp["h1"]), int(hp["h2"]), int(hp["h3"]))

    model = PatientMLP(
        hidden_dims=hidden_dims,
        drop=hp.get("dropout", 0.5),
        activation=hp.get("activation", "relu"),
        batch_norm=hp.get("batch_norm", True),
    )
    model.load_state_dict(state_dict)

    device = "cpu"
    model = model.to(device).eval()

    df = df[~df["bundle"].isin(["left_ventricle", "right_ventricle"])].copy()

    df_z = _compute_z(df, value_col="mean_no_cov")

    mat = _pivot_features(df_z, value_col="zscore")
    sid_order = mat.index
    x = mat.values.astype(np.float32)

    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32).to(device))
        proba = torch.sigmoid(logits).cpu().numpy().reshape(-1)

    return pd.DataFrame(
        {
            "sid": sid_order.to_numpy(),
            "prob_outlier": proba.astype(float),
        }
    )


class PatientMLP(nn.Module):
    def __init__(
        self,
        in_features=430,
        hidden_dims=(256, 128, 64),
        drop=0.3,
        activation="relu",
        batch_norm=True,
        config=None,
    ):
        """
        Build a configurable MLP for binary classification.

        in_features: int
            Number of input features.

        hidden_dims: sequence
            Sizes of the hidden layers.

        drop: float or sequence
            Dropout probability, either a single value or one per hidden layer.

        activation: str
            Activation name:
            'relu', 'gelu', 'leaky_relu', 'elu', or 'tanh'.

        batch_norm: bool
            If True, insert BatchNorm1d between Linear and activation.

        config: dict
            Optional dict overriding the parameters above.

        Returns
        -------
        None
        """
        super().__init__()

        if config is not None:
            in_features = config.get("in_features", in_features)
            hidden_dims = config.get("hidden_dims", hidden_dims)
            drop = config.get("dropout", config.get("drop", drop))
            activation = config.get("activation", activation)
            batch_norm = config.get("batch_norm", batch_norm)

        def _act(name):
            name = (name or "relu").lower()
            if name == "relu":
                return nn.ReLU()
            if name == "gelu":
                return nn.GELU()
            if name == "leaky_relu":
                return nn.LeakyReLU(negative_slope=0.01)
            if name == "elu":
                return nn.ELU()
            if name == "tanh":
                return nn.Tanh()
            return nn.ReLU()

        if isinstance(drop, (int, float)):
            drop_list = [float(drop)] * len(hidden_dims)
        else:
            drop_list = [float(d) for d in drop]
            if len(drop_list) != len(hidden_dims):
                raise ValueError(
                    "The 'drop' list must have the same length as 'hidden_dims'."
                )

        layers: List[nn.Module] = []
        prev = int(in_features)

        for h, pdrop in zip(hidden_dims, drop_list):
            h = int(h)
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(_act(activation))
            if pdrop and pdrop > 0:
                layers.append(nn.Dropout(pdrop))
            prev = h

        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        x: tensor
            Input tensor of shape (n_samples, n_features).

        Returns
        -------
        out: tensor
            Logits of shape (n_samples,).
        """
        return self.net(x).squeeze(1)


DEFAULT_MODEL_CONFIG = {
    "in_features": 430,
    "hidden_dims": [256, 128, 64],
    "activation": "relu",
    "dropout": 0.3,
    "batch_norm": True,
}


def build_mlp_from_config(cfg=None):
    """
    Build a PatientMLP from a configuration dict.

    cfg: dict
        Optional overrides, for example:
        {'hidden_dims': [512, 256], 'activation': 'gelu', 'dropout': 0.4}

    Returns
    -------
    model: PatientMLP
        Instantiated model with merged configuration.
    """
    merged = dict(DEFAULT_MODEL_CONFIG)
    if cfg:
        merged.update(cfg)

    return PatientMLP(
        in_features=merged.get("in_features", 430),
        hidden_dims=merged.get("hidden_dims", (256, 128, 64)),
        drop=merged.get("dropout", 0.3),
        activation=merged.get("activation", "relu"),
        batch_norm=merged.get("batch_norm", True),
    )