import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


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
                    "The 'drop' list must match the length of 'hidden_dims'."
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
    "dropout": 0.5,
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


class PatientDataset(Dataset):
    """
    Torch Dataset wrapping patient feature matrices and binary labels.

    X: array-like
        Feature matrix with shape (n_samples, n_features).
    y: array-like
        Binary labels aligned with X rows.
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        """Return the number of samples."""
        return len(self.X)

    def __getitem__(self, idx):
        """Return the feature vector and label at index idx."""
        return self.X[idx], self.y[idx]


def make_loaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build DataLoaders for train, validation, and test splits.

    X_train, y_train: array-like
        Training features and labels.
    X_val, y_val: array-like
        Validation features and labels.
    X_test, y_test: array-like
        Test features and labels.
    batch_size: int
        Batch size used for all DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader: DataLoader
        Torch DataLoaders for each dataset split.
    """
    train = DataLoader(
        PatientDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val = DataLoader(
        PatientDataset(X_val, y_val),
        batch_size=batch_size,
    )
    test = DataLoader(
        PatientDataset(X_test, y_test),
        batch_size=batch_size,
    )
    return train, val, test


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    crit: nn.Module,
    opt: torch.optim.Optimizer,
    device: str = "cpu",
    neg_weight: float = 10.0,
) -> float:
    """
    Run one training epoch with class-weighted BCE loss.

    model: torch.nn.Module
        Model to train.
    loader: DataLoader
        Training DataLoader.
    crit: nn.Module
        Loss function returning per-sample losses.
    opt: torch.optim.Optimizer
        Optimizer updating model parameters.
    device: str
        Device identifier, e.g., 'cpu' or 'cuda'.
    neg_weight: float
        Weight applied to negative class samples.

    Returns
    -------
    mean_loss: float
        Average loss over the training dataset.
    """
    model.train()
    running = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device).float()
        opt.zero_grad()
        logits = model(xb)
        base_loss = crit(logits, yb)
        weights = torch.where(yb == 0, float(neg_weight), 1.0)
        loss = (base_loss * weights).mean()
        loss.backward()
        opt.step()
        running += loss.item() * xb.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    crit: nn.Module,
    device: str = "cpu",
    neg_weight: float = 10.0,
):
    """
    Evaluate the model on a dataset split.

    model: torch.nn.Module
        Model to evaluate.
    loader: DataLoader
        DataLoader for the evaluation split.
    crit: nn.Module
        Loss function returning per-sample losses.
    device: str
        Device identifier, e.g., 'cpu' or 'cuda'.
    neg_weight: float
        Weight applied to negative class samples.

    Returns
    -------
    mean_loss: float
        Average weighted loss over the dataset.
    auc: float
        ROC-AUC for the split; NaN if labels are single-class.
    f1: float
        F1 score at 0.5 threshold; NaN if labels are single-class.
    probs: ndarray
        Predicted probabilities for each sample.
    labels: ndarray
        Ground-truth labels for each sample.
    """
    model.eval()
    losses, probs, labels = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.float()
        logits = model(xb)
        base_loss = crit(logits, yb.to(device))
        weights = torch.where(yb == 0, float(neg_weight), 1.0)
        loss = (base_loss * weights).mean().item()
        losses.append(loss * xb.size(0))
        probs.append(torch.sigmoid(logits).cpu())
        labels.append(yb)

    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()
    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, (probs > 0.5).astype(int))
    else:
        auc = float("nan")
        f1 = float("nan")

    return np.sum(losses) / len(loader.dataset), auc, f1, probs, labels


def fit(
    model: torch.nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    wd: float = 1e-4,
    patience: int = 10,
    device: str = "cpu",
    neg_weight: float = 10.0,
    run_name: Optional[str] = None,
):
    """
    Train the model and keep the checkpoint with the best validation AUC.

    model: torch.nn.Module
        PatientMLP or compatible binary classifier.
    train_dl: DataLoader
        Training DataLoader.
    val_dl: DataLoader
        Validation DataLoader.
    epochs: int
        Maximum number of training epochs.
    lr: float
        Learning rate for the optimizer.
    wd: float
        Weight decay for the optimizer.
    patience: int
        Early stopping patience on validation AUC.
    device: str
        Device identifier, e.g., 'cpu' or 'cuda'.
    neg_weight: float
        Weight applied to negative class samples.
    run_name: Optional[str]
        Optional run name for TensorBoard logging.

    Returns
    -------
    best_state: dict or None
        State dict of the best epoch (None if no improvement was logged).
    train_losses: list
        Per-epoch training losses.
    val_losses: list
        Per-epoch validation losses.
    best_auc: float
        Best validation AUC achieved during training.
    """
    crit = nn.BCEWithLogitsLoss(reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        patience=5,
        factor=0.5,
    )

    writer = None
    if run_name and SummaryWriter is not None:
        writer = SummaryWriter(f"{MODEL_DIR}/runs/{run_name}")

    best_auc, best_state, counter = -1.0, None, 0
    tr_losses, val_losses = [], []
    for ep in range(1, epochs + 1):
        tr_loss = train_epoch(
            model,
            train_dl,
            crit,
            opt,
            device=device,
            neg_weight=neg_weight,
        )
        val_loss, val_auc, _f1, _p, _l = eval_epoch(
            model,
            val_dl,
            crit,
            device=device,
            neg_weight=neg_weight,
        )
        tr_losses.append(tr_loss)
        val_losses.append(val_loss)
        sched.step(val_loss)

        if writer is not None:
            writer.add_scalar("Loss/train", tr_loss, ep)
            writer.add_scalar("Loss/val", val_loss, ep)
            if not np.isnan(val_auc):
                writer.add_scalar("AUC/val", val_auc, ep)

        if not np.isnan(val_auc) and val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_state, tr_losses, val_losses, best_auc
