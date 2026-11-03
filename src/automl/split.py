"""Train / validation / test split with an encoded target.

* train: fits the models;
* validation: early stopping and hyperparameter search (never the test set, see stage 1c);
* test: touched exactly once, to report the final quality of the chosen model.

Classification splits are stratified, so every part keeps the class balance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from automl.task import TaskSpec

MIN_ROWS_PER_CLASS = 3  # stratification needs at least one row of every class in each part


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_valid: pd.DataFrame
    y_valid: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    label_encoder: LabelEncoder | None  # classification only: maps labels <-> 0..k-1
    n_rows_without_target: int

    @property
    def classes(self) -> np.ndarray | None:
        return None if self.label_encoder is None else self.label_encoder.classes_

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.X_train), "valid": len(self.X_valid), "test": len(self.X_test)}


def split_dataset(
    df: pd.DataFrame,
    spec: TaskSpec,
    feature_columns: list[str],
    test_size: float = 0.2,
    valid_size: float = 0.2,
    random_state: int = 0,
) -> DataSplits:
    """Split rows into train/valid/test. Sizes are shares of all rows with a target."""
    if not 0 < test_size < 1 or not 0 < valid_size < 1 or test_size + valid_size >= 1:
        raise ValueError("test_size and valid_size must be in (0, 1) and sum to less than 1")

    has_target = df[spec.target].notna()
    data = df.loc[has_target]
    X = data[feature_columns]

    label_encoder = None
    if spec.is_classification:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(data[spec.target].astype(str))
        counts = np.bincount(y)
        if counts.min() < MIN_ROWS_PER_CLASS:
            rare = list(label_encoder.classes_[counts < MIN_ROWS_PER_CLASS])
            raise ValueError(
                f"Classes with fewer than {MIN_ROWS_PER_CLASS} rows cannot be split: {rare}"
            )
    else:
        y = data[spec.target].to_numpy(dtype=float)

    stratify = y if spec.is_classification else None
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    valid_share_of_rest = valid_size / (1 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_rest,
        y_rest,
        test_size=valid_share_of_rest,
        random_state=random_state,
        stratify=y_rest if spec.is_classification else None,
    )
    return DataSplits(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        label_encoder=label_encoder,
        n_rows_without_target=int((~has_target).sum()),
    )
