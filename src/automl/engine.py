"""The AutoML engine: from a table to a fitted, saved model with an honest leaderboard.

    automl = AutoML(time_budget=120).fit(df, target="churned")
    automl.leaderboard_          # every candidate: validation and test score
    automl.model_.predict(new_df)
    automl.save("model.joblib")

The order of operations is the whole point:

1. understand the task: profile, TaskSpec (inferred or given), checked against the data;
2. split once: development rows (train + validation) and test rows;
3. every candidate is tuned on validation data (``models.py``): a holdout validation split
   for large tables, k-fold cross-validation over the development rows for small ones, where
   a single validation split is too noisy to choose by;
4. the best candidate is chosen by its **validation** score;
5. every candidate is refitted on all development rows with its chosen settings and scored on
   **test**, once. Test scores are reported, they never influence a choice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from automl.data import DatasetProfile, profile_dataset
from automl.explain import explain
from automl.metrics import get_metric, model_output
from automl.models import (
    Candidate,
    Fold,
    SearchData,
    search_dummy,
    search_linear,
    search_random_forest,
    search_xgboost,
    search_xgboost_default,
)
from automl.preprocess import FeaturePlan, make_preprocessor, plan_features
from automl.split import DataSplits, split_dataset
from automl.task import TaskSpec, TaskSpecError, infer_task_spec

MIN_XGBOOST_SECONDS = 5.0  # the tuned model always gets at least this long
CV_MAX_ROWS = 5_000  # validation="auto": k-fold CV below this many development rows


@dataclass
class FittedModel:
    """What gets saved: preprocessing + model in one pipeline, plus what it needs to be used."""

    pipeline: Pipeline
    spec: TaskSpec
    feature_columns: list[str]
    classes: np.ndarray | None
    model_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._features(df)
        if self.classes is None:
            return np.asarray(self.pipeline.predict(X), dtype=float)
        # Same rule the metrics use: the most probable class.
        return self.classes[np.asarray(self.pipeline.predict_proba(X)).argmax(axis=1)]

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.classes is None:
            raise ValueError("predict_proba is only available for classification")
        proba = np.asarray(self.pipeline.predict_proba(self._features(df)))
        return pd.DataFrame(proba, columns=[str(c) for c in self.classes], index=df.index)

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns missing from the input: {missing}")
        return df[self.feature_columns]


class AutoML:
    def __init__(
        self,
        spec: TaskSpec | None = None,
        time_budget: float = 120.0,
        n_trials: int = 100,
        test_size: float = 0.2,
        valid_size: float = 0.2,
        check_leakage: bool = True,
        validation: str = "auto",
        cv_folds: int = 5,
        selection: str = "best",
        random_state: int = 0,
        n_jobs: int = -1,
    ) -> None:
        """
        ``validation``: "holdout", "cv" or "auto" (cv below ``CV_MAX_ROWS`` rows).
        ``selection``: "best" (highest validation score) or "one_se" (the simplest candidate
        within one standard error of the best; needs cv). The benchmark found no gain from
        "one_se" on average (benchmarks/results.md), so "best" is the default.
        """
        if validation not in ("auto", "holdout", "cv"):
            raise ValueError('validation must be "auto", "holdout" or "cv"')
        if selection not in ("best", "one_se"):
            raise ValueError('selection must be "best" or "one_se"')
        if selection == "one_se" and validation == "holdout":
            raise ValueError("the one-standard-error rule needs cross-validation")
        self.selection = selection
        self.spec = spec
        self.time_budget = time_budget
        self.n_trials = n_trials
        self.test_size = test_size
        self.valid_size = valid_size
        self.check_leakage = check_leakage
        self.validation = validation
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, df: pd.DataFrame, target: str | None = None) -> AutoML:
        started = time.perf_counter()

        # 1. Understand the task.
        self.profile_: DatasetProfile = profile_dataset(df)
        spec = self.spec or infer_task_spec(
            df, target=target, profile=self.profile_, check_leakage=self.check_leakage
        )
        if target is not None and spec.target != target:
            raise TaskSpecError([f"target {target!r} differs from the spec ({spec.target!r})"])
        spec.check_against(df, self.profile_)
        self.spec_ = spec
        self.plan_: FeaturePlan = plan_features(self.profile_, spec)
        metric = get_metric(spec.metric)

        # 2. Split once.
        splits = split_dataset(
            df,
            spec,
            self.plan_.columns,
            test_size=self.test_size,
            valid_size=self.valid_size,
            random_state=self.random_state,
        )
        self.split_sizes_ = splits.sizes()

        # 3. Tune every candidate on validation data.
        n_development = self.split_sizes_["train"] + self.split_sizes_["valid"]
        use_cv = self.validation == "cv" or (
            self.validation == "auto" and n_development < CV_MAX_ROWS
        )
        self.validation_ = f"{self.cv_folds}-fold CV" if use_cv else "holdout"
        data = self._search_data(splits, metric.name, use_cv)
        candidates = [
            search(data)
            for search in (
                search_dummy,
                search_linear,
                search_random_forest,
                search_xgboost_default,
            )
        ]
        remaining = self.time_budget - (time.perf_counter() - started)
        candidates.append(
            search_xgboost(data, self.n_trials, timeout=max(remaining, MIN_XGBOOST_SECONDS))
        )
        self.candidates_ = candidates

        # 4. Choose by validation. Candidates are ordered from simplest to most complex.
        objectives = [
            c.valid_score if metric.greater_is_better else -c.valid_score for c in candidates
        ]
        best = candidates[self._choose(candidates, objectives, use_cv)]

        # 5. Refit on train + validation, score on test once.
        X_full = pd.concat([splits.X_train, splits.X_valid])
        y_full = np.concatenate([splits.y_train, splits.y_valid])
        rows, pipelines = [], {}
        for candidate in candidates:
            pipeline = self._refit(candidate, X_full, y_full)
            pipelines[candidate.name] = pipeline
            test_score = metric.score(
                splits.y_test, model_output(pipeline, splits.X_test, spec.is_classification)
            )
            rows.append(
                {
                    "model": candidate.name,
                    f"valid_{metric.name}": candidate.valid_score,
                    f"test_{metric.name}": test_score,
                    "fit_seconds": round(candidate.fit_seconds, 2),
                    "selected": candidate is best,
                }
            )
        self.leaderboard_ = (
            pd.DataFrame(rows)
            .sort_values(f"valid_{metric.name}", ascending=not metric.greater_is_better)
            .reset_index(drop=True)
        )

        # Every refitted candidate stays available (for comparison and explanations);
        # the chosen one is ``model_``.
        self.models_ = {
            candidate.name: FittedModel(
                pipeline=pipelines[candidate.name],
                spec=spec,
                feature_columns=self.plan_.columns,
                classes=splits.classes,
                model_name=candidate.name,
            )
            for candidate in candidates
        }
        self.model_ = self.models_[best.name]
        self.model_.metadata = self._metadata(best, started)
        # Global importance on the test rows: data the model has not been trained on.
        self.model_.metadata["explanation"] = explain(self.model_, splits.X_test).to_dict()
        return self

    def save(self, path: str | Path) -> Path:
        """Save the chosen model with its leaderboard. Load with ``load_model``.

        The file is a pickle: load only files you created yourself.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model_.metadata["leaderboard"] = self.leaderboard_.to_dict(orient="records")
        joblib.dump(self.model_, path)
        return path

    # ----- helpers -------------------------------------------------------------------------

    def _search_data(self, splits: DataSplits, metric_name: str, use_cv: bool) -> SearchData:
        if use_cv:
            X = pd.concat([splits.X_train, splits.X_valid])
            y = np.concatenate([splits.y_train, splits.y_valid])
            splitter_class = StratifiedKFold if self.spec_.is_classification else KFold
            splitter = splitter_class(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            pairs = [
                (X.iloc[train], X.iloc[valid], y[train], y[valid])
                for train, valid in splitter.split(X, y)
            ]
        else:
            pairs = [(splits.X_train, splits.X_valid, splits.y_train, splits.y_valid)]
        return SearchData(
            problem_type=self.spec_.problem_type.value,
            metric=get_metric(metric_name),
            folds=[self._fold(*pair) for pair in pairs],
            n_development_rows=len(splits.y_train) + len(splits.y_valid),
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def _choose(self, candidates: list[Candidate], objectives: list[float], use_cv: bool) -> int:
        """Index of the chosen candidate.

        "best": the highest validation score (ties: the simpler model).
        "one_se": the one-standard-error rule (Breiman et al., 1984). A tuned model's
        cross-validated score is a little optimistic (it was the best of many trials) and has
        noise of its own; a simpler model whose score is within one standard error of the
        best is statistically as good and safer, so the simplest such model wins.
        """
        best = int(np.argmax(objectives))
        one_se = self.selection == "one_se" and use_cv
        self.selection_ = "one-standard-error rule" if one_se else "best validation score"
        if not one_se:
            return best
        std = candidates[best].details.get("valid_std", 0.0)
        threshold = objectives[best] - std / np.sqrt(self.cv_folds)
        return next(i for i, value in enumerate(objectives) if value >= threshold)

    def _fold(
        self, X_train: pd.DataFrame, X_valid: pd.DataFrame, y_train: np.ndarray, y_valid: np.ndarray
    ) -> Fold:
        # Preprocessing is fitted on the fold's train part only, like everything else.
        trees = make_preprocessor(self.plan_).fit(X_train)
        linear = make_preprocessor(self.plan_, for_linear=True).fit(X_train)
        return Fold(
            X_train_trees=trees.transform(X_train),
            X_valid_trees=trees.transform(X_valid),
            X_train_linear=linear.transform(X_train),
            X_valid_linear=linear.transform(X_valid),
            y_train=y_train,
            y_valid=y_valid,
        )

    def _refit(self, candidate: Candidate, X: pd.DataFrame, y: np.ndarray) -> Pipeline:
        pipeline = Pipeline(
            [
                ("preprocess", make_preprocessor(self.plan_, for_linear=candidate.linear_features)),
                ("model", clone(candidate.estimator)),
            ]
        )
        return pipeline.fit(X, y)

    def _metadata(self, best: Candidate, started: float) -> dict[str, Any]:
        return {
            "metric": self.spec_.metric,
            "valid_score": best.valid_score,
            "details": best.details,
            "split_sizes": self.split_sizes_,
            "validation": self.validation_,
            "selection": self.selection_,
            "fit_seconds_total": round(time.perf_counter() - started, 1),
            "versions": {
                name: version(name) for name in ("automl", "xgboost", "scikit-learn", "pandas")
            },
        }


def load_model(path: str | Path) -> FittedModel:
    """Load a model saved by ``AutoML.save``. Only load files you created yourself."""
    model = joblib.load(Path(path))
    if not isinstance(model, FittedModel):
        raise TypeError(f"{path} does not contain an automl model")
    return model
