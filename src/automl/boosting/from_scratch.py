"""Classic gradient boosting written from scratch (Friedman, 2001).

The whole algorithm fits in one loop:

    F_0(x) = best constant                       (mean for MSE, log-odds for log-loss)
    for m = 1..M:
        r_i  = -dL(y_i, F(x_i)) / dF             ("pseudo-residuals": where to move F)
        h_m  = regression tree fitted to (x_i, r_i)
        for every leaf j of h_m:
            gamma_j = best constant step inside the leaf
        F(x) = F(x) + learning_rate * gamma_{leaf(x)}

Only two things depend on the loss function: the gradient ``r`` and the leaf values ``gamma``.

* Squared error, L = (y - F)^2 / 2: the gradient is the ordinary residual ``y - F``, and the
  best leaf value is the mean residual in the leaf, which the tree already stores.
* Log-loss (binary classification), with p = sigmoid(F): the gradient is ``y - p``. The tree
  splits on it, but the leaf value is one Newton step: sum(y - p) / sum(p * (1 - p)).
  This second-order step is the idea XGBoost later builds its whole tree on.

The split search itself is delegated to scikit-learn's ``DecisionTreeRegressor``; the next
learning step (stage 1b) replaces it with an XGBoost-style tree built from gradients and
hessians. With the same parameters these classes reproduce scikit-learn's
``GradientBoostingRegressor`` / ``GradientBoostingClassifier``, which the tests check.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit
from sklearn.tree import DecisionTreeRegressor

FloatArray = NDArray[np.float64]


class _ScratchGradientBoosting:
    """Shared boosting loop. Subclasses define the loss-specific pieces."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    # ----- loss-specific pieces ------------------------------------------------------------

    def _initial_raw_prediction(self, y: FloatArray) -> float:
        raise NotImplementedError

    def _negative_gradient(self, y: FloatArray, raw: FloatArray) -> FloatArray:
        raise NotImplementedError

    def _leaf_values(
        self,
        tree: DecisionTreeRegressor,
        leaves: NDArray[np.intp],
        y: FloatArray,
        raw: FloatArray,
        residual: FloatArray,
    ) -> FloatArray:
        """Value of every tree node (indexed by node id); only leaf entries are used."""
        raise NotImplementedError

    def _loss(self, y: FloatArray, raw: FloatArray) -> float:
        raise NotImplementedError

    # ----- the boosting loop ---------------------------------------------------------------

    def _fit_raw(self, X: FloatArray, y: FloatArray) -> None:
        self._validate_params()
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]

        self.init_ = self._initial_raw_prediction(y)
        raw = np.full(n_samples, self.init_)  # F(x_i) for every training row

        self.trees_: list[DecisionTreeRegressor] = []
        self.leaf_values_: list[FloatArray] = []
        self.train_loss_ = np.empty(self.n_estimators)

        for m in range(self.n_estimators):
            residual = self._negative_gradient(y, raw)

            # Stochastic gradient boosting: every tree sees a random part of the rows.
            rows = self._sample_rows(rng, n_samples)

            tree = DecisionTreeRegressor(
                criterion="squared_error",
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=int(rng.integers(np.iinfo(np.int32).max)),
            )
            tree.fit(X[rows], residual[rows])

            # The tree decides the regions (leaves); the loss decides the value in each leaf.
            leaves = tree.apply(X)
            values = self._leaf_values(tree, leaves[rows], y[rows], raw[rows], residual[rows])

            # Shrinkage: take only a small step in the direction the tree suggests.
            raw += self.learning_rate * values[leaves]

            self.trees_.append(tree)
            self.leaf_values_.append(values)
            self.train_loss_[m] = self._loss(y, raw)

    def _sample_rows(self, rng: np.random.Generator, n_samples: int) -> NDArray[np.intp]:
        if self.subsample >= 1.0:
            return np.arange(n_samples)
        n_rows = max(1, int(self.subsample * n_samples))
        return np.sort(rng.choice(n_samples, size=n_rows, replace=False))

    def _staged_raw_predict(self, X: ArrayLike) -> Iterator[FloatArray]:
        """Raw score F(x) after each boosting stage: useful to watch the model learn."""
        X = _as_2d_float(X)
        raw = np.full(X.shape[0], self.init_)
        for tree, values in zip(self.trees_, self.leaf_values_, strict=True):
            raw = raw + self.learning_rate * values[tree.apply(X)]
            yield raw

    def _raw_predict(self, X: ArrayLike) -> FloatArray:
        X = _as_2d_float(X)
        raw = np.full(X.shape[0], self.init_)
        for tree, values in zip(self.trees_, self.leaf_values_, strict=True):
            raw += self.learning_rate * values[tree.apply(X)]
        return raw

    def _validate_params(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if not 0.0 < self.subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1]")


class ScratchGradientBoostingRegressor(_ScratchGradientBoosting):
    """Gradient boosting for regression with squared error."""

    def fit(self, X: ArrayLike, y: ArrayLike) -> ScratchGradientBoostingRegressor:
        X = _as_2d_float(X)
        y = np.asarray(y, dtype=np.float64)
        self._fit_raw(X, y)
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        return self._raw_predict(X)

    def staged_predict(self, X: ArrayLike) -> Iterator[FloatArray]:
        yield from self._staged_raw_predict(X)

    def _initial_raw_prediction(self, y: FloatArray) -> float:
        # The constant that minimises squared error is the mean.
        return float(np.mean(y))

    def _negative_gradient(self, y: FloatArray, raw: FloatArray) -> FloatArray:
        # L = (y - F)^2 / 2  ->  -dL/dF = y - F: the plain residual.
        return y - raw

    def _leaf_values(
        self,
        tree: DecisionTreeRegressor,
        leaves: NDArray[np.intp],
        y: FloatArray,
        raw: FloatArray,
        residual: FloatArray,
    ) -> FloatArray:
        # For squared error the best step in a leaf is the mean residual there,
        # which is exactly what a regression tree already predicts.
        return tree.tree_.value[:, 0, 0]

    def _loss(self, y: FloatArray, raw: FloatArray) -> float:
        return float(np.mean((y - raw) ** 2))


class ScratchGradientBoostingClassifier(_ScratchGradientBoosting):
    """Gradient boosting for binary classification with log-loss.

    The model predicts the log-odds F(x); the probability of the positive class is sigmoid(F).
    Labels can be any two values (strings included): they are encoded as 0/1 internally.
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> ScratchGradientBoostingClassifier:
        X = _as_2d_float(X)
        self.classes_, y_encoded = np.unique(np.asarray(y), return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError(
                f"Only binary classification is supported, got {len(self.classes_)} classes"
            )
        self._fit_raw(X, y_encoded.astype(np.float64))
        return self

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        p = expit(self._raw_predict(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: ArrayLike) -> NDArray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    def staged_predict_proba(self, X: ArrayLike) -> Iterator[FloatArray]:
        for raw in self._staged_raw_predict(X):
            p = expit(raw)
            yield np.column_stack([1.0 - p, p])

    def _initial_raw_prediction(self, y: FloatArray) -> float:
        # The best constant is the log-odds of the positive class share.
        p = np.clip(np.mean(y), 1e-15, 1 - 1e-15)
        return float(np.log(p / (1.0 - p)))

    def _negative_gradient(self, y: FloatArray, raw: FloatArray) -> FloatArray:
        # L = -[y log p + (1 - y) log(1 - p)], p = sigmoid(F)  ->  -dL/dF = y - p.
        return y - expit(raw)

    def _leaf_values(
        self,
        tree: DecisionTreeRegressor,
        leaves: NDArray[np.intp],
        y: FloatArray,
        raw: FloatArray,
        residual: FloatArray,
    ) -> FloatArray:
        # One Newton step per leaf: sum of gradients / sum of hessians,
        # where the hessian of log-loss is d2L/dF2 = p * (1 - p).
        p = expit(raw)
        n_nodes = tree.tree_.node_count
        gradient_sum = np.bincount(leaves, weights=residual, minlength=n_nodes)
        hessian_sum = np.bincount(leaves, weights=p * (1.0 - p), minlength=n_nodes)
        values = np.zeros(n_nodes)
        nonzero = hessian_sum > 1e-150
        values[nonzero] = gradient_sum[nonzero] / hessian_sum[nonzero]
        return values

    def _loss(self, y: FloatArray, raw: FloatArray) -> float:
        # Numerically stable log-loss: log(1 + e^F) - y * F.
        return float(np.mean(np.logaddexp(0.0, raw) - y * raw))


def _as_2d_float(X: ArrayLike) -> FloatArray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
    return X
