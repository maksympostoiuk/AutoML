"""A small XGBoost written from scratch: second-order boosting with its own tree.

Classic gradient boosting (``from_scratch.py``) fits an ordinary regression tree to the
residuals and only then fixes the leaf values. XGBoost goes further: it approximates the loss
around the current prediction with a second-order Taylor expansion and grows the tree directly
on that approximation. For every row it needs just two numbers:

    g_i = dL/dF      (gradient)        squared error: F - y        log-loss: p - y
    h_i = d2L/dF2    (hessian)         squared error: 1            log-loss: p * (1 - p)

For a group of rows with sums G = sum(g) and H = sum(h), the best leaf value and how much
that leaf "scores" are closed-form:

    weight = -G / (H + lambda)
    score  =  G^2 / (H + lambda)

A split of a node into left and right children is worth

    gain = score(left) + score(right) - score(node)

and the tree is grown greedily: in every node try every feature and every threshold
(the "exact" method), keep the split with the largest gain. The regularisation parameters
all live in these formulas:

* ``reg_lambda`` (L2) is added to H: it shrinks leaf weights and makes splits less attractive;
* ``reg_alpha`` (L1) shrinks |G| by alpha before squaring (soft thresholding);
* ``gamma`` is the minimum gain a split must bring: after the tree is grown, splits whose gain
  is below gamma are pruned bottom-up;
* ``min_child_weight`` is the minimum H in a child. For squared error h = 1, so it is a
  minimum number of rows; for log-loss confident rows (p near 0 or 1) weigh almost nothing.

Missing values (NaN) are handled by learning a default direction per split: the search tries
sending all missing rows left and right and keeps the better option.

Like the real library, gains are compared without the 1/2 factor from the paper, so ``gamma``
has the same meaning as in ``xgboost``; features are stored as float32. With
``tree_method="exact"`` the real ``xgboost`` builds the same trees, which the tests check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, logit

FloatArray = NDArray[np.float64]

# XGBoost ignores splits that improve the objective by less than this (kRtEps in its source).
_MIN_SPLIT_IMPROVEMENT = 1e-6


@dataclass(frozen=True)
class TreeParams:
    max_depth: int = 6
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    gamma: float = 0.0


def _soft_threshold(G: FloatArray | float, alpha: float) -> FloatArray | float:
    """L1 regularisation: move G towards zero by alpha, stopping at zero."""
    return np.sign(G) * np.maximum(np.abs(G) - alpha, 0.0)


def leaf_weight(G: float, H: float, p: TreeParams) -> float:
    """Optimal leaf value for gradient sum G and hessian sum H: -G / (H + lambda)."""
    if H < p.min_child_weight or H <= 0.0:
        return 0.0
    return float(-_soft_threshold(G, p.reg_alpha) / (H + p.reg_lambda))


def leaf_score(G: FloatArray | float, H: FloatArray | float, p: TreeParams) -> FloatArray:
    """How much a leaf reduces the loss (up to a factor of 1/2): G^2 / (H + lambda)."""
    return _soft_threshold(G, p.reg_alpha) ** 2 / (H + p.reg_lambda)


@dataclass
class _Split:
    gain: float
    feature: int
    threshold: float
    default_left: bool


@dataclass
class XGBTree:
    """One regression tree grown on gradients and hessians.

    Nodes are stored in flat arrays (as in the real library); node 0 is the root and
    ``left[i] == -1`` marks a leaf. ``value`` holds leaf outputs already multiplied by the
    learning rate.
    """

    params: TreeParams
    feature: list[int] = field(default_factory=list)
    threshold: list[float] = field(default_factory=list)
    default_left: list[bool] = field(default_factory=list)
    left: list[int] = field(default_factory=list)
    right: list[int] = field(default_factory=list)
    value: list[float] = field(default_factory=list)
    gain: list[float] = field(default_factory=list)
    hessian_sum: list[float] = field(default_factory=list)

    # ----- growing -------------------------------------------------------------------------

    def fit(
        self,
        X: FloatArray,
        g: FloatArray,
        h: FloatArray,
        learning_rate: float,
        rows: NDArray[np.intp] | None = None,
        features: NDArray[np.intp] | None = None,
    ) -> XGBTree:
        rows = np.arange(X.shape[0]) if rows is None else rows
        features = np.arange(X.shape[1]) if features is None else features
        self._grow(X, g, h, rows, features, depth=0)
        self._prune(0)
        self.value = [learning_rate * v for v in self.value]
        self._freeze()
        return self

    def _new_node(self, G: float, H: float) -> int:
        self.feature.append(-1)
        self.threshold.append(np.nan)
        self.default_left.append(True)
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(leaf_weight(G, H, self.params))
        self.gain.append(0.0)
        self.hessian_sum.append(H)
        return len(self.value) - 1

    def _grow(
        self,
        X: FloatArray,
        g: FloatArray,
        h: FloatArray,
        rows: NDArray[np.intp],
        features: NDArray[np.intp],
        depth: int,
    ) -> int:
        G, H = float(g[rows].sum()), float(h[rows].sum())
        node = self._new_node(G, H)
        if depth >= self.params.max_depth:
            return node

        split = self._best_split(X, g, h, rows, features, G, H)
        if split is None or split.gain <= _MIN_SPLIT_IMPROVEMENT:
            return node

        x = X[rows, split.feature]
        goes_left = np.where(np.isnan(x), split.default_left, x < split.threshold)
        self.feature[node] = split.feature
        self.threshold[node] = split.threshold
        self.default_left[node] = split.default_left
        self.gain[node] = split.gain
        self.left[node] = self._grow(X, g, h, rows[goes_left], features, depth + 1)
        self.right[node] = self._grow(X, g, h, rows[~goes_left], features, depth + 1)
        return node

    def _best_split(
        self,
        X: FloatArray,
        g: FloatArray,
        h: FloatArray,
        rows: NDArray[np.intp],
        features: NDArray[np.intp],
        G: float,
        H: float,
    ) -> _Split | None:
        """Exact greedy search: every feature, every threshold between two distinct values."""
        p = self.params
        parent_score = leaf_score(G, H, p)
        best: _Split | None = None

        for f in features:
            x = X[rows, f]
            present = ~np.isnan(x)
            has_missing = not present.all()
            order = np.argsort(x[present], kind="stable")
            xs = x[present][order]
            if xs.size == 0 or (xs.size == 1 and not has_missing):
                continue
            gs, hs = g[rows][present][order], h[rows][present][order]

            # Sums for "the first k+1 sorted rows go left", for every k at once.
            G_prefix, H_prefix = np.cumsum(gs)[:-1], np.cumsum(hs)[:-1]
            G_missing, H_missing = G - gs.sum(), H - hs.sum()
            can_split = xs[1:] != xs[:-1]  # only between two different values
            thresholds = (xs[1:] + xs[:-1]) / 2

            for missing_left in (True, False) if has_missing else (True,):
                GL = G_prefix + (G_missing if missing_left else 0.0)
                HL = H_prefix + (H_missing if missing_left else 0.0)
                ok_position, split_at = can_split, thresholds
                if has_missing:
                    # One more candidate: "missing vs. everything else". The threshold is
                    # placed just outside the observed values, the same way xgboost does it.
                    if missing_left:
                        GL, HL = np.append(GL, G_missing), np.append(HL, H_missing)
                        outside = xs[0] - (abs(xs[0]) + _MIN_SPLIT_IMPROVEMENT)
                    else:
                        GL, HL = np.append(GL, gs.sum()), np.append(HL, hs.sum())
                        outside = xs[-1] + (abs(xs[-1]) + _MIN_SPLIT_IMPROVEMENT)
                    ok_position = np.append(can_split, True)
                    split_at = np.append(thresholds, outside)
                GR, HR = G - GL, H - HL
                ok = ok_position & (HL >= p.min_child_weight) & (HR >= p.min_child_weight)
                if not ok.any():
                    continue
                gains = leaf_score(GL, HL, p) + leaf_score(GR, HR, p) - parent_score
                gains = np.where(ok, gains, -np.inf)
                k = int(np.argmax(gains))
                # Strictly better only: on ties the earlier feature wins, as in xgboost.
                if best is None or gains[k] > best.gain:
                    best = _Split(float(gains[k]), int(f), float(split_at[k]), missing_left)
        return best

    def _prune(self, node: int) -> None:
        """Remove splits that gain less than gamma, from the bottom of the tree up."""
        if self.left[node] == -1:
            return
        self._prune(self.left[node])
        self._prune(self.right[node])
        children_are_leaves = self.left[self.left[node]] == -1 and self.left[self.right[node]] == -1
        if children_are_leaves and self.gain[node] < self.params.gamma:
            self.left[node] = self.right[node] = -1  # the node keeps its own weight

    def _freeze(self) -> None:
        """Turn the node lists into arrays for fast vectorised prediction."""
        self._feature = np.asarray(self.feature)
        self._threshold = np.asarray(self.threshold)
        self._default_left = np.asarray(self.default_left)
        self._left = np.asarray(self.left)
        self._right = np.asarray(self.right)
        self._value = np.asarray(self.value)

    # ----- using the tree ------------------------------------------------------------------

    def predict(self, X: FloatArray) -> FloatArray:
        """Walk all rows down the tree together, one level per iteration."""
        node = np.zeros(X.shape[0], dtype=np.intp)
        while True:
            active = np.flatnonzero(self._left[node] != -1)
            if active.size == 0:
                return self._value[node]
            current = node[active]
            x = X[active, self._feature[current]]
            goes_left = np.where(
                np.isnan(x), self._default_left[current], x < self._threshold[current]
            )
            node[active] = np.where(goes_left, self._left[current], self._right[current])

    @property
    def n_leaves(self) -> int:
        return sum(self._reachable_leaves(0))

    def _reachable_leaves(self, node: int) -> list[int]:
        if self.left[node] == -1:
            return [1]
        return self._reachable_leaves(self.left[node]) + self._reachable_leaves(self.right[node])

    def to_text(self, feature_names: list[str] | None = None) -> str:
        """Readable dump in the spirit of ``Booster.get_dump()``."""
        lines: list[str] = []

        def walk(node: int, depth: int) -> None:
            indent = "    " * depth
            if self.left[node] == -1:
                lines.append(f"{indent}leaf = {self.value[node]:.4f}")
                return
            f = self.feature[node]
            name = feature_names[f] if feature_names else f"f{f}"
            missing = "left" if self.default_left[node] else "right"
            lines.append(
                f"{indent}[{name} < {self.threshold[node]:.4g}] "
                f"gain = {self.gain[node]:.3f}, missing -> {missing}"
            )
            walk(self.left[node], depth + 1)
            walk(self.right[node], depth + 1)

        walk(0, 0)
        return "\n".join(lines)


class _MiniXGBoost:
    """Boosting loop shared by the regressor and the classifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.3,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        gamma: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        base_score: float | None = None,
        random_state: int | None = None,
    ) -> None:
        # Defaults are the same as in xgboost.
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.base_score = base_score
        self.random_state = random_state

    # Loss-specific pieces, defined by subclasses.
    def _initial_margin(self, y: FloatArray) -> float:
        raise NotImplementedError

    def _gradients(self, y: FloatArray, margin: FloatArray) -> tuple[FloatArray, FloatArray]:
        raise NotImplementedError

    def _loss(self, y: FloatArray, margin: FloatArray) -> float:
        raise NotImplementedError

    def _fit_margin(self, X: FloatArray, y: FloatArray) -> None:
        self._validate_params()
        rng = np.random.default_rng(self.random_state)
        n_rows, n_features = X.shape
        tree_params = TreeParams(
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            gamma=self.gamma,
        )

        self.base_margin_ = self._initial_margin(y)
        margin = np.full(n_rows, self.base_margin_)
        self.trees_: list[XGBTree] = []
        self.train_loss_ = np.empty(self.n_estimators)

        for m in range(self.n_estimators):
            g, h = self._gradients(y, margin)
            rows = _sample(rng, n_rows, self.subsample)
            features = _sample(rng, n_features, self.colsample_bytree)
            tree = XGBTree(tree_params).fit(X, g, h, self.learning_rate, rows, features)
            margin += tree.predict(X)
            self.trees_.append(tree)
            self.train_loss_[m] = self._loss(y, margin)

    def _margin(self, X: ArrayLike) -> FloatArray:
        X = _as_features(X)
        margin = np.full(X.shape[0], self.base_margin_)
        for tree in self.trees_:
            margin += tree.predict(X)
        return margin

    def _validate_params(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if self.reg_lambda < 0.0 or self.reg_alpha < 0.0 or self.gamma < 0.0:
            raise ValueError("reg_lambda, reg_alpha and gamma must be >= 0")
        for name in ("subsample", "colsample_bytree"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")


class MiniXGBRegressor(_MiniXGBoost):
    """Squared-error regression (``objective="reg:squarederror"``)."""

    def fit(self, X: ArrayLike, y: ArrayLike) -> MiniXGBRegressor:
        self._fit_margin(_as_features(X), np.asarray(y, dtype=np.float64))
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        return self._margin(X)

    def _initial_margin(self, y: FloatArray) -> float:
        return float(np.mean(y)) if self.base_score is None else float(self.base_score)

    def _gradients(self, y: FloatArray, margin: FloatArray) -> tuple[FloatArray, FloatArray]:
        return margin - y, np.ones_like(y)

    def _loss(self, y: FloatArray, margin: FloatArray) -> float:
        return float(np.mean((y - margin) ** 2))


class MiniXGBClassifier(_MiniXGBoost):
    """Binary classification with log-loss (``objective="binary:logistic"``).

    As in xgboost, ``base_score`` is a probability; the trees work on the log-odds margin.
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> MiniXGBClassifier:
        self.classes_, y_encoded = np.unique(np.asarray(y), return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError(
                f"Only binary classification is supported, got {len(self.classes_)} classes"
            )
        self._fit_margin(_as_features(X), y_encoded.astype(np.float64))
        return self

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        p = expit(self._margin(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: ArrayLike) -> NDArray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    def _initial_margin(self, y: FloatArray) -> float:
        p = float(np.mean(y)) if self.base_score is None else float(self.base_score)
        return float(logit(np.clip(p, 1e-15, 1 - 1e-15)))

    def _gradients(self, y: FloatArray, margin: FloatArray) -> tuple[FloatArray, FloatArray]:
        p = expit(margin)
        return p - y, np.maximum(p * (1.0 - p), 1e-16)

    def _loss(self, y: FloatArray, margin: FloatArray) -> float:
        return float(np.mean(np.logaddexp(0.0, margin) - y * margin))


def _sample(rng: np.random.Generator, n: int, fraction: float) -> NDArray[np.intp]:
    if fraction >= 1.0:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max(1, int(fraction * n)), replace=False))


def _as_features(X: ArrayLike) -> FloatArray:
    # xgboost keeps features as float32; rounding the same way keeps the split points equal.
    X = np.asarray(X, dtype=np.float32).astype(np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
    return X
