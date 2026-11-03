import numpy as np
import pytest
import xgboost as xgb
from sklearn.datasets import make_classification, make_friedman1

from automl.boosting import MiniXGBClassifier, MiniXGBRegressor, XGBTree
from automl.boosting.mini_xgboost import TreeParams, leaf_score, leaf_weight

BASE = {"n_estimators": 20, "max_depth": 4, "learning_rate": 0.3}


@pytest.fixture(scope="module")
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    return make_friedman1(n_samples=300, noise=1.0, random_state=0)


@pytest.fixture(scope="module")
def classification_data() -> tuple[np.ndarray, np.ndarray]:
    return make_classification(n_samples=300, n_features=6, n_informative=4, random_state=0)


@pytest.fixture(scope="module")
def data_with_missing(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    X, y = classification_data
    X = X.copy()
    X[np.random.default_rng(0).random(X.shape) < 0.2] = np.nan
    return X, y


# ----- the formulas on a case small enough to check by hand ---------------------------------


def test_leaf_formulas() -> None:
    p = TreeParams(reg_lambda=1.0)
    assert leaf_weight(G=-6.0, H=2.0, p=p) == pytest.approx(2.0)  # -G / (H + lambda)
    assert leaf_score(G=-6.0, H=2.0, p=p) == pytest.approx(12.0)  # G^2 / (H + lambda)

    l1 = TreeParams(reg_lambda=1.0, reg_alpha=2.0)
    assert leaf_weight(G=-6.0, H=2.0, p=l1) == pytest.approx(4 / 3)  # |G| shrunk by alpha
    assert leaf_weight(G=-1.5, H=2.0, p=l1) == 0.0  # small gradients are zeroed by L1

    assert leaf_weight(G=-6.0, H=0.5, p=TreeParams(min_child_weight=1.0)) == 0.0


def test_one_tree_by_hand() -> None:
    # y is a step function. Starting from the mean (3), gradients F - y are [2, 2, -2, -2].
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    g = np.array([2.0, 2.0, -2.0, -2.0])
    h = np.ones(4)
    tree = XGBTree(TreeParams(max_depth=1, reg_lambda=0.0)).fit(X, g, h, learning_rate=1.0)

    assert tree.threshold[0] == 2.5  # the split goes between the two groups
    assert tree.gain[0] == pytest.approx(16.0)  # 4^2/2 + 4^2/2 - 0^2/4
    np.testing.assert_allclose(tree.predict(X), [-2.0, -2.0, 2.0, 2.0])  # -G/H in each leaf


def test_gamma_prunes_splits_that_gain_too_little() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    g, h = np.array([2.0, 2.0, -2.0, -2.0]), np.ones(4)
    kept = XGBTree(TreeParams(max_depth=1, reg_lambda=0.0, gamma=15.0)).fit(X, g, h, 1.0)
    pruned = XGBTree(TreeParams(max_depth=1, reg_lambda=0.0, gamma=17.0)).fit(X, g, h, 1.0)
    assert kept.n_leaves == 2
    assert pruned.n_leaves == 1  # the gain of 16 is below gamma = 17


def test_missing_values_learn_a_default_direction() -> None:
    # Rows with NaN behave like the large values, so they should be sent right.
    X = np.array([[1.0], [2.0], [3.0], [4.0], [np.nan], [np.nan]])
    g = np.array([2.0, 2.0, -2.0, -2.0, -2.0, -2.0])
    tree = XGBTree(TreeParams(max_depth=1, reg_lambda=0.0)).fit(X, g, np.ones(6), 1.0)
    assert tree.default_left[0] is False
    assert tree.predict(np.array([[np.nan]]))[0] == pytest.approx(2.0)


def test_split_on_missingness_alone() -> None:
    # All observed values are equal, so the only useful split is "missing vs. present".
    X = np.array([[1.0], [1.0], [1.0], [np.nan], [np.nan]])
    g = np.array([1.0, 1.0, 1.0, -1.5, -1.5])
    tree = XGBTree(TreeParams(max_depth=1, reg_lambda=0.0)).fit(X, g, np.ones(5), 1.0)
    assert tree.n_leaves == 2
    np.testing.assert_allclose(tree.predict(X), [-1.0, -1.0, -1.0, 1.5, 1.5])


# ----- the same trees as the real xgboost ---------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"gamma": 20.0},
        {"reg_alpha": 5.0, "reg_lambda": 3.0},
        {"min_child_weight": 10.0},
    ],
    ids=["defaults", "gamma", "l1+l2", "min_child_weight"],
)
def test_regressor_matches_xgboost(
    regression_data: tuple[np.ndarray, np.ndarray], extra: dict[str, float]
) -> None:
    X, y = regression_data
    params = {**BASE, **extra, "base_score": float(np.mean(y))}
    ours = MiniXGBRegressor(**params).fit(X, y)
    reference = xgb.XGBRegressor(**params, tree_method="exact", n_jobs=1).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), reference.predict(X), rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize(
    "extra",
    [{}, {"gamma": 1.0}, {"min_child_weight": 3.0}],
    ids=["defaults", "gamma", "min_child_weight"],
)
@pytest.mark.parametrize("with_missing", [False, True], ids=["complete", "missing"])
def test_classifier_matches_xgboost(
    classification_data: tuple[np.ndarray, np.ndarray],
    data_with_missing: tuple[np.ndarray, np.ndarray],
    extra: dict[str, float],
    with_missing: bool,
) -> None:
    X, y = data_with_missing if with_missing else classification_data
    params = {**BASE, **extra, "base_score": float(np.mean(y))}
    ours = MiniXGBClassifier(**params).fit(X, y)
    reference = xgb.XGBClassifier(**params, tree_method="exact", n_jobs=1).fit(X, y)
    np.testing.assert_allclose(ours.predict_proba(X), reference.predict_proba(X), atol=1e-5)


def test_regularisation_makes_trees_smaller(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = regression_data

    def leaves(**extra: float) -> int:
        model = MiniXGBRegressor(**BASE, **extra).fit(X, y)
        return sum(tree.n_leaves for tree in model.trees_)

    plain = leaves()
    assert leaves(gamma=20.0) < plain
    assert leaves(min_child_weight=10.0) < plain


# ----- behaviour ----------------------------------------------------------------------------


def test_training_loss_goes_down(classification_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = classification_data
    model = MiniXGBClassifier(**BASE).fit(X, y)
    assert model.train_loss_[-1] < model.train_loss_[0] < np.log(2)


def test_classifier_keeps_original_labels(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = classification_data
    labels = np.where(y == 1, "churn", "stay")
    model = MiniXGBClassifier(n_estimators=5).fit(X, labels)
    assert set(model.predict(X)) <= {"churn", "stay"}


def test_sampling_is_reproducible(regression_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = regression_data

    def predict(seed: int) -> np.ndarray:
        model = MiniXGBRegressor(
            n_estimators=5, subsample=0.7, colsample_bytree=0.5, random_state=seed
        )
        return model.fit(X, y).predict(X)

    np.testing.assert_array_equal(predict(1), predict(1))
    assert not np.allclose(predict(1), predict(2))


def test_to_text_shows_the_tree() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    tree = XGBTree(TreeParams(max_depth=1, reg_lambda=0.0)).fit(
        X, np.array([2.0, 2.0, -2.0, -2.0]), np.ones(4), 1.0
    )
    text = tree.to_text(["tenure"])
    assert "[tenure < 2.5] gain = 16.000" in text
    assert text.count("leaf = ") == 2


@pytest.mark.parametrize(
    "params",
    [{"n_estimators": 0}, {"learning_rate": 0.0}, {"gamma": -1.0}, {"colsample_bytree": 0.0}],
)
def test_invalid_params_are_rejected(
    params: dict[str, float], regression_data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = regression_data
    with pytest.raises(ValueError):
        MiniXGBRegressor(**params).fit(X, y)
