import numpy as np
import pytest
from sklearn.datasets import make_classification, make_friedman1
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from automl.boosting import ScratchGradientBoostingClassifier, ScratchGradientBoostingRegressor

PARAMS = {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3}


@pytest.fixture
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    return make_friedman1(n_samples=300, noise=1.0, random_state=0)


@pytest.fixture
def classification_data() -> tuple[np.ndarray, np.ndarray]:
    return make_classification(n_samples=300, n_features=6, n_informative=4, random_state=0)


def test_regressor_matches_sklearn(regression_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = regression_data
    ours = ScratchGradientBoostingRegressor(**PARAMS, random_state=0).fit(X, y)
    reference = GradientBoostingRegressor(**PARAMS, random_state=0).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), reference.predict(X), rtol=1e-6, atol=1e-8)


def test_classifier_matches_sklearn(classification_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = classification_data
    ours = ScratchGradientBoostingClassifier(**PARAMS, random_state=0).fit(X, y)
    reference = GradientBoostingClassifier(**PARAMS, random_state=0).fit(X, y)
    np.testing.assert_allclose(ours.predict_proba(X), reference.predict_proba(X), atol=1e-6)


def test_regression_training_loss_never_increases(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    # With squared error and learning_rate <= 1 every tree can only reduce the training loss.
    X, y = regression_data
    model = ScratchGradientBoostingRegressor(**PARAMS).fit(X, y)
    assert np.all(np.diff(model.train_loss_) <= 1e-12)
    assert model.train_loss_[-1] < np.var(y) / 5


def test_classification_training_loss_goes_down(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = classification_data
    model = ScratchGradientBoostingClassifier(**PARAMS).fit(X, y)
    baseline = np.log(2)  # log-loss of always predicting 50/50 on balanced classes
    assert model.train_loss_[-1] < model.train_loss_[0] < baseline


def test_staged_predict_ends_with_predict(regression_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = regression_data
    model = ScratchGradientBoostingRegressor(n_estimators=10).fit(X, y)
    stages = list(model.staged_predict(X))
    assert len(stages) == 10
    np.testing.assert_allclose(stages[-1], model.predict(X))


def test_classifier_keeps_original_labels(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = classification_data
    labels = np.where(y == 1, "churn", "stay")
    model = ScratchGradientBoostingClassifier(n_estimators=20).fit(X, labels)
    assert set(model.predict(X)) <= {"churn", "stay"}
    assert list(model.classes_) == ["churn", "stay"]


def test_classifier_rejects_multiclass() -> None:
    X = np.arange(12, dtype=float).reshape(6, 2)
    with pytest.raises(ValueError, match="binary"):
        ScratchGradientBoostingClassifier().fit(X, [0, 1, 2, 0, 1, 2])


def test_subsample_is_reproducible_and_changes_the_model(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = regression_data

    def fit(subsample: float, seed: int) -> np.ndarray:
        model = ScratchGradientBoostingRegressor(
            n_estimators=20, subsample=subsample, random_state=seed
        )
        return model.fit(X, y).predict(X)

    np.testing.assert_array_equal(fit(0.5, seed=1), fit(0.5, seed=1))
    assert not np.allclose(fit(0.5, seed=1), fit(1.0, seed=1))


@pytest.mark.parametrize(
    "params",
    [{"n_estimators": 0}, {"learning_rate": 0.0}, {"subsample": 0.0}, {"subsample": 1.5}],
)
def test_invalid_params_are_rejected(
    params: dict[str, float], regression_data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = regression_data
    with pytest.raises(ValueError):
        ScratchGradientBoostingRegressor(**params).fit(X, y)
