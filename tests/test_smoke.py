import numpy as np
import pytest
import xgboost as xgb
from sklearn.datasets import make_classification

import xautoml
from xautoml.cli import main


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert xautoml.__version__ in capsys.readouterr().out


def test_core_stack_trains_xgboost() -> None:
    """The ML stack is installed and XGBoost learns something on an easy problem."""
    X, y = make_classification(n_samples=400, n_features=8, random_state=0)
    model = xgb.XGBClassifier(n_estimators=30, max_depth=3, random_state=0)
    model.fit(X[:300], y[:300])
    accuracy = np.mean(model.predict(X[300:]) == y[300:])
    assert accuracy > 0.8
