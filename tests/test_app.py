from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture
def app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=120)


def test_first_screen_renders(app: AppTest) -> None:
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.title[0].value == "AutoML for tables, honestly"
    assert {h.value for h in app.header} == {"Data", "1. The data", "2. The task"}


def test_recorded_claude_reply_then_training(app: AppTest) -> None:
    app.run()
    app.button[0].click().run()  # "Use a recorded reply from Claude"
    assert not app.exception, [e.value for e in app.exception]
    assert app.success[0].value == "The reply fits the data."

    app.sidebar.slider[0].set_value(10)
    next(b for b in app.button if b.label == "Train models").click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert "4. Why the model predicts what it predicts" in [h.value for h in app.header]
    board = app.dataframe[-1].value
    assert set(board["model"]) == {
        "dummy",
        "logistic_regression",
        "random_forest",
        "xgboost_default",
        "xgboost",
    }


def test_rules_mode_and_a_bad_reply(app: AppTest) -> None:
    app.run()
    app.text_area[0].set_value('{"target": "nope"}').run()
    assert app.error, "an invalid reply is explained, not accepted"

    task_mode = next(r for r in app.radio if r.label == "How to set up the task")
    task_mode.set_value("Automatic rules").run()
    assert not app.exception, [e.value for e in app.exception]
    assert any("Predict" in m.value for m in app.markdown)
