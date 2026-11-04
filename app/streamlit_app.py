"""AutoML demo: from a table to an honest leaderboard and SHAP explanations.

uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
# On Streamlit Community Cloud the package is used from the repository, not installed.
sys.path.insert(0, str(APP_DIR.parent / "src"))

from automl.data import DatasetProfile, profile_dataset  # noqa: E402
from automl.engine import AutoML  # noqa: E402
from automl.explain import explain  # noqa: E402
from automl.llm import build_chat_prompt, check_answer  # noqa: E402
from automl.llm.propose import automatic_flags  # noqa: E402
from automl.metrics import get_metric  # noqa: E402
from automl.plots import plot_importance, plot_row  # noqa: E402
from automl.preprocess import plan_features  # noqa: E402
from automl.task import TaskSpec, TaskSpecError, guess_target, infer_problem_type  # noqa: E402

EXAMPLES = {
    "Titanic (OpenML, 1,309 passengers)": {
        "file": "titanic.csv",
        "target": "survived",
        "description": "Predict which passengers survived the Titanic disaster.",
        "recorded_reply": "titanic_claude_reply.json",
    },
    "Customer churn (synthetic, 400 customers)": {
        "file": "churn.csv",
        "target": "churned",
        "description": "Predict which customers will cancel their contract.",
        "recorded_reply": None,
    },
}
UPLOAD = "Upload a CSV"
REPO_URL = "https://github.com/maksympostoiuk/AutoML"


@st.cache_data(show_spinner=False)
def load_example(file: str) -> pd.DataFrame:
    return pd.read_csv(APP_DIR / "examples" / file)


def main() -> None:
    st.set_page_config(page_title="AutoML", layout="wide")
    st.title("AutoML for tables, honestly")
    st.markdown(
        "XGBoost tuned with Optuna next to simple baselines, a test split that never "
        "influences a choice, leakage checks, SHAP explanations, and an LLM that reads the "
        f"columns' meaning. [Source code and benchmark]({REPO_URL})."
    )

    with st.sidebar:
        st.header("Data")
        choice = st.radio("Dataset", [*EXAMPLES, UPLOAD], label_visibility="collapsed")
        budget = st.slider("Training time budget, seconds", 10, 60, 20, step=5)
        st.caption("Small tables use 5-fold cross-validation, which takes longer per trial.")

    example = EXAMPLES.get(choice)
    if example:
        df = load_example(example["file"])
    else:
        uploaded = st.file_uploader("CSV file", type="csv")
        if uploaded is None:
            st.info("Upload a CSV file with one row per example and a column to predict.")
            return
        df = pd.read_csv(uploaded)

    profile = profile_dataset(df)
    data_section(df, profile)
    spec = task_section(df, profile, example)
    if spec is None:
        return
    automl = train_section(df, spec, budget, key=f"{choice}|{spec.model_dump_json()}|{budget}")
    if automl is None:
        return
    explain_section(df, automl)


def data_section(df: pd.DataFrame, profile: DatasetProfile) -> None:
    st.header("1. The data")
    st.caption(
        f"{profile.n_rows:,} rows × {profile.n_columns} columns. Every column gets a kind; "
        "columns the models cannot use say why."
    )
    table = pd.DataFrame(
        [
            {
                "column": c.name,
                "kind": c.kind.value,
                "missing": f"{c.missing_share:.0%}",
                "distinct": c.n_unique,
                "note": c.reason,
            }
            for c in profile.columns
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")
    with st.expander("First rows"):
        st.dataframe(df.head(20), width="stretch")


def task_section(
    df: pd.DataFrame, profile: DatasetProfile, example: dict | None
) -> TaskSpec | None:
    st.header("2. The task")
    mode = st.radio(
        "How to set up the task",
        ["Describe it in words (LLM via a chat window)", "Automatic rules"],
        horizontal=True,
    )
    if mode == "Automatic rules":
        return rules_task(df, profile, example)
    return llm_task(df, profile, example)


def rules_task(df: pd.DataFrame, profile: DatasetProfile, example: dict | None) -> TaskSpec | None:
    from automl.task import infer_task_spec

    default = example["target"] if example else guess_target(profile)[0]
    target = st.selectbox("Column to predict", profile.names, index=profile.names.index(default))
    try:
        spec = infer_task_spec(df, target=target, profile=profile)
        spec.check_against(df, profile)
    except TaskSpecError as error:
        st.error("This column cannot be predicted: " + "; ".join(error.problems))
        return None
    show_spec(spec, profile)
    return spec


def llm_task(df: pd.DataFrame, profile: DatasetProfile, example: dict | None) -> TaskSpec | None:
    st.markdown(
        "The LLM gets the column profile above (never the rows) and your description. Copy "
        "the prompt into a [Claude chat](https://claude.ai), paste the reply below, and the "
        "code checks it against the data."
    )
    description = st.text_input(
        "What do you want to predict?", value=example["description"] if example else ""
    )
    hint = st.selectbox(
        "Column to predict, if you know it (adds the leakage checks for it to the prompt)",
        ["Let the LLM decide", *profile.names],
        index=1 + profile.names.index(example["target"]) if example else 0,
    )
    if not description.strip():
        return None
    flags = None
    if hint != "Let the LLM decide":
        try:
            problem_type, _ = infer_problem_type(df[hint])
            flags = automatic_flags(df, hint, problem_type.value, profile)
        except TaskSpecError:
            flags = None
    with st.expander("Prompt to copy into the chat", expanded=False):
        st.code(build_chat_prompt(profile, description, flags), language="markdown")

    reply_key = f"reply|{example['file'] if example else 'upload'}"
    if example and example["recorded_reply"] and st.button("Use a recorded reply from Claude"):
        st.session_state[reply_key] = (APP_DIR / "examples" / example["recorded_reply"]).read_text(
            encoding="utf-8"
        )
    reply = st.text_area("The chat's reply (JSON)", key=reply_key, height=180)
    if not reply.strip():
        return None

    result = check_answer(reply, df, profile)
    if not result.ok:
        st.error(
            "The reply does not fit the data:\n\n" + "\n".join(f"- {p}" for p in result.problems)
        )
        st.markdown("Send this follow-up to the same chat and paste the new reply:")
        st.code(result.repair_prompt(), language="markdown")
        return None
    st.success("The reply fits the data.")
    show_spec(result.spec, profile)
    return result.spec


def show_spec(spec: TaskSpec, profile: DatasetProfile) -> None:
    plan = plan_features(profile, spec)
    st.markdown(
        f"**Predict** `{spec.target}` · **{spec.problem_type.value}** · metric "
        f"**{spec.metric}** · {len(plan.columns)} feature columns"
    )
    if plan.ignored:
        st.dataframe(
            pd.DataFrame({"not used": list(plan.ignored), "why": list(plan.ignored.values())}),
            hide_index=True,
            width="stretch",
        )
    for note in spec.notes:
        st.caption(note)


def train_section(df: pd.DataFrame, spec: TaskSpec, budget: int, key: str) -> AutoML | None:
    st.header("3. Training")
    st.caption(
        "Five candidates, simplest first. Each is tuned on validation data, the best validation "
        "score wins, and every model is scored once on a test split no choice has seen."
    )
    if st.button("Train models", type="primary"):
        with st.spinner(f"Training for about {budget} seconds…"):
            st.session_state["automl"] = AutoML(spec=spec, time_budget=budget, n_trials=60).fit(df)
            st.session_state["automl_key"] = key
    if st.session_state.get("automl_key") != key:
        return None

    automl: AutoML = st.session_state["automl"]
    metric = get_metric(spec.metric)
    board = automl.leaderboard_.copy()
    board["selected"] = board["selected"].map({True: "✓", False: ""})
    board = board.round({c: 4 for c in board.columns if c.startswith(("valid_", "test_"))})
    sizes = automl.split_sizes_
    st.markdown(
        f"Chosen: **{automl.model_.model_name}** · selection by {automl.validation_} on "
        f"{sizes['train'] + sizes['valid']:,} rows · test on {sizes['test']:,} rows · "
        f"{metric.name}: {'higher' if metric.greater_is_better else 'lower'} is better"
    )
    st.dataframe(board, hide_index=True, width="stretch")
    return automl


def explain_section(df: pd.DataFrame, automl: AutoML) -> None:
    st.header("4. Why the model predicts what it predicts")
    explanation = explain(automl.model_, df, max_rows=len(df))
    left, right = st.columns(2)
    with left:
        st.pyplot(plot_importance(explanation), width="stretch")
    with right:
        row = st.number_input("Row to explain", 0, len(df) - 1, 0)
        st.markdown(f"**Row {row}:** {explanation.describe_row(int(row))}")
        st.pyplot(plot_row(explanation, int(row)), width="stretch")

    predictions = pd.DataFrame({"prediction": automl.model_.predict(df)})
    if automl.model_.classes is not None:
        predictions = pd.concat(
            [predictions, automl.model_.predict_proba(df).add_prefix("proba_")], axis=1
        )
    st.download_button(
        "Download predictions (CSV)",
        predictions.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
    )


main()
