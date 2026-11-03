import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def churn_df() -> pd.DataFrame:
    """A small churn table with every kind of column the profiler must recognise."""
    rng = np.random.default_rng(0)
    n = 400
    tenure = rng.integers(1, 72, n)
    monthly = rng.normal(70, 20, n).round(2)
    contract = rng.choice(["month-to-month", "one year", "two year"], n, p=[0.5, 0.3, 0.2])
    score = -0.05 * tenure + 0.02 * monthly + (contract == "month-to-month") * 1.5
    churned = np.where(rng.random(n) < 1 / (1 + np.exp(-(score - 0.5))), "yes", "no")

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1000, 1000 + n),
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "contract": contract,
            "payment_method": rng.choice(["card", "bank", "check", None], n),
            "has_partner": rng.random(n) < 0.5,
            "signup_date": pd.date_range("2020-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
            "support_notes": [
                f"Customer called about billing issue number {i} and asked for a callback"
                for i in range(n)
            ],
            "country": "CZ",
            # Recorded after the customer left: a textbook leak.
            "cancellation_date": np.where(
                churned == "yes",
                pd.Series(pd.date_range("2024-01-01", periods=n, freq="D")).dt.strftime("%Y-%m-%d"),
                None,
            ),
            "churned": churned,
        }
    )
    df.loc[rng.choice(n, 30, replace=False), "monthly_charges"] = np.nan
    return df
