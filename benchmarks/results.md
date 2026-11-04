# Benchmark

The AutoML engine run end to end, the way a user runs it, on 11 public
[OpenML](https://www.openml.org/) datasets: 748 to 48,842 rows, binary and multiclass
classification and regression, numeric and categorical features. Three seeds per dataset;
every seed is a different train/validation/test split. 60 runs, 48 minutes on a laptop.

```bash
uv run python benchmarks/run.py     # resumable; --quick for a smoke test
uv run python benchmarks/report.py  # rebuilds the tables below
```

Settings: 45 s budget for small datasets (120 s above 5,000 rows), at most 60 Optuna trials.
Every candidate is refitted on train + validation and scored once on the test split; the test
split never influences any choice.

## Findings

1. **Gradient boosting wins on larger data, linear models win on small data.** Tuned XGBoost
   has the best test score on 5 of 11 datasets, all of them with 1,000+ rows. On the three
   smallest (blood-transfusion, diabetes, vehicle) a regularised logistic regression is best.
   This is why the engine always trains simple baselines next to XGBoost.
2. **Tuning helps, a little.** Tuned XGBoost beats XGBoost with default settings in 30 of 33
   runs, by +0.002 to +0.036 ROC AUC (house_prices: 33,296 -> 28,421 RMSE). On `car` it is
   worse: with 5-fold CV only 5 trials fitted into 45 s, too few for the search to beat
   the defaults.
3. **Cross-validation did not pick better models on average, and neither did the
   one-standard-error rule.** All three strategies capture about 97% of the achievable gain.
   The 1-SE rule (pick the simplest model within one standard error of the best) helps where
   it should, on tiny noisy data (blood-transfusion +0.029 ROC AUC), but costs 16% RMSE on
   house_prices by choosing a forest over tuned XGBoost too often. Measured, not assumed: the
   engine keeps 5-fold CV for small data (steadier estimates, a spread per candidate) and
   chooses by the best validation score; `selection="one_se"` stays available.
4. **The benchmark found a real bug.** OpenML credit-g has categories such as `<0` and
   `0<=X<200`; they became one-hot feature names that XGBoost rejects. Feature names are now
   made safe (`automl.preprocess.safe_name`), with a regression test.

Limitations: three seeds give a rough spread, not significance; with 5-fold CV the 45 s budget
leaves few Optuna trials on wider datasets (`trials` column below); no dataset here has the
time-ordered structure where random splits mislead.

<!-- GENERATED: everything below is rebuilt by benchmarks/report.py -->

## Test scores by model

Mean ± standard deviation over seeds, on the test split (never used for any choice). Small datasets use 5-fold CV for tuning, large ones a holdout split. ↑ higher is better, ↓ lower is better; the best mean is in bold.

| dataset | rows | task | metric | dummy | linear | random_forest | xgboost_default | xgboost |
|---|---:|---|---|---:|---:|---:|---:|---:|
| blood-transfusion-service-center | 748 | binary | roc_auc ↑ | 0.500 ± 0.000 | **0.753 ± 0.032** | 0.675 ± 0.064 | 0.699 ± 0.049 | 0.717 ± 0.046 |
| diabetes | 768 | binary | roc_auc ↑ | 0.500 ± 0.000 | **0.832 ± 0.047** | 0.827 ± 0.040 | 0.793 ± 0.031 | 0.829 ± 0.019 |
| vehicle | 846 | multiclass | log_loss ↓ | 1.386 ± 0.000 | **0.391 ± 0.016** | 0.504 ± 0.030 | 0.491 ± 0.067 | 0.464 ± 0.059 |
| credit-g | 1,000 | binary | roc_auc ↑ | 0.500 ± 0.000 | 0.788 ± 0.075 | 0.788 ± 0.054 | 0.779 ± 0.052 | **0.794 ± 0.064** |
| house_prices | 1,460 | regression | rmse ↓ | 83,104 ± 1,591 | 36,655 ± 7,583 | 29,862 ± 3,661 | 33,296 ± 5,585 | **28,421 ± 2,679** |
| car | 1,728 | multiclass | log_loss ↓ | 0.838 ± 0.000 | 0.158 ± 0.043 | 0.174 ± 0.018 | **0.028 ± 0.020** | 0.050 ± 0.036 |
| kc1 | 2,109 | binary | roc_auc ↑ | 0.500 ± 0.000 | 0.798 ± 0.005 | **0.840 ± 0.002** | 0.815 ± 0.007 | 0.835 ± 0.005 |
| abalone | 4,177 | regression | rmse ↓ | 3.310 ± 0.188 | 2.278 ± 0.121 | 2.197 ± 0.093 | 2.224 ± 0.057 | **2.184 ± 0.074** |
| phoneme | 5,404 | binary | roc_auc ↑ | 0.500 ± 0.000 | 0.812 ± 0.012 | **0.963 ± 0.006** | 0.954 ± 0.006 | 0.961 ± 0.004 |
| bank-marketing | 45,211 | binary | roc_auc ↑ | 0.500 ± 0.000 | 0.905 ± 0.002 | 0.930 ± 0.002 | 0.933 ± 0.001 | **0.936 ± 0.002** |
| adult | 48,842 | binary | roc_auc ↑ | 0.500 ± 0.000 | 0.907 ± 0.001 | 0.906 ± 0.002 | 0.930 ± 0.002 | **0.932 ± 0.002** |

## Does cross-validation pick better models?

Small datasets, same seeds. Each cell is the mean test score of the model a strategy picks. *Share of the achievable gain*: (picked − dummy) / (best − dummy), where *best* is the model with the best test score in hindsight; 100% means the strategy always picked it.

| dataset | holdout, best | CV, best (default) | CV, 1-SE |
|---|---:|---:|---:|
| blood-transfusion-service-center | 0.718 | 0.717 | 0.746 |
| diabetes | 0.837 | 0.829 | 0.823 |
| vehicle | 0.382 | 0.391 | 0.391 |
| credit-g | 0.783 | 0.794 | 0.788 |
| house_prices | 27,979 | 28,421 | 33,025 |
| car | 0.024 | 0.027 | 0.028 |
| kc1 | 0.821 | 0.836 | 0.840 |
| abalone | 2.185 | 2.184 | 2.232 |
| phoneme | 0.963 | 0.963 | 0.963 |
| **share of the achievable gain** | **97.0%** | **97.4%** | **96.9%** |

Which model each strategy picked (runs):

| strategy | linear | random_forest | xgboost_default | xgboost |
|---|---:|---:|---:|---:|
| holdout, best | 3 | 0 | 1 | 23 |
| CV, best (default) | 3 | 4 | 2 | 18 |
| CV, 1-SE | 11 | 7 | 3 | 6 |

## How much does tuning add?

| dataset | metric | xgboost_default | xgboost (tuned) | tuned better in | trials |
|---|---|---:|---:|---:|---:|
| blood-transfusion-service-center | roc_auc | 0.699 | 0.717 | 3 of 3 seeds | 60 |
| diabetes | roc_auc | 0.793 | 0.829 | 3 of 3 seeds | 54 |
| vehicle | log_loss | 0.491 | 0.464 | 2 of 3 seeds | 8 |
| credit-g | roc_auc | 0.779 | 0.794 | 3 of 3 seeds | 23 |
| house_prices | rmse | 33,296 | 28,421 | 3 of 3 seeds | 6 |
| car | log_loss | 0.028 | 0.050 | 1 of 3 seeds | 5 |
| kc1 | roc_auc | 0.815 | 0.835 | 3 of 3 seeds | 50 |
| abalone | rmse | 2.224 | 2.184 | 3 of 3 seeds | 36 |
| phoneme | roc_auc | 0.954 | 0.961 | 3 of 3 seeds | 18 |
| bank-marketing | roc_auc | 0.933 | 0.936 | 3 of 3 seeds | 57 |
| adult | roc_auc | 0.930 | 0.932 | 3 of 3 seeds | 41 |

_60 runs, 11 datasets, generated from `benchmarks/results/results.csv`._
