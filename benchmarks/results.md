# Benchmark

<!-- GENERATED: everything below is rebuilt by benchmarks/report.py -->

## Test scores by model

Mean ± standard deviation over seeds, on the test split (never used for any choice). Small datasets use 5-fold CV for tuning, large ones a holdout split. ↑ higher is better, ↓ lower is better; the best mean is in bold.

| dataset | rows | task | metric | dummy | linear | random_forest | xgboost_default | xgboost |
|---|---:|---|---|---:|---:|---:|---:|---:|
| blood-transfusion-service-center | 748 | binary | roc_auc ↑ | 0.500 ± – | **0.731 ± –** | 0.602 ± – | 0.652 ± – | 0.680 ± – |

## Does cross-validation pick better models?

Small datasets, same seeds. Each cell is the mean test score of the model a strategy picks. *Share of the achievable gain*: (picked − dummy) / (best − dummy), where *best* is the model with the best test score in hindsight; 100% means the strategy always picked it.

_No datasets with both holdout and CV runs yet._

Which model each strategy picked (runs):



## How much does tuning add?

| dataset | metric | xgboost_default | xgboost (tuned) | tuned better in | trials |
|---|---|---:|---:|---:|---:|
| blood-transfusion-service-center | roc_auc | 0.652 | 0.680 | 1 of 1 seeds | 60 |

_1 runs, 1 datasets, generated from `benchmarks/results/results.csv`._
