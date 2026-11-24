# Loan Approval Classification

[![notebook](https://github.com/th3realAyo/loan_approval_classification/actions/workflows/notebook.yml/badge.svg)](https://github.com/th3realAyo/loan_approval_classification/actions/workflows/notebook.yml)

Predicts whether a loan application is approved from the applicant's income, employment,
credit profile and loan details. Trained on 45,000 applications
([Kaggle](https://www.kaggle.com/datasets/muhammadmusharraf444/loan-approval-dataset)), 22% approved.

## Approach

- **Policy rule + model.** Every applicant with a previous loan is rejected in the data, with no
  exceptions. That is a hard rule, not something to learn, so it is applied directly and the model
  only scores the remaining applicants.
- **One sklearn pipeline** handles log transforms, scaling and encoding, so nothing leaks between
  train and test and the saved model takes raw input.
- **Model selection** with 5-fold stratified CV across logistic regression, random forest and
  gradient boosting; the winner (`HistGradientBoostingClassifier`) is tuned with a randomised search.
- **Decision threshold** tuned on out-of-fold predictions, then checked once on a held-out test set.

## Results

Held-out test set (8,999 applications), approved class:

| | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| First version (random forest) | 0.90 | 0.71 | 0.90 | 0.79 | n/a |
| **Current** | **0.934** | **0.861** | **0.836** | **0.849** | **0.978** |

Loan-to-income ratio, interest rate, home ownership and income drive the predictions. Gender and
education have no measurable effect. Full metrics are in [`artifacts/metrics.json`](artifacts/metrics.json).

## Run

```bash
pip install -r requirements.txt
jupyter notebook loan_approval_classification_model.ipynb
```

The dataset downloads automatically on first run. CI re-runs the notebook on every push and fails
if test ROC-AUC drops below 0.95 or F1 below 0.80.

## Future work

- Fairness audit across gender and age, and a version trained without protected attributes
- SHAP explanations for individual decisions
- Cost-based threshold (price a bad approval against a missed good applicant)
- Confirm `loan_interest_rate` is known before the decision, not set after it
- Serve the exported pipeline behind a small prediction API

## Contributing

Ideas, issues and pull requests are welcome. If you're picking up one of the items above, open an
issue first so we can agree on the approach.
