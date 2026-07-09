# Loan Approval Classification

Predicting loan approval decisions from applicant demographic, financial,
employment, and credit-related data, using classical ML classification models.

## Status

🚧 In progress — feature encoding complete, currently investigating data
leakage flagged during feature importance analysis (see Notes below) before
finalizing model comparison.

## Data

Source: [Kaggle — loan-approval-dataset](https://www.kaggle.com/datasets/muhammadmusharraf444/loan-approval-dataset)
by muhammadmusharraf444.

Data is **not** committed to this repo, it's downloaded at runtime via
`kagglehub`. Run the notebook top to bottom to fetch it fresh.

## Project structure

```
notebooks/    → main analysis + modeling notebook (.ipynb)
data/         → runtime download target (gitignored, not tracked)
reports/      → figures / exported charts, if any
```

## Setup

```bash
pip install -r requirements.txt
```

You'll need a Kaggle account and API token configured for `kagglehub` to
download the dataset (`kagglehub.login()` in the first cell handles this
interactively in Colab).

## Pipeline overview

1. Data cleaning & standardization (column names, casing)
2. Univariate + exploratory analysis
3. Train/test split (stratified on target, 80/20)
4. Feature encoding:
   - Ordinal: `education` (explicit rank order)
   - Binary: `gender`, `previous_loan`
   - One-hot: `loan_intent`, `home_ownership`
   - Numeric: log-transform (`person_income`, `loan_amount`) + `StandardScaler`
5. Baseline model: Logistic Regression (`class_weight="balanced"`)
6. Comparison model: Random Forest (`class_weight="balanced"`)
7. Feature importance review

## Notes / open issues

- **`previous_loan` leakage flagged**: crosstab against the target shows a
  deterministic split (`previous_loan == "yes"` → 100% rejected, 0 exceptions).
  This is being treated as leakage — likely a hard rule baked into how labels
  were generated rather than a genuine independent predictor. Current
  model metrics (below) are **inflated** and will be re-run once this is
  resolved.
- `credit_history` and `loan_percentage` are being checked for the same
  pattern before finalizing the feature set.
- `loan_interest_rate` was investigated for leakage and cleared — overlapping
  distributions across both outcome classes with no hard cutoff, consistent
  with risk-based pricing rather than a post-decision artifact.

## Results (baseline — pre-leakage-fix, will be updated)

| Model | Accuracy | Precision (approved) | Recall (approved) | F1 (approved) |
|---|---|---|---|---|
| Logistic Regression | 0.86 | 0.63 | 0.93 | 0.75 |
| Random Forest | 0.90 | 0.71 | 0.90 | 0.79 |

## License

Add a license if you plan to make this public (MIT is a common default for
portfolio/coursework projects).
