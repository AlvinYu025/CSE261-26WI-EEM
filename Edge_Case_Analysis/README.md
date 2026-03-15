# Step 4: Edge Case Analysis

This folder contains the final analysis stage that tests whether mismatch features
improve listing-level prediction tasks.

## What is in this folder

- `joint_classification_regression.py`
  - Main script for classification/regression experiments.
  - Compares text-only (`TF-IDF`) vs text + mismatch features.
- `*.ipynb`
  - Exploratory notebooks for distributions, correlations, and error analysis.

## Inputs

- Listing-level file with `id` and target columns (for example `listings.csv`)
- Mismatch outputs from Step 3:
  - `../Mismatch_Score/al/llm_mismatch_score.csv`
  - `../Mismatch_Score/al/baseline_mismatch_score.csv`

## Main command examples

Run classification (default target):

```bash
python joint_classification_regression.py \
  --listings-path listings.csv \
  --mismatch-path ../Mismatch_Score/al/llm_mismatch_score.csv \
  --baseline-path ../Mismatch_Score/al/baseline_mismatch_score.csv \
  --mode classification \
  --targets joint_superhost_rating
```

Run regression:

```bash
python joint_classification_regression.py \
  --listings-path listings.csv \
  --mismatch-path ../Mismatch_Score/al/llm_mismatch_score.csv \
  --baseline-path ../Mismatch_Score/al/baseline_mismatch_score.csv \
  --mode regression \
  --regression-targets host_price_log,user_rating \
  --regression-joint
```

Run both:

```bash
python joint_classification_regression.py \
  --listings-path listings.csv \
  --mismatch-path ../Mismatch_Score/al/llm_mismatch_score.csv \
  --baseline-path ../Mismatch_Score/al/baseline_mismatch_score.csv \
  --mode both
```
