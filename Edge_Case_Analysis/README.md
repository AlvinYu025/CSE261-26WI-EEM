# README_ADDED

## Overall Goal
The core objective of this newly added and refactored code is to validate the effectiveness of `mismatch` features, particularly their incremental value across different tasks.

Since `mismatch` originates from the alignment/divergence between host-side and guest-side information, the experimental design aims to cover signals from both sides rather than relying on a single-sided metric.

## 1) `ny_joint_classification_regression.py`
### Why
This script serves as the main experiment entry point. The goal is not to run a single metric, but to simultaneously examine on the same NY dataset:
1. Whether `mismatch` outperforms or at least complements the baseline (GloVe route) signal in classification tasks.
2. Whether `mismatch` outperforms or at least complements the baseline (GloVe route) signal in regression tasks.

If a target cannot be aligned with the expectation-experience mechanism, or if results are unstable, it will not be used as a primary conclusion target.

### Classification Design
Uses the joint label `joint_superhost_rating` (4 classes), combining host and guest signals:
1. `host_is_superhost` (host side)
2. `review_scores_rating` binarized by threshold (`<= 4.8` vs `> 4.8`, guest side)

4-class definition:
1. non-superhost + high-rating
2. non-superhost + low-rating
3. superhost + high-rating
4. superhost + low-rating

The motivation is to align the classification target with the two-sided definition of `mismatch`, rather than predicting a single-sided variable.

### Regression Design
Regression defaults to a joint target (rather than two separate regressions):
1. host side: `host_price_log`
2. guest side: `user_rating` (`review_scores_rating`)

Processing: each is z-scored separately, then averaged to form a joint regression target.

### What Is Compared
The script uniformly compares three routes:
1. `TF-IDF only`
2. `TF-IDF + glove_mismatch` (from `baseline_mismatch_score.csv`)
3. `TF-IDF + customized_mismatch` (`mismatch_proxy/mabs/mover/munder`)

Default filter: `n_reviews > 1`.

## 2) `mismatch_balanced_classification.py`
### Why
This script focuses on validating the interpretable gain of mismatch under a class-balanced setting.

### What Was Done
1. Ran balanced classification experiments across three datasets.
2. Results are recorded in the comments at the end of the file.

### Interpretation
1. The balanced version makes it easier to observe mismatch contributions by reducing the majority-class masking effect.
2. The imbalanced version better reflects the true distribution, but improvements remain observable in this setting as well.

## 3) `mismatch_balanced_stratified_regression.py`
### Why
Balanced vs. imbalanced comparison is also conducted on the regression side to check the robustness of conclusions.

### Finding
Overall differences are small, and the imbalanced version tends to perform better.

### Possible Reason
Forcing balance in regression tasks alters the original target distribution and variance structure, weakening the model's fit to the true data-generating mechanism. Preserving the original distribution (imbalanced) is generally closer to the true signal.

## 4) `regression_correlation.ipynb`
### What Was Updated
1. Fixed CSV loading paths and retested.

Data can be downloaded from Drive.

## 5) `tfidf_feature_ablation.py`
### Why
Ablation experiment comparing the relative contribution of `TF-IDF + mismatch` vs. `TF-IDF + other features`.

### Finding
1. Mismatch provides improvement, but in some cases underperforms certain other structured features.
2. One possible reason is that the prediction target is `rating` (more guest-side), which introduces target-side bias.

### Value
Even if not the strongest single feature, mismatch still provides independent incremental information, which remains valuable for subsequent joint tasks.

## 6) `tfidf_mismatch_5class_compare.py`
### Why
A lightweight sanity check experiment.

### What Was Done
1. Binned `rating` into bins with roughly equal sample sizes.
2. Compared `TF-IDF` vs. `TF-IDF + mismatch`.

Purpose: quick directional validation, not intended as a final main experiment.

## 7) `ny_mismatch_classification.py`
Early NY classification script, primarily used for mismatch classification baseline replication and quick reference comparison.

## 8) `mismatch_stratified_regression.py`
Stratified regression by rating tier, observing changes in coefficients and significance of mismatch-related variables across different tiers.

## 9) `retest_ny_dual_regression.py`
A retained standalone dual-regression script for historical reproduction and independent regression debugging.

## Core Data Files
1. `listings.csv`
2. `Mismatch_Score/ny/llm_mismatch_score.csv` (customized mismatch)
3. `Mismatch_Score/ny/baseline_mismatch_score.csv` (glove mismatch)

## Naming Notes
1. `glove_mismatch`: mismatch metric derived from the baseline GloVe route.
2. `customized_mismatch`: mismatch metric derived from the current customized route (10D/slot-based).