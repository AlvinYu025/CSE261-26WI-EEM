# README_ADDED

## Overall Goal
本次新增与重构代码的核心目标是验证 `mismatch` 特征是否有效，尤其是它在不同任务中的增量价值。

`mismatch` 本身来自 host 端与 guest 端信息的对齐/偏差，因此实验设计尽量覆盖两端信号，而不是只看单一侧指标。

## 1) `ny_joint_classification_regression.py`
### Why
这个脚本是主实验入口。目的不是只跑一个指标，而是在同一套 NY 数据上同时检查：
1. 分类任务中 `mismatch` 是否有稳定增益。
2. 回归任务中 `mismatch` 是否优于或至少补充 baseline（GloVe 路线）信号。

如果某个 target 无法和 expectation-experience（预期-体验）机制对齐，或结果不稳定，就不作为主结论目标。

### Classification Design
使用联合标签 `joint_superhost_rating`（4 类），将 host 与 guest 两侧融合：
1. `host_is_superhost`（host side）
2. `review_scores_rating` 经阈值二值化（`<= 4.8` vs `> 4.8`，guest side）

4 类定义：
1. non-superhost + high-rating
2. non-superhost + low-rating
3. superhost + high-rating
4. superhost + low-rating

这样做的动机是：让分类目标尽量与 `mismatch` 的双侧定义对齐，而不是只预测单一侧变量。

### Regression Design
回归默认使用联合目标（而非分开两个回归）：
1. host side: `host_price_log`
2. guest side: `user_rating`（`review_scores_rating`）

处理方式：先分别 z-score，再取平均，形成一个 joint regression target。

### What Is Compared
脚本统一比较三条路线：
1. `TF-IDF only`
2. `TF-IDF + glove_mismatch`（来自 `baseline_mismatch_score.csv`）
3. `TF-IDF + customized_mismatch`（`mismatch_proxy/mabs/mover/munder`）

默认过滤：`n_reviews > 1`。

## 2) `mismatch_balanced_classification.py`
### Why
该脚本重点验证类别平衡设置下 mismatch 的可解释增益。

### What Was Done
1. 在三个数据集上做 balanced 分类实验。
2. 结果记录在文件末尾注释。

### Interpretation
1. balanced 版本更容易展示 mismatch 的贡献（减少多数类掩盖效应）。
2. imbalanced 更贴近真实分布，但在该设定下仍可观察到提升。

## 3) `mismatch_balanced_stratified_regression.py`
### Why
回归侧也做 balanced vs imbalanced 对比，检查结论是否稳健。

### Finding
差异整体不大，且 imbalanced 版本往往更好。

### Possible Reason
回归任务里强行平衡会改变原始目标分布和方差结构，导致模型对真实数据生成机制的拟合变弱；保持原分布（imbalanced）通常更接近真实信号。

## 4) `regression_correlation_retest.ipynb`
### What Was Updated
1. 修正了 CSV 加载路径并重新测试。
2. 不再使用 `inner merge`（当前数据已是最新版本，不需要再通过 inner merge 对齐筛掉样本）。

数据可从 drive 下载。

## 5) `tfidf_feature_ablation.py`
### Why
做消融实验，比较 `TF-IDF + mismatch` 与 `TF-IDF + 其他特征` 的相对贡献。

### Finding
1. mismatch 有提升，但部分情况下不如某些其他结构化特征提升大。
2. 一个可能原因是预测目标是 `rating`（更偏 guest-side），会带来目标侧偏置。

### Value
即使不是最强单特征，mismatch 仍提供了独立增量信息，这一点对后续联合任务仍有价值。

## 6) `tfidf_mismatch_5class_compare.py`
### Why
做一个轻量级 sanity check 小实验。

### What Was Done
1. 对 `rating` 分 bin（每个 bin 数据量大致接近）。
2. 比较 `TF-IDF` 与 `TF-IDF + mismatch`。

结论用途：快速验证方向，不作为最终主实验。

## 7) `ny_mismatch_classification.py`
早期 NY 分类脚本，主要用于 mismatch 分类基线复验与快速对照。

## 8) `mismatch_stratified_regression.py`
按评分层进行分层回归，观察 mismatch 相关变量在不同层中的系数与显著性变化。

## 9) `retest_ny_dual_regression.py`
保留的独立双回归脚本，便于做历史复现和单独回归调试。

## Core Data Files
1. `listings.csv`
2. `Mismatch_Score/ny/llm_mismatch_score.csv`（customized mismatch）
3. `Mismatch_Score/ny/baseline_mismatch_score.csv`（glove mismatch）

## Naming Notes
1. `glove_mismatch`: baseline GloVe 路线得到的 mismatch 指标。
2. `customized_mismatch`: 当前自定义路线（10D/slot-based）得到的 mismatch 指标。
