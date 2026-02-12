# Step 3: *Define Mismatch Score*

**`mismatch_proxy_score.py`** computes the **rating-anchored mismatch proxy** for both embedding spaces (GloVe baseline and 10-d LLM semantic space), and additionally computes dimension-level delta metrics exclusively for the 10-d space.

---

## 3.1 Compute Mismatch Proxy (Dual-Supervision Regression)

### What the script does

- Loads the GloVe baseline embeddings (`airbnb_glove_embeddings.csv`) and the 10-d student embeddings (`student_scores_13k.csv`) separately
- For each space, fits two independent **linear regression** models:
  - `model_desc`: description embeddings → rating
  - `model_rev`: review embeddings → rating
- Both models are fitted on the **full dataset** (no train/test split), as the regressors serve as projection functions rather than generalization models — the fitted values themselves are the signal of interest
- Computes `mismatch_proxy` for every item in **both** spaces:

$$\text{mismatch\_proxy}_i = \hat{y}_i^{\text{desc}} - \hat{y}_i^{\text{rev}}$$

where $\hat{y}^{\text{desc}}$ and $\hat{y}^{\text{rev}}$ are the rating predictions from the description and review regressors respectively.

### Interpretation

| Value | Meaning |
|---|---|
| `mismatch_proxy > 0` | **Over-description**: the description implies higher quality than the review reflects |
| `mismatch_proxy < 0` | **Under-description**: the description implies lower quality than the review reflects |
| `mismatch_proxy ≈ 0` | Description and review are broadly aligned in their implied rating |

### Design note

Because both embedding spaces are projected onto the same rating target, `mismatch_proxy` yields a **unified and directly comparable mismatch signal** regardless of the underlying representational space — whether GloVe-based (100-d) or the data-driven 10-d semantic space. This is the central motivation for using rating as a supervisory anchor rather than computing embedding distances directly, which would be neither directional nor cross-space comparable.

---

## 3.2 Delta_k & Aggregate Metrics (10-d Space Only)

This analysis is performed **exclusively on the 10-d LLM semantic space**, as it requires interpretable per-dimension scores that are not available in the GloVe baseline.

### What the script does

- For each of the $K = 10$ semantic dimensions, computes the per-dimension mismatch:

$$\delta_k = d_k - r_k$$

where $d_k$ and $r_k$ are the description and review scores for dimension $k$ respectively.

- Computes three **item-level aggregate metrics** across all dimensions:

| Metric | Formula | Interpretation |
|---|---|---|
| `mabs` | $m^{\text{abs}} = \dfrac{1}{K}\displaystyle\sum_{k=1}^{K} \lvert \delta_k \rvert$ | Overall mismatch magnitude, direction-agnostic |
| `mover` | $m^{\text{over}} = \dfrac{1}{K}\displaystyle\sum_{k=1}^{K} \max(\delta_k,\ 0)$ | Mean excess across dimensions where description oversells |
| `munder` | $m^{\text{under}} = \dfrac{1}{K}\displaystyle\sum_{k=1}^{K} \max(-\delta_k,\ 0)$ | Mean deficit across dimensions where description undersells |

- Appends all delta columns (`delta_S01` … `delta_S10`) and the three aggregate columns to the output CSV

### Relationship to the main experiment

The `mismatch_proxy` from §3.1 is a single scalar that summarises alignment via a rating anchor. The delta metrics here complement it by providing **dimension-level granularity**: while the proxy answers *how much* mismatch exists in aggregate, the delta analysis answers *in which semantic dimensions* the mismatch occurs and *in which direction*. These two analyses are therefore not redundant — the former enables cross-space comparison, the latter enables aspect-level interpretability within the 10-d space.

---

## 3.3 Extract Top Over- and Under-Description Examples

### What the script does

- Sorts all items by `mismatch_proxy` in descending order → selects the top 0.1‰ as **over-description** examples
- Sorts all items by `mismatch_proxy` in ascending order → selects the top 0.1‰ as **under-description** examples
- Runs on **both** embedding spaces (GloVe and 10-d) independently
- Retains the following columns for inspection:

| Column | Description |
|---|---|
| `row_idx` / `review_id` | Row identifier in the original dataset |
| `mismatch_proxy` | Dual-supervision regression proxy score |
| `mismatch_type` | `'over'` or `'under'` |
| `rating` | Actual user rating |
| `description` | Product description text |
| `review` | User review text |

---

## Command

```bash
python mismatch_proxy_score.py
```

---

## Outputs

| File | Contents |
|---|---|
| `baseline_mismatch_score.csv` | Full GloVe dataset with `mismatch_proxy` column appended |
| `llm_mismatch_score.csv` | Full 10-d dataset with `mismatch_proxy`, `delta_S01`…`delta_S10`, `mabs`, `mover`, `munder` appended |
| `baseline_mismatch_top10.csv` | Top 0.1‰ over- and under-description examples from the GloVe space |
| `llm_mismatch_top10.csv` | Top 0.1‰ over- and under-description examples from the 10-d space |