# Step 3: Mismatch Score

`mismatch_proxy_score.py` computes mismatch signals from two embedding spaces:

- GloVe baseline embeddings
- 10-d student semantic embeddings

## What the script computes

1. `mismatch_proxy` (both spaces)
   - Fits two regressions to predict rating:
     - description embedding -> predicted rating
     - review embedding -> predicted rating
   - Defines mismatch as:
     - `mismatch_proxy = pred_desc - pred_rev`

2. 10-d dimension deltas (student space only)
   - `delta_S01 ... delta_S10`
   - Aggregates:
     - `mabs`: average absolute mismatch across dimensions
     - `mover`: average positive mismatch
     - `munder`: average negative mismatch (absolute direction)

3. Top examples
   - Saves top over-description and under-description rows.

## Important configuration

This script currently uses a hard-coded dataset switch:

```python
dataset = "mo"   # al / am / mo
```

Set it before running to choose the city split you want to process.

## Command

```bash
python mismatch_proxy_score.py
```

## Inputs expected

- `../Baseline/airbnb_glove_embeddings-{dataset}.csv`
- `../Space_Formation_Embedding_Extraction/student_scores_{dataset}.csv`

## Outputs written to

- `{dataset}/baseline_mismatch_score.csv`
- `{dataset}/baseline_mismatch_top10.csv`
- `{dataset}/llm_mismatch_score.csv`
- `{dataset}/llm_mismatch_top10.csv`
