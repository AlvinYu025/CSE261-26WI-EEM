#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# --------------------------
# Utility function: convert string representation of list to np.array
# --------------------------
def str_to_array(s):
    if isinstance(s, str):
        s = s.strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        s = np.array([float(x) for x in s.split()])
    return s

# --------------------------
# Compute mismatch_proxy via dual-supervision regression
# --------------------------
def compute_mismatch(df, desc_emb_cols, rev_emb_cols, rating_col, output_csv):
    out_dir = os.path.dirname(output_csv)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    # Convert string embedding to array if single column
    if len(desc_emb_cols) == 1:
        desc_emb_array = df[desc_emb_cols[0]].apply(str_to_array)
        rev_emb_array = df[rev_emb_cols[0]].apply(str_to_array)
        X_desc = np.stack(desc_emb_array.values)
        X_rev  = np.stack(rev_emb_array.values)
    else:
        X_desc = df[desc_emb_cols].values
        X_rev  = df[rev_emb_cols].values

    y = df[rating_col].values

    # Dual-supervision linear regression
    model_desc = LinearRegression()
    model_desc.fit(X_desc, y)
    pred_desc = model_desc.predict(X_desc)

    model_rev = LinearRegression()
    model_rev.fit(X_rev, y)
    pred_rev = model_rev.predict(X_rev)

    df['mismatch_proxy'] = pred_desc - pred_rev

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved mismatch CSV to {output_csv}")
    return df

# --------------------------
# Compute delta_k, mabs, mover, munder
# --------------------------
def compute_delta_metrics(df, K=10):
    desc_cols = [f"desc_S{i:02d}_score" for i in range(1, K+1)]
    rev_cols  = [f"rev_S{i:02d}_score"  for i in range(1, K+1)]

    delta_k = df[desc_cols].values - df[rev_cols].values
    delta_k_df = pd.DataFrame(delta_k, columns=[f"delta_S{i:02d}" for i in range(1, K+1)])

    df['mabs']   = np.mean(np.abs(delta_k), axis=1)
    df['mover']  = np.mean(np.maximum(delta_k, 0), axis=1)
    df['munder'] = np.mean(np.maximum(-delta_k, 0), axis=1)

    df = pd.concat([df, delta_k_df], axis=1)
    return df

# --------------------------
# Extract top 0.1% examples (over/under) for inspection
# --------------------------
def extract_top_examples(df, row_id_col, output_csv):
    top_count = max(1, int(len(df) * 0.001))  # at least 1 row

    top_over = df.sort_values("mismatch_proxy", ascending=False).head(top_count)
    top_under = df.sort_values("mismatch_proxy", ascending=True).head(top_count)

    top_over = top_over.copy()
    top_under = top_under.copy()
    top_over['mismatch_type'] = 'over'
    top_under['mismatch_type'] = 'under'

    # Columns to keep
    columns_to_keep = [row_id_col, "mismatch_proxy", "mismatch_type", "rating", "description", "review"]
    top_examples = pd.concat([top_over[columns_to_keep], top_under[columns_to_keep]], ignore_index=True)

    top_examples.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved top 0.1‰ over- and under-description examples to {output_csv}")

# --------------------------
# Main function
# --------------------------
def main():
    dataset = "mo"      # al / am / mo
    # --------------------------
    # 1. GloVe baseline
    # --------------------------
    baseline_path = f"../Baseline/airbnb_glove_embeddings-{dataset}.csv"
    df_baseline = pd.read_csv(baseline_path)
    desc_emb_col = ['desc_emb']
    rev_emb_col  = ['review_emb']
    rating_col   = 'rating'
    output_baseline_csv = f"{dataset}/baseline_mismatch_score.csv"

    df_baseline = compute_mismatch(df_baseline, desc_emb_col, rev_emb_col, rating_col, output_baseline_csv)
    extract_top_examples(df_baseline, row_id_col="review_id", output_csv=f"{dataset}/baseline_mismatch_top10.csv")

    # --------------------------
    # 2. 10-d student embedding
    # --------------------------
    student_path = f"../Space_Formation_Embedding_Extraction/student_scores_{dataset}.csv"
    df_student = pd.read_csv(student_path)

    desc_emb_cols = [f"desc_S{str(i).zfill(2)}_score" for i in range(1, 11)]
    rev_emb_cols  = [f"rev_S{str(i).zfill(2)}_score" for i in range(1, 11)]
    rating_col    = 'rating'
    output_student_csv = f"{dataset}/llm_mismatch_score.csv"

    df_student = compute_mismatch(df_student, desc_emb_cols, rev_emb_cols, rating_col, output_student_csv)

    # Compute delta_k and aggregate metrics
    df_student = compute_delta_metrics(df_student, K=10)
    df_student.to_csv(output_student_csv, index=False, encoding="utf-8-sig")
    print(f"Updated CSV with delta_k and aggregates saved to {output_student_csv}")

    extract_top_examples(df_student, row_id_col="row_idx", output_csv=f"{dataset}/llm_mismatch_top10.csv")

if __name__ == "__main__":
    main()
