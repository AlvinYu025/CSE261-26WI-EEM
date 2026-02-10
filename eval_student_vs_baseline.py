#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a trained multi-output student regressor vs. a constant-mean baseline.

- Loads your best model weights saved by Trainer (model.safetensors or pytorch_model.bin)
- Reconstructs the exact expand-to-2N logic: each row -> (DESC, y_desc_vec) and (REV, y_rev_vec)
- Rebuilds the train/eval split with the same seed + val_ratio
- Computes:
  * baseline (train-mean vector) MAE/MSE on eval
  * student MAE/MSE on eval
  * per-dim MAE, and how many dims student beats baseline
  * bootstrap 95% CI for MAE improvement

Expected teacher CSV columns:
  row_idx, description, review,
  desc_S01_score..desc_S10_score,
  rev_S01_score..rev_S10_score
"""

from __future__ import annotations
import os
import json
import argparse
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoConfig, AutoModel


# -----------------------
# Utilities
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_slot_ids(K: int) -> List[str]:
    return [f"S{i:02d}" for i in range(1, K + 1)]


def expand_teacher_df(df: pd.DataFrame, K: int, desc_col: str, review_col: str) -> pd.DataFrame:
    slot_ids = make_slot_ids(K)
    desc_cols = [f"desc_{sid}_score" for sid in slot_ids]
    rev_cols  = [f"rev_{sid}_score"  for sid in slot_ids]

    for c in ["row_idx", desc_col, review_col] + desc_cols + rev_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # description samples
    df_desc = pd.DataFrame({
        "row_idx": df["row_idx"].values,
        "text_type": "desc",
        "text": df[desc_col].fillna("").astype(str).values,
    })
    y_desc = df[desc_cols].to_numpy(np.float32)
    for j in range(K):
        df_desc[f"y{j}"] = y_desc[:, j]

    # review samples
    df_rev = pd.DataFrame({
        "row_idx": df["row_idx"].values,
        "text_type": "rev",
        "text": df[review_col].fillna("").astype(str).values,
    })
    y_rev = df[rev_cols].to_numpy(np.float32)
    for j in range(K):
        df_rev[f"y{j}"] = y_rev[:, j]

    out = pd.concat([df_desc, df_rev], axis=0).reset_index(drop=True)
    return out


def train_eval_split_indices(n: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(val_ratio * n))
    va_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return tr_idx, va_idx


def mse_mae(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.float32)
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return {"mse": mse, "mae": mae}


def per_dim_mae(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.float32)
    return np.mean(np.abs(y_pred - y_true), axis=0)


def bootstrap_mae_delta(
    y_true: np.ndarray,
    y_pred_student: np.ndarray,
    y_pred_base: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Bootstrap CI for delta = MAE(base) - MAE(student).
    """
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    deltas = np.empty(n_boot, dtype=np.float32)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m_base = np.mean(np.abs(y_pred_base[idx] - y_true[idx]))
        m_stu  = np.mean(np.abs(y_pred_student[idx] - y_true[idx]))
        deltas[b] = m_base - m_stu

    mean = float(np.mean(deltas))
    lo = float(np.quantile(deltas, 0.025))
    hi = float(np.quantile(deltas, 0.975))
    p_pos = float(np.mean(deltas > 0.0))
    return {"delta_mean": mean, "delta_ci_lo": lo, "delta_ci_hi": hi, "p_delta_pos": p_pos}


# -----------------------
# Dataset / Collator (same as training)
# -----------------------
class ScoreDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int, K: int):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_length = int(max_length)
        self.K = int(K)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        text_type = str(row["text_type"])
        text = str(row["text"])
        inp = f"[{text_type.upper()}]\n{text}"

        enc = self.tok(
            inp,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        y = np.array([row[f"y{j}"] for j in range(self.K)], dtype=np.float32)
        enc["labels"] = y
        return enc


@dataclass
class DataCollatorWithLabels:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = [f.pop("labels") for f in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = torch.from_numpy(np.stack(labels, axis=0).astype(np.float32))
        return batch


# -----------------------
# Model definition (same structure as training)
# -----------------------
class MultiOutputRegressor(nn.Module):
    def __init__(self, base_model_name: str, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name, config=self.config)

        hidden = getattr(self.config, "hidden_size", None)
        if hidden is None:
            hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(hidden, int(out_dim))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
        )
        cls = out.last_hidden_state[:, 0, :]
        x = self.dropout(cls)
        preds = self.head(x)
        return preds


def load_student_model(save_dir: str, device: str | torch.device = "cuda") -> Tuple[nn.Module, AutoTokenizer, dict]:
    meta_path = os.path.join(save_dir, "student_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"student_meta.json not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    base_model_name = meta.get("model_name", "microsoft/deberta-v3-large")
    K = int(meta.get("K", 10))
    max_length = int(meta.get("max_length", 512))

    tok = AutoTokenizer.from_pretrained(save_dir, use_fast=True)
    model = MultiOutputRegressor(base_model_name, out_dim=K, dropout=0.1)

    # Prefer safetensors if present
    st_path = os.path.join(save_dir, "model.safetensors")
    bin_path = os.path.join(save_dir, "pytorch_model.bin")

    if os.path.exists(st_path):
        from safetensors.torch import load_file
        sd = load_file(st_path)
    elif os.path.exists(bin_path):
        sd = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found in {save_dir} (expected model.safetensors or pytorch_model.bin)")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("[warn] missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("[warn] unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    model.to(device).eval()

    meta["_loaded_base_model_name"] = base_model_name
    meta["_loaded_K"] = K
    meta["_loaded_max_length"] = max_length
    return model, tok, meta


@torch.inference_mode()
def predict(model: nn.Module, dataloader: DataLoader, device: str | torch.device) -> Tuple[np.ndarray, np.ndarray]:
    preds_all = []
    labels_all = []
    for batch in dataloader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = model(**batch)
        preds_all.append(preds.detach().float().cpu().numpy())
        labels_all.append(labels.detach().float().cpu().numpy())
    return np.concatenate(preds_all, axis=0), np.concatenate(labels_all, axis=0)


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_dir", type=str, required=True, help="Folder containing model.safetensors + student_meta.json")
    ap.add_argument("--teacher_csv", type=str, required=True, help="teacher_scores_xxx.csv")
    ap.add_argument("--desc_col", type=str, default="description")
    ap.add_argument("--review_col", type=str, default="review")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--val_ratio", type=float, default=None, help="If None, read from student_meta.json or default 0.2")
    ap.add_argument("--seed", type=int, default=None, help="If None, read from student_meta.json or default 0")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load student + meta
    model, tok, meta = load_student_model(args.student_dir, device=device)

    K = int(args.K)
    if "_loaded_K" in meta:
        # keep consistent with trained head
        K = int(meta["_loaded_K"])

    # Resolve seed / val_ratio
    seed = int(args.seed) if args.seed is not None else int(meta.get("seed", 0))
    val_ratio = float(args.val_ratio) if args.val_ratio is not None else float(meta.get("val_ratio", 0.2))
    max_length = int(meta.get("_loaded_max_length", meta.get("max_length", 512)))

    set_seed(seed)

    # Load teacher and expand
    df0 = pd.read_csv(args.teacher_csv)
    df_exp = expand_teacher_df(df0, K=K, desc_col=args.desc_col, review_col=args.review_col)

    # Split like training
    tr_idx, va_idx = train_eval_split_indices(len(df_exp), val_ratio=val_ratio, seed=seed)
    df_tr = df_exp.iloc[tr_idx].reset_index(drop=True)
    df_va = df_exp.iloc[va_idx].reset_index(drop=True)

    # Baseline: constant train-mean vector
    y_train = df_tr[[f"y{j}" for j in range(K)]].to_numpy(np.float32)
    mean_train = np.mean(y_train, axis=0, keepdims=True)  # [1,K]

    y_eval = df_va[[f"y{j}" for j in range(K)]].to_numpy(np.float32)  # [N,K]
    y_pred_base = np.repeat(mean_train, repeats=y_eval.shape[0], axis=0)  # [N,K]

    base_metrics = mse_mae(y_pred_base, y_eval)
    base_mae_dim = per_dim_mae(y_pred_base, y_eval)

    # Student predictions
    eval_ds = ScoreDataset(df_va, tok, max_length=max_length, K=K)
    collator = DataCollatorWithLabels(tok)
    dl = DataLoader(
        eval_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    y_pred_student, y_true = predict(model, dl, device=device)
    stu_metrics = mse_mae(y_pred_student, y_true)
    stu_mae_dim = per_dim_mae(y_pred_student, y_true)

    # Compare per-dim
    wins = int(np.sum(stu_mae_dim < base_mae_dim))
    ties = int(np.sum(np.isclose(stu_mae_dim, base_mae_dim)))
    losses = K - wins - ties

    # Bootstrap CI for MAE improvement
    boot = bootstrap_mae_delta(
        y_true=y_true,
        y_pred_student=y_pred_student,
        y_pred_base=y_pred_base,
        n_boot=int(args.bootstrap),
        seed=seed,
    )

    # Print report
    print("\n=== Evaluation Split Info ===")
    print(f"N_total_expanded={len(df_exp)}  N_train={len(df_tr)}  N_eval={len(df_va)}")
    print(f"K={K}  seed={seed}  val_ratio={val_ratio}  max_length={max_length}")
    print(f"student_dir={args.student_dir}")
    print(f"teacher_csv={args.teacher_csv}")

    print("\n=== Baseline (Constant train-mean) ===")
    print(f"eval_mae={base_metrics['mae']:.6f}  eval_mse={base_metrics['mse']:.6f}")

    print("\n=== Student (Loaded best model.safetensors) ===")
    print(f"eval_mae={stu_metrics['mae']:.6f}  eval_mse={stu_metrics['mse']:.6f}")

    delta = base_metrics["mae"] - stu_metrics["mae"]
    rel = delta / max(1e-12, base_metrics["mae"])
    print("\n=== Improvement ===")
    print(f"MAE_reduction = {delta:.6f}  (relative {rel*100:.2f}%)")
    print(f"Per-dim wins: {wins}/{K}  (ties={ties}, losses={losses})")

    print("\n=== Per-dimension MAE (baseline vs student) ===")
    rows = []
    for j in range(K):
        rows.append((j, float(base_mae_dim[j]), float(stu_mae_dim[j]), float(base_mae_dim[j] - stu_mae_dim[j])))
    df_dim = pd.DataFrame(rows, columns=["dim", "mae_base", "mae_student", "delta(base-student)"])
    print(df_dim.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n=== Bootstrap significance (delta = MAE(base) - MAE(student)) ===")
    print(f"delta_mean={boot['delta_mean']:.6f}  95%CI=[{boot['delta_ci_lo']:.6f}, {boot['delta_ci_hi']:.6f}]")
    print(f"P(delta>0)={boot['p_delta_pos']:.4f}  (closer to 1.0 => stronger evidence student beats baseline)")

    # Optional: save a csv report next to the model
    out_csv = os.path.join(args.student_dir, "eval_student_vs_baseline_report.csv")
    df_dim.to_csv(out_csv, index=False)
    print(f"\n[saved] per-dim report -> {out_csv}")


if __name__ == "__main__":
    main()
