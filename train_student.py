#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a 10-dim student regressor in ONE shared aspect space.

Input: single teacher CSV with columns:
  row_idx, description, review,
  desc_S01_score..desc_S10_score,
  rev_S01_score..rev_S10_score

Training trick:
  Expand each row into TWO samples:
    (text=description, y=desc_vec)
    (text=review,      y=rev_vec)

So the student learns ONE function:
  f(text) -> R^10
shared for desc/review.
"""

from __future__ import annotations
import os, json, argparse, random, inspect
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

# -----------------------
# helpers
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_slot_ids(K: int) -> List[str]:
    return [f"S{i:02d}" for i in range(1, K + 1)]

def compute_metrics_mse(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    mse = float(np.mean((preds - labels) ** 2))
    mae = float(np.mean(np.abs(preds - labels)))
    return {"mse": mse, "mae": mae}

def build_training_args_compat(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    params = set(sig.parameters.keys())
    if "evaluation_strategy" in params and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    if "eval_strategy" in params and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    return TrainingArguments(**kwargs)

# -----------------------
# Expand teacher rows -> 2N samples
# -----------------------
def expand_teacher_df(df: pd.DataFrame, K: int, desc_col: str, review_col: str) -> pd.DataFrame:
    slot_ids = make_slot_ids(K)
    desc_cols = [f"desc_{sid}_score" for sid in slot_ids]
    rev_cols  = [f"rev_{sid}_score"  for sid in slot_ids]

    for c in [desc_col, review_col] + desc_cols + rev_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # desc samples
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

# -----------------------
# Dataset / Collator
# -----------------------
class ScoreDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int, K: int):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_length = max_length
        self.K = K

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        text_type = str(row["text_type"])
        text = str(row["text"])

        # optional prefix helps model distinguish style, but still same space
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
# Model: encoder + regression head
# -----------------------
class MultiOutputRegressor(nn.Module):
    def __init__(self, base_model_name: str, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name, config=self.config)

        hidden = getattr(self.config, "hidden_size", None)
        if hidden is None:
            hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, out_dim)
        self.loss_fn = nn.MSELoss()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
        )
        cls = out.last_hidden_state[:, 0, :]
        x = self.dropout(cls)
        preds = self.head(x)

        loss = None
        if labels is not None:
            loss = self.loss_fn(preds, labels)

        return {"loss": loss, "logits": preds}

# -----------------------
# Callback: epoch-level train/eval metrics + csv
# -----------------------
class EpochEvalCallback(TrainerCallback):
    def __init__(self, train_ds, eval_ds, out_csv_path: str):
        self.train_ds = train_ds
        self.eval_ds = eval_ds
        self.out_csv_path = out_csv_path
        self.rows = []
        self.trainer = None  # <-- add

    def set_trainer(self, trainer: Trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        if trainer is None:
            print("[warn] EpochEvalCallback has no trainer; skip epoch eval.", flush=True)
            return control

        epoch = float(state.epoch) if state.epoch is not None else None

        tr = trainer.evaluate(eval_dataset=self.train_ds, metric_key_prefix="train")
        ev = trainer.evaluate(eval_dataset=self.eval_ds, metric_key_prefix="eval") if self.eval_ds is not None else {}

        row = {"epoch": epoch}
        for k in ["train_loss","train_mse","train_mae","eval_loss","eval_mse","eval_mae"]:
            if k in tr: row[k] = float(tr[k])
            if k in ev: row[k] = float(ev[k])

        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.out_csv_path, index=False)

        msg = f"[epoch {epoch:.2f}] train mse={row.get('train_mse',np.nan):.4f} mae={row.get('train_mae',np.nan):.4f}"
        msg += f" | eval mse={row.get('eval_mse',np.nan):.4f} mae={row.get('eval_mae',np.nan):.4f}"
        print(msg, flush=True)
        return control

def save_curves(csv_path: str, out_prefix: str):
    import matplotlib.pyplot as plt
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return

    x = df["epoch"].to_numpy()

    # MAE
    plt.figure()
    if "train_mae" in df: plt.plot(x, df["train_mae"].to_numpy(), label="train_mae")
    if "eval_mae" in df:  plt.plot(x, df["eval_mae"].to_numpy(),  label="eval_mae")
    plt.xlabel("epoch"); plt.ylabel("MAE"); plt.legend(); plt.tight_layout()
    plt.savefig(out_prefix + "_mae.png", dpi=200)
    plt.close()

    # MSE
    plt.figure()
    if "train_mse" in df: plt.plot(x, df["train_mse"].to_numpy(), label="train_mse")
    if "eval_mse" in df:  plt.plot(x, df["eval_mse"].to_numpy(),  label="eval_mse")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend(); plt.tight_layout()
    plt.savefig(out_prefix + "_mse.png", dpi=200)
    plt.close()

    print("[saved]", out_prefix + "_mae.png")
    print("[saved]", out_prefix + "_mse.png")

def quick_label_sanity_and_baseline(df0: pd.DataFrame, K: int, desc_col: str, review_col: str,
                                    val_ratio: float, seed: int):
    """
    Sanity check for teacher labels + baseline performance.

    Baseline: predict a constant vector mu (mean of train labels) for every sample.
    Since you assume desc/rev share the same 10-d space, we use ONE shared mu over the expanded 2N train split.

    Reports:
    - Per-dim mean/std for train/eval/all
    - Baseline MAE/MSE on eval split (same split method as your training code)
    """
    # ----- expand exactly like training -----
    df = expand_teacher_df(df0, K=K, desc_col=desc_col, review_col=review_col)

    # ----- split exactly like training -----
    n = len(df)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(val_ratio * n))
    va_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_va = df.iloc[va_idx].reset_index(drop=True)

    # ----- collect label matrices -----
    ycols = [f"y{j}" for j in range(K)]
    Y_tr = df_tr[ycols].to_numpy(np.float32)   # [Ntr, K]
    Y_va = df_va[ycols].to_numpy(np.float32)   # [Nva, K]
    Y_all = df[ycols].to_numpy(np.float32)

    # ----- per-dim stats -----
    def stats(Y):
        return Y.mean(axis=0), Y.std(axis=0)

    mu_tr, sd_tr = stats(Y_tr)
    mu_va, sd_va = stats(Y_va)
    mu_all, sd_all = stats(Y_all)

    # ----- baseline: predict constant mu_tr for all eval samples -----
    pred_va = np.broadcast_to(mu_tr[None, :], Y_va.shape)
    base_mse = float(np.mean((pred_va - Y_va) ** 2))
    base_mae = float(np.mean(np.abs(pred_va - Y_va)))

    # ----- extra: label scale summary -----
    # mean per-dim std: how much signal/variation exists on average
    mean_sd_all = float(sd_all.mean())
    mean_sd_va  = float(sd_va.mean())

    print("\n=== [Sanity] Teacher label distribution (10-d space) ===")
    print(f"[sizes] expanded samples: total={len(df)}  train={len(df_tr)}  eval={len(df_va)}")
    print(f"[label std] mean(std) over dims: all={mean_sd_all:.4f}  eval={mean_sd_va:.4f}")

    # Show per-dim mean/std compactly
    def fmt(v): return "[" + ", ".join(f"{x:.3f}" for x in v.tolist()) + "]"
    print("\n[per-dim mean] train:", fmt(mu_tr))
    print("[per-dim mean]  eval:", fmt(mu_va))
    print("[per-dim mean]   all:", fmt(mu_all))

    print("\n[per-dim  std] train:", fmt(sd_tr))
    print("[per-dim  std]  eval:", fmt(sd_va))
    print("[per-dim  std]   all:", fmt(sd_all))

    print("\n=== [Baseline] Constant-mean predictor on eval ===")
    print(f"baseline_eval_mae = {base_mae:.6f}")
    print(f"baseline_eval_mse = {base_mse:.6f}")

    print("\n[How to interpret]")
    print("- If your DeBERTa eval MAE is only slightly better than this baseline, it is not learning much beyond label priors.")
    print("- If DeBERTa improves a lot (e.g., >10-20% relative MAE reduction), it is learning meaningful text->score mapping.")

    return {
        "baseline_eval_mae": base_mae,
        "baseline_eval_mse": base_mse,
        "mu_train": mu_tr,
        "std_all": sd_all,
        "n_train": len(df_tr),
        "n_eval": len(df_va),
    }


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True,
                    help="Teacher CSV with text+labels (row_idx, description, review, desc_Sxx_score, rev_Sxx_score)")
    ap.add_argument("--desc_col", type=str, default="description")
    ap.add_argument("--review_col", type=str, default="review")
    ap.add_argument("--K", type=int, default=10)

    ap.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large")
    ap.add_argument("--save_dir", type=str, default="student_deberta_k10_shared")
    ap.add_argument("--max_length", type=int, default=512)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_ratio", type=float, default=0.2)

    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--train_bs", type=int, default=8)
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.01)

    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--num_workers", type=int, default=2)

    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    # load teacher
    df0 = pd.read_csv(args.train_csv)
    if "row_idx" not in df0.columns:
        raise ValueError("teacher csv must contain row_idx")

    # expand to 2N samples
    df = expand_teacher_df(df0, K=args.K, desc_col=args.desc_col, review_col=args.review_col)
    print(f"[data] expanded: {len(df0)} rows -> {len(df)} samples (desc+rev)", flush=True)

    _ = quick_label_sanity_and_baseline(
        df0=df0,
        K=args.K,
        desc_col=args.desc_col,
        review_col=args.review_col,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    # exit()

    # split
    n = len(df)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_val = max(1, int(args.val_ratio * n))
    va_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_va = df.iloc[va_idx].reset_index(drop=True)

    # tokenizer/ds
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_ds = ScoreDataset(df_tr, tok, args.max_length, args.K)
    val_ds   = ScoreDataset(df_va, tok, args.max_length, args.K)
    collator = DataCollatorWithLabels(tok)

    # model
    model = MultiOutputRegressor(args.model_name, out_dim=args.K, dropout=0.1)

    metrics_csv = os.path.join(args.save_dir, "metrics_epoch.csv")
    curves_prefix = os.path.join(args.save_dir, "curves")

    targs = build_training_args_compat(
        output_dir=os.path.join(args.save_dir, "runs"),
        overwrite_output_dir=True,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        save_total_limit=2,
        report_to="none",
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tok,
        compute_metrics=compute_metrics_mse,
    )

    # untrained baseline
    print("\n=== Untrained performance ===", flush=True)
    tr0 = trainer.evaluate(eval_dataset=train_ds, metric_key_prefix="train")
    va0 = trainer.evaluate(eval_dataset=val_ds,   metric_key_prefix="eval")
    print(f"[untrained] train mse={tr0['train_mse']:.4f} mae={tr0['train_mae']:.4f}", flush=True)
    print(f"[untrained]  eval mse={va0['eval_mse']:.4f} mae={va0['eval_mae']:.4f}\n", flush=True)

    # epoch callback
    cb = EpochEvalCallback(train_ds=train_ds, eval_ds=val_ds, out_csv_path=metrics_csv)
    cb.set_trainer(trainer)
    trainer.add_callback(cb)

    # train
    trainer.train()

    # save final (best already loaded)
    trainer.save_model(args.save_dir)
    tok.save_pretrained(args.save_dir)

    meta = {
        "K": args.K,
        "space": "shared_10d",
        "slot_ids": make_slot_ids(args.K),
        "model_name": args.model_name,
        "max_length": args.max_length,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "train_samples": len(train_ds),
        "eval_samples": len(val_ds),
        "teacher_csv": args.train_csv,
    }
    with open(os.path.join(args.save_dir, "student_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if os.path.exists(metrics_csv):
        save_curves(metrics_csv, curves_prefix)

    print("[saved]", args.save_dir, flush=True)
    print("[meta]", meta, flush=True)

if __name__ == "__main__":
    main()
