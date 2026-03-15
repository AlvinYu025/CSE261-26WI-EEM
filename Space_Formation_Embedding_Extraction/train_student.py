#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a JOINT student regressor that matches your teacher generation:

Teacher produces (d, r) in ONE call:
  (d, r) = T(description, review)
where d and r are both length-K vectors in [0,1].

So student should learn:
  f([DESC]\n desc \n [REV]\n rev) -> R^(2K)
  first K dims = desc (d)
  next  K dims = review (r)

Input teacher CSV columns:
  row_idx, description, review,
  desc_S01_score..desc_S10_score,
  rev_S01_score..rev_S10_score
"""

from __future__ import annotations
import os, json, argparse, random, inspect
from dataclasses import dataclass
from typing import Dict, Any, List

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
# Dataset / Collator
# -----------------------
class JointScoreDataset(Dataset):
    """
    One row => one sample:
      input:  [DESC]\n{description}\n[REV]\n{review}
      label:  concat(d, r) in R^(2K)
    """
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int, K: int,
                 desc_col: str, review_col: str):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_length = max_length
        self.K = K
        self.desc_col = desc_col
        self.review_col = review_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        desc = "" if pd.isna(row[self.desc_col]) else str(row[self.desc_col])
        rev  = "" if pd.isna(row[self.review_col]) else str(row[self.review_col])

        inp = f"[DESC]\n{desc}\n[REV]\n{rev}"

        enc = self.tok(
            inp,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        # labels: [2K] = [d0..dK-1, r0..rK-1]
        y = np.array([row[f"y{j}"] for j in range(2 * self.K)], dtype=np.float32)
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
        self.trainer = None

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

def quick_label_sanity_and_baseline_joint(df0: pd.DataFrame, K: int,
                                         desc_col: str, review_col: str,
                                         val_ratio: float, seed: int):
    """
    Baseline: predict constant mean vector mu (over TRAIN) in R^(2K) for every sample.
    """
    slot_ids = make_slot_ids(K)
    desc_cols = [f"desc_{sid}_score" for sid in slot_ids]
    rev_cols  = [f"rev_{sid}_score"  for sid in slot_ids]
    for c in ["row_idx", desc_col, review_col] + desc_cols + rev_cols:
        if c not in df0.columns:
            raise ValueError(f"Missing column: {c}")

    df = df0.copy().reset_index(drop=True)
    y_desc = df[desc_cols].to_numpy(np.float32)
    y_rev  = df[rev_cols].to_numpy(np.float32)
    Y = np.concatenate([y_desc, y_rev], axis=1)  # [N, 2K]

    n = len(df)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(val_ratio * n))
    va_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    Y_tr = Y[tr_idx]
    Y_va = Y[va_idx]

    mu_tr = Y_tr.mean(axis=0)
    pred_va = np.broadcast_to(mu_tr[None, :], Y_va.shape)
    base_mse = float(np.mean((pred_va - Y_va) ** 2))
    base_mae = float(np.mean(np.abs(pred_va - Y_va)))

    sd_all = Y.std(axis=0)
    print("\n=== [Sanity] Joint teacher label distribution (2K-d) ===")
    print(f"[sizes] rows: total={n} train={len(tr_idx)} eval={len(va_idx)}")
    print(f"[label std] mean(std) over dims: all={float(sd_all.mean()):.4f}")
    print("\n=== [Baseline] Constant-mean predictor on eval (2K-d) ===")
    print(f"baseline_eval_mae = {base_mae:.6f}")
    print(f"baseline_eval_mse = {base_mse:.6f}")

    return {"baseline_eval_mae": base_mae, "baseline_eval_mse": base_mse}

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--desc_col", type=str, default="description")
    ap.add_argument("--review_col", type=str, default="review")
    ap.add_argument("--K", type=int, default=10)

    ap.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large")
    ap.add_argument("--save_dir", type=str, default="student_deberta_joint20")
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

    df0 = pd.read_csv(args.train_csv)
    if "row_idx" not in df0.columns:
        raise ValueError("teacher csv must contain row_idx")

    # build joint labels y0..y(2K-1)
    slot_ids = make_slot_ids(args.K)
    desc_cols = [f"desc_{sid}_score" for sid in slot_ids]
    rev_cols  = [f"rev_{sid}_score"  for sid in slot_ids]
    for c in [args.desc_col, args.review_col] + desc_cols + rev_cols:
        if c not in df0.columns:
            raise ValueError(f"Missing column: {c}")

    df = df0.copy().reset_index(drop=True)
    y_desc = df[desc_cols].to_numpy(np.float32)
    y_rev  = df[rev_cols].to_numpy(np.float32)
    Y = np.concatenate([y_desc, y_rev], axis=1)  # [N,2K]
    for j in range(2 * args.K):
        df[f"y{j}"] = Y[:, j]

    print(f"[data] joint rows={len(df)} labels_dim={2*args.K}", flush=True)

    _ = quick_label_sanity_and_baseline_joint(
        df0=df0,
        K=args.K,
        desc_col=args.desc_col,
        review_col=args.review_col,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # split on rows (NOT 2N anymore)
    n = len(df)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_val = max(1, int(args.val_ratio * n))
    va_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_va = df.iloc[va_idx].reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_ds = JointScoreDataset(df_tr, tok, args.max_length, args.K, args.desc_col, args.review_col)
    val_ds   = JointScoreDataset(df_va, tok, args.max_length, args.K, args.desc_col, args.review_col)
    collator = DataCollatorWithLabels(tok)

    out_dim = 2 * args.K
    model = MultiOutputRegressor(args.model_name, out_dim=out_dim, dropout=0.1)

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

    print("\n=== Untrained performance ===", flush=True)
    tr0 = trainer.evaluate(eval_dataset=train_ds, metric_key_prefix="train")
    va0 = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="eval")
    print(f"[untrained] train mse={tr0['train_mse']:.4f} mae={tr0['train_mae']:.4f}", flush=True)
    print(f"[untrained]  eval mse={va0['eval_mse']:.4f} mae={va0['eval_mae']:.4f}\n", flush=True)

    cb = EpochEvalCallback(train_ds=train_ds, eval_ds=val_ds, out_csv_path=metrics_csv)
    cb.set_trainer(trainer)
    trainer.add_callback(cb)

    trainer.train()

    trainer.save_model(args.save_dir)
    tok.save_pretrained(args.save_dir)

    meta = {
        "K": args.K,
        "space": "joint_20d",
        "slot_ids": slot_ids,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "target": "both_joint",
        "out_dim": out_dim,
        "train_samples": len(train_ds),
        "eval_samples": len(val_ds),
        "teacher_csv": args.train_csv,
        "label_space": "d_then_r",
        "range": "[0,1] for both d/r (per teacher prompt)",
    }
    with open(os.path.join(args.save_dir, "student_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if os.path.exists(metrics_csv):
        save_curves(metrics_csv, curves_prefix)

    print("[saved]", args.save_dir, flush=True)
    print("[meta]", meta, flush=True)

if __name__ == "__main__":
    main()