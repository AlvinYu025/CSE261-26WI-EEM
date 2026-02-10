#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference for the trained student regressor.

Input:
- base CSV with description/review
- trained student directory (from train_student.py)

Output:
- scores.csv with columns:
  row_idx,
  desc_S01_score..desc_S10_score (if target includes desc)
  rev_S01_score..rev_S10_score   (if target includes rev)

This can replace the LLM scoring stage:
- You can plug the produced scores.csv into your downstream analysis.
"""

import os
import json
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoConfig, AutoModel

# -----------------------
# helpers
# -----------------------
def make_slot_ids(K: int) -> List[str]:
    return [f"S{i:02d}" for i in range(1, K + 1)]

def clamp01_np(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def clamp11_np(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -1.0, 1.0)

# -----------------------
# Dataset
# -----------------------
class InferDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, desc_col: str, review_col: str, max_length: int):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.desc_col = desc_col
        self.review_col = review_col
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        desc = "" if pd.isna(row[self.desc_col]) else str(row[self.desc_col])
        rev  = "" if pd.isna(row[self.review_col]) else str(row[self.review_col])
        text = f"[DESC] {desc}\n[REVIEW] {rev}"
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        enc["__idx__"] = idx
        return enc

class Collator:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        idxs = [f.pop("__idx__") for f in features]
        batch = self.tok.pad(features, padding=True, return_tensors="pt")
        batch["__idx__"] = torch.tensor(idxs, dtype=torch.long)
        return batch

# -----------------------
# Model (must match train_student.py)
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

# -----------------------
# Main
# -----------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_dir", type=str, required=True, help="Saved model dir from train_student.py")
    ap.add_argument("--csv_path", type=str, required=True, help="CSV with description/review")
    ap.add_argument("--out_csv", type=str, required=True, help="Where to save scores.csv")
    ap.add_argument("--desc_col", type=str, default="description")
    ap.add_argument("--review_col", type=str, default="review")
    ap.add_argument("--row_id_col", type=str, default="row_idx")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=None, help="Override meta max_length")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--clamp", action="store_true", help="Clamp outputs to valid ranges")
    args = ap.parse_args()

    meta_path = os.path.join(args.student_dir, "student_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}. Did you save with train_student.py?")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    K = int(meta["K"])
    target = meta["target"]
    slot_ids = meta["slot_ids"]
    out_dim = int(meta["out_dim"])
    base_model_name = meta["model_name"]
    max_length = int(args.max_length) if args.max_length is not None else int(meta["max_length"])

    df = pd.read_csv(args.csv_path)
    # ensure row_idx exists for stable alignment
    if args.row_id_col not in df.columns:
        df = df.reset_index(drop=True)
        df[args.row_id_col] = np.arange(len(df))

    tokenizer = AutoTokenizer.from_pretrained(args.student_dir, use_fast=True)

    # Load weights: we saved the whole Trainer model, so easiest is torch.load via transformers?
    # But we used a custom nn.Module; we will reconstruct and load state dict from pytorch_model.bin
    model = MultiOutputRegressor(base_model_name, out_dim=out_dim, dropout=0.1)
    state_path = os.path.join(args.student_dir, "pytorch_model.bin")
    if not os.path.exists(state_path):
        # newer HF saves model.safetensors sometimes
        st2 = os.path.join(args.student_dir, "model.safetensors")
        raise FileNotFoundError(f"Missing {state_path} (or {st2}). If you saved safetensors, adapt loader.")
    sd = torch.load(state_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)

    model.to(args.device).eval()

    ds = InferDataset(df, tokenizer, args.desc_col, args.review_col, max_length=max_length)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=Collator(tokenizer))

    preds_all = np.zeros((len(df), out_dim), dtype=np.float32)

    for batch in dl:
        idxs = batch.pop("__idx__").cpu().numpy()
        batch = {k: v.to(args.device) for k, v in batch.items()}
        y = model(**batch)  # [B, D]
        y = y.detach().float().cpu().numpy()
        preds_all[idxs] = y

    # Build output scores.csv with the SAME column convention you used earlier.
    out = pd.DataFrame({args.row_id_col: df[args.row_id_col].to_numpy()})

    if target == "desc":
        # 10-d (desc only)
        for j, sid in enumerate(slot_ids):
            col = f"desc_{sid}_score"
            out[col] = preds_all[:, j]
        if args.clamp:
            for sid in slot_ids:
                out[f"desc_{sid}_score"] = clamp01_np(out[f"desc_{sid}_score"].to_numpy())

    elif target == "rev":
        # 10-d (rev only)
        for j, sid in enumerate(slot_ids):
            col = f"rev_{sid}_score"
            out[col] = preds_all[:, j]
        if args.clamp:
            for sid in slot_ids:
                out[f"rev_{sid}_score"] = clamp11_np(out[f"rev_{sid}_score"].to_numpy())

    else:
        # 20-d: first 10 desc, next 10 rev
        for j, sid in enumerate(slot_ids):
            out[f"desc_{sid}_score"] = preds_all[:, j]
        for j, sid in enumerate(slot_ids):
            out[f"rev_{sid}_score"] = preds_all[:, K + j]
        if args.clamp:
            for sid in slot_ids:
                out[f"desc_{sid}_score"] = clamp01_np(out[f"desc_{sid}_score"].to_numpy())
                out[f"rev_{sid}_score"] = clamp11_np(out[f"rev_{sid}_score"].to_numpy())

    out.to_csv(args.out_csv, index=False)
    print("[saved]", args.out_csv, "rows=", len(out), "dim=", out_dim, "target=", target)

if __name__ == "__main__":
    main()
