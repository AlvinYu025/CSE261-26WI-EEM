#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference for JOINT 20-d student:
  f([DESC]\n desc \n [REV]\n rev) -> [d(10), r(10)]

- Loads student_meta.json + tokenizer from student_dir
- Loads weights from model.safetensors OR pytorch_model.bin
- Writes: row_idx, desc_Sxx_score..., rev_Sxx_score...
- --clamp clamps BOTH to [0,1] (matches your teacher prompt)
- Prints progress via tqdm
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
from tqdm import tqdm

def clamp01_np(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def load_state_dict_any(student_dir: str) -> Dict[str, torch.Tensor]:
    st_path = os.path.join(student_dir, "model.safetensors")
    bin_path = os.path.join(student_dir, "pytorch_model.bin")

    if os.path.exists(st_path):
        from safetensors.torch import load_file
        return load_file(st_path)
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"Missing {st_path} and {bin_path} under {student_dir}.")

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
        text = f"[DESC]\n{desc}\n[REV]\n{rev}"
        enc = self.tok(text, truncation=True, max_length=self.max_length, padding=False, return_tensors=None)
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

class MultiOutputRegressor(nn.Module):
    def __init__(self, base_model_name: str, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name, config=self.config)
        hidden = getattr(self.config, "hidden_size", None) or self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
        )
        cls = out.last_hidden_state[:, 0, :]
        return self.head(self.dropout(cls))

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_dir", type=str, required=True)
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--desc_col", type=str, default="description")
    ap.add_argument("--review_col", type=str, default="review")
    ap.add_argument("--row_id_col", type=str, default="row_idx")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--clamp", action="store_true")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    meta_path = os.path.join(args.student_dir, "student_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}. Did you save with train_student_joint20.py?")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    K = int(meta["K"])
    slot_ids = meta.get("slot_ids", [f"S{i:02d}" for i in range(1, K + 1)])
    base_model_name = meta["model_name"]
    out_dim = int(meta["out_dim"])
    max_length = int(args.max_length) if args.max_length is not None else int(meta["max_length"])
    assert out_dim == 2 * K, f"Expected out_dim=2K, got out_dim={out_dim}, K={K}"

    df = pd.read_csv(args.csv_path)
    if args.row_id_col not in df.columns:
        df = df.reset_index(drop=True)
        df[args.row_id_col] = np.arange(len(df))

    tokenizer = AutoTokenizer.from_pretrained(args.student_dir, use_fast=True)

    model = MultiOutputRegressor(base_model_name, out_dim=out_dim, dropout=0.1)
    sd = load_state_dict_any(args.student_dir)
    model.load_state_dict(sd, strict=True)
    model.to(args.device).eval()

    ds = InferDataset(df, tokenizer, args.desc_col, args.review_col, max_length=max_length)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=Collator(tokenizer),
        num_workers=args.num_workers,
        pin_memory=("cuda" in args.device),
    )

    preds_all = np.zeros((len(df), out_dim), dtype=np.float32)

    print(f"[infer] rows={len(df)} K={K} out_dim={out_dim} device={args.device} batch={args.batch_size} max_length={max_length}")
    pbar = tqdm(dl, desc="JOINT infer", dynamic_ncols=True)

    for step, batch in enumerate(pbar, start=1):
        idxs = batch.pop("__idx__").cpu().numpy()
        batch = {k: v.to(args.device) for k, v in batch.items()}
        y = model(**batch).detach().float().cpu().numpy()  # [B,2K]
        preds_all[idxs] = y

        if (step % args.log_every) == 0:
            pbar.set_postfix({"batches": step, "seen": int(idxs.max()) + 1})

    if args.clamp:
        preds_all = clamp01_np(preds_all)

    # build output: keep original columns, append scores
    out = df.copy()

    for j, sid in enumerate(slot_ids):
        out[f"desc_{sid}_score"] = preds_all[:, j]
    for j, sid in enumerate(slot_ids):
        out[f"rev_{sid}_score"] = preds_all[:, K + j]

    out.to_csv(args.out_csv, index=False)
    print("[saved]", args.out_csv, "rows=", len(out), "added_cols=", 2 * K)

if __name__ == "__main__":
    main()