#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data-driven K=10 aspect space with a local open-source LLM (e.g., Qwen2.5-Instruct):
1) Mine candidate phrases (2-3 grams) from descriptions + reviews (debiased by listing)
2) Ask LLM to DEFINE 10 slots (aspect names + definitions) from candidates (data-driven)
   - IMPORTANT: slots contain only {slot_id, name, definition} (NO example_phrases here)
3) Ask LLM to ASSIGN each phrase to one of the 10 slots (or DROP)
4) Ask LLM to SCORE each row (desc + review) ONCE, output full 10-d vectors:
   - desc_expectation in [0,1] with mentioned flag
   - review_experience in [-1,1] with mentioned flag
   - Every slot MUST appear; if no evidence -> mentioned=false, score=0

Outputs in out_dir:
- phrases.json                     (candidate phrases)
- slots.json                       (10 slots: id/name/definition)
- phrase_assignment.json           (phrase -> slot_id/DROP)
- taxonomy.json                    (slots + assigned phrases + auto example_phrases)
- scores.csv                       (per-row vectors)

Assumes CSV has: listing_id, description, review (names configurable).
"""

from __future__ import annotations

import argparse, json, os, re, time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# -----------------------
# env/cache (optional)
# -----------------------
HF_CACHE = os.environ.get("HF_CACHE", "/data/fengfei/hf_cache")
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE, "hub")


# -----------------------
# utils
# -----------------------
_WS_RE = re.compile(r"\s+")
_NONASCII_RE = re.compile(r"[^\x00-\x7F]+")

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u200b", " ")
    s = _NONASCII_RE.sub(" ", s)
    s = s.lower()
    s = _WS_RE.sub(" ", s).strip()
    return s

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None

def extract_first_json(text: str) -> Optional[dict]:
    """
    Best-effort: extract first valid JSON object from a possibly messy model output.
    """
    if not text:
        return None
    t = text.strip()
    t = t.replace("```json", "").replace("```", "").strip()

    obj = safe_json_loads(t)
    if isinstance(obj, dict):
        return obj

    first = t.find("{")
    last = t.rfind("}")
    if first != -1 and last != -1 and last > first:
        cand = t[first:last+1]
        obj = safe_json_loads(cand)
        if isinstance(obj, dict):
            return obj

    starts = [i for i, ch in enumerate(t) if ch == "{"]

    for s in starts:
        depth = 0
        for i in range(s, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    cand = t[s:i+1]
                    obj = safe_json_loads(cand)
                    if isinstance(obj, dict):
                        return obj
                    break
    return None

def chunk_list(xs: List[Any], n: int) -> List[List[Any]]:
    return [xs[i:i+n] for i in range(0, len(xs), n)]


# -----------------------
# debias sampling
# -----------------------
def debiased_sample_by_listing(df: pd.DataFrame, listing_col: str, max_rows_per_listing: int, seed: int) -> pd.DataFrame:
    if max_rows_per_listing <= 0:
        return df.copy()
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby(listing_col, sort=False):
        if len(g) <= max_rows_per_listing:
            parts.append(g)
        else:
            idx = rng.choice(g.index.to_numpy(), size=max_rows_per_listing, replace=False)
            parts.append(df.loc[idx])
    return pd.concat(parts, axis=0).reset_index(drop=True)


# -----------------------
# phrase mining (2-3 grams only)
# -----------------------
@dataclass
class PhraseMiningConfig:
    ngram_min: int = 2
    ngram_max: int = 3
    stop_words: str = "english"
    max_features: int = 200000
    min_df: int = 8
    max_df: float = 0.4

def mine_top_phrases(texts: List[str], top_n: int, cfg: PhraseMiningConfig) -> List[str]:
    texts = [clean_text(t) for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return []
    cv = CountVectorizer(
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        stop_words=cfg.stop_words,
        max_features=cfg.max_features,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
    )
    X = cv.fit_transform(texts)  # counts

    # tf * idf global ranking
    df = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float64)
    N = float(X.shape[0])
    idf = np.log((1.0 + N) / (1.0 + df)) + 1.0
    tf = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
    scores = tf * idf
    vocab = np.array(cv.get_feature_names_out())
    idx = np.argsort(-scores)[: min(top_n, len(scores))]
    phrases = vocab[idx].tolist()
    return phrases


# -----------------------
# LLM wrapper (Transformers, chat-template + generate)
# -----------------------
@dataclass
class LLMConfig:
    model_id: str = "Qwen/Qwen2.5-14B-Instruct"
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    max_new_tokens: int = 512
    max_retries: int = 3
    sleep_between_retries: float = 0.3

class LLM:
    def __init__(self, cfg: LLMConfig):
        print("[LLM] loading:", cfg.model_id, flush=True)
        self.cfg = cfg
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype="auto",
            device_map={"": 0},
        ).eval()

        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id

    def call(self, messages: List[Dict[str, str]]) -> str:
        # IMPORTANT: return_dict=True to get attention_mask
        enc = self.tok.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        in_len = enc["input_ids"].shape[1]

        gen = self.model.generate(
            **enc,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True if self.cfg.temperature > 0 else False,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            repetition_penalty=self.cfg.repetition_penalty,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id,
            use_cache=True,
        )

        new_tokens = gen[0, in_len:]
        text = self.tok.decode(new_tokens, skip_special_tokens=True).strip()

        low = text.lower()
        if low.startswith("analysis"):
            text = text[len("analysis"):].lstrip()
        if text.lower().startswith("final"):
            text = text[len("final"):].lstrip()
        return text

    def call_json(self, messages):
        last_err = None
        msgs = list(messages)

        for attempt in range(self.cfg.max_retries):
            txt = self.call(msgs)

            obj = extract_first_json(txt)
            if isinstance(obj, dict):
                return obj

            last_err = f"JSON parse failed. Head: {txt[:200]}"

            msgs = list(messages) + [{
                "role": "user",
                "content": (
                    "Your last response was not valid JSON.\n"
                    "Now output ONLY a single valid JSON object.\n"
                    "Output must start with '{' and end with '}'.\n"
                    "No prose, no markdown, no explanation."
                )
            }]

            time.sleep(self.cfg.sleep_between_retries)

        return {"_error": last_err or "unknown"}

# -----------------------
# Step 1: LLM defines 10 slots (data-driven, NO example_phrases)
# -----------------------
def llm_define_slots(llm: LLM, phrases: List[str], K: int, sample_size: int) -> Dict[str, Any]:
    phrases = [p for p in phrases if p and isinstance(p, str)]
    rng = np.random.default_rng(0)
    if len(phrases) > sample_size:
        phrases_in = rng.choice(np.array(phrases), size=sample_size, replace=False).tolist()
    else:
        phrases_in = phrases

    # Strong JSON-only with a tiny few-shot (forces it to actually output JSON)
    sys = (
        "You MUST output ONLY a single JSON object. No prose.\n"
        "FIRST char must be '{' and LAST must be '}'.\n"
        "Use double quotes for JSON strings. No trailing commas.\n"
        "Goal: define K aspect slots (semantic dimensions) for Airbnb stays.\n"
        "These slots will be used to score BOTH host descriptions (expectations) and guest reviews (experiences).\n"
    )

    few_user = (
        "Define K=2 slots.\n"
        "Candidate phrases:\n- clean apartment\n- spotless place\n- great location\n- minutes walk\n"
        "Return JSON with schema: {\"K\":2,\"slots\":[{\"slot_id\":\"S01\",\"name\":\"...\",\"definition\":\"...\"}]}\n"
        "No extra keys."
    )
    few_assistant = (
        "{\"K\":2,\"slots\":["
        "{\"slot_id\":\"S01\",\"name\":\"cleanliness\",\"definition\":\"How clean and tidy the space is\"},"
        "{\"slot_id\":\"S02\",\"name\":\"location\",\"definition\":\"Convenience and proximity to nearby places\"}"
        "]}"
    )

    user = (
        f"Define exactly K={K} aspect slots from the candidate phrases below.\n"
        "Return EXACTLY one JSON object with schema (no extra keys):\n"
        f'{{"K":{K},"slots":[{{"slot_id":"S01","name":"...","definition":"..."}}]}}\n'
        "Hard rules:\n"
        f"- Output exactly {K} slots with slot_id S01..S{K:02d}.\n"
        "- Each name must be short (2-4 words), snake_case or simple words.\n"
        "- Each definition must be 1 sentence describing what evidence counts.\n"
        "- Slots must be mutually distinct and cover the major guest-relevant aspects.\n\n"
        "Candidate phrases:\n- " + "\n- ".join(phrases_in)
    )

    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": few_user},
        {"role": "assistant", "content": few_assistant},
        {"role": "user", "content": user},
    ]
    return llm.call_json(messages)


# -----------------------
# Step 2: LLM assigns phrases to slots (semantic clustering)
# -----------------------
def llm_assign_phrases(llm: LLM, slots: List[dict], phrases: List[str], batch_size: int) -> Dict[str, str]:
    """
    Returns mapping: phrase -> slot_id (or "DROP")
    """
    slot_lines = []
    for s in slots:
        sid = s["slot_id"]
        name = s.get("name", "").strip()
        definition = s.get("definition", "").strip()
        slot_lines.append(f"{sid}: {name} — {definition}")
    slot_spec = "\n".join(slot_lines)

    mapping: Dict[str, str] = {}

    sys = (
        "You assign phrases into one of the given slots by meaning.\n"
        "Output JSON only. No prose.\n"
        "Rules:\n"
        "- Choose exactly one slot_id for each phrase, or 'DROP' if too generic/unclear.\n"
        "- Do not invent new slots.\n"
        "- Use the slot definitions to decide, not surface word overlap.\n"
    )

    for chunk in chunk_list(phrases, batch_size):
        chunk = [p for p in chunk if p and isinstance(p, str)]
        if not chunk:
            continue

        user = (
            "Slots:\n" + slot_spec + "\n\n"
            "Assign each phrase.\n"
            "Return JSON:\n"
            '{"assignments":[{"phrase":"...","slot_id":"S01|S02|...|DROP"}, ...]}\n\n'
            "Phrases:\n- " + "\n- ".join(chunk)
        )
        obj = llm.call_json([{"role": "system", "content": sys}, {"role": "user", "content": user}])
        assigns = obj.get("assignments", [])
        if isinstance(assigns, list):
            for a in assigns:
                if not isinstance(a, dict):
                    continue
                ph = clean_text(a.get("phrase", ""))
                sid = str(a.get("slot_id", "")).strip()
                if ph:
                    mapping[ph] = sid

    return mapping


# -----------------------
# Post-process: auto example_phrases from assigned phrases (data-driven)
# -----------------------
def select_examples_for_slot(phrases: List[str], max_examples: int) -> List[str]:
    """
    Simple heuristic: prefer longer phrases (more specific), de-duplicate by token set.
    """
    uniq = []
    seen = set()
    # sort: longer first, then lex
    for p in sorted(set(phrases), key=lambda x: (-len(x.split()), -len(x), x)):
        key = tuple(p.split())  # crude; good enough
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
        if len(uniq) >= max_examples:
            break
    return uniq


# -----------------------
# Step 3: Score each row once (desc+review) on full K=10
# -----------------------
def build_scoring_prompt(slots: List[dict]) -> Tuple[str, str]:
    """
    Teacher-friendly scoring prompt (same space):
    - only uses aspect names (no definitions)
    - allows mild inference to avoid all-zeros
    - outputs compact arrays d/r
    - BOTH d and r are experience-level vectors in [0,1]
    """
    aspect_order = [s["slot_id"] for s in slots]
    aspect_names = [s.get("name", "").strip() for s in slots]

    aspects_text = "\n".join(
        [f"{i+1}. {sid} | {name}" for i, (sid, name) in enumerate(zip(aspect_order, aspect_names))]
    )

    system = (
        "Output ONLY valid JSON. No prose. No markdown.\n"
        "Return exactly one JSON object.\n"
    )

    user_prefix = (
        f"You will score Airbnb texts on K={len(slots)} aspects.\n"
        "You will receive items with:\n"
        "- description: host-written listing description (CLAIMED experience level)\n"
        "- review: guest-written review (ACTUAL experienced level)\n\n"

        "CRITICAL: d and r MUST be in the SAME numeric space [0,1].\n"
        "They both represent EXPERIENCE LEVEL / QUALITY LEVEL for each aspect.\n"
        "- 0.0 = clearly bad / absent / unusable / strong complaint about this aspect.\n"
        "- 0.2 = weak / limited / minor issues.\n"
        "- 0.5 = normal/okay, clearly present with acceptable quality.\n"
        "- 0.8 = very good.\n"
        "- 1.0 = exceptional / superlative.\n\n"

        "Output for each item:\n"
        "- d: length-K vector in [0,1] for the DESCRIPTION's claimed level for each aspect.\n"
        "- r: length-K vector in [0,1] for the REVIEW's actual experienced level for each aspect.\n\n"

        "IMPORTANT RULES (teacher mode):\n"
        "1) Do NOT be overly strict. Mild inference is allowed when it is natural in Airbnb context.\n"
        "   Examples:\n"
        "   - 'slept great' => sleep_quality around 0.7-0.9\n"
        "   - 'easy walk to downtown' => location/accessibility around 0.6-0.8\n"
        "   - 'spotless' / 'pristine' => cleanliness around 0.9-1.0\n"
        "2) Do NOT fabricate: if there is no evidence at all, use 0.0.\n"
        "3) Use 0.0 ONLY if the aspect is neither mentioned nor clearly implied.\n"
        "   Use 0.1–0.2 if weakly implied.\n"
        "   Use 0.4–0.6 when clearly mentioned.\n"
        "   Use 0.8–1.0 only for strong emphasis / multiple mentions / superlatives.\n"
        "4) Negative review evidence MUST reduce r:\n"
        "   - 'dirty', 'bugs', 'smelled' => cleanliness near 0.0-0.2\n"
        "   - 'noisy', 'couldn't sleep' => noise_level/sleep_quality near 0.0-0.3\n"
        "5) Mixed review evidence (both good and bad): pick a middle value (0.3-0.6).\n"
        "6) Generic overall praise/complaint (e.g., 'great stay', 'would return'):\n"
        "   - Put it only into an overall/guest_experience aspect IF it exists.\n"
        "   - Otherwise do NOT spread it to all aspects.\n"
        "7) Do not conflate aspects (kitchen vs bathroom; location vs neighborhood vs noise).\n\n"

        "Output JSON format (exact):\n"
        "{\"results\":[{\"i\":0,\"d\":[K numbers],\"r\":[K numbers]}, ...]}\n"
        "Numbers must be decimals in [0,1]. No strings. No NaN.\n\n"

        "Aspects (fixed order for arrays):\n"
        f"{aspects_text}\n\n"
    )

    return system, user_prefix

def llm_score_rows(
    llm: LLM,
    slots: List[dict],
    df: pd.DataFrame,
    desc_col: str,
    review_col: str,
    batch_size_rows: int = 8,
    max_chars_desc: int = 600,
    max_chars_review: int = 500,
) -> List[dict]:
    system, user_prefix = build_scoring_prompt(slots)
    K = len(slots)

    outputs: List[dict] = []
    rows = list(df.itertuples(index=False))

    for chunk in chunk_list(rows, batch_size_rows):
        items = []
        for r in chunk:
            d = getattr(r, desc_col)
            v = getattr(r, review_col)
            d = (str(d) if d is not None else "")[:max_chars_desc]
            v = (str(v) if v is not None else "")[:max_chars_review]
            items.append({"description": d, "review": v})

        user = user_prefix + "Items JSON:\n" + json.dumps(items, ensure_ascii=False)
        obj = llm.call_json([{"role":"system","content":system},{"role":"user","content":user}])

        if isinstance(obj, dict) and "_error" in obj:
            print("[warn] call_json failed:", obj["_error"])
            continue

        res = obj.get("results", [])
        if not isinstance(res, list):
            print("[warn] results is not list. head:", str(obj)[:200])
            continue

        for rr in res:
            if not isinstance(rr, dict):
                continue

            d = rr.get("d", [0.0]*K)
            r = rr.get("r", [0.0]*K)

            if not isinstance(d, list): d = [0.0]*K
            if not isinstance(r, list): r = [0.0]*K

            d = (d + [0.0]*K)[:K]
            r = (r + [0.0]*K)[:K]

            d = [float(x) if isinstance(x, (int, float)) else 0.0 for x in d]
            r = [float(x) if isinstance(x, (int, float)) else 0.0 for x in r]
            d = [max(0.0, min(1.0, x)) for x in d]
            r = [max(-1.0, min(1.0, x)) for x in r]

            rr["d"] = d
            rr["r"] = r

            outputs.append(rr)

    return outputs

# -----------------------
# Build taxonomy object (adds example_phrases AFTER assignment)
# -----------------------
def build_taxonomy(K: int, slots_obj: dict, phrase_to_slot: Dict[str, str], max_examples: int = 10) -> dict:
    slots = slots_obj.get("slots", [])

    # normalize slots
    norm_slots = []
    for s in slots:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("slot_id", "")).strip()
        if not sid:
            continue
        norm_slots.append({
            "slot_id": sid,
            "name": str(s.get("name", "")).strip(),
            "definition": str(s.get("definition", "")).strip(),
            "phrases": [],
            "example_phrases": [],
        })

    slot_index = {s["slot_id"]: s for s in norm_slots}
    dropped = []

    for ph_raw, sid in phrase_to_slot.items():
        ph = clean_text(ph_raw)
        if not ph:
            continue
        if sid == "DROP" or sid not in slot_index:
            dropped.append(ph)
        else:
            slot_index[sid]["phrases"].append(ph)

    for s in norm_slots:
        s["phrases"] = sorted(set(s["phrases"]))
        s["example_phrases"] = select_examples_for_slot(s["phrases"], max_examples=max_examples)

    taxonomy = {
        "meta": {"K": K, "llm_datadriven_slots": True, "examples_from_assignment": True},
        "slots": norm_slots,
        "dropped_phrases": sorted(set(dropped)),
    }
    return taxonomy


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, default="./airbnb.csv")
    ap.add_argument("--listing_col", type=str, default="listing_id")
    ap.add_argument("--desc_col", type=str, default="description")
    ap.add_argument("--review_col", type=str, default="review")

    ap.add_argument("--out_dir", type=str, default="out_airbnb_k10")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--top_phrases_each", type=int, default=800)
    ap.add_argument("--max_rows_per_listing_mining", type=int, default=5)

    ap.add_argument("--llm_model_id", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--llm_temperature", type=float, default=0.0)
    ap.add_argument("--llm_max_new_tokens_slots", type=int, default=650)
    ap.add_argument("--llm_max_new_tokens_assign", type=int, default=450)
    ap.add_argument("--llm_max_new_tokens_score", type=int, default=650)

    ap.add_argument("--slot_sample_size", type=int, default=120, help="phrases shown to LLM when defining slots")
    ap.add_argument("--assign_phrase_batch", type=int, default=40, help="phrases per LLM call when assigning")
    ap.add_argument("--score_row_batch", type=int, default=2, help="rows per LLM call when scoring")
    ap.add_argument("--limit_rows", type=int, default=-1, help="debug: limit rows for scoring")
    ap.add_argument("--score_k_per_listing", type=int, default=2,
                help="when --score_unique_listing is on, take up to k rows per listing_id")

    ap.add_argument("--example_phrases_per_slot", type=int, default=10, help="examples selected from assigned phrases")

    ap.add_argument("--score_n", type=int, default=500, help="how many rows to score with LLM (<=0 means score all)")
    ap.add_argument("--score_seed", type=int, default=0)
    ap.add_argument("--score_unique_listing", action="store_true", help="sample by unique listing_id first")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    print("[args.llm_model_id] =", args.llm_model_id, flush=True)

    df = pd.read_csv(args.csv_path)
    for c in [args.listing_col, args.desc_col, args.review_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}. Columns: {list(df.columns)}")

    if args.limit_rows and args.limit_rows > 0:
        df = df.head(args.limit_rows).copy()

    # ---- phrase mining (debiased) ----
    df_mine = debiased_sample_by_listing(df, args.listing_col, args.max_rows_per_listing_mining, seed=args.seed)
    desc_texts = df_mine[args.desc_col].fillna("").astype(str).tolist()
    rev_texts  = df_mine[args.review_col].fillna("").astype(str).tolist()

    pm = PhraseMiningConfig(
        ngram_min=1,
        ngram_max=3,
        min_df=max(10, int(0.005 * len(df_mine))),
        max_df=0.25,
    )

    desc_ph = mine_top_phrases(desc_texts, args.top_phrases_each, pm)
    rev_ph  = mine_top_phrases(rev_texts,  args.top_phrases_each, pm)
    candidates = sorted(set([clean_text(p) for p in (desc_ph + rev_ph) if clean_text(p)]))

    with open(os.path.join(args.out_dir, "phrases.json"), "w", encoding="utf-8") as f:
        json.dump({"num_candidates": len(candidates), "candidates": candidates}, f, indent=2)
    print(f"[saved] {os.path.join(args.out_dir,'phrases.json')} (candidates={len(candidates)})")

    # ---- Step 1: define slots (NO example_phrases) ----
    llm_slots = LLM(LLMConfig(
        model_id=args.llm_model_id,
        temperature=args.llm_temperature,
        max_new_tokens=args.llm_max_new_tokens_slots,
    ))
    slots_obj = llm_define_slots(llm_slots, candidates, K=args.K, sample_size=args.slot_sample_size)
    with open(os.path.join(args.out_dir, "slots.json"), "w", encoding="utf-8") as f:
        json.dump(slots_obj, f, indent=2)
    print(f"[saved] {os.path.join(args.out_dir,'slots.json')}")

    if "_error" in slots_obj or "slots" not in slots_obj:
        raise RuntimeError(f"LLM slot definition failed: {slots_obj.get('_error')}")

    slots = slots_obj["slots"]

    # ---- Step 2: assign phrases to slots ----
    llm_assign = LLM(LLMConfig(
        model_id=args.llm_model_id,
        temperature=args.llm_temperature,
        max_new_tokens=args.llm_max_new_tokens_assign,
    ))
    phrase_to_slot = llm_assign_phrases(llm_assign, slots, candidates, batch_size=args.assign_phrase_batch)
    with open(os.path.join(args.out_dir, "phrase_assignment.json"), "w", encoding="utf-8") as f:
        json.dump({"num_assigned": len(phrase_to_slot), "mapping": phrase_to_slot}, f, indent=2)
    print(f"[saved] {os.path.join(args.out_dir,'phrase_assignment.json')} (assigned={len(phrase_to_slot)})")

    taxonomy = build_taxonomy(args.K, slots_obj, phrase_to_slot, max_examples=args.example_phrases_per_slot)
    with open(os.path.join(args.out_dir, "taxonomy.json"), "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2)
    print(f"[saved] {os.path.join(args.out_dir,'taxonomy.json')}")


    # ---- load existing taxonomy (skip mining/slot/assignment) ----
    taxonomy_path = os.path.join(args.out_dir, "taxonomy.json")
    if not os.path.exists(taxonomy_path):
        raise FileNotFoundError(f"taxonomy.json not found at {taxonomy_path}. Run taxonomy build first or provide correct --out_dir.")

    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    if "slots" not in taxonomy:
        raise ValueError("taxonomy.json missing 'slots' field.")
    print(f"[load] {taxonomy_path} (K={len(taxonomy['slots'])})")


    # ---- choose rows for LLM scoring (teacher labels) ----
    df_all = df.copy().reset_index(drop=True)
    df_all["row_idx"] = np.arange(len(df_all))  # keep global row id for later joining

    if args.score_n is not None and args.score_n > 0:
        rng = np.random.default_rng(args.score_seed)

        if args.score_unique_listing:
            # pick up to k rows per listing for diversity
            k = args.score_k_per_listing  # e.g. 3 or 4

            df_shuf = df_all.sample(frac=1.0, random_state=args.score_seed).reset_index(drop=True)
            # groupby on listing_id, take up to k rows each
            df_k = (
                df_shuf.groupby(args.listing_col, sort=False, group_keys=False)
                    .head(k)
                    .reset_index(drop=True)
            )

            take = min(args.score_n, len(df_k))
            df_score = df_k.head(take).copy()
        else:
            take = min(args.score_n, len(df_all))
            idx = rng.choice(df_all.index.to_numpy(), size=take, replace=False)
            df_score = df_all.loc[idx].copy().reset_index(drop=True)

    else:
        df_score = df_all.copy()


    # ---- Step 3: score rows (desc+review) ----
    llm_score = LLM(LLMConfig(
        model_id=args.llm_model_id,
        temperature=args.llm_temperature,
        max_new_tokens=args.llm_max_new_tokens_score,
    ))

    score_objs = llm_score_rows(
        llm_score,
        slots=taxonomy["slots"],
        df=df_score[[args.desc_col, args.review_col]].copy(),
        desc_col=args.desc_col,
        review_col=args.review_col,
        batch_size_rows=args.score_row_batch
    )

    # ---- flatten scores to CSV ----
    if not score_objs:
        print("[error] score_objs is empty. LLM scoring returned 0 results.")
        print("[hint] likely JSON parse failures or model returned empty results.")
        return
    print("[debug] score_objs[0] =", score_objs[0])


    Kslots = [s["slot_id"] for s in taxonomy["slots"]]
    rows_out = []

    n_out = min(len(df_score), len(score_objs))  # safety
    if n_out < len(df_score):
        print(f"[warn] LLM returned {len(score_objs)} results for {len(df_score)} inputs")

    for local_i in range(n_out):
        if (local_i + 1) % 20 == 0 or local_i == 0:
            print(f"[progress] processed {local_i+1}/{n_out} rows", flush=True)

        global_row_idx = int(df_score.iloc[local_i]["row_idx"])
        so = score_objs[local_i] if isinstance(score_objs[local_i], dict) else {}

        row = {
            "row_idx": global_row_idx,
            "description": df_score.iloc[local_i][args.desc_col],
            "review": df_score.iloc[local_i][args.review_col],
        }

        d = so.get("d", [])
        r = so.get("r", [])
        if not isinstance(d, list): d = []
        if not isinstance(r, list): r = []

        for j, sid in enumerate(Kslots):
            d_s = float(d[j]) if j < len(d) else 0.0
            r_s = float(r[j]) if j < len(r) else 0.0
            row[f"desc_{sid}_score"] = max(0.0, min(1.0, d_s))
            row[f"rev_{sid}_score"]  = max(-1.0, min(1.0, r_s))

        rows_out.append(row)

    teacher_df = pd.DataFrame(rows_out)
    teacher_path = os.path.join(args.out_dir, f"teacher_scores_{len(teacher_df)}.csv")
    teacher_df.to_csv(teacher_path, index=False)
    print(f"[saved] {teacher_path} (rows={len(teacher_df)}, K={len(Kslots)})")

if __name__ == "__main__":
    main()
