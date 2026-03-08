import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
RATING_BINS_5 = [-np.inf, 4.72, 4.81, 4.88, 4.93, np.inf]
RATING_LABELS_5 = [0, 1, 2, 3, 4]


def parse_args():
    parser = argparse.ArgumentParser(
        description="5-class rating classification: TF-IDF vs TF-IDF + mismatch_proxy"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Mismatch_Score/am/llm_mismatch_score.csv",
        help="CSV path containing review text, rating, and mismatch_proxy",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="review",
        help="Text column for TF-IDF features",
    )
    parser.add_argument(
        "--mismatch-col",
        type=str,
        default="mismatch_proxy",
        help="Mismatch feature column",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=30000,
        help="Max TF-IDF features",
    )
    return parser.parse_args()


def make_y5(rating_series: pd.Series) -> pd.Series:
    y5 = pd.cut(
        rating_series,
        bins=RATING_BINS_5,
        labels=RATING_LABELS_5,
        right=True,
        include_lowest=True,
    )
    return y5.astype("int64")


def build_xy(df: pd.DataFrame, text_col: str, mismatch_col: str):
    missing_cols = [c for c in [text_col, "rating", mismatch_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    work = df[[text_col, "rating", mismatch_col]].copy()
    work[text_col] = work[text_col].fillna("").astype(str)
    work[mismatch_col] = pd.to_numeric(work[mismatch_col], errors="coerce")
    work["rating"] = pd.to_numeric(work["rating"], errors="coerce")
    work = work.dropna(subset=[mismatch_col, "rating"]).reset_index(drop=True)

    y = make_y5(work["rating"])
    x_text = work[text_col]
    x_mismatch = work[mismatch_col].astype(float).values.reshape(-1, 1)
    return x_text, x_mismatch, y


def train_eval(x_train, x_test, y_train, y_test):
    clf = LogisticRegression(
        max_iter=2000,
        solver="saga",
        multi_class="multinomial",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "weighted_f1": f1_score(y_test, pred, average="weighted"),
        "report": classification_report(y_test, pred, digits=4),
    }


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path)
    x_text, x_mismatch, y = build_xy(df, args.text_col, args.mismatch_col)

    print(f"Input: {input_path}")
    print(f"Rows used: {len(y)}")
    print("Fixed 5-class bins:", RATING_BINS_5)
    print("Class distribution:")
    for cls, cnt in y.value_counts().sort_index().items():
        print(f"  class {cls}: {cnt} ({cnt / len(y):.4%})")

    x_train_text, x_test_text, y_train, y_test, m_train, m_test = train_test_split(
        x_text,
        y,
        x_mismatch,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        max_features=args.max_features,
    )
    x_train_tfidf = vec.fit_transform(x_train_text)
    x_test_tfidf = vec.transform(x_test_text)

    print("\n=== Model A: TF-IDF only ===")
    res_a = train_eval(x_train_tfidf, x_test_tfidf, y_train, y_test)
    print(f"accuracy   : {res_a['accuracy']:.4f}")
    print(f"macro_f1   : {res_a['macro_f1']:.4f}")
    print(f"weighted_f1: {res_a['weighted_f1']:.4f}")
    print(res_a["report"])

    print("\n=== Model B: TF-IDF + mismatch_proxy ===")
    x_train_mix = hstack([x_train_tfidf, csr_matrix(m_train)], format="csr")
    x_test_mix = hstack([x_test_tfidf, csr_matrix(m_test)], format="csr")
    res_b = train_eval(x_train_mix, x_test_mix, y_train, y_test)
    print(f"accuracy   : {res_b['accuracy']:.4f}")
    print(f"macro_f1   : {res_b['macro_f1']:.4f}")
    print(f"weighted_f1: {res_b['weighted_f1']:.4f}")
    print(res_b["report"])

    print("\n=== Delta (B - A) ===")
    print(f"accuracy   : {res_b['accuracy'] - res_a['accuracy']:+.4f}")
    print(f"macro_f1   : {res_b['macro_f1'] - res_a['macro_f1']:+.4f}")
    print(f"weighted_f1: {res_b['weighted_f1'] - res_a['weighted_f1']:+.4f}")


if __name__ == "__main__":
    main()
