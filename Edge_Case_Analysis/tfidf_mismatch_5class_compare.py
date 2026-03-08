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

# results

# NY
# Input: Mismatch_Score\ny\llm_mismatch_score.csv
# Rows used: 13561
# Fixed 5-class bins: [-inf, 4.72, 4.81, 4.88, 4.93, inf]
# Class distribution:
#   class 0: 2501 (18.4426%)
#   class 1: 3159 (23.2947%)
#   class 2: 1413 (10.4196%)
#   class 3: 2309 (17.0268%)
#   class 4: 4179 (30.8163%)

# === Model A: TF-IDF only ===
# accuracy   : 0.5691
# macro_f1   : 0.4997
# weighted_f1: 0.5474
#               precision    recall  f1-score   support

#            0     0.5032    0.4700    0.4860       500
#            1     0.5022    0.5538    0.5267       632
#            2     0.5625    0.1590    0.2479       283
#            3     0.6415    0.4416    0.5231       462
#            4     0.6169    0.8493    0.7146       836

#     accuracy                         0.5691      2713
#    macro avg     0.5652    0.4947    0.4997      2713
# weighted avg     0.5677    0.5691    0.5474      2713


# === Model B: TF-IDF + mismatch_proxy ===
# accuracy   : 0.5717
# macro_f1   : 0.5021
# weighted_f1: 0.5501
#               precision    recall  f1-score   support

#            0     0.5064    0.4780    0.4918       500
#            1     0.5028    0.5585    0.5292       632
#            2     0.5696    0.1590    0.2486       283
#            3     0.6395    0.4416    0.5224       462
#            4     0.6223    0.8493    0.7183       836

#     accuracy                         0.5717      2713
#    macro avg     0.5681    0.4973    0.5021      2713
# weighted avg     0.5705    0.5717    0.5501      2713


# === Delta (B - A) ===
# accuracy   : +0.0026
# macro_f1   : +0.0024
# weighted_f1: +0.0027



# AM
# Input: Mismatch_Score\am\llm_mismatch_score.csv
# Rows used: 152146
# Fixed 5-class bins: [-inf, 4.72, 4.81, 4.88, 4.93, inf]
# Class distribution:
#   class 0: 23798 (15.6416%)
#   class 1: 26562 (17.4582%)
#   class 2: 34309 (22.5501%)
#   class 3: 31832 (20.9220%)
#   class 4: 35645 (23.4282%)

# === Model A: TF-IDF only ===
# accuracy   : 0.5635
# macro_f1   : 0.5549
# weighted_f1: 0.5606
#               precision    recall  f1-score   support

#            0     0.5196    0.5368    0.5281      4760
#            1     0.5446    0.4335    0.4828      5312
#            2     0.5589    0.5532    0.5560      6862
#            3     0.6253    0.5472    0.5836      6367
#            4     0.5615    0.7026    0.6242      7129

#     accuracy                         0.5635     30430
#    macro avg     0.5620    0.5547    0.5549     30430
# weighted avg     0.5648    0.5635    0.5606     30430


# === Model B: TF-IDF + mismatch_proxy ===
# accuracy   : 0.5637
# macro_f1   : 0.5552
# weighted_f1: 0.5608
#               precision    recall  f1-score   support

#            0     0.5192    0.5384    0.5287      4760
#            1     0.5455    0.4354    0.4843      5312
#            2     0.5603    0.5525    0.5564      6862
#            3     0.6244    0.5470    0.5832      6367
#            4     0.5614    0.7016    0.6237      7129

#     accuracy                         0.5637     30430
#    macro avg     0.5622    0.5550    0.5552     30430
# weighted avg     0.5650    0.5637    0.5608     30430


# === Delta (B - A) ===
# accuracy   : +0.0002
# macro_f1   : +0.0003
# weighted_f1: +0.0002



# MO
# Input: Mismatch_Score\mo\llm_mismatch_score.csv
# Rows used: 140236
# Fixed 5-class bins: [-inf, 4.72, 4.81, 4.88, 4.93, inf]
# Class distribution:
#   class 0: 30371 (21.6571%)
#   class 1: 30318 (21.6193%)
#   class 2: 33275 (23.7279%)
#   class 3: 22572 (16.0957%)
#   class 4: 23700 (16.9001%)
    
# === Model A: TF-IDF only ===
# accuracy   : 0.4272
# macro_f1   : 0.4272
# weighted_f1: 0.4248
#               precision    recall  f1-score   support

#            0     0.4410    0.4960    0.4669      6074
#            1     0.3671    0.3386    0.3522      6064
#            2     0.3738    0.4488    0.4079      6655
#            3     0.4574    0.2999    0.3623      4515
#            4     0.5508    0.5430    0.5469      4740

#     accuracy                         0.4272     28048
#    macro avg     0.4380    0.4253    0.4272     28048
# weighted avg     0.4303    0.4272    0.4248     28048


# === Model B: TF-IDF + mismatch_proxy ===
# accuracy   : 0.4275
# macro_f1   : 0.4277
# weighted_f1: 0.4252
#               precision    recall  f1-score   support

#            0     0.4433    0.4952    0.4678      6074
#            1     0.3688    0.3386    0.3531      6064
#            2     0.3724    0.4488    0.4071      6655
#            3     0.4579    0.3021    0.3640      4515
#            4     0.5491    0.5441    0.5466      4740

#     accuracy                         0.4275     28048
#    macro avg     0.4383    0.4258    0.4277     28048
# weighted avg     0.4306    0.4275    0.4252     28048


# === Delta (B - A) ===
# accuracy   : +0.0004
# macro_f1   : +0.0005
# weighted_f1: +0.0004