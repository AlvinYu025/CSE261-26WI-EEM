import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


RANDOM_STATE = 42
RATING_BINS_5 = [-np.inf, 4.72, 4.81, 4.88, 4.93, np.inf]
RATING_LABELS_5 = [0, 1, 2, 3, 4]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation: TF-IDF + different numeric features for 5-class rating classification"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Mismatch_Score/am/llm_mismatch_score.csv",
        help="Input CSV path",
    )
    parser.add_argument("--text-col", type=str, default="review", help="Text column")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--max-features", type=int, default=12000, help="Max TF-IDF vocab size")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=60000,
        help="Optional stratified sample size before train/test split; set <=0 to use full data",
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


def zscore_fit_transform(x_train: np.ndarray, x_test: np.ndarray):
    mu = np.nanmean(x_train, axis=0, keepdims=True)
    sd = np.nanstd(x_train, axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (x_train - mu) / sd, (x_test - mu) / sd


def train_eval(x_train, x_test, y_train, y_test):
    clf = LinearSVC(random_state=RANDOM_STATE, max_iter=3000)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "weighted_f1": f1_score(y_test, pred, average="weighted"),
    }


def build_extra_matrix(work: pd.DataFrame, idx_train, idx_test, num_cols, cat_cols):
    parts_train = []
    parts_test = []

    if num_cols:
        train_num = work.loc[idx_train, num_cols].copy()
        test_num = work.loc[idx_test, num_cols].copy()
        train_num = train_num.fillna(train_num.mean())
        test_num = test_num.fillna(train_num.mean())
        x_train_num = train_num.values.astype(float)
        x_test_num = test_num.values.astype(float)
        x_train_num, x_test_num = zscore_fit_transform(x_train_num, x_test_num)
        parts_train.append(csr_matrix(x_train_num))
        parts_test.append(csr_matrix(x_test_num))

    if cat_cols:
        train_cat = work.loc[idx_train, cat_cols].copy().fillna("__MISSING__").astype(str)
        test_cat = work.loc[idx_test, cat_cols].copy().fillna("__MISSING__").astype(str)
        train_cat_oh = pd.get_dummies(train_cat, columns=cat_cols, drop_first=False)
        test_cat_oh = pd.get_dummies(test_cat, columns=cat_cols, drop_first=False)
        train_cat_oh, test_cat_oh = train_cat_oh.align(test_cat_oh, join="outer", axis=1, fill_value=0)
        parts_train.append(csr_matrix(train_cat_oh.values.astype(float)))
        parts_test.append(csr_matrix(test_cat_oh.values.astype(float)))

    if not parts_train:
        return None, None
    if len(parts_train) == 1:
        return parts_train[0], parts_test[0]
    return hstack(parts_train, format="csr"), hstack(parts_test, format="csr")


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path)
    needed = [args.text_col, "rating"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    review_score_cols = [
        "review_scores_accuracy",
        "review_scores_cleanliness",
        "review_scores_checkin",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_value",
    ]

    feature_sets = {
        "TF-IDF only": {"num": [], "cat": []},
        "TF-IDF + mismatch_proxy": {"num": ["mismatch_proxy"], "cat": []},
        "TF-IDF + mabs": {"num": ["mabs"], "cat": []},
        "TF-IDF + mover": {"num": ["mover"], "cat": []},
        "TF-IDF + munder": {"num": ["munder"], "cat": []},
        "TF-IDF + price": {"num": ["price"], "cat": []},
        "TF-IDF + availability_30": {"num": ["availability_30"], "cat": []},
        "TF-IDF + availability_60": {"num": ["availability_60"], "cat": []},
        "TF-IDF + neighborhood": {"num": [], "cat": ["neighborhood"]},
        "TF-IDF + review_scores_all": {"num": review_score_cols, "cat": []},
    }

    available_sets = {}
    for name, spec in feature_sets.items():
        need_cols = spec["num"] + spec["cat"]
        if all(c in df.columns for c in need_cols):
            available_sets[name] = spec

    keep_cols = [args.text_col, "rating"]
    for spec in available_sets.values():
        keep_cols.extend(spec["num"])
        keep_cols.extend(spec["cat"])
    keep_cols = sorted(set(keep_cols))
    work = df[keep_cols].copy()
    work[args.text_col] = work[args.text_col].fillna("").astype(str)
    work["rating"] = pd.to_numeric(work["rating"], errors="coerce")
    numeric_cols = sorted({c for spec in available_sets.values() for c in spec["num"]})
    for c in numeric_cols:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.dropna(subset=["rating"]).reset_index(drop=True)
    y = make_y5(work["rating"])

    if args.sample_size and args.sample_size > 0 and args.sample_size < len(work):
        frac = args.sample_size / len(work)
        sampled = (
            work.assign(_y=y)
            .groupby("_y", group_keys=False)
            .apply(lambda g: g.sample(max(1, int(round(len(g) * frac))), random_state=RANDOM_STATE))
            .reset_index(drop=True)
        )
        y = sampled["_y"].astype("int64")
        work = sampled.drop(columns=["_y"]).reset_index(drop=True)

    x_text_train, x_text_test, y_train, y_test, idx_train, idx_test = train_test_split(
        work[args.text_col],
        y,
        work.index.values,
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
    x_train_tfidf = vec.fit_transform(x_text_train)
    x_test_tfidf = vec.transform(x_text_test)

    results = []
    for name, spec in available_sets.items():
        num_cols = spec["num"]
        cat_cols = spec["cat"]
        if not num_cols and not cat_cols:
            x_train = x_train_tfidf
            x_test = x_test_tfidf
        else:
            extra_train, extra_test = build_extra_matrix(
                work=work,
                idx_train=idx_train,
                idx_test=idx_test,
                num_cols=num_cols,
                cat_cols=cat_cols,
            )
            x_train = hstack([x_train_tfidf, extra_train], format="csr")
            x_test = hstack([x_test_tfidf, extra_test], format="csr")

        metrics = train_eval(x_train, x_test, y_train, y_test)
        feature_desc = []
        if num_cols:
            feature_desc.extend(num_cols)
        if cat_cols:
            feature_desc.extend([f"{c}(onehot)" for c in cat_cols])
        results.append(
            {
                "model": name,
                "features": ",".join(feature_desc) if feature_desc else "(none)",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
        print(
            f"{name:<36} acc={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} weighted_f1={metrics['weighted_f1']:.4f}"
        )

    res_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    base = res_df.loc[res_df["model"] == "TF-IDF only"].iloc[0]
    res_df["delta_acc_vs_base"] = res_df["accuracy"] - base["accuracy"]
    res_df["delta_macro_f1_vs_base"] = res_df["macro_f1"] - base["macro_f1"]
    res_df["delta_weighted_f1_vs_base"] = res_df["weighted_f1"] - base["weighted_f1"]

    print("\n=== Ranked by macro_f1 ===")
    print(
        res_df[
            [
                "model",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "delta_acc_vs_base",
                "delta_macro_f1_vs_base",
                "delta_weighted_f1_vs_base",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )


if __name__ == "__main__":
    main()
