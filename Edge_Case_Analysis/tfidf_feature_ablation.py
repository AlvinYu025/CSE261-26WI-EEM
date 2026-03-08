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

# results:

# NY
# TF-IDF only                          acc=0.5717 macro_f1=0.5226 weighted_f1=0.5642
# TF-IDF + mismatch_proxy              acc=0.5735 macro_f1=0.5239 weighted_f1=0.5662
# TF-IDF + mabs                        acc=0.5757 macro_f1=0.5260 weighted_f1=0.5679
# TF-IDF + mover                       acc=0.5710 macro_f1=0.5221 weighted_f1=0.5638
# TF-IDF + munder                      acc=0.5765 macro_f1=0.5263 weighted_f1=0.5688
# TF-IDF + price                       acc=0.5776 macro_f1=0.5254 weighted_f1=0.5702
# TF-IDF + availability_30             acc=0.5905 macro_f1=0.5435 weighted_f1=0.5834
# TF-IDF + availability_60             acc=0.5990 macro_f1=0.5485 weighted_f1=0.5921
# TF-IDF + neighborhood                acc=0.6226 macro_f1=0.5804 weighted_f1=0.6180
# TF-IDF + review_scores_all           acc=0.7593 macro_f1=0.7119 weighted_f1=0.7522


# AM
# TF-IDF only                          acc=0.5098 macro_f1=0.5013 weighted_f1=0.5073
# TF-IDF + mismatch_proxy              acc=0.5095 macro_f1=0.5012 weighted_f1=0.5071
# TF-IDF + mabs                        acc=0.5098 macro_f1=0.5013 weighted_f1=0.5073
# TF-IDF + mover                       acc=0.5100 macro_f1=0.5016 weighted_f1=0.5075
# TF-IDF + munder                      acc=0.5108 macro_f1=0.5026 weighted_f1=0.5084
# TF-IDF + price                       acc=0.5108 macro_f1=0.5024 weighted_f1=0.5084
# TF-IDF + availability_30             acc=0.5128 macro_f1=0.5046 weighted_f1=0.5102
# TF-IDF + availability_60             acc=0.5135 macro_f1=0.5051 weighted_f1=0.5109
# TF-IDF + neighborhood                acc=0.5185 macro_f1=0.5109 weighted_f1=0.5163
# TF-IDF + review_scores_all           acc=0.6985 macro_f1=0.6928 weighted_f1=0.6919

# === Ranked by macro_f1 ===
#                      model  accuracy  macro_f1  weighted_f1  delta_acc_vs_base  delta_macro_f1_vs_base  delta_weighted_f1_vs_base
# TF-IDF + review_scores_all    0.6985    0.6928       0.6919             0.1887                  0.1915                     0.1846
#      TF-IDF + neighborhood    0.5185    0.5109       0.5163             0.0087                  0.0096                     0.0091
#   TF-IDF + availability_60    0.5135    0.5051       0.5109             0.0037                  0.0038                     0.0037
#   TF-IDF + availability_30    0.5128    0.5046       0.5102             0.0030                  0.0032                     0.0029
#            TF-IDF + munder    0.5108    0.5026       0.5084             0.0011                  0.0013                     0.0011
#             TF-IDF + price    0.5108    0.5024       0.5084             0.0011                  0.0011                     0.0011
#             TF-IDF + mover    0.5100    0.5016       0.5075             0.0002                  0.0003                     0.0003
#                TF-IDF only    0.5098    0.5013       0.5073             0.0000                  0.0000                     0.0000
#              TF-IDF + mabs    0.5098    0.5013       0.5073             0.0000                 -0.0000                    -0.0000
#    TF-IDF + mismatch_proxy    0.5095    0.5012       0.5071            -0.0002                 -0.0001                    -0.0002


# MO
# TF-IDF only                          acc=0.3851 macro_f1=0.3846 weighted_f1=0.3824
# TF-IDF + mismatch_proxy              acc=0.3867 macro_f1=0.3860 weighted_f1=0.3839
# TF-IDF + mabs                        acc=0.3853 macro_f1=0.3849 weighted_f1=0.3827
# TF-IDF + mover                       acc=0.3863 macro_f1=0.3858 weighted_f1=0.3836
# TF-IDF + munder                      acc=0.3861 macro_f1=0.3855 weighted_f1=0.3833
# TF-IDF + price                       acc=0.3883 macro_f1=0.3876 weighted_f1=0.3853
# TF-IDF + availability_30             acc=0.3897 macro_f1=0.3891 weighted_f1=0.3867
# TF-IDF + availability_60             acc=0.3890 macro_f1=0.3882 weighted_f1=0.3857
# TF-IDF + neighborhood                acc=0.3961 macro_f1=0.3971 weighted_f1=0.3943
# TF-IDF + review_scores_all           acc=0.5996 macro_f1=0.5895 weighted_f1=0.5898

# === Ranked by macro_f1 ===
#                      model  accuracy  macro_f1  weighted_f1  delta_acc_vs_base  delta_macro_f1_vs_base  delta_weighted_f1_vs_base
# TF-IDF + review_scores_all    0.5996    0.5895       0.5898             0.2145                 
#  0.2049                     0.2074
#      TF-IDF + neighborhood    0.3961    0.3971       0.3943             0.0110                 
#  0.0125                     0.0118
#   TF-IDF + availability_30    0.3897    0.3891       0.3867             0.0047                  0.0045                     0.0043        
#   TF-IDF + availability_60    0.3890    0.3882       0.3857             0.0039                  0.0036                     0.0033        
#             TF-IDF + price    0.3883    0.3876       0.3853             0.0032                  0.0030                     0.0029        
#    TF-IDF + mismatch_proxy    0.3867    0.3860       0.3839             0.0016                  0.0014                     0.0015        
#             TF-IDF + mover    0.3863    0.3858       0.3836             0.0013                  0.0012                     0.0012        
#            TF-IDF + munder    0.3861    0.3855       0.3833             0.0010                  0.0008                     0.0009        
#              TF-IDF + mabs    0.3853    0.3849       0.3827             0.0002                  0.0003                     0.0003        
#                TF-IDF only    0.3851    0.3846       0.3824             0.0000                  0.0000                     0.0000   