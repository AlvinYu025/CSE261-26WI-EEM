import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Listing-level targets: TF-IDF vs TF-IDF + mismatch (using existing data)."
    )
    parser.add_argument(
        "--listings-path",
        type=str,
        default="listings.csv",
        help="Path to listings.csv",
    )
    parser.add_argument(
        "--mismatch-path",
        type=str,
        default="Mismatch_Score/al/llm_mismatch_score.csv",
        help="Path to AL mismatch CSV",
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        default="Mismatch_Score/al/baseline_mismatch_score.csv",
        help="Path to AL baseline mismatch CSV (computed from GloVe baseline).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Test split ratio",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=8000,
        help="Max TF-IDF features",
    )
    parser.add_argument(
        "--max-reviews-per-listing",
        type=int,
        default=50,
        help="Max reviews to concatenate for each listing text",
    )
    parser.add_argument(
        "--min-reviews",
        type=int,
        default=4,
        help="Minimum review count per listing (use 4 for review > 3).",
    )
    parser.add_argument(
        "--min-reviews-gt",
        type=int,
        default=1,
        help="If set, use strict filter n_reviews > this value (overrides --min-reviews).",
    )
    parser.add_argument(
        "--binning",
        type=str,
        default="median",
        choices=["median", "cut", "qcut"],
        help="Target binning method. 'median' is the most stable for binary tasks.",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=2,
        help="Number of classes for all targets.",
    )
    parser.add_argument(
        "--tune-grid",
        action="store_true",
        help="Grid-search min_reviews=1..3 and n_classes=2..5, then pick one config where all targets improve.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="joint_superhost_rating",
        help=(
            "Comma-separated target columns. "
            "Examples: joint_superhost_rating or host_is_superhost,review_scores_rating"
        ),
    )
    parser.add_argument(
        "--rating-threshold",
        type=float,
        default=4.8,
        help="Threshold for review_scores_rating binary target: class1 if <= threshold else class0.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="classification",
        choices=["classification", "regression", "both"],
        help="Run classification, regression, or both in one script.",
    )
    parser.add_argument(
        "--regression-targets",
        type=str,
        default="host_price_log,user_rating",
        help="Comma-separated regression targets.",
    )
    parser.add_argument(
        "--regression-joint",
        action="store_true",
        default=True,
        help="Use a single joint regression target built from the two regression targets.",
    )
    parser.add_argument(
        "--no-regression-joint",
        action="store_false",
        dest="regression_joint",
        help="Disable joint regression target and run each regression target separately.",
    )
    return parser.parse_args()


def zscore_fit_transform(x_train: np.ndarray, x_test: np.ndarray):
    mu = np.nanmean(x_train, axis=0, keepdims=True)
    sd = np.nanstd(x_train, axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (x_train - mu) / sd, (x_test - mu) / sd


def train_eval(x_train, x_test, y_train, y_test):
    clf = LinearSVC(random_state=RANDOM_STATE, max_iter=4000)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "weighted_f1": f1_score(y_test, pred, average="weighted"),
    }


def train_eval_reg(x_train, x_test, y_train, y_test):
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    return {
        "mae": mean_absolute_error(y_test, pred),
        "rmse": rmse,
        "r2": r2_score(y_test, pred),
    }


def make_target(series: pd.Series, n_class: int, method: str):
    if method == "median":
        if n_class != 2:
            raise ValueError("binning=median only supports n_classes=2")
        med = series.median()
        y = (series > med).astype("int64")
        return y
    if method == "qcut":
        y = pd.qcut(series, q=n_class, labels=False, duplicates="drop")
    else:
        y = pd.cut(series, bins=n_class, labels=False, include_lowest=True, duplicates="drop")
    return y.astype("int64")


def resolve_binning(method: str, n_class: int) -> str:
    if method == "median" and n_class > 2:
        return "qcut"
    return method


def build_target_series(
    work: pd.DataFrame,
    target_col: str,
    n_class: int,
    binning: str,
    rating_threshold: float,
):
    # Host-side binary label
    if target_col == "host_is_superhost":
        s = work[target_col].astype(str).str.strip().str.lower()
        mapped = s.map({"t": 1, "f": 0, "true": 1, "false": 0})
        valid = mapped.notna()
        return mapped[valid].astype("int64"), valid
    # Joint host+user label:
    # class 0: non-superhost & high-rating
    # class 1: non-superhost & low-rating
    # class 2: superhost & high-rating
    # class 3: superhost & low-rating
    if target_col == "joint_superhost_rating":
        if "host_is_superhost" not in work.columns or "review_scores_rating" not in work.columns:
            raise ValueError(
                "joint_superhost_rating requires columns: host_is_superhost and review_scores_rating"
            )
        s_host = work["host_is_superhost"].astype(str).str.strip().str.lower()
        host = s_host.map({"t": 1, "f": 0, "true": 1, "false": 0})
        s_rating = pd.to_numeric(work["review_scores_rating"], errors="coerce")
        valid = host.notna() & s_rating.notna()
        low_rating = (s_rating[valid] <= rating_threshold).astype("int64")
        y = (host[valid].astype("int64") * 2 + low_rating).astype("int64")
        return y, valid
    if target_col == "review_scores_rating":
        s = pd.to_numeric(work[target_col], errors="coerce")
        valid = s.notna()
        y = (s[valid] <= rating_threshold).astype("int64")
        return y, valid

    # Generic numeric target
    num = pd.to_numeric(work[target_col], errors="coerce")
    valid = num.notna()
    y = make_target(num[valid], n_class=n_class, method=binning)
    return y, valid


def build_listing_level_df(listings_path: Path, mismatch_path: Path, max_reviews_per_listing: int):
    listings = pd.read_csv(listings_path)
    mismatch = pd.read_csv(mismatch_path)

    if "id" not in listings.columns:
        raise ValueError("listings.csv must contain column: id")
    if "listing_id" not in mismatch.columns:
        raise ValueError("mismatch file must contain column: listing_id")

    metric_cols = [c for c in ["mismatch_proxy", "mabs", "mover", "munder"] if c in mismatch.columns]
    if not metric_cols:
        raise ValueError("No mismatch metric columns found in mismatch file.")

    if "review" not in mismatch.columns:
        raise ValueError("mismatch file must contain review text column: review")

    mismatch["review"] = mismatch["review"].fillna("").astype(str)

    review_text = (
        mismatch.groupby("listing_id")["review"]
        .apply(lambda s: " ".join(s.head(max_reviews_per_listing)))
        .rename("review_text")
    )

    agg_spec = {"review_id": "count", **{c: "mean" for c in metric_cols}}
    agg = mismatch.groupby("listing_id").agg(agg_spec).rename(columns={"review_id": "n_reviews"}).reset_index()
    agg = agg.merge(review_text, on="listing_id", how="left")

    merged = listings.merge(agg, left_on="id", right_on="listing_id", how="inner")
    return merged


def load_baseline_mismatch_listing_feature(baseline_path: Path):
    if not baseline_path.exists():
        return None
    raw = pd.read_csv(baseline_path, usecols=["listing_id", "mismatch_proxy"])
    raw["mismatch_proxy"] = pd.to_numeric(raw["mismatch_proxy"], errors="coerce")
    raw = raw.dropna(subset=["listing_id", "mismatch_proxy"]).copy()
    if raw.empty:
        return None
    listing_base = (
        raw.groupby("listing_id", as_index=False)["mismatch_proxy"]
        .mean()
        .rename(columns={"mismatch_proxy": "baseline_mismatch_proxy"})
    )
    return listing_base


def run_one_target(
    df: pd.DataFrame,
    target_col: str,
    n_class: int,
    test_size: float,
    max_features: int,
    binning: str,
    rating_threshold: float,
):
    work = df.copy()
    work = work.dropna(subset=["review_text"]).reset_index(drop=True)
    y, valid_mask = build_target_series(work, target_col, n_class, binning, rating_threshold)
    work = work[valid_mask].reset_index(drop=True)
    y = y.reset_index(drop=True)

    # if qcut drops classes due to ties, skip very small/degenerate tasks
    if y.nunique() < 2:
        print(f"\n[{target_col}] skipped: <2 classes after {binning}.")
        return None
    cls_counts = y.value_counts()
    if cls_counts.min() < 2:
        print(
            f"\n[{target_col}] skipped: least populated class has {int(cls_counts.min())} sample(s), "
            "cannot use stratified split safely."
        )
        print("Class distribution:")
        for cls, cnt in cls_counts.sort_index().items():
            print(f"  class {cls}: {cnt} ({cnt / len(y):.2%})")
        return None
    n_test = int(round(len(y) * test_size))
    if n_test < y.nunique():
        print(
            f"\n[{target_col}] skipped: test split too small for stratification "
            f"(n_test={n_test}, n_classes={y.nunique()})."
        )
        return None

    x_text = work["review_text"].fillna("").astype(str)
    x_num = work[["mismatch_proxy", "mabs", "mover", "munder"]].copy()
    x_num = x_num.fillna(x_num.mean())
    has_baseline = "baseline_mismatch_proxy" in work.columns
    # Naive structured features (no TF-IDF): host + rating
    host_raw = work.get("host_is_superhost", pd.Series(index=work.index, dtype=object)).astype(str).str.strip().str.lower()
    host_bin = host_raw.map({"t": 1, "f": 0, "true": 1, "false": 0})
    rating_num = pd.to_numeric(work.get("review_scores_rating", pd.Series(index=work.index, dtype=float)), errors="coerce")
    naive_ok = host_bin.notna().sum() > 0 and rating_num.notna().sum() > 0

    x_tr_text, x_te_text, y_tr, y_te, idx_tr, idx_te = train_test_split(
        x_text,
        y,
        work.index.values,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=max_features,
    )
    x_tr_tfidf = vec.fit_transform(x_tr_text)
    x_te_tfidf = vec.transform(x_te_text)

    base = train_eval(x_tr_tfidf, x_te_tfidf, y_tr, y_te)

    baseline = None
    if has_baseline:
        b_train = work.loc[idx_tr, ["baseline_mismatch_proxy"]].copy()
        b_test = work.loc[idx_te, ["baseline_mismatch_proxy"]].copy()
        b_train = b_train.fillna(b_train.mean())
        b_test = b_test.fillna(b_train.mean())
        x_b_tr = b_train.values.astype(float)
        x_b_te = b_test.values.astype(float)
        x_b_tr, x_b_te = zscore_fit_transform(x_b_tr, x_b_te)
        x_tr_b = hstack([x_tr_tfidf, csr_matrix(x_b_tr)], format="csr")
        x_te_b = hstack([x_te_tfidf, csr_matrix(x_b_te)], format="csr")
        baseline = train_eval(x_tr_b, x_te_b, y_tr, y_te)

    num_tr = x_num.loc[idx_tr].values.astype(float)
    num_te = x_num.loc[idx_te].values.astype(float)
    num_tr, num_te = zscore_fit_transform(num_tr, num_te)
    x_tr_full = hstack([x_tr_tfidf, csr_matrix(num_tr)], format="csr")
    x_te_full = hstack([x_te_tfidf, csr_matrix(num_te)], format="csr")
    full = train_eval(x_tr_full, x_te_full, y_tr, y_te)

    naive = None
    if naive_ok:
        naive_df = pd.DataFrame(
            {
                "host_is_superhost_bin": host_bin,
                "review_scores_rating": rating_num,
            },
            index=work.index,
        )
        naive_train = naive_df.loc[idx_tr].copy()
        naive_test = naive_df.loc[idx_te].copy()
        naive_train = naive_train.fillna(naive_train.mean(numeric_only=True))
        naive_test = naive_test.fillna(naive_train.mean(numeric_only=True))
        x_naive_tr = naive_train.values.astype(float)
        x_naive_te = naive_test.values.astype(float)
        x_naive_tr, x_naive_te = zscore_fit_transform(x_naive_tr, x_naive_te)
        naive = train_eval(csr_matrix(x_naive_tr), csr_matrix(x_naive_te), y_tr, y_te)

    print(f"\nTarget: {target_col}  (binning={binning}, classes={y.nunique()}, n={len(work)})")
    print("Class distribution:")
    for cls, cnt in y.value_counts().sort_index().items():
        print(f"  class {cls}: {cnt} ({cnt / len(y):.2%})")
    print(
        f"TF-IDF only                     acc={base['accuracy']:.4f} "
        f"macro_f1={base['macro_f1']:.4f} weighted_f1={base['weighted_f1']:.4f}"
    )
    print(
        f"TF-IDF + customized_mismatch    acc={full['accuracy']:.4f} "
        f"macro_f1={full['macro_f1']:.4f} weighted_f1={full['weighted_f1']:.4f}"
    )
    if baseline is not None:
        print(
            f"TF-IDF + glove_mismatch         acc={baseline['accuracy']:.4f} "
            f"macro_f1={baseline['macro_f1']:.4f} weighted_f1={baseline['weighted_f1']:.4f}"
        )
    print(
        f"Delta (full - base)             acc={full['accuracy']-base['accuracy']:+.4f} "
        f"macro_f1={full['macro_f1']-base['macro_f1']:+.4f} "
        f"weighted_f1={full['weighted_f1']-base['weighted_f1']:+.4f}"
    )
    if baseline is not None:
        print(
            f"Delta (glove - base)            acc={baseline['accuracy']-base['accuracy']:+.4f} "
            f"macro_f1={baseline['macro_f1']-base['macro_f1']:+.4f} "
            f"weighted_f1={baseline['weighted_f1']-base['weighted_f1']:+.4f}"
        )
        print(
            f"Delta (custom - glove)          acc={full['accuracy']-baseline['accuracy']:+.4f} "
            f"macro_f1={full['macro_f1']-baseline['macro_f1']:+.4f} "
            f"weighted_f1={full['weighted_f1']-baseline['weighted_f1']:+.4f}"
        )
    if naive is not None:
        print(
            f"Naive(host+rating) only         acc={naive['accuracy']:.4f} "
            f"macro_f1={naive['macro_f1']:.4f} weighted_f1={naive['weighted_f1']:.4f}"
        )
        if target_col == "joint_superhost_rating":
            print("[WARN] Naive(host+rating) overlaps with target definition; interpret as upper-bound style reference.")
    else:
        print("Naive(host+rating) only         skipped: missing host/rating values.")

    return {
        "target": target_col,
        "n": len(work),
        "classes": int(y.nunique()),
        "acc_base": base["accuracy"],
        "acc_glove": (baseline["accuracy"] if baseline is not None else np.nan),
        "acc_full": full["accuracy"],
        "delta_acc": full["accuracy"] - base["accuracy"],
        "macro_f1_base": base["macro_f1"],
        "macro_f1_glove": (baseline["macro_f1"] if baseline is not None else np.nan),
        "macro_f1_full": full["macro_f1"],
        "delta_macro_f1": full["macro_f1"] - base["macro_f1"],
        "macro_f1_naive": (naive["macro_f1"] if naive is not None else np.nan),
        "weighted_f1_base": base["weighted_f1"],
        "weighted_f1_glove": (baseline["weighted_f1"] if baseline is not None else np.nan),
        "weighted_f1_full": full["weighted_f1"],
        "delta_weighted_f1": full["weighted_f1"] - base["weighted_f1"],
        "delta_macro_f1_glove_vs_base": (
            baseline["macro_f1"] - base["macro_f1"] if baseline is not None else np.nan
        ),
        "delta_macro_f1_custom_vs_glove": (
            full["macro_f1"] - baseline["macro_f1"] if baseline is not None else np.nan
        ),
    }


def run_targets(df, targets, n_classes, binning, test_size, max_features, rating_threshold):
    records = []
    for col in targets:
        synthetic_targets = {"joint_superhost_rating"}
        if (col not in df.columns) and (col not in synthetic_targets):
            print(f"\n[{col}] skipped: column not found.")
            continue
        rec = run_one_target(
            df=df,
            target_col=col,
            n_class=n_classes,
            test_size=test_size,
            max_features=max_features,
            binning=binning,
            rating_threshold=rating_threshold,
        )
        if rec is not None:
            records.append(rec)
    return records


def normalize_reg_target(df: pd.DataFrame, target_col: str):
    if target_col == "user_rating":
        return pd.to_numeric(df["review_scores_rating"], errors="coerce")
    if target_col == "host_price_log":
        price = pd.to_numeric(
            df["price"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
            errors="coerce",
        )
        return np.log1p(price)
    if target_col == "host_min_nights_policy":
        if "minimum_nights_avg_ntm" in df.columns:
            return pd.to_numeric(df["minimum_nights_avg_ntm"], errors="coerce")
        return pd.to_numeric(df["minimum_nights"], errors="coerce")
    if target_col == "host_demand_conversion":
        rev = pd.to_numeric(df["number_of_reviews_ltm"], errors="coerce")
        avail = pd.to_numeric(df["availability_90"], errors="coerce")
        return rev / (avail + 1.0)
    if target_col == "host_acceptance_rate":
        return pd.to_numeric(
            df[target_col].astype(str).str.replace("%", "", regex=False),
            errors="coerce",
        )
    return pd.to_numeric(df[target_col], errors="coerce")


def build_joint_regression_target(df: pd.DataFrame, target_a: str, target_b: str):
    a = normalize_reg_target(df, target_a)
    b = normalize_reg_target(df, target_b)
    valid = a.notna() & b.notna()
    if valid.sum() == 0:
        return None, None

    a_v = a[valid].astype(float)
    b_v = b[valid].astype(float)
    a_sd = float(a_v.std()) if float(a_v.std()) > 1e-8 else 1.0
    b_sd = float(b_v.std()) if float(b_v.std()) > 1e-8 else 1.0
    a_z = (a_v - float(a_v.mean())) / a_sd
    b_z = (b_v - float(b_v.mean())) / b_sd
    joint = 0.5 * a_z + 0.5 * b_z
    return joint, valid


def run_one_regression_target(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    max_features: int,
    y_override=None,
    valid_override=None,
):
    work = df.copy()
    if y_override is not None and valid_override is not None:
        work = work[valid_override].copy().reset_index(drop=True)
        work["target"] = pd.Series(y_override.values, index=work.index)
    else:
        work["target"] = normalize_reg_target(work, target_col)
    work = work.dropna(subset=["target", "review_text"]).reset_index(drop=True)
    if len(work) < 40:
        print(f"\n[{target_col}] skipped: too few rows ({len(work)}).")
        return None

    x_text = work["review_text"].astype(str)
    y = work["target"].values.astype(float)

    x_tr_text, x_te_text, y_tr, y_te, idx_tr, idx_te = train_test_split(
        x_text, y, work.index.values, test_size=test_size, random_state=RANDOM_STATE
    )

    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=max_features,
    )
    x_tr_tfidf = vec.fit_transform(x_tr_text)
    x_te_tfidf = vec.transform(x_te_text)

    base = train_eval_reg(x_tr_tfidf, x_te_tfidf, y_tr, y_te)

    baseline = None
    if "baseline_mismatch_proxy" in work.columns:
        b_train = work.loc[idx_tr, ["baseline_mismatch_proxy"]].copy()
        b_test = work.loc[idx_te, ["baseline_mismatch_proxy"]].copy()
        b_train = b_train.fillna(b_train.mean())
        b_test = b_test.fillna(b_train.mean())
        x_b_tr = b_train.values.astype(float)
        x_b_te = b_test.values.astype(float)
        x_b_tr, x_b_te = zscore_fit_transform(x_b_tr, x_b_te)
        x_tr_b = hstack([x_tr_tfidf, csr_matrix(x_b_tr)], format="csr")
        x_te_b = hstack([x_te_tfidf, csr_matrix(x_b_te)], format="csr")
        baseline = train_eval_reg(x_tr_b, x_te_b, y_tr, y_te)

    c_train = work.loc[idx_tr, ["mismatch_proxy", "mabs", "mover", "munder"]].copy()
    c_test = work.loc[idx_te, ["mismatch_proxy", "mabs", "mover", "munder"]].copy()
    c_train = c_train.fillna(c_train.mean())
    c_test = c_test.fillna(c_train.mean())
    x_c_tr = c_train.values.astype(float)
    x_c_te = c_test.values.astype(float)
    x_c_tr, x_c_te = zscore_fit_transform(x_c_tr, x_c_te)
    x_tr_c = hstack([x_tr_tfidf, csr_matrix(x_c_tr)], format="csr")
    x_te_c = hstack([x_te_tfidf, csr_matrix(x_c_te)], format="csr")
    custom = train_eval_reg(x_tr_c, x_te_c, y_tr, y_te)

    print(f"\n[Regression] Target: {target_col} (n={len(work)})")
    print(f"TF-IDF only                  MAE={base['mae']:.4f} RMSE={base['rmse']:.4f} R2={base['r2']:.4f}")
    if baseline is not None:
        print(
            f"TF-IDF + glove_mismatch      MAE={baseline['mae']:.4f} RMSE={baseline['rmse']:.4f} R2={baseline['r2']:.4f}"
        )
    print(
        f"TF-IDF + customized_mismatch MAE={custom['mae']:.4f} RMSE={custom['rmse']:.4f} R2={custom['r2']:.4f}"
    )
    print(
        f"Delta(custom - base)         dMAE={custom['mae']-base['mae']:+.4f} "
        f"dRMSE={custom['rmse']-base['rmse']:+.4f} dR2={custom['r2']-base['r2']:+.4f}"
    )
    if baseline is not None:
        print(
            f"Delta(custom - glove)        dMAE={custom['mae']-baseline['mae']:+.4f} "
            f"dRMSE={custom['rmse']-baseline['rmse']:+.4f} dR2={custom['r2']-baseline['r2']:+.4f}"
        )

    return {
        "target": target_col,
        "n": len(work),
        "r2_base": base["r2"],
        "r2_glove": (baseline["r2"] if baseline is not None else np.nan),
        "r2_custom": custom["r2"],
        "delta_r2_custom_base": custom["r2"] - base["r2"],
        "delta_r2_custom_glove": (custom["r2"] - baseline["r2"] if baseline is not None else np.nan),
    }


def tune_parameters(df_raw, targets, args):
    rows = []
    print("\n" + "=" * 80)
    print("GRID TUNING: min_reviews in [1,3], n_classes in [2,5]")
    print("=" * 80)

    for min_reviews in range(1, 4):
        df = df_raw[df_raw["n_reviews"] >= min_reviews].reset_index(drop=True)
        if df.empty:
            continue
        for n_classes in range(2, 6):
            binning = resolve_binning(args.binning, n_classes)
            print(
                f"\n--- Config: min_reviews={min_reviews}, n_classes={n_classes}, "
                f"binning={binning}, n_listings={len(df)} ---"
            )
            recs = run_targets(
                df=df,
                targets=targets,
                n_classes=n_classes,
                binning=binning,
                test_size=args.test_size,
                max_features=args.max_features,
                rating_threshold=args.rating_threshold,
            )
            if not recs:
                rows.append(
                    {
                        "min_reviews": min_reviews,
                        "n_classes": n_classes,
                        "binning": binning,
                        "n_targets": 0,
                        "all_improved": False,
                        "min_delta_macro_f1": np.nan,
                        "mean_delta_macro_f1": np.nan,
                    }
                )
                continue

            deltas = [r["delta_macro_f1"] for r in recs]
            all_improved = all(d > 0 for d in deltas)
            rows.append(
                {
                    "min_reviews": min_reviews,
                    "n_classes": n_classes,
                    "binning": binning,
                    "n_targets": len(recs),
                    "all_improved": all_improved,
                    "min_delta_macro_f1": float(np.min(deltas)),
                    "mean_delta_macro_f1": float(np.mean(deltas)),
                }
            )

            print("Per-target delta_macro_f1:")
            for r in recs:
                print(f"  {r['target']:<28} {r['delta_macro_f1']:+.4f}")
            print(
                f"Summary: all_improved={all_improved} | "
                f"min_delta_macro_f1={np.min(deltas):+.4f} | mean_delta_macro_f1={np.mean(deltas):+.4f}"
            )

    res = pd.DataFrame(rows)
    if res.empty:
        print("\nNo valid configurations were evaluated.")
        return

    print("\n" + "=" * 80)
    print("GRID RESULTS")
    print("=" * 80)
    print(
        res.sort_values(
            ["all_improved", "min_delta_macro_f1", "mean_delta_macro_f1"],
            ascending=[False, False, False],
        ).to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    feasible = res[(res["all_improved"]) & (res["n_targets"] == len(targets))]
    if feasible.empty:
        print("\nNo configuration improved all targets simultaneously.")
        return

    best = feasible.sort_values(
        ["min_delta_macro_f1", "mean_delta_macro_f1"],
        ascending=[False, False],
    ).iloc[0]
    print("\n" + "=" * 80)
    print("BEST CONFIG (all targets improved)")
    print("=" * 80)
    print(
        f"min_reviews={int(best['min_reviews'])}, n_classes={int(best['n_classes'])}, "
        f"binning={best['binning']}, n_targets={int(best['n_targets'])}, "
        f"min_delta_macro_f1={best['min_delta_macro_f1']:+.4f}, "
        f"mean_delta_macro_f1={best['mean_delta_macro_f1']:+.4f}"
    )


def main():
    args = parse_args()
    listings_path = Path(args.listings_path)
    mismatch_path = Path(args.mismatch_path)
    baseline_path = Path(args.baseline_path)
    if not listings_path.exists():
        raise FileNotFoundError(f"Missing file: {listings_path}")
    if not mismatch_path.exists():
        raise FileNotFoundError(f"Missing file: {mismatch_path}")

    df_raw = build_listing_level_df(
        listings_path=listings_path,
        mismatch_path=mismatch_path,
        max_reviews_per_listing=args.max_reviews_per_listing,
    )
    baseline_listing = load_baseline_mismatch_listing_feature(baseline_path=baseline_path)
    if baseline_listing is not None:
        df_raw = df_raw.merge(baseline_listing, on="listing_id", how="left")

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    if args.tune_grid and args.mode != "classification":
        print("tune-grid is only for classification mode. Switch mode to classification or remove --tune-grid.")
        return
    if args.tune_grid:
        tune_parameters(df_raw=df_raw, targets=targets, args=args)
        return

    if args.min_reviews_gt is not None:
        df = df_raw[df_raw["n_reviews"] > args.min_reviews_gt].reset_index(drop=True)
        min_reviews_msg = f"n_reviews > {args.min_reviews_gt}"
    else:
        df = df_raw[df_raw["n_reviews"] >= args.min_reviews].reset_index(drop=True)
        min_reviews_msg = f"n_reviews >= {args.min_reviews}"
    binning = resolve_binning(args.binning, args.n_classes)

    print(f"Merged listing rows: {len(df)}")
    print(f"Unique listing_id: {df['listing_id'].nunique()}")
    print(f"Min reviews filter: {min_reviews_msg}")
    print(f"GloVe mismatch: {baseline_path} | loaded={baseline_listing is not None}")
    if binning != args.binning:
        print(f"Binning auto-adjusted: {args.binning} -> {binning} (n_classes={args.n_classes})")

    if args.mode in ("classification", "both"):
        records = run_targets(
            df=df,
            targets=targets,
            n_classes=args.n_classes,
            binning=binning,
            test_size=args.test_size,
            max_features=args.max_features,
            rating_threshold=args.rating_threshold,
        )
        if records:
            out = pd.DataFrame(records).sort_values("delta_macro_f1", ascending=False).reset_index(drop=True)
            print("\n=== Classification Summary (sorted by delta_macro_f1) ===")
            print(
                out[
                    [
                        "target",
                        "n",
                        "classes",
                        "acc_base",
                        "acc_glove",
                        "acc_full",
                        "delta_acc",
                        "macro_f1_base",
                        "macro_f1_glove",
                        "macro_f1_full",
                        "delta_macro_f1",
                        "delta_macro_f1_glove_vs_base",
                        "delta_macro_f1_custom_vs_glove",
                    ]
                ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
            )

    if args.mode in ("regression", "both"):
        reg_targets = [t.strip() for t in args.regression_targets.split(",") if t.strip()]
        synthetic_reg_targets = {"user_rating", "host_demand_conversion", "host_price_log", "host_min_nights_policy"}
        reg_rows = []
        if args.regression_joint and len(reg_targets) >= 2:
            t1, t2 = reg_targets[0], reg_targets[1]
            t1_ok = (t1 in df.columns) or (t1 in synthetic_reg_targets)
            t2_ok = (t2 in df.columns) or (t2 in synthetic_reg_targets)
            if (not t1_ok) or (not t2_ok):
                print(f"\nJoint regression skipped: missing columns in {t1},{t2}.")
            else:
                y_joint, valid_joint = build_joint_regression_target(df, t1, t2)
                if y_joint is None:
                    print("\nJoint regression skipped: no valid rows after target merge.")
                else:
                    print(f"\n[Joint regression target] built from: {t1} + {t2} (z-score average)")
                    rec = run_one_regression_target(
                        df=df,
                        target_col=f"joint_regression_{t1}_{t2}",
                        test_size=args.test_size,
                        max_features=args.max_features,
                        y_override=y_joint,
                        valid_override=valid_joint,
                    )
                    if rec is not None:
                        reg_rows.append(rec)
        else:
            for t in reg_targets:
                if (t not in df.columns) and (t not in synthetic_reg_targets):
                    print(f"\n[{t}] skipped: column not found.")
                    continue
                rec = run_one_regression_target(
                    df=df,
                    target_col=t,
                    test_size=args.test_size,
                    max_features=args.max_features,
                )
                if rec is not None:
                    reg_rows.append(rec)
        if reg_rows:
            reg_out = pd.DataFrame(reg_rows).sort_values("delta_r2_custom_base", ascending=False).reset_index(drop=True)
            print("\n=== Regression Summary (sorted by delta_r2_custom_base) ===")
            print(
                reg_out[
                    [
                        "target",
                        "n",
                        "r2_base",
                        "r2_glove",
                        "r2_custom",
                        "delta_r2_custom_base",
                        "delta_r2_custom_glove",
                    ]
                ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
            )


if __name__ == "__main__":
    main()