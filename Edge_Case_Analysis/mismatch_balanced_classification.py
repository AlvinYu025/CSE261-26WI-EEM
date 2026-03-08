import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

# Keep settings aligned with regression_correlation.ipynb cell 15
RANDOM_STATE = 42
TEST_SIZE = 0.2

BASE_FEATURES = [
    "price",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "availability_eoy",
]

MISMATCH_FEATURES = ["mismatch_proxy", "mabs", "mover", "munder"]


def load_city_data():
    root = Path(__file__).resolve().parents[1]
    am_path = root / "Mismatch_Score" / "am" / "llm_mismatch_score.csv"
    ny_path = root / "Mismatch_Score" / "ny" / "llm_mismatch_score.csv"

    am = pd.read_csv(am_path)
    ny = pd.read_csv(ny_path)
    return {"NY": ny, "AM": am}


def prepare_data(df: pd.DataFrame):
    out = df.copy()
    out["disappointed"] = (out["rating"] < 4.5).astype(int)
    out["price"] = np.log1p(out["price"])
    out["avail_diff_365_30"] = out["availability_365"] - out["availability_30"]
    out["avail_diff_90_30"] = out["availability_90"] - out["availability_30"]
    out = pd.get_dummies(out, columns=["neighborhood"], drop_first=True)
    neighborhood_cols = [c for c in out.columns if c.startswith("neighborhood_")]
    base_cols = BASE_FEATURES + ["avail_diff_365_30", "avail_diff_90_30"] + neighborhood_cols
    full_cols = base_cols + MISMATCH_FEATURES
    return out, base_cols, full_cols


def make_balanced_index(y: pd.Series, seed: int):
    """Return 1:1 class-balanced indices by undersampling the majority class."""
    idx_pos = y[y == 1].index
    idx_neg = y[y == 0].index
    n = min(len(idx_pos), len(idx_neg))
    if n == 0:
        raise ValueError("Cannot balance classes because one class has zero samples.")

    rng = np.random.default_rng(seed)
    pos_keep = rng.choice(idx_pos.to_numpy(), size=n, replace=False)
    neg_keep = rng.choice(idx_neg.to_numpy(), size=n, replace=False)
    keep = np.concatenate([pos_keep, neg_keep])
    rng.shuffle(keep)
    return keep


def run_setting(setting_name, X_train, X_test, y_train, y_test):
    runs = [
        ("Baseline", X_train["base"], X_test["base"]),
        ("Full (+mismatch)", X_train["full"], X_test["full"]),
    ]
    metrics = {}

    print(f"\n{'-' * 60}")
    print(f"SETTING: {setting_name}")
    print(f"{'-' * 60}")
    print(f"Train samples: {len(y_train)} | class0={(y_train == 0).sum()} class1={(y_train == 1).sum()}")
    print(f"Test  samples: {len(y_test)} | class0={(y_test == 0).sum()} class1={(y_test == 1).sum()}")

    for label, X_tr, X_te in runs:
        model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced" if setting_name == "imbalanced" else None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_tr, y_train)

        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_minor = f1_score(y_test, y_pred, pos_label=1)

        print(f"\n--- {label} ---")
        print(f"AUC-ROC:          {auc:.4f}")
        print(f"F1 (macro):       {f1_macro:.4f}")
        print(f"F1 (disappointed):{f1_minor:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["not disappointed", "disappointed"]))

        metrics[label] = {"auc": auc, "f1_macro": f1_macro, "f1_minority": f1_minor}

    delta_auc = metrics["Full (+mismatch)"]["auc"] - metrics["Baseline"]["auc"]
    print("\n--- Incremental Value of Mismatch ---")
    print(f"dAUC: {delta_auc:+.4f}")
    metrics["delta_auc"] = delta_auc
    return metrics


def train_eval(city: str, df: pd.DataFrame):
    print("\n" + "=" * 60)
    print(f"CITY: {city}")
    print("=" * 60)

    df_prep, base_cols, full_cols = prepare_data(df)
    needed = full_cols + ["disappointed", "listing_id"]
    model_df = df_prep[needed].dropna()

    X_base = model_df[base_cols]
    X_full = model_df[full_cols]
    y = model_df["disappointed"]
    groups = model_df["listing_id"]

    print("\nDataset size:")
    print(f"  Rows after dropna: {len(model_df)}")
    print(f"  Unique listings:   {groups.nunique()}")
    print("\nTarget distribution:")
    print(f"  Not disappointed (0): {(y == 0).sum()} ({(y == 0).mean() * 100:.1f}%)")
    print(f"  Disappointed    (1): {(y == 1).sum()} ({(y == 1).mean() * 100:.1f}%)")

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_base, y, groups=groups))

    print(f"\nTrain listings: {groups.iloc[train_idx].nunique()}")
    print(f"Test  listings: {groups.iloc[test_idx].nunique()}")

    y_train_raw = y.iloc[train_idx]
    y_test_raw = y.iloc[test_idx]

    X_train_raw = {"base": X_base.iloc[train_idx], "full": X_full.iloc[train_idx]}
    X_test_raw = {"base": X_base.iloc[test_idx], "full": X_full.iloc[test_idx]}

    imbalanced = run_setting("imbalanced", X_train_raw, X_test_raw, y_train_raw, y_test_raw)

    train_balanced_idx = make_balanced_index(y_train_raw, RANDOM_STATE)
    test_balanced_idx = make_balanced_index(y_test_raw, RANDOM_STATE)

    X_train_bal = {
        "base": X_train_raw["base"].loc[train_balanced_idx],
        "full": X_train_raw["full"].loc[train_balanced_idx],
    }
    X_test_bal = {
        "base": X_test_raw["base"].loc[test_balanced_idx],
        "full": X_test_raw["full"].loc[test_balanced_idx],
    }
    y_train_bal = y_train_raw.loc[train_balanced_idx]
    y_test_bal = y_test_raw.loc[test_balanced_idx]

    balanced = run_setting("balanced", X_train_bal, X_test_bal, y_train_bal, y_test_bal)
    return {"imbalanced": imbalanced, "balanced": balanced}


def main():
    datasets = load_city_data()
    summary = {}
    for city, df in datasets.items():
        summary[city] = train_eval(city, df)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'City':<6} {'Setting':<11} {'AUC_base':>10} {'AUC_full':>10} {'dAUC':>8} {'F1m_base':>10} {'F1m_full':>10}")
    print("-" * 80)
    for city, city_result in summary.items():
        for setting in ["imbalanced", "balanced"]:
            r = city_result[setting]
            auc_base = r["Baseline"]["auc"]
            auc_full = r["Full (+mismatch)"]["auc"]
            delta = r["delta_auc"]
            f1m_base = r["Baseline"]["f1_minority"]
            f1m_full = r["Full (+mismatch)"]["f1_minority"]
            print(
                f"{city:<6} {setting:<11} {auc_base:>10.4f} {auc_full:>10.4f} {delta:>+8.4f} "
                f"{f1m_base:>10.4f} {f1m_full:>10.4f}"
            )


if __name__ == "__main__":
    main()

# command
# python Edge_Case_Analysis/tfidf_feature_ablation_5class.py --input Mismatch_Score/ny/llm_mismatch_score.csv --text-col review --max-features 12000 --sample-size 


# results
# New York
# TF-IDF only                          acc=0.5717 macro_f1=0.5226 weighted_f1=0.5642
# TF-IDF + mismatch_proxy              acc=0.5735 macro_f1=0.5239 weighted_f1=0.5662
# TF-IDF + mabs                        acc=0.5757 macro_f1=0.5260 weighted_f1=0.5679
# TF-IDF + mover                       acc=0.5710 macro_f1=0.5221 weighted_f1=0.5638
# TF-IDF + munder                      acc=0.5765 macro_f1=0.5263 weighted_f1=0.5688
# TF-IDF + price                       acc=0.5776 macro_f1=0.5254 weighted_f1=0.5702
# TF-IDF + availability_30             acc=0.5905 macro_f1=0.5435 weighted_f1=0.5834
# TF-IDF + availability_60             acc=0.5990 macro_f1=0.5485 weighted_f1=0.5921
# TF-IDF + neighborhood                acc=0.6226 macro_f1=0.5804 weighted_f1=0.6180


# Amsterdam:
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


# Montreal
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