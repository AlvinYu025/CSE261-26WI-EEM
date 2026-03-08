import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
RATING_THRESHOLD = 4.5

DATA_FILES = {
    "NY": "Mismatch_Score/ny/llm_mismatch_score.csv",
    "MO": "Mismatch_Score/mo/llm_mismatch_score.csv",
    "AM": "Mismatch_Score/am/llm_mismatch_score.csv",
}


def load_datasets():
    root = Path(__file__).resolve().parents[1]
    dfs = {}
    sources = {}
    for city, rel in DATA_FILES.items():
        notebook_name = f"{city.lower()}_llm_mismatch_score.csv"
        candidates = [
            root / "Edge_Case_Analysis" / notebook_name,
            root / notebook_name,
            root / rel,
        ]
        p = next((x for x in candidates if x.exists()), None)
        if p is None:
            raise FileNotFoundError(f"{city}: file not found")
        df = pd.read_csv(p)
        df["city"] = city
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        dfs[city] = df
        sources[city] = p
    return dfs, sources


def build_listing_df(df):
    agg = (
        df.groupby("listing_id")
        .agg(
            total_reviews=("review_id", "count"),
            avg_mover=("mover", "mean"),
            avg_munder=("munder", "mean"),
            avg_mabs=("mabs", "mean"),
            avg_rating=("rating", "mean"),
            price=("price", "first"),
            neighborhood=("neighborhood", "first"),
        )
        .reset_index()
    )

    use_age = "date" in df.columns
    if use_age:
        ages = (
            df.groupby("listing_id")
            .agg(first_review=("date", "min"), last_review=("date", "max"))
            .reset_index()
        )
        ages["listing_age_months"] = ((ages["last_review"] - ages["first_review"]).dt.days / 30).clip(lower=1)
        agg = agg.merge(ages[["listing_id", "listing_age_months"]], on="listing_id", how="left")
    else:
        agg["listing_age_months"] = np.nan

    agg["log_reviews"] = np.log1p(agg["total_reviews"])

    agg["price"] = (
        agg["price"].astype(str).str.replace(r"[\$,]", "", regex=True).str.strip()
    )
    agg["price"] = pd.to_numeric(agg["price"], errors="coerce")
    agg["log_price"] = np.log1p(agg["price"])

    drop_cols = ["avg_mover", "log_price", "neighborhood"]
    if use_age:
        drop_cols.append("listing_age_months")
    agg = agg.dropna(subset=drop_cols)
    return agg, use_age


def make_formula(use_age: bool):
    f = "log_reviews ~ avg_mover + avg_munder + log_price"
    if use_age:
        f += " + listing_age_months"
    f += " + C(neighborhood)"
    return f


def maybe_balance(low, high):
    n = min(len(low), len(high))
    if n == 0:
        return low.iloc[0:0], high.iloc[0:0]
    return (
        low.sample(n=n, random_state=RANDOM_STATE, replace=False),
        high.sample(n=n, random_state=RANDOM_STATE, replace=False),
    )


def run_stratified(listing_dfs, use_age_map, balanced=False):
    title = (
        f"STRATIFIED REGRESSION (BALANCED): rating < {RATING_THRESHOLD} vs >= {RATING_THRESHOLD}"
        if balanced
        else f"STRATIFIED REGRESSION: rating < {RATING_THRESHOLD} vs >= {RATING_THRESHOLD}"
    )
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    summary_records = []

    for city, ldf in listing_dfs.items():
        low = ldf[ldf["avg_rating"] < RATING_THRESHOLD].copy()
        high = ldf[ldf["avg_rating"] >= RATING_THRESHOLD].copy()
        if balanced:
            low, high = maybe_balance(low, high)

        print(f"\n--- {city}  |  Low-rating n={len(low)}  |  High-rating n={len(high)} ---")

        for label, subdf, tag in [
            ("Low rating  (<4.5)", low, "Low  (<4.5)"),
            ("High rating (>=4.5)", high, "High (>=4.5)"),
        ]:
            if len(subdf) < 30:
                print(f"  [{label}] Too few samples ({len(subdf)}), skipping.")
                summary_records.append(
                    {"city": city, "group": tag, "coef": np.nan, "p": np.nan, "n": len(subdf), "note": "N too small"}
                )
                continue

            hood_counts = subdf["neighborhood"].value_counts()
            valid_hoods = hood_counts[hood_counts > 1].index
            subdf = subdf[subdf["neighborhood"].isin(valid_hoods)]

            try:
                formula = make_formula(use_age_map[city])
                model = smf.ols(formula, data=subdf).fit()

                print(f"\n  [{label}]  N={int(model.nobs)}  R2={model.rsquared:.4f}")
                print(f"  {'Variable':<25} {'Coef':>10} {'Std Err':>10} {'t':>8} {'p':>8}")
                print(f"  {'-'*65}")

                key_vars = ["avg_mover", "avg_munder", "log_price"]
                if use_age_map[city]:
                    key_vars.append("listing_age_months")

                for var in key_vars:
                    if var in model.params:
                        coef = model.params[var]
                        se = model.bse[var]
                        t = model.tvalues[var]
                        p = model.pvalues[var]
                        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                        print(f"  {var:<25} {coef:>10.4f} {se:>10.4f} {t:>8.3f} {p:>8.4f} {sig}")

                coef = model.params.get("avg_mover", np.nan)
                p = model.pvalues.get("avg_mover", np.nan)
                summary_records.append({"city": city, "group": tag, "coef": coef, "p": p, "n": int(model.nobs), "note": ""})
            except Exception as e:
                print(f"  [{label}] Error: {e}")
                summary_records.append(
                    {"city": city, "group": tag, "coef": np.nan, "p": np.nan, "n": len(subdf), "note": f"Error: {e}"}
                )

    print("\n" + "=" * 60)
    sum_title = "STRATIFIED SUMMARY: avg_mover coefficient by rating group"
    if balanced:
        sum_title += " (BALANCED)"
    print(sum_title)
    print("=" * 60)
    print(f"  {'City':<6} {'Group':<20} {'Coef':>10} {'p-value':>10} {'Significant?':>14}")
    print(f"  {'-'*65}")

    for city in ["NY", "MO", "AM"]:
        for group in ["Low  (<4.5)", "High (>=4.5)"]:
            recs = [r for r in summary_records if r["city"] == city and r["group"] == group]
            if not recs:
                continue
            r = recs[0]
            if r["note"] == "N too small":
                print(f"  {city:<6} {group:<20} {'N too small':>35}")
                continue
            if np.isnan(r["coef"]) or np.isnan(r["p"]):
                print(f"  {city:<6} {group:<20} {'Error':>35}")
                continue
            sig = "Yes ***" if r["p"] < 0.001 else ("Yes **" if r["p"] < 0.01 else ("Yes *" if r["p"] < 0.05 else "No"))
            print(f"  {city:<6} {group:<20} {r['coef']:>10.4f} {r['p']:>10.4f} {sig:>14}")

    return summary_records


def main():
    dfs, sources = load_datasets()

    print("=" * 60)
    print("DATA SOURCES")
    print("=" * 60)
    for city in ["NY", "MO", "AM"]:
        print(f"{city}: {sources[city]}")

    listing_dfs = {}
    use_age_map = {}
    for city, df in dfs.items():
        ldf, use_age = build_listing_df(df)
        listing_dfs[city] = ldf
        use_age_map[city] = use_age

    if not all(use_age_map.values()):
        print("\n[NOTE] Some files do not contain 'date'.")
        print("[NOTE] Output format is aligned to notebook, but coefficients cannot exactly match notebook outputs.")

    imbalanced_records = run_stratified(listing_dfs, use_age_map, balanced=False)
    balanced_records = run_stratified(listing_dfs, use_age_map, balanced=True)

    print("\n" + "=" * 60)
    print("COMPARISON (balanced - imbalanced, avg_mover coef)")
    print("=" * 60)
    for city in ["NY", "MO", "AM"]:
        for group in ["Low  (<4.5)", "High (>=4.5)"]:
            i = next((r for r in imbalanced_records if r["city"] == city and r["group"] == group), None)
            b = next((r for r in balanced_records if r["city"] == city and r["group"] == group), None)
            if i is None or b is None or np.isnan(i["coef"]) or np.isnan(b["coef"]):
                print(f"{city} {group}: N/A")
            else:
                print(f"{city} {group}: {b['coef'] - i['coef']:+.4f}")

    print("\n" + "=" * 60)
    print("EVAL TABLE (imbalanced vs balanced)")
    print("=" * 60)
    print(
        f"{'City':<5} {'Group':<13} {'Set':<11} {'N':>6} {'Coef':>10} {'p':>10} {'Sig':>8}"
    )
    print("-" * 60)

    def sig_label(p):
        if pd.isna(p):
            return "N/A"
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    for city in ["NY", "MO", "AM"]:
        for group in ["Low  (<4.5)", "High (>=4.5)"]:
            i = next((r for r in imbalanced_records if r["city"] == city and r["group"] == group), None)
            b = next((r for r in balanced_records if r["city"] == city and r["group"] == group), None)
            for label, r in [("imbalanced", i), ("balanced", b)]:
                if r is None:
                    continue
                coef_str = "N/A" if pd.isna(r["coef"]) else f"{r['coef']:+.4f}"
                p_str = "N/A" if pd.isna(r["p"]) else f"{r['p']:.4f}"
                n_val = int(r["n"]) if not pd.isna(r["n"]) else -1
                print(f"{city:<5} {group:<13} {label:<11} {n_val:>6} {coef_str:>10} {p_str:>10} {sig_label(r['p']):>8}")

    print("\n" + "=" * 60)
    print("WHICH SETTING BETTER SUPPORTS CONCLUSIONS?")
    print("=" * 60)
    for city in ["NY", "MO", "AM"]:
        for group in ["Low  (<4.5)", "High (>=4.5)"]:
            i = next((r for r in imbalanced_records if r["city"] == city and r["group"] == group), None)
            b = next((r for r in balanced_records if r["city"] == city and r["group"] == group), None)
            if i is None or b is None or pd.isna(i["coef"]) or pd.isna(b["coef"]):
                print(f"{city} {group}: insufficient data for strict comparison.")
                continue

            same_sign = np.sign(i["coef"]) == np.sign(b["coef"])
            delta = abs(b["coef"] - i["coef"])
            i_sig = (i["p"] < 0.05) if not pd.isna(i["p"]) else False
            b_sig = (b["p"] < 0.05) if not pd.isna(b["p"]) else False

            if same_sign and i_sig == b_sig and delta < 1.0:
                verdict = "Both support robustly (stable across settings)."
            elif same_sign and (i["n"] >= b["n"] * 2):
                verdict = "Imbalanced is better as primary evidence (larger N, more stable estimate)."
            elif not same_sign or delta >= 3.0:
                verdict = "Conclusion is sensitive to rebalancing; report as non-robust and avoid strong claims."
            else:
                verdict = "Use imbalanced as main result, balanced as sensitivity check."

            print(f"{city} {group}: {verdict}")


if __name__ == "__main__":
    main()
