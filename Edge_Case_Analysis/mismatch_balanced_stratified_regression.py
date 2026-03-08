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

# results

# ============================================================
# STRATIFIED REGRESSION: rating < 4.5 vs >= 4.5
# ============================================================

# --- NY  |  Low-rating n=31  |  High-rating n=271 ---

#   [Low rating  (<4.5)]  N=29  R2=0.5072
#   Variable                        Coef    Std Err        t        p
#   -----------------------------------------------------------------
#   avg_mover                    -2.9908     1.3283   -2.252   0.0371 *
#   avg_munder                  -11.0508    10.4726   -1.055   0.3053
#   log_price                     0.5543     0.3620    1.531   0.1431

#   [High rating (>=4.5)]  N=271  R2=0.0957
#   Variable                        Coef    Std Err        t        p
#   -----------------------------------------------------------------
#   avg_mover                    -6.0009     3.7531   -1.599   0.1111 
#   avg_munder                    1.5550     7.9779    0.195   0.8456
#   log_price                     0.2216     0.1775    1.248   0.2131

# --- MO  |  Low-rating n=588  |  High-rating n=4155 ---

#   [Low rating  (<4.5)]  N=581  R2=0.1232
#   Variable                        Coef    Std Err        t        p
#   -----------------------------------------------------------------
#   avg_mover                    -2.4817     0.7031   -3.530   0.0005 ***
#   avg_munder                    9.8402     3.4451    2.856   0.0044 **
#   log_price                     0.0372     0.0564    0.659   0.5100

#   [High rating (>=4.5)]  N=4155  R2=0.1154
#   Variable                        Coef    Std Err        t        p
#   -----------------------------------------------------------------
#   avg_mover                    -5.6340     2.1481   -2.623   0.0088 **
#   avg_munder                   -5.5310     2.0485   -2.700   0.0070 **
#   log_price                     0.4727     0.0303   15.576   0.0000 ***

# --- AM  |  Low-rating n=231  |  High-rating n=3882 ---

#   [Low rating  (<4.5)]  N=231  R2=0.2382
#   Variable                        Coef    Std Err        t        p
#   -----------------------------------------------------------------
#   avg_mover                    -6.8281     2.8552   -2.391   0.0177 *
#   avg_munder                   -2.9261     3.7312   -0.784   0.4338
#   log_price                    -0.3981     0.1144   -3.481   0.0006 ***

#   [High rating (>=4.5)]  N=3882  R2=0.1667
#   Variable                        Coef    Std Err        t        p
#   -----------------------------------------------------------------
#   avg_mover                    -1.2410     1.7983   -0.690   0.4902
#   avg_munder                   -1.8104     1.1975   -1.512   0.1307
#   log_price                    -0.7877     0.0410  -19.215   0.0000 ***

# ============================================================
# STRATIFIED SUMMARY: avg_mover coefficient by rating group
# ============================================================
#   City   Group                      Coef    p-value   Significant?
#   -----------------------------------------------------------------
#   NY     Low  (<4.5)             -2.9908     0.0371          Yes *
#   NY     High (>=4.5)            -6.0009     0.1111             No
#   MO     Low  (<4.5)             -2.4817     0.0005        Yes ***
#   MO     High (>=4.5)            -5.6340     0.0088         Yes **
#   AM     Low  (<4.5)             -6.8281     0.0177          Yes *
#   AM     High (>=4.5)            -1.2410     0.4902             No




# ============================================================
# STRATIFIED REGRESSION (BALANCED): rating < 4.5 vs >= 4.5
# ============================================================

# --- NY  |  Low-rating n=31  |  High-rating n=31 ---

#   [Low rating  (<4.5)]  N=29  R2=0.5072
#   Variable                        Coef    Std Err        t        p
#   -----------------------------------------------------------------
#   avg_mover                    -2.9908     1.3283   -2.252   0.0371 *
#   avg_munder                  -11.0508    10.4726   -1.055   0.3053
#   log_price                     0.5543     0.3620    1.531   0.1431

#   [High rating (>=4.5)]  N=25  R2=0.5039
#   Variable                        Coef    Std Err        t        p
#   -----------------------------------------------------------------
#   avg_mover                     1.8571    34.1165    0.054   0.9573
#   avg_munder                  -60.0484    23.9935   -2.503   0.0244 *
#   log_price                    -0.6866     0.4926   -1.394   0.1837

# --- MO  |  Low-rating n=588  |  High-rating n=588 ---

#   [Low rating  (<4.5)]  N=581  R2=0.1232
#   Variable                        Coef    Std Err        t        p
#   -----------------------------------------------------------------
#   avg_mover                    -2.4817     0.7031   -3.530   0.0005 ***
#   avg_munder                    9.8402     3.4451    2.856   0.0044 **
#   log_price                     0.0372     0.0564    0.659   0.5100

#   [High rating (>=4.5)]  N=579  R2=0.1477
#   Variable                        Coef    Std Err        t        p
#   -----------------------------------------------------------------
#   avg_mover                    -9.2206     5.9660   -1.546   0.1228
#   avg_munder                  -15.8598     5.7854   -2.741   0.0063 **
#   log_price                     0.5532     0.0845    6.547   0.0000 ***

# --- AM  |  Low-rating n=231  |  High-rating n=231 ---

#   [Low rating  (<4.5)]  N=231  R2=0.2382
#   Variable                        Coef    Std Err        t        p
#   -----------------------------------------------------------------
#   avg_mover                    -6.8281     2.8552   -2.391   0.0177 *
#   avg_munder                   -2.9261     3.7312   -0.784   0.4338
#   log_price                    -0.3981     0.1144   -3.481   0.0006 ***

#   [High rating (>=4.5)]  N=229  R2=0.2799
#   Variable                        Coef    Std Err        t        p
#   -----------------------------------------------------------------
#   avg_mover                     9.6038     9.3076    1.032   0.3034
#   avg_munder                   -1.5195     5.6863   -0.267   0.7896
#   log_price                    -1.1581     0.1702   -6.803   0.0000 ***

# ============================================================
# STRATIFIED SUMMARY: avg_mover coefficient by rating group (BALANCED)
# ============================================================
#   City   Group                      Coef    p-value   Significant?
#   -----------------------------------------------------------------
#   NY     Low  (<4.5)             -2.9908     0.0371          Yes *
#   NY     High (>=4.5)             1.8571     0.9573             No
#   MO     Low  (<4.5)             -2.4817     0.0005        Yes ***
#   MO     High (>=4.5)            -9.2206     0.1228             No
#   AM     Low  (<4.5)             -6.8281     0.0177          Yes *
#   AM     High (>=4.5)             9.6038     0.3034             No




# ============================================================
# COMPARISON (balanced - imbalanced, avg_mover coef)
# ============================================================
# NY Low  (<4.5): +0.0000
# NY High (>=4.5): +7.8580
# MO Low  (<4.5): -0.0000
# MO High (>=4.5): -3.5866
# AM Low  (<4.5): -0.0000
# AM High (>=4.5): +10.8447

# ============================================================
# EVAL TABLE (imbalanced vs balanced)
# ============================================================
# City  Group         Set              N       Coef          p      Sig
# ------------------------------------------------------------
# NY    Low  (<4.5)   imbalanced      29    -2.9908     0.0371        *
# NY    Low  (<4.5)   balanced        29    -2.9908     0.0371        *
# NY    High (>=4.5)  imbalanced     271    -6.0009     0.1111       ns
# NY    High (>=4.5)  balanced        25    +1.8571     0.9573       ns
# MO    Low  (<4.5)   imbalanced     581    -2.4817     0.0005      ***
# MO    Low  (<4.5)   balanced       581    -2.4817     0.0005      ***
# MO    High (>=4.5)  imbalanced    4155    -5.6340     0.0088       **
# MO    High (>=4.5)  balanced       579    -9.2206     0.1228       ns
# AM    Low  (<4.5)   imbalanced     231    -6.8281     0.0177        *
# AM    Low  (<4.5)   balanced       231    -6.8281     0.0177        *
# AM    High (>=4.5)  imbalanced    3882    -1.2410     0.4902       ns
# AM    High (>=4.5)  balanced       229    +9.6038     0.3034       ns

# ============================================================
# WHICH SETTING BETTER SUPPORTS CONCLUSIONS?
# ============================================================
# NY Low  (<4.5): Both support robustly (stable across settings).
# NY High (>=4.5): Conclusion is sensitive to rebalancing; report as non-robust and avoid strong claims.
# MO Low  (<4.5): Both support robustly (stable across settings).
# MO High (>=4.5): Imbalanced is better as primary evidence (larger N, more stable estimate).
# AM Low  (<4.5): Both support robustly (stable across settings).
# AM High (>=4.5): Conclusion is sensitive to rebalancing; report as non-robust and avoid strong claims.