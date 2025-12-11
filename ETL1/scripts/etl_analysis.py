#!/usr/bin/env python3
"""
etl_analysis.py

Reads telco_data from Supabase and generates:
 - Metrics:
    * churn_percentage
    * average_monthly_charges_per_contract
    * counts of customer segments (new, regular, loyal, champion)
    * internet service distribution
    * pivot churn vs tenure_group
 - Optional visualizations (saved as PNGs)
 - Summary CSV saved to data/processed/analysis_summary.csv

Run from project root: python etl_analysis.py
"""
import os
import sys
from dotenv import load_dotenv
from supabase import create_client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Config & helper functions
# ---------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = "telco_data"

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in .env. Put them in your .env file.")

# output folder
OUT_DIR = os.path.join("data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_all_table(table_name):
    """Fetch all rows from Supabase table. Uses pagination if necessary."""
    sb = get_supabase_client()

    # Try to fetch all rows. If the client enforces a 1000 limit, page through.
    limit = 1000
    offset = 0
    rows = []
    while True:
        resp = sb.table(table_name).select("*").limit(limit).offset(offset).execute()
        data = resp.data if hasattr(resp, "data") else (resp.get("data") if isinstance(resp, dict) else None)
        if data is None:
            raise RuntimeError("Failed to fetch data from Supabase. Response had no 'data'.")
        rows.extend(data)
        if len(data) < limit:
            break
        offset += limit
    return pd.DataFrame(rows)

def ensure_columns(df, cols):
    """Return list of missing columns."""
    return [c for c in cols if c not in df.columns]

def safe_series(df, col, default=None):
    return df[col] if col in df.columns else pd.Series([default]*len(df))

# ---------------------------
# Analysis logic
# ---------------------------
def compute_metrics(df: pd.DataFrame):
    # Normalize common column names defensively
    cols_lower = {c.lower(): c for c in df.columns}
    def colname(*variants):
        for v in variants:
            if v in df.columns:
                return v
            if v.lower() in cols_lower:
                return cols_lower[v.lower()]
        return None

    churn_col = colname("churn", "Churn")
    monthly_col = colname("monthly_charges", "MonthlyCharges", "monthlyCharges")
    total_col = colname("total_charges", "TotalCharges", "totalCharges")
    tenure_col = colname("tenure", "tenure")
    tenure_group_col = colname("tenure_group", "tenureGroup", "tenure_group")
    contract_col = colname("contract", "Contract", "contract_code_normalized")
    internet_col = colname("internet_service", "InternetService", "internet_service")

    # Basic data presence checks
    missing = ensure_columns(df, [c for c in [churn_col, monthly_col, total_col, tenure_col] if c is not None])
    if missing:
        print("Warning: some expected columns missing from fetched table:", missing)

    metrics = {}

    # Churn percentage
    if churn_col is not None:
        try:
            churn_vals = pd.to_numeric(df[churn_col], errors="coerce")
            churn_pct = churn_vals.dropna().mean() * 100 if len(churn_vals.dropna())>0 else np.nan
        except Exception:
            churn_pct = np.nan
    else:
        churn_pct = np.nan
    metrics["churn_percentage"] = round(float(churn_pct) if not pd.isna(churn_pct) else np.nan, 3)

    # Average monthly charges per contract
    if monthly_col is not None and contract_col is not None:
        avg_monthly_by_contract = (
            df[[contract_col, monthly_col]]
            .dropna(subset=[contract_col, monthly_col])
            .groupby(contract_col)[monthly_col]
            .mean()
            .round(3)
            .to_dict()
        )
    else:
        avg_monthly_by_contract = {}
    metrics["avg_monthly_by_contract"] = avg_monthly_by_contract

    # Customer segments: new, regular, loyal, champion - based on tenure (simple rule)
    # new: 0-12, regular: 13-24, loyal: 25-48, champion: 49+
    seg_counts = {"new": 0, "regular": 0, "loyal": 0, "champion": 0}
    if tenure_col is not None:
        tenure_vals = pd.to_numeric(df[tenure_col], errors="coerce").fillna(0).astype(int)
        seg_counts["new"] = int((tenure_vals <= 12).sum())
        seg_counts["regular"] = int(((tenure_vals >= 13) & (tenure_vals <= 24)).sum())
        seg_counts["loyal"] = int(((tenure_vals >= 25) & (tenure_vals <= 48)).sum())
        seg_counts["champion"] = int((tenure_vals >= 49).sum())
    metrics["customer_segments_counts"] = seg_counts

    # Internet service distribution
    if internet_col is not None:
        internet_dist = df[internet_col].fillna("Unknown").astype(str).value_counts(dropna=False).to_dict()
    else:
        internet_dist = {}
    metrics["internet_service_distribution"] = internet_dist

    # Pivot: Churn vs Tenure Group
    if churn_col is not None and tenure_group_col is not None:
        pivot = (pd.pivot_table(df, index=tenure_group_col, columns=churn_col, values=monthly_col,
                                aggfunc='count', fill_value=0))
        # Convert pivot to dict-of-dicts for saving
        pivot_dict = {str(idx): {str(col): int(pivot.loc[idx, col]) for col in pivot.columns} for idx in pivot.index}
    else:
        pivot_dict = {}
    metrics["pivot_churn_vs_tenure_group"] = pivot_dict

    return metrics

# ---------------------------
# Visualizations
# ---------------------------
def make_visualizations(df: pd.DataFrame):
    # Normalize names again
    cols_lower = {c.lower(): c for c in df.columns}
    def colname(*variants):
        for v in variants:
            if v in df.columns:
                return v
            if v.lower() in cols_lower:
                return cols_lower[v.lower()]
        return None

    churn_col = colname("churn", "Churn")
    monthly_col = colname("monthly_charges", "MonthlyCharges", "monthlyCharges")
    total_col = colname("total_charges", "TotalCharges", "totalCharges")
    tenure_group_col = colname("tenure_group", "tenureGroup", "tenure_group")
    contract_col = colname("contract", "Contract", "contract_code_normalized")

    plots = {}

    # 1) Churn rate by Monthly Charge Segment (derive monthly_charge_segment using quantiles)
    try:
        if monthly_col:
            df_tmp = df.copy()
            # safe conversion
            df_tmp[monthly_col] = pd.to_numeric(df_tmp[monthly_col], errors="coerce")
            df_tmp["monthly_charge_segment"] = pd.qcut(df_tmp[monthly_col].fillna(0) + 1e-9, q=3, labels=["low", "medium", "high"])
            if churn_col:
                seg = (df_tmp.groupby("monthly_charge_segment")[churn_col].mean() * 100).reindex(["low", "medium", "high"])
            else:
                seg = pd.Series(dtype=float)
            fig, ax = plt.subplots()
            seg.plot(kind="bar", ax=ax)
            ax.set_title("Churn rate by Monthly Charge Segment (%)")
            ax.set_ylabel("Churn %")
            ax.set_xlabel("Monthly charge segment")
            fname = os.path.join(OUT_DIR, "churn_by_monthly_segment.png")
            fig.tight_layout()
            fig.savefig(fname)
            plt.close(fig)
            plots["churn_by_monthly_segment"] = fname
    except Exception as e:
        print("Warning: could not create churn_by_monthly_segment plot:", e)

    # 2) Histogram of TotalCharges
    try:
        if total_col:
            fig, ax = plt.subplots()
            df_tmp = df.copy()
            df_tmp[total_col] = pd.to_numeric(df_tmp[total_col], errors="coerce")
            df_tmp[total_col].dropna().plot(kind="hist", bins=30, ax=ax)
            ax.set_title("Histogram of TotalCharges")
            ax.set_xlabel("TotalCharges")
            fname = os.path.join(OUT_DIR, "hist_total_charges.png")
            fig.tight_layout()
            fig.savefig(fname)
            plt.close(fig)
            plots["hist_total_charges"] = fname
    except Exception as e:
        print("Warning: could not create hist_total_charges plot:", e)

    # 3) Bar plot of Contract types
    try:
        if contract_col:
            fig, ax = plt.subplots()
            df_tmp = df.copy()
            df_tmp[contract_col] = df_tmp[contract_col].astype(str).fillna("Unknown")
            df_tmp[contract_col].value_counts().plot(kind="bar", ax=ax)
            ax.set_title("Contract Types")
            ax.set_xlabel("Contract")
            fname = os.path.join(OUT_DIR, "bar_contract_types.png")
            fig.tight_layout()
            fig.savefig(fname)
            plt.close(fig)
            plots["bar_contract_types"] = fname
    except Exception as e:
        print("Warning: could not create bar_contract_types plot:", e)

    return plots

# ---------------------------
# Save summary CSV
# ---------------------------
def save_summary_csv(metrics: dict, out_path: str):
    # Flatten some nested metrics into a table-friendly form
    rows = []
    # churn percentage
    rows.append({"metric": "churn_percentage", "value": metrics.get("churn_percentage")})
    # avg monthly by contract
    ambc = metrics.get("avg_monthly_by_contract", {})
    for k, v in ambc.items():
        rows.append({"metric": f"avg_monthly_charges_contract_{k}", "value": v})
    # segments
    segs = metrics.get("customer_segments_counts", {})
    for k, v in segs.items():
        rows.append({"metric": f"segment_count_{k}", "value": v})
    # internet distribution
    inet = metrics.get("internet_service_distribution", {})
    for k, v in inet.items():
        rows.append({"metric": f"internet_dist_{k}", "value": v})
    # pivot churn vs tenure group - store as JSON string per group
    pivot = metrics.get("pivot_churn_vs_tenure_group", {})
    for tg, vals in pivot.items():
        rows.append({"metric": f"pivot_tenure_{tg}", "value": str(vals)})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    return df_out

# ---------------------------
# Main
# ---------------------------
def main():
    print("Fetching data from Supabase...")
    df = fetch_all_table(TABLE_NAME)
    if df.empty:
        print("No rows fetched from Supabase table:", TABLE_NAME)
        sys.exit(1)

    print(f"Fetched {len(df)} rows.")

    print("Computing metrics...")
    metrics = compute_metrics(df)

    print("Generating visualizations...")
    plots = make_visualizations(df)

    summary_csv_path = os.path.join(OUT_DIR, "analysis_summary.csv")
    print("Saving summary CSV to:", summary_csv_path)
    summary_df = save_summary_csv(metrics, summary_csv_path)

    # Print a short summary to the console
    print("\n=== Analysis Summary ===")
    print("Rows fetched:", len(df))
    print("Churn %:", metrics.get("churn_percentage"))
    print("Avg monthly by contract:", metrics.get("avg_monthly_by_contract"))
    print("Customer segments:", metrics.get("customer_segments_counts"))
    print("Internet distribution:", metrics.get("internet_service_distribution"))
    print("Pivot churn vs tenure group (sample):", {k: v for k, v in list(metrics.get("pivot_churn_vs_tenure_group", {}).items())[:5]})
    print("Saved summary CSV:", summary_csv_path)
    if plots:
        print("Saved plots:")
        for k, v in plots.items():
            print(" -", k, "->", v)
    else:
        print("No plots generated.")

if __name__ == "__main__":
    main()
