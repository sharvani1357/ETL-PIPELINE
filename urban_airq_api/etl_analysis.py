#!/usr/bin/env python3
"""
etl_analysis.py

Reads air_quality_data from Supabase, computes KPIs, generates reports and plots,
and writes CSVs + PNGs into data/processed/.
"""
from __future__ import annotations
import os
from pathlib import Path
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[0]
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", BASE_DIR / "data" / "processed"))
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = os.getenv("TABLE_NAME", "air_quality_data")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("etl_analysis")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("Please set SUPABASE_URL and SUPABASE_KEY in your .env")

def fetch_table_from_supabase() -> pd.DataFrame:
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Fetching table '%s' from Supabase...", TABLE_NAME)
    res = client.table(TABLE_NAME).select("*").execute()
    data = None
    if hasattr(res, "data"):
        data = res.data
    elif isinstance(res, dict) and "data" in res:
        data = res["data"]
    else:
        try:
            data = res.json()
        except Exception as e:
            logger.exception("Could not parse Supabase response: %s", e)
            raise RuntimeError("Unexpected Supabase response shape; inspect 'res' object") from e

    df = pd.DataFrame(data)
    if df.empty:
        logger.warning("Supabase returned no rows.")
        return df
    df.columns = [c.lower() for c in df.columns]
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    numeric_cols = ["pm2_5", "pm10", "ozone", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "uv_index", "severity_score", "hour"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "risk_flag" not in df.columns and "risk_level" in df.columns:
        df = df.rename(columns={"risk_level":"risk_flag"})
    return df

def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    metrics = {}
    if "pm2_5" in df.columns and not df["pm2_5"].dropna().empty:
        avg_pm25_by_city = df.groupby("city")["pm2_5"].mean().dropna()
        metrics["city_highest_avg_pm2_5"] = avg_pm25_by_city.idxmax()
        metrics["highest_avg_pm2_5"] = float(round(avg_pm25_by_city.max(), 3))
    else:
        metrics["city_highest_avg_pm2_5"] = None
        metrics["highest_avg_pm2_5"] = None

    if "severity_score" in df.columns and not df["severity_score"].dropna().empty:
        avg_sev = df.groupby("city")["severity_score"].mean().dropna()
        metrics["city_highest_avg_severity"] = avg_sev.idxmax()
        metrics["highest_avg_severity"] = float(round(avg_sev.max(), 3))
    else:
        metrics["city_highest_avg_severity"] = None
        metrics["highest_avg_severity"] = None

    if "risk_flag" in df.columns and not df["risk_flag"].dropna().empty:
        total = len(df)
        def pct(level):
            return round(100.0 * len(df[df["risk_flag"] == level]) / total, 2) if total else 0.0
        metrics["pct_high_risk"] = pct("High Risk")
        metrics["pct_moderate_risk"] = pct("Moderate Risk")
        metrics["pct_low_risk"] = pct("Low Risk")
    else:
        metrics["pct_high_risk"] = metrics["pct_moderate_risk"] = metrics["pct_low_risk"] = None

    if "hour" in df.columns and "pm2_5" in df.columns:
        hourly = df.groupby("hour")["pm2_5"].mean().dropna()
        if not hourly.empty:
            metrics["hour_worst_avg_pm2_5"] = int(hourly.idxmax())
            metrics["worst_hour_avg_pm2_5"] = float(round(hourly.max(), 3))
        else:
            metrics["hour_worst_avg_pm2_5"] = None
            metrics["worst_hour_avg_pm2_5"] = None
    else:
        metrics["hour_worst_avg_pm2_5"] = None
        metrics["worst_hour_avg_pm2_5"] = None

    return pd.DataFrame([metrics])

def city_risk_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if "risk_flag" not in df.columns:
        return pd.DataFrame(columns=["city","risk_flag","count","total","pct"])
    table = df.groupby(["city","risk_flag"]).size().reset_index(name="count")
    totals = df.groupby("city").size().reset_index(name="total")
    merged = table.merge(totals, on="city", how="left")
    merged["pct"] = (merged["count"] / merged["total"] * 100).round(2)
    merged = merged.sort_values(["city","risk_flag"], ascending=[True, False]).reset_index(drop=True)
    return merged

def pollution_trends(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        return pd.DataFrame()
    df2 = df.dropna(subset=["time"]).copy()
    df2["time_hour"] = df2["time"].dt.floor("H")
    cols = [c for c in ["pm2_5","pm10","ozone"] if c in df2.columns]
    if not cols:
        return pd.DataFrame()
    agg = df2.groupby(["city","time_hour"])[cols].mean().reset_index()
    agg = agg.rename(columns={"time_hour":"time"})
    agg = agg.sort_values(["city","time"]).reset_index(drop=True)
    return agg

def plot_histogram_pm25(df: pd.DataFrame, out_path: Path):
    if "pm2_5" not in df.columns or df["pm2_5"].dropna().empty:
        logger.warning("No PM2.5 data for histogram.")
        return
    plt.figure(figsize=(8,5))
    plt.hist(df["pm2_5"].dropna(), bins=30)
    plt.title("Histogram of PM2.5")
    plt.xlabel("PM2.5 (µg/m³)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info("Wrote %s", out_path)

def plot_bar_risk_flags_per_city(df: pd.DataFrame, out_path: Path):
    if "risk_flag" not in df.columns:
        logger.warning("No risk_flag column for bar plot.")
        return
    ct = df.groupby(["city","risk_flag"]).size().unstack(fill_value=0)
    if ct.empty:
        logger.warning("No counts for risk flags.")
        return
    ax = ct.plot(kind="bar", figsize=(10,6))
    ax.set_title("Risk Flags per City")
    ax.set_xlabel("City")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info("Wrote %s", out_path)

def plot_line_hourly_pm25(trends_df: pd.DataFrame, out_path: Path):
    if trends_df.empty:
        logger.warning("No trends data for line plot.")
        return
    pivot = trends_df.pivot(index="time", columns="city", values="pm2_5")
    if pivot.empty:
        logger.warning("Pivot resulted in empty DataFrame for trends.")
        return
    pivot.plot(figsize=(12,6))
    plt.title("Hourly PM2.5 Trends (avg)")
    plt.xlabel("Time")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info("Wrote %s", out_path)

def plot_scatter_severity_vs_pm25(df: pd.DataFrame, out_path: Path):
    if "severity_score" not in df.columns or "pm2_5" not in df.columns:
        logger.warning("No data for severity vs pm2_5 scatter.")
        return
    df2 = df.dropna(subset=["severity_score","pm2_5"])
    if df2.empty:
        logger.warning("No rows with both severity_score and pm2_5.")
        return
    plt.figure(figsize=(8,6))
    plt.scatter(df2["pm2_5"], df2["severity_score"])
    plt.title("Severity Score vs PM2.5")
    plt.xlabel("PM2.5 (µg/m³)")
    plt.ylabel("Severity Score")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info("Wrote %s", out_path)

def main():
    df = fetch_table_from_supabase()
    if df.empty:
        logger.error("No data loaded from Supabase. Exiting.")
        return

    kpis = compute_kpis(df)
    kpi_csv = PROCESSED_DIR / "summary_metrics.csv"
    kpis.to_csv(kpi_csv, index=False)
    logger.info("Saved KPIs -> %s", kpi_csv)

    risk_df = city_risk_distribution(df)
    risk_csv = PROCESSED_DIR / "city_risk_distribution.csv"
    risk_df.to_csv(risk_csv, index=False)
    logger.info("Saved city risk distribution -> %s", risk_csv)

    trends = pollution_trends(df)
    trends_csv = PROCESSED_DIR / "pollution_trends.csv"
    trends.to_csv(trends_csv, index=False)
    logger.info("Saved pollution trends -> %s", trends_csv)

    plot_histogram_pm25(df, PROCESSED_DIR / "histogram_pm25.png")
    plot_bar_risk_flags_per_city(df, PROCESSED_DIR / "bar_risk_flags_per_city.png")
    plot_line_hourly_pm25(trends, PROCESSED_DIR / "line_hourly_pm25_trends.png")
    plot_scatter_severity_vs_pm25(df, PROCESSED_DIR / "scatter_severity_vs_pm25.png")

    logger.info("Analysis finished. All outputs saved in %s", PROCESSED_DIR)

if __name__ == "__main__":
    main()
