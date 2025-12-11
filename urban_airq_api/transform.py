#!/usr/bin/env python3
"""
transform.py

Flatten OpenAQ v3 and Open-Meteo JSON raw files into one-hour rows and produce
data/staged/air_quality_transformed.csv with derived features.

Required output columns:
city, time, pm10, pm2_5, carbon_monoxide, nitrogen_dioxide,
sulphur_dioxide, ozone, uv_index, aqi_category, severity_score, risk_flag, hour
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw")
STAGED_DIR = Path("data/staged")
STAGED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = STAGED_DIR / "air_quality_transformed.csv"

PARAMETER_MAP = {
    "pm25": "pm2_5",
    "pm2_5": "pm2_5",
    "pm10": "pm10",
    "co": "carbon_monoxide",
    "carbon_monoxide": "carbon_monoxide",
    "no2": "nitrogen_dioxide",
    "nitrogen_dioxide": "nitrogen_dioxide",
    "so2": "sulphur_dioxide",
    "sulphur_dioxide": "sulphur_dioxide",
    "o3": "ozone",
    "ozone": "ozone",
    "uv": "uv_index",
    "uv_index": "uv_index",
}

REQUIRED_POLLUTANTS = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "uv_index"]

def list_raw_files() -> List[Path]:
    return sorted([p for p in RAW_DIR.glob("*.json")])

def detect_api_format(payload: Dict[str, Any]) -> str:
    if isinstance(payload, dict) and "results" in payload:
        return "openaq"
    if isinstance(payload, dict) and "hourly" in payload:
        return "openmeteo"
    return "unknown"

def flatten_openaq(payload: Dict[str, Any], city: str) -> pd.DataFrame:
    rows = []
    for station in payload.get("results", []):
        measurements = station.get("measurements", []) or []
        station_time = station.get("lastUpdated") or (station.get("date") or {}).get("utc")
        for m in measurements:
            param = str(m.get("parameter", "")).lower()
            value = m.get("value")
            time_val = m.get("lastUpdated") or station_time or (m.get("date") or {}).get("utc")
            rows.append({"city": city, "time": time_val, "parameter": param, "value": value})

    if not rows:
        return pd.DataFrame(columns=["city", "time"] + REQUIRED_POLLUTANTS)

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index=["city", "time"], columns="parameter", values="value", aggfunc="first").reset_index()

    # rename parameters to canonical names
    rename_map = {col: PARAMETER_MAP[col] for col in pivot.columns if isinstance(col, str) and col in PARAMETER_MAP}
    pivot = pivot.rename(columns=rename_map)

    # ensure required pollutant columns are present
    for p in REQUIRED_POLLUTANTS:
        if p not in pivot.columns:
            pivot[p] = np.nan

    cols = ["city", "time"] + REQUIRED_POLLUTANTS
    pivot = pivot[cols]
    return pivot

def flatten_openmeteo(payload: Dict[str, Any], city: str) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    if not hourly:
        return pd.DataFrame(columns=["city", "time"] + REQUIRED_POLLUTANTS)
    df = pd.DataFrame(hourly)
    df["city"] = city
    for p in REQUIRED_POLLUTANTS:
        if p not in df.columns:
            df[p] = np.nan
    cols = ["city", "time"] + REQUIRED_POLLUTANTS
    df = df[cols]
    return df

def parse_time_col(df: pd.DataFrame) -> pd.DataFrame:
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df

def enforce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQUIRED_POLLUTANTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def drop_all_missing_pollutants(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=REQUIRED_POLLUTANTS, how="all")

def compute_aqi_category(pm2_5: float) -> str:
    if pd.isna(pm2_5):
        return "Unknown"
    if pm2_5 <= 50:
        return "Good"
    if pm2_5 <= 100:
        return "Moderate"
    if pm2_5 <= 200:
        return "Unhealthy"
    if pm2_5 <= 300:
        return "Very Unhealthy"
    return "Hazardous"

def compute_severity_score(row: pd.Series) -> float:
    vals = {k: row.get(k, 0) or 0 for k in ["pm2_5","pm10","nitrogen_dioxide","sulphur_dioxide","carbon_monoxide","ozone"]}
    return (vals["pm2_5"] * 5.0) + (vals["pm10"] * 3.0) + (vals["nitrogen_dioxide"] * 4.0) + \
           (vals["sulphur_dioxide"] * 4.0) + (vals["carbon_monoxide"] * 2.0) + (vals["ozone"] * 3.0)

def classify_risk(severity: float) -> str:
    if pd.isna(severity):
        return "Unknown"
    if severity > 400:
        return "High Risk"
    if severity > 200:
        return "Moderate Risk"
    return "Low Risk"

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_time_col(df)
    df = enforce_numeric(df)
    df = drop_all_missing_pollutants(df)
    if df.empty:
        return df
    df["aqi_category"] = df["pm2_5"].apply(compute_aqi_category)
    df["severity_score"] = df.apply(compute_severity_score, axis=1)
    df["risk_flag"] = df["severity_score"].apply(classify_risk)
    df["hour"] = df["time"].dt.hour
    all_cols = ["city","time"] + REQUIRED_POLLUTANTS + ["aqi_category","severity_score","risk_flag","hour"]
    for c in all_cols:
        if c not in df.columns:
            df[c] = None
    return df[all_cols]

def transform_all() -> None:
    files = list_raw_files()
    frames: List[pd.DataFrame] = []
    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️ Failed to read {f.name}: {e}")
            continue
        city = f.stem.split("_")[0].strip()
        fmt = detect_api_format(payload)
        if fmt == "openaq":
            df = flatten_openaq(payload, city)
        elif fmt == "openmeteo":
            df = flatten_openmeteo(payload, city)
        else:
            print(f"⚠️ Unknown format for {f.name}; skipping.")
            continue
        frames.append(df)

    if not frames:
        print("❌ No raw data found to transform.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined["city"] = combined["city"].astype(str).str.strip().replace({"bangalore":"Bengaluru","Bangalore":"Bengaluru"})
    final = add_features(combined)
    if final.empty:
        print("❌ No valid pollutant rows after cleaning.")
        return
    # re-order and save
    final = final[["city","time","pm10","pm2_5","carbon_monoxide","nitrogen_dioxide","sulphur_dioxide","ozone","uv_index","aqi_category","severity_score","risk_flag","hour"]]
    final.to_csv(OUTPUT_FILE, index=False, date_format="%Y-%m-%dT%H:%M:%S")
    print(f"✅ Saved transformed dataset → {OUTPUT_FILE} (rows: {len(final)})")

if __name__ == "__main__":
    transform_all()
