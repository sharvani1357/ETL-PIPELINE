#!/usr/bin/env python3
"""
load.py (robust)

- Reads staged CSV (data/staged/air_quality_transformed.csv)
- Cleans NaN / inf / numpy types -> JSON-friendly Python types
- Batch inserts into supabase.public.air_quality_data (default batch_size=200)
- Retries failed batches (default retries=2)
- Prints diagnostics and summary
"""
from __future__ import annotations
import os
import time
import logging
from pathlib import Path
import math
from typing import List, Any, Dict
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[0]
STAGED_DIR = Path(os.getenv("STAGED_DIR", BASE_DIR / "data" / "staged"))
STAGED_CSV = STAGED_DIR / "air_quality_transformed.csv"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("Please set SUPABASE_URL and SUPABASE_KEY in your .env")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "200"))
LOAD_MAX_RETRIES = int(os.getenv("LOAD_MAX_RETRIES", "2"))
LOAD_BACKOFF_SECONDS = float(os.getenv("LOAD_BACKOFF_SECONDS", "3"))

TABLE_NAME = "air_quality_data"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS public.{TABLE_NAME} (
    id BIGSERIAL PRIMARY KEY,
    city TEXT,
    time TIMESTAMP,
    pm10 DOUBLE PRECISION,
    pm2_5 DOUBLE PRECISION,
    carbon_monoxide DOUBLE PRECISION,
    nitrogen_dioxide DOUBLE PRECISION,
    sulphur_dioxide DOUBLE PRECISION,
    ozone DOUBLE PRECISION,
    uv_index DOUBLE PRECISION,
    aqi_category TEXT,
    severity_score DOUBLE PRECISION,
    risk_flag TEXT,
    hour INTEGER
);
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("load")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def create_table_if_not_exists():
    try:
        logger.info("Attempting to create table via RPC (if permitted)...")
        supabase.rpc("execute_sql", {"query": CREATE_TABLE_SQL}).execute()
        logger.info("create_table_if_not_exists: RPC executed (or table exists).")
    except Exception as e:
        logger.warning(f"Could not create table via RPC: {e}")
        logger.info("Please run the following SQL in Supabase SQL editor if the table doesn't exist:")
        logger.info(CREATE_TABLE_SQL)

def _read_staged_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Staged CSV not found at {path}")
    df = pd.read_csv(path)
    required_cols = [
        "city","time","pm10","pm2_5","carbon_monoxide","nitrogen_dioxide",
        "sulphur_dioxide","ozone","uv_index","aqi_category","severity_score","risk_flag","hour"
    ]
    for c in required_cols:
        if c not in df.columns:
            df[c] = None
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["time"] = df["time"].dt.tz_localize(None)
    numeric_cols = ["pm10","pm2_5","carbon_monoxide","nitrogen_dioxide","sulphur_dioxide","ozone","uv_index","severity_score","hour"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[required_cols]

def _clean_record_value(v: Any) -> Any:
    if v is pd.NaT:
        return None
    if isinstance(v, (np.floating, float)):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(v)
    if isinstance(v, (np.integer, int)):
        if pd.isna(v):
            return None
        return int(v)
    if isinstance(v, (pd.Timestamp, )):
        if pd.isna(v):
            return None
        return v.isoformat()
    if v is pd.NA:
        return None
    if isinstance(v, (np.bool_, )):
        return bool(v)
    try:
        if isinstance(v, (np.ndarray, )):
            return v.tolist()
    except Exception:
        pass
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v

def _records_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    df_clean = df.copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.replace({pd.NA: np.nan})
    records = []
    for rec in df_clean.to_dict(orient="records"):
        newrec = {}
        for k, v in rec.items():
            newrec[k] = _clean_record_value(v)
        records.append(newrec)
    return records

def _diagnostics(df: pd.DataFrame):
    logger.info(f"Staged rows: {len(df)}")
    null_counts = df.isna().sum().to_dict()
    logger.info("Null counts per column:")
    for k, v in null_counts.items():
        logger.info(f" - {k}: {v}")
    logger.info("Sample rows (first 3):")
    logger.info(df.head(3).to_dict(orient="records"))

def load_to_supabase(staged_csv_path: Path, batch_size: int = BATCH_SIZE, retries: int = LOAD_MAX_RETRIES):
    df = _read_staged_csv(staged_csv_path)
    total = len(df)
    if total == 0:
        logger.info("No rows to load.")
        return
    _diagnostics(df)
    records = _records_from_df(df)
    logger.info(f"Prepared {len(records)} JSON-ready records for insertion.")
    inserted = 0
    failed_batches = 0
    for i in range(0, total, batch_size):
        batch = records[i:i+batch_size]
        attempt = 0
        max_attempts = 1 + retries
        while attempt < max_attempts:
            attempt += 1
            try:
                res = supabase.table(TABLE_NAME).insert(batch).execute()
                err = getattr(res, "error", None)
                if err:
                    raise RuntimeError(err)
                end = min(i+batch_size, total)
                inserted += (end - i)
                logger.info(f"✅ Inserted rows {i+1}-{end} of {total}")
                break
            except Exception as e:
                logger.warning(f"Batch {i//batch_size + 1} attempt {attempt}/{max_attempts} failed: {e}")
                if attempt < max_attempts:
                    backoff = LOAD_BACKOFF_SECONDS * attempt
                    logger.info(f"Retrying batch after {backoff}s ...")
                    time.sleep(backoff)
                else:
                    logger.error(f"❌ Batch {i//batch_size + 1} failed after {max_attempts} attempts. Skipping batch.")
                    failed_batches += 1
                    break
    logger.info("🎯 Load complete.")
    logger.info(f"Inserted rows: {inserted}/{total}")
    if failed_batches:
        logger.warning(f"Failed batches: {failed_batches}")
    else:
        logger.info("All batches inserted successfully.")

if __name__ == "__main__":
    create_table_if_not_exists()
    try:
        load_to_supabase(STAGED_CSV, batch_size=BATCH_SIZE, retries=LOAD_MAX_RETRIES)
    except Exception as e:
        logger.exception("Load failed: %s", e)
