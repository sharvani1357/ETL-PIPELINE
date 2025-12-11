#!/usr/bin/env python3
"""
extract.py

Extract step for Urban Air Quality Monitoring using Open-Meteo Air Quality API.

- Endpoint: https://air-quality-api.open-meteo.com/v1/air-quality
- Query params:
    ?latitude=<lat>&longitude=<lon>&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide,uv_index

Behavior:
- Fetch hourly pollutant data for 5 cities (Delhi, Mumbai, Bengaluru, Hyderabad, Kolkata).
- Retry logic with exponential backoff (default 3 attempts).
- Save raw JSON responses to data/raw/<city>_raw_<timestamp>.json
- On failure, save a failed file with error message: <city>_raw_<timestamp>_failed.json
- Return list of saved file paths.
"""
from __future__ import annotations

import json
import os
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# --------------- Configuration ---------------
BASE_DIR = Path(__file__).resolve().parents[0]
RAW_DIR = Path(os.getenv("RAW_DIR", BASE_DIR / "data" / "raw"))
RAW_DIR.mkdir(parents=True, exist_ok=True)

OPEN_METEO_BASE = os.getenv("OPEN_METEO_BASE", "https://air-quality-api.open-meteo.com/v1/air-quality")

# cities with coordinates (exact list per your requirement)
CITIES: Dict[str, Dict[str, float]] = {
    "Delhi": {"latitude": 28.7041, "longitude": 77.1025},
    "Mumbai": {"latitude": 19.0760, "longitude": 72.8777},
    "Bengaluru": {"latitude": 12.9716, "longitude": 77.5946},
    "Hyderabad": {"latitude": 17.3850, "longitude": 78.4867},
    "Kolkata": {"latitude": 22.5726, "longitude": 88.3639},
}

# Environment-configurable defaults
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))         # attempts
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "10"))
SLEEP_BETWEEN_CALLS = float(os.getenv("SLEEP_BETWEEN_CALLS", "0.5"))

# hourly variables we request (includes uv_index)
HOURLY_VARS = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide,uv_index"

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract")


# --------------- Helpers ---------------
def _now_ts() -> str:
    """Compact UTC timestamp for filenames: 20251211T053045Z"""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _save_json(payload: object, city: str, failed: bool = False) -> str:
    """Save payload to RAW_DIR and return absolute path. If failed=True, add _failed suffix."""
    ts = _now_ts()
    suffix = "failed.json" if failed else "json"
    fname = f"{city.replace(' ', '_').lower()}_raw_{ts}" + (f"_{suffix}" if failed else f".{suffix}")
    path = RAW_DIR / fname
    try:
        with open(path, "w", encoding="utf-8") as f:
            # payload may be an exception string or a dict; dump gracefully
            if isinstance(payload, (dict, list)):
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
            else:
                json.dump({"error": str(payload)}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        # fallback to plain text file if json.dump fails
        alt = RAW_DIR / f"{city.replace(' ', '_').lower()}_raw_{ts}_failed.txt"
        with open(alt, "w", encoding="utf-8") as f:
            f.write(repr(payload))
        return str(alt.resolve())
    return str(path.resolve())


def _call_api(lat: float, lon: float, timeout: int = TIMEOUT_SECONDS) -> requests.Response:
    """Make the HTTP request (no retries here)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_VARS,
    }
    resp = requests.get(OPEN_METEO_BASE, params=params, timeout=timeout)
    return resp


# --------------- Public functions ---------------
def fetch_city_open_meteo(city: str, coords: Dict[str, float]) -> Optional[str]:
    """
    Fetch Open-Meteo air-quality for single city with retries+backoff.
    Returns saved file path (string) on success, or path to failed file on failure.
    """
    lat = coords["latitude"]
    lon = coords["longitude"]
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Requesting %s (attempt %d/%d) params={'latitude':%s,'longitude':%s}", OPEN_METEO_BASE, attempt, MAX_RETRIES, lat, lon)
            resp = _call_api(lat, lon, timeout=TIMEOUT_SECONDS)
            # success 200
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                except ValueError:
                    # non-json body — save raw text
                    payload = {"raw_text": resp.text}
                saved = _save_json(payload, city, failed=False)
                logger.info("✅ [%s] fetched and saved to: %s", city, saved)
                return saved
            else:
                # non-200 — log and maybe retry
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.warning("⚠️ [%s] non-200 response (attempt %d): %s", city, attempt, last_err)
        except requests.RequestException as e:
            last_err = str(e)
            logger.warning("⚠️ [%s] RequestException on attempt %d: %s", city, attempt, last_err)
        except Exception as e:
            last_err = str(e)
            logger.warning("⚠️ [%s] Unexpected error on attempt %d: %s", city, attempt, last_err)

        # backoff before next attempt
        backoff = 2 ** (attempt - 1)
        logger.info("⏳ [%s] sleeping %ds before retrying...", city, backoff)
        time.sleep(backoff)

    # exhausted retries -> save failed payload describing error
    logger.error("❌ [%s] All %d attempts failed. Saving failed payload.", city, MAX_RETRIES)
    failed_path = _save_json({"error": last_err}, city, failed=True)
    logger.info("Saved failed payload to: %s", failed_path)
    return failed_path


def fetch_all_cities(cities: Dict[str, Dict[str, float]] = CITIES) -> List[str]:
    """
    Fetch air-quality data for all cities in 'cities' dict.
    Returns list of saved file paths (successful plus failed).
    """
    saved_paths: List[str] = []
    for city, coords in cities.items():
        path = fetch_city_open_meteo(city, coords)
        saved_paths.append(path)
        time.sleep(SLEEP_BETWEEN_CALLS)
    return saved_paths


# --------------- CLI ---------------
if __name__ == "__main__":
    logger.info("Starting Open-Meteo extraction for cities: %s", list(CITIES.keys()))
    saved = fetch_all_cities()
    logger.info("Extraction complete. Saved files:")
    for p in saved:
        logger.info(" - %s", p)
