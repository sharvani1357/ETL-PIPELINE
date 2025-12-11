#!/usr/bin/env python3
"""
Simple ETL runner: extract -> transform -> load -> analysis

This keeps the interface tiny (like your example) but will try a couple
of common function names so you don't have to change your other files.
"""
import time
from pathlib import Path

import extract as extract_mod
import transform as transform_mod
import load as load_mod
import etl_analysis as analysis_mod

# default staged CSV that transform should produce
DEFAULT_STAGED = Path("data/staged/air_quality_transformed.csv")


def _call_extract():
    # prefer fetch_all_cities -> fetch_all -> fetch_all_raw -> fetch_all (module)
    for name in ("fetch_all_cities", "fetch_all", "fetch_all_raw", "extract_all", "fetch_all"):
        fn = getattr(extract_mod, name, None)
        if callable(fn):
            return fn()
    # last resort: try a function named 'fetch' or 'extract'
    for name in ("fetch","extract"):
        fn = getattr(extract_mod, name, None)
        if callable(fn):
            return fn()
    raise RuntimeError("No extract function found in extract.py")


def _call_transform(raw_files):
    # try transform_data(list) -> transform_all() -> transform()
    fn = getattr(transform_mod, "transform_data", None)
    if callable(fn):
        try:
            return fn(raw_files)
        except TypeError:
            return fn()
    fn = getattr(transform_mod, "transform_all", None)
    if callable(fn):
        fn()
        return str(DEFAULT_STAGED)
    fn = getattr(transform_mod, "transform", None)
    if callable(fn):
        try:
            out = fn(raw_files)
            return out if out else str(DEFAULT_STAGED)
        except TypeError:
            fn()
            return str(DEFAULT_STAGED)
    raise RuntimeError("No transform function found in transform.py")


def _call_load(staged_csv):
    # create table if exists
    if hasattr(load_mod, "create_table_if_not_exists"):
        try:
            load_mod.create_table_if_not_exists()
        except Exception as e:
            print("create_table_if_not_exists() returned error (continuing):", e)
    # call loader
    fn = getattr(load_mod, "load_to_supabase", None) or getattr(load_mod, "load", None)
    if not callable(fn):
        raise RuntimeError("No load function (load_to_supabase/load) found in load.py")
    # many loaders accept (path, batch_size=...), so pass path first
    try:
        fn(staged_csv)
    except TypeError:
        fn(str(staged_csv))


def _call_analysis():
    # try main() then run_analysis()
    if callable(getattr(analysis_mod, "main", None)):
        analysis_mod.main()
        return
    if callable(getattr(analysis_mod, "run_analysis", None)):
        analysis_mod.run_analysis()
        return
    # fallback: try any callable named 'run' or 'run_all'
    for name in ("run","run_all","execute"):
        fn = getattr(analysis_mod, name, None)
        if callable(fn):
            fn()
            return
    raise RuntimeError("No analysis entrypoint found in etl_analysis.py")


def run_full_pipeline():
    print("=== RUN: full pipeline ===")
    # 1) Extract
    print("1) Extracting...")
    extracted = _call_extract()
    # normalized list of raw paths (function may return list/dict/etc.)
    raw_paths = []
    if isinstance(extracted, dict):
        # look for raw_path values
        for v in extracted.values():
            if isinstance(v, dict) and "raw_path" in v:
                raw_paths.append(v["raw_path"])
            elif isinstance(v, str):
                raw_paths.append(v)
    elif isinstance(extracted, (list, tuple)):
        for item in extracted:
            if isinstance(item, str):
                raw_paths.append(item)
            elif isinstance(item, dict) and "raw_path" in item:
                raw_paths.append(item["raw_path"])
    elif isinstance(extracted, str):
        raw_paths = [extracted]

    print(" -> extracted files:", raw_paths)
    time.sleep(1)

    # 2) Transform
    print("2) Transforming...")
    staged = _call_transform(raw_paths)
    if isinstance(staged, (list, tuple)):
        staged = staged[0] if staged else DEFAULT_STAGED
    staged = Path(staged) if staged else DEFAULT_STAGED
    print(" -> staged csv:", staged)
    time.sleep(1)

    # 3) Load
    print("3) Loading to Supabase...")
    try:
        _call_load(staged)
    except Exception as e:
        print("Load failed:", e)
    time.sleep(1)

    # 4) Analysis
    print("4) Running analysis...")
    try:
        _call_analysis()
    except Exception as e:
        print("Analysis failed:", e)

    print("=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    run_full_pipeline()
