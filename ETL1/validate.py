'''#!/usr/bin/env python3
"""
validate.py

Validation checks after loading Telco dataset to Supabase.

Checks:
 - No missing values in: tenure, MonthlyCharges, TotalCharges (in staged CSV)
 - Unique count of rows == original dataset (based on customer_id)
 - Row count in Supabase table == staged CSV row count
 - All segments exist: tenure_group, monthly_charge_segment (derive monthly_charge_segment if missing)
 - Contract codes are only {0,1,2} (accepts textual contract labels and maps them)

Prints a validation summary and exits with code 0 on success, 1 on failure.
"""
import os
import sys
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# Config - adjust if your paths differ
RAW_CSV = os.path.join("data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
STAGED_CSV = os.path.join("data", "staged", "telco_transformed.csv")
SUPABASE_TABLE = "telco_data"

# Supabase client helper
def get_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env")
    return create_client(url, key)

# Load CSV safely
def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)

# Fetch all rows from Supabase table
def fetch_supabase_table(table_name):
    sb = get_supabase()
    resp = sb.table(table_name).select("*").execute()
    data = None
    # handle response depending on supabase-py shape
    if hasattr(resp, "data"):
        data = resp.data
    elif isinstance(resp, dict):
        data = resp.get("data")
    else:
        # last resort: try attribute 'get' (older clients)
        try:
            data = resp.get("data")
        except Exception:
            data = None
    if data is None:
        raise RuntimeError("Could not fetch data from Supabase (response missing 'data').")
    return pd.DataFrame(data)

# Derive monthly_charge_segment if missing: create quantile-based 3 segments (low/medium/high)
def ensure_monthly_segment(df):
    col = "monthly_charge_segment"
    if col in df.columns and df[col].notna().any():
        return df
    if "monthly_charges" not in df.columns:
        # try alternate names
        for alt in ["MonthlyCharges", "monthlyCharges"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "monthly_charges"})
                break
    if "monthly_charges" not in df.columns:
        # cannot create
        return df
    # Create 3 quantile buckets
    try:
        df[col] = pd.qcut(df["monthly_charges"].astype(float), q=3, labels=["low", "medium", "high"])
    except Exception:
        # fallback to simple cut
        df[col] = pd.cut(df["monthly_charges"].astype(float), bins=3, labels=["low", "medium", "high"])
    return df

# Map textual contract labels to codes if necessary
def normalize_contract_codes(df):
    # Accepted codes are 0,1,2. If contract uses strings, map common ones.
    if "contract" not in df.columns:
        # try variants
        for alt in ["Contract", "contract_code", "contract_type"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "contract"})
                break
    if "contract" not in df.columns:
        # nothing to check
        return df

    # If numeric-like, coerce to ints
    try:
        df["contract_code_normalized"] = pd.to_numeric(df["contract"], errors="coerce").astype("Int64")
    except Exception:
        df["contract_code_normalized"] = pd.to_numeric(df["contract"], errors="coerce").astype("Int64")

    # If many NaNs, maybe contract is textual; map common labels
    if df["contract_code_normalized"].isna().sum() > len(df) * 0.1:
        mapping = {
            "Month-to-month": 0,
            "Month to month": 0,
            "month-to-month": 0,
            "Month To Month": 0,
            "One year": 1,
            "One Year": 1,
            "one year": 1,
            "Two year": 2,
            "Two Year": 2,
            "two year": 2,
            "Month-to-month ": 0,
            "Month to month ": 0
        }
        df["contract_code_normalized"] = df["contract"].map(mapping).astype("Int64")
    return df

# Run validations
def run_validations():
    errors = []
    warnings = []

    # Load staged CSV
    try:
        staged = load_csv(STAGED_CSV)
    except Exception as e:
        print(f"❌ Failed to load staged CSV: {e}")
        sys.exit(1)

    # Load raw CSV (original)
    try:
        raw = load_csv(RAW_CSV)
    except Exception as e:
        print(f"⚠️ Could not load raw CSV: {e}")
        raw = None

    # 1) No missing in tenure, MonthlyCharges, TotalCharges (in staged)
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        # try lower variants
        if col not in staged.columns:
            alt = col.lower()
            if alt in staged.columns:
                col_use = alt
            else:
                # not present — treat as error
                errors.append(f"Missing expected column in staged CSV: {col}")
                continue
        else:
            col_use = col
        null_count = staged[col_use].isnull().sum()
        if null_count > 0:
            errors.append(f"Column '{col_use}' has {null_count} missing values in staged CSV")
        else:
            print(f"OK: '{col_use}' has no missing values")

    # 2) Unique count of rows == original dataset (based on customer_id if available, else full-row unique)
    if raw is not None:
        # find unique key
        if "customer_id" in staged.columns and "customerID" in raw.columns:
            staged_unique = staged["customer_id"].nunique()
            raw_unique = raw["customerID"].nunique()
            if staged_unique != raw_unique:
                errors.append(f"Unique count mismatch: staged customer_id unique={staged_unique} vs raw customerID unique={raw_unique}")
            else:
                print(f"OK: Unique customer count matches ({staged_unique})")
        elif "customer_id" in staged.columns and "customer_id" in raw.columns:
            staged_unique = staged["customer_id"].nunique()
            raw_unique = raw["customer_id"].nunique()
            if staged_unique != raw_unique:
                errors.append(f"Unique count mismatch: staged customer_id unique={staged_unique} vs raw customer_id unique={raw_unique}")
            else:
                print(f"OK: Unique customer count matches ({staged_unique})")
        else:
            # fallback: compare number of rows
            if len(staged) != len(raw):
                errors.append(f"Row count mismatch between staged ({len(staged)}) and raw ({len(raw)})")
            else:
                print(f"OK: Row counts match ({len(staged)})")

    # 3) Row count matches Supabase table
    try:
        supadb = fetch_supabase_table(SUPABASE_TABLE)
        supa_count = len(supadb)
        staged_count = len(staged)
        if supa_count != staged_count:
            errors.append(f"Supabase row count {supa_count} != staged CSV row count {staged_count}")
        else:
            print(f"OK: Supabase row count matches staged ({supa_count})")
    except Exception as e:
        errors.append(f"Failed to fetch table '{SUPABASE_TABLE}' from Supabase: {e}")
        supadb = pd.DataFrame()

    # 4) All segments exist: tenure_group, monthly_charge_segment
    # tenure_group exists in staged
    if "tenure_group" not in staged.columns or staged["tenure_group"].isnull().all():
        errors.append("Missing or empty 'tenure_group' in staged CSV")
    else:
        print(f"OK: 'tenure_group' present with {staged['tenure_group'].nunique()} values")

    staged = ensure_monthly_segment(staged)
    if "monthly_charge_segment" not in staged.columns or staged["monthly_charge_segment"].isnull().all():
        errors.append("Missing or empty 'monthly_charge_segment' in staged CSV (couldn't derive)")
    else:
        print(f"OK: 'monthly_charge_segment' present with values: {staged['monthly_charge_segment'].unique()}")

    # 5) Contract codes are only {0,1,2}
    # Check in staged first, then in supabase if available
    def check_contracts_frame(df, label):
        if "contract" not in df.columns and "contract_code_normalized" not in df.columns:
            return (False, f"No 'contract' column in {label}")
        df2 = df.copy()
        df2 = normalize_contract_codes(df2)
        vals = df2["contract_code_normalized"].dropna().unique().tolist()
        vals_int = [int(v) for v in vals if v is not None and str(v).strip() != ""]
        invalid = [v for v in vals_int if v not in (0, 1, 2)]
        if invalid:
            return (False, f"{label}: found invalid contract codes {invalid} (allowed: 0,1,2)")
        return (True, f"{label}: contract codes OK (found: {sorted(vals_int)})")

    # staged check
    ok_staged, msg_staged = check_contracts_frame(staged, "staged CSV")
    if ok_staged:
        print("OK:", msg_staged)
    else:
        errors.append(msg_staged)

    # supabase check (if we fetched it)
    if not supadb.empty:
        ok_supa, msg_supa = check_contracts_frame(supadb, "Supabase")
        if ok_supa:
            print("OK:", msg_supa)
        else:
            errors.append(msg_supa)

    # Final summary
    print("\n" + "=" * 60)
    if errors:
        print("VALIDATION FAILED with the following issues:")
        for e in errors:
            print(" -", e)
        print("\nPlease fix the issues and re-run the ETL/load.")
        sys.exit(1)
    else:
        print("ALL VALIDATIONS PASSED ✅")
        sys.exit(0)

if __name__ == "__main__":
    try:
        run_validations()
    except Exception as ex:
        print("Unexpected error during validation:", ex)
        sys.exit(2)'''
#!/usr/bin/env python3
"""
validate.py (updated)

Validation checks after loading Telco dataset to Supabase.

Checks:
 - No missing values in: tenure, MonthlyCharges, TotalCharges (in staged CSV)
 - Unique count of rows == original dataset (based on customer_id)
 - Row count in Supabase table == staged CSV row count (uses exact count)
 - All segments exist: tenure_group, monthly_charge_segment (derive monthly_charge_segment if missing)
 - Contract codes are only {0,1,2} (accepts textual contract labels and maps them)

Prints a validation summary and exits with code 0 on success, 1 on failure.
"""
import os
import sys
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# Config - adjust if your paths differ
RAW_CSV = os.path.join("data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
STAGED_CSV = os.path.join("data", "staged", "telco_transformed.csv")
SUPABASE_TABLE = "telco_data"

# Supabase client helper
def get_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env")
    return create_client(url, key)

# Load CSV safely
def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)

# Fetch a sample and the exact table count from Supabase
def fetch_supabase_table_sample_and_count(table_name, sample_limit=1000):
    sb = get_supabase()
    # Try to get exact count via count="exact"
    total_count = None
    try:
        resp = sb.table(table_name).select("id", count="exact").limit(1).execute()
        # resp may have .count or dict field 'count'
        if hasattr(resp, "count") and resp.count is not None:
            total_count = int(resp.count)
        elif isinstance(resp, dict) and resp.get("count") is not None:
            total_count = int(resp.get("count"))
    except Exception:
        total_count = None

    # Fetch a sample (useful for validating presence of columns)
    try:
        sample_resp = sb.table(table_name).select("*").limit(sample_limit).execute()
        sample_data = sample_resp.data if hasattr(sample_resp, "data") else sample_resp.get("data")
        sample_df = pd.DataFrame(sample_data)
    except Exception as e:
        raise RuntimeError(f"Could not retrieve sample from Supabase: {e}")

    return sample_df, total_count

# Derive monthly_charge_segment if missing: create quantile-based 3 segments (low/medium/high)
def ensure_monthly_segment(df):
    col = "monthly_charge_segment"
    if col in df.columns and df[col].notna().any():
        return df
    if "monthly_charges" not in df.columns:
        for alt in ["MonthlyCharges", "monthlyCharges"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "monthly_charges"})
                break
    if "monthly_charges" not in df.columns:
        return df
    try:
        df[col] = pd.qcut(df["monthly_charges"].astype(float), q=3, labels=["low", "medium", "high"])
    except Exception:
        df[col] = pd.cut(df["monthly_charges"].astype(float), bins=3, labels=["low", "medium", "high"])
    return df

# Map textual contract labels to numeric codes (0,1,2)
def normalize_contract_codes(df):
    df = df.copy()
    if "contract" not in df.columns:
        # try some variants
        for alt in ["Contract", "contract_type", "contract_code"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "contract"})
                break
    if "contract" not in df.columns:
        df["contract_code_normalized"] = pd.array([pd.NA]*len(df), dtype="Int64")
        return df

    # Try numeric coercion
    df["contract_code_normalized"] = pd.to_numeric(df["contract"], errors="coerce").astype("Int64")

    # If too many NaNs, map textual labels
    if df["contract_code_normalized"].isna().sum() > len(df) * 0.1:
        mapping = {
            "month-to-month": 0, "month to month": 0, "monthtomonth": 0, "month": 0,
            "one year": 1, "one-year": 1, "oneyear": 1,
            "two year": 2, "two-year": 2, "twoyear": 2
        }
        df["contract_code_normalized"] = df["contract"].str.strip().str.lower().map(mapping).astype("Int64")
        # Still try numeric fallback
        mask_na = df["contract_code_normalized"].isna()
        if mask_na.any():
            df.loc[mask_na, "contract_code_normalized"] = pd.to_numeric(df.loc[mask_na, "contract"], errors="coerce").astype("Int64")
    return df

# Run validations
def run_validations():
    errors = []
    warnings = []

    # Load staged CSV
    try:
        staged = load_csv(STAGED_CSV)
    except Exception as e:
        print(f"⚠️ Could not load raw CSV: {e}")
        staged = None

    if staged is None:
        print("Aborting because staged CSV could not be loaded.")
        sys.exit(1)

    # Load raw CSV (original) if possible
    try:
        raw = load_csv(RAW_CSV)
    except Exception as e:
        print(f"⚠️ Could not load raw CSV: {e}")
        raw = None

    # 1) No missing in tenure, MonthlyCharges, TotalCharges (in staged)
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        col_use = col
        if col not in staged.columns and col.lower() in staged.columns:
            col_use = col.lower()
        if col_use not in staged.columns:
            errors.append(f"Missing expected column in staged CSV: {col} (tried '{col_use}')")
            continue
        null_count = staged[col_use].isnull().sum()
        if null_count > 0:
            errors.append(f"Column '{col_use}' has {null_count} missing values in staged CSV")
        else:
            print(f"OK: '{col_use}' has no missing values")

    # 2) Unique count of rows == original dataset (based on customer_id)
    if raw is not None:
        if "customer_id" in staged.columns and ("customerID" in raw.columns or "customer_id" in raw.columns):
            staged_unique = staged["customer_id"].nunique()
            raw_key = "customerID" if "customerID" in raw.columns else "customer_id"
            raw_unique = raw[raw_key].nunique()
            if staged_unique != raw_unique:
                errors.append(f"Unique count mismatch: staged customer_id unique={staged_unique} vs raw {raw_key} unique={raw_unique}")
            else:
                print(f"OK: Unique customer count matches ({staged_unique})")
        else:
            if len(staged) != len(raw):
                errors.append(f"Row count mismatch between staged ({len(staged)}) and raw ({len(raw)})")
            else:
                print(f"OK: Row counts match ({len(staged)})")

    # 3) Row count matches Supabase table (use exact count when possible)
    try:
        sample_df, supa_count = fetch_supabase_table_sample_and_count(SUPABASE_TABLE)
        staged_count = len(staged)
        if supa_count is None:
            warnings.append("Could not retrieve exact Supabase row count; using sample for checks.")
            print(f"INFO: sample rows fetched from Supabase (showing up to {len(sample_df)} rows).")
        else:
            if supa_count != staged_count:
                errors.append(f"Supabase row count {supa_count} != staged CSV row count {staged_count}")
            else:
                print(f"OK: Supabase row count matches staged ({supa_count})")
    except Exception as e:
        errors.append(f"Failed to fetch table '{SUPABASE_TABLE}' from Supabase: {e}")
        sample_df = pd.DataFrame()

    # 4) All segments exist: tenure_group, monthly_charge_segment
    if "tenure_group" not in staged.columns or staged["tenure_group"].isnull().all():
        errors.append("Missing or empty 'tenure_group' in staged CSV")
    else:
        print(f"OK: 'tenure_group' present with {staged['tenure_group'].nunique()} values")

    staged = ensure_monthly_segment(staged)
    if "monthly_charge_segment" not in staged.columns or staged["monthly_charge_segment"].isnull().all():
        errors.append("Missing or empty 'monthly_charge_segment' in staged CSV (couldn't derive)")
    else:
        print(f"OK: 'monthly_charge_segment' present with values: {staged['monthly_charge_segment'].unique()}")

    # 5) Contract codes are only {0,1,2}
    ok_staged = False
    try:
        staged_norm = normalize_contract_codes(staged)
        vals = staged_norm["contract_code_normalized"].dropna().unique().tolist()
        vals_int = sorted([int(v) for v in vals if v is not None and str(v).strip() != ""])
        invalid = [v for v in vals_int if v not in (0, 1, 2)]
        if invalid:
            errors.append(f"staged CSV: found invalid contract codes {invalid} (allowed: 0,1,2)")
        else:
            print(f"OK: staged CSV contract codes OK (found: {vals_int})")
            ok_staged = True
    except Exception as e:
        errors.append(f"Error validating contract codes in staged CSV: {e}")

    # also validate sample from supabase for contract presence/validity
    if not sample_df.empty:
        try:
            sample_norm = normalize_contract_codes(sample_df)
            vals = sample_norm["contract_code_normalized"].dropna().unique().tolist()
            vals_int = sorted([int(v) for v in vals if v is not None and str(v).strip() != ""])
            invalid = [v for v in vals_int if v not in (0, 1, 2)]
            if invalid:
                errors.append(f"Supabase sample: found invalid contract codes {invalid} (allowed: 0,1,2)")
            else:
                print(f"OK: Supabase sample contract codes OK (found: {vals_int})")
        except Exception as e:
            warnings.append(f"Could not validate contracts in Supabase sample: {e}")

    # Final summary
    print("\n" + "=" * 60)
    if errors:
        print("VALIDATION FAILED with the following issues:")
        for e in errors:
            print(" -", e)
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(" -", w)
        sys.exit(1)
    else:
        print("ALL VALIDATIONS PASSED ✅")
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(" -", w)
        sys.exit(0)

if __name__ == "__main__":
    try:
        run_validations()
    except Exception as ex:
        print("Unexpected error during validation:", ex)
        sys.exit(2)

