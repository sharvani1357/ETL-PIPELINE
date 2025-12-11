import os
import re
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

# -------------------------
def get_supabase_client():
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("❌ Missing SUPABASE_URL or SUPABASE_KEY in .env")
    return create_client(url, key)

# -------------------------
def to_snake_case(name: str) -> str:
    """Convert column name to snake_case and lower-case; remove non-alphanum underscores."""
    if not isinstance(name, str):
        return name
    s = name.strip()
    # common replacements
    s = s.replace(" ", "_").replace("-", "_")
    # convert camelCase / PascalCase to snake_case
    s = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', s)
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    s = re.sub(r'__+', '_', s)  # collapse multiple underscores
    s = re.sub(r'[^0-9a-zA-Z_]', '', s)  # remove other chars
    return s.lower()

# -------------------------
def create_table_if_not_exists():
    """Try to create table via RPC if available; not required if you create table manually."""
    try:
        supabase = get_supabase_client()
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS public.telco_data (
            id BIGSERIAL PRIMARY KEY,
            customer_id TEXT,
            gender TEXT,
            senior_citizen INTEGER,
            partner TEXT,
            dependents TEXT,
            tenure INTEGER,
            phone_service TEXT,
            multiple_lines TEXT,
            internet_service TEXT,
            online_security TEXT,
            online_backup TEXT,
            device_protection TEXT,
            tech_support TEXT,
            streaming_tv TEXT,
            streaming_movies TEXT,
            contract TEXT,
            paperless_billing TEXT,
            payment_method TEXT,
            monthly_charges DOUBLE PRECISION,
            total_charges DOUBLE PRECISION,
            churn INTEGER,
            is_senior INTEGER,
            tenure_group TEXT,
            services_count INTEGER,
            has_internet INTEGER,
            monthly_total_ratio DOUBLE PRECISION
        );
        """
        try:
            supabase.rpc('execute_sql', {'query': create_table_sql}).execute()
            print("✅ Table 'telco_data' created or already exists (via RPC).")
        except Exception as e:
            # Not fatal: the RPC often isn't present (as your logs showed)
            print(f"ℹ️  RPC create-table note: {e}")
            print("ℹ️  If the table doesn't exist, create it manually in Supabase's SQL editor (see instructions).")
    except Exception as e:
        print("⚠️  Could not attempt RPC table creation:", e)

# -------------------------
def normalize_and_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase/convert columns to snake_case and apply a few manual mappings."""
    df = df.copy()

    # First, rename columns using automatic snake_case
    new_cols = {c: to_snake_case(c) for c in df.columns}
    df.rename(columns=new_cols, inplace=True)

    # Manual mapping for common names (if different)
    manual_map = {
        'customerid': 'customer_id',
        'seniorcitizen': 'senior_citizen',
        'monthlycharges': 'monthly_charges',
        'totalcharges': 'total_charges',
        'phoneservice': 'phone_service',
        'multipllines': 'multiple_lines',  # handle possible typos
        'streamingtv': 'streaming_tv',
        'streamingmovies': 'streaming_movies',
        'paymentmethod': 'payment_method',
        'paperlessbilling': 'paperless_billing',
        'internetservice': 'internet_service',
        'onlinesecurity': 'online_security',
        'onlinebackup': 'online_backup',
        'deviceprotection': 'device_protection',
        'techsupport': 'tech_support',
        'tenure_group': 'tenure_group'  # keep as-is
    }

    # apply manual map if keys present
    to_rename = {k: v for k, v in manual_map.items() if k in df.columns and df.columns.tolist().count(k) > 0}
    if to_rename:
        df.rename(columns=to_rename, inplace=True)

    # Ensure `churn` column exists (could be labelled 'Churn' originally)
    if 'churn' not in df.columns:
        # try other common variants
        for alt in ['churned', 'is_churn', 'churn_flag', 'Churn']:
            if alt in df.columns:
                df.rename(columns={alt: 'churn'}, inplace=True)
                break

    # Ensure churn is numeric 0/1
    if 'churn' in df.columns:
        df['churn'] = df['churn'].map({'Yes': 1, 'No': 0}).fillna(df['churn'])  # map Yes/No -> 1/0 when present
        try:
            df['churn'] = pd.to_numeric(df['churn'], errors='coerce').fillna(0).astype(int)
        except Exception:
            pass

    return df

# -------------------------
def load_to_supabase(staged_path: str, table_name: str = "telco_data", batch_size: int = 50):
    # make absolute path relative to script if needed
    if not os.path.isabs(staged_path):
        staged_path = os.path.abspath(os.path.join(os.path.dirname(__file__), staged_path))

    print(f"🔍 Looking for data file at: {staged_path}")
    if not os.path.exists(staged_path):
        print(f"❌ File not found at {staged_path}. Run transform.py first.")
        return

    df = pd.read_csv(staged_path)
    if df.empty:
        print("⚠️  Staged CSV is empty. Aborting.")
        return

    # Normalize columns to snake_case and map common names
    df = normalize_and_map_columns(df)

    # Optional: drop any columns you'd rather not insert (example: drop unnamed index columns)
    drop_candidates = [c for c in df.columns if c.lower().startswith('unnamed') or c == 'index']
    if drop_candidates:
        df.drop(columns=drop_candidates, inplace=True)

    print("ℹ️  Columns that will be inserted:", df.columns.tolist())

    # Initialize supabase client
    supabase = get_supabase_client()

    total_rows = len(df)
    print(f"📊 Loading {total_rows} rows into '{table_name}' (batch size {batch_size})...")

    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i + batch_size].copy()
        batch = batch.where(pd.notnull(batch), None)  # NaN -> None
        records = batch.to_dict('records')
        try:
            resp = supabase.table(table_name).insert(records).execute()
            # Supabase-py responses can vary. Check for common error structure:
            if isinstance(resp, dict) and resp.get('error'):
                print(f"⚠️  Error inserting batch {i//batch_size + 1}: {resp.get('error')}")
            else:
                end = min(i + batch_size, total_rows)
                print(f"✅ Inserted rows {i+1}-{end} of {total_rows}")
        except Exception as e:
            print(f"⚠️  Exception while inserting batch {i//batch_size + 1}: {e}")
            # If the error mentions missing column(s), print them for debugging
            if isinstance(e, dict) and e.get('message'):
                print("Supabase message:", e.get('message'))
            continue

    print(f"🎯 Finished loading data into '{table_name}'.")

# -------------------------
if __name__ == "__main__":
    staged_csv_path = os.path.join("..", "data", "staged", "telco_transformed.csv")
    create_table_if_not_exists()  # attempt RPC creation (non-fatal)
    load_to_supabase(staged_csv_path)
