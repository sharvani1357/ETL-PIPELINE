import os
import pandas as pd

def transform_data(raw_path):
    # Ensure the path is relative to project root (same pattern as your Titanic code)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up one level
    staged_dir = os.path.join(base_dir, "data", "staged")
    os.makedirs(staged_dir, exist_ok=True)

    df = pd.read_csv(raw_path)

    # --- 1️⃣ Normalize whitespace & basic types ---
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # --- 2️⃣ Handle TotalCharges which may be blank for some rows ---
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # --- 3️⃣ Fill missing numeric fields ---
    # tenure, MonthlyCharges, TotalCharges may contain NaNs
    if 'tenure' in df.columns:
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0).astype(int)
    if 'MonthlyCharges' in df.columns:
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(df['MonthlyCharges'].median())
    if 'TotalCharges' in df.columns:
        # where TotalCharges is NaN, approximate as MonthlyCharges * tenure
        mask = df['TotalCharges'].isna()
        df.loc[mask, 'TotalCharges'] = (df.loc[mask, 'MonthlyCharges'] * df.loc[mask, 'tenure']).fillna(0)
        # final fallback
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # --- 4️⃣ Standardize certain service text values ---
    # Replace "No internet service" / "No phone service" with "No" for simple binary columns
    replace_map = {
        'No internet service': 'No',
        'No phone service': 'No'
    }
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isin(replace_map.keys()).any():
            df[col] = df[col].replace(replace_map)

    # --- 5️⃣ Feature engineering ---
    # is_senior: make sure SeniorCitizen is 0/1 int (some datasets already have it numeric)
    if 'SeniorCitizen' in df.columns:
        try:
            df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
        except Exception:
            # try mapping 'Yes'/'No' if present
            df['SeniorCitizen'] = df['SeniorCitizen'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    df['is_senior'] = df['SeniorCitizen'].apply(lambda x: 1 if x == 1 else 0) if 'SeniorCitizen' in df.columns else 0

    # tenure_group: simple buckets
    if 'tenure' in df.columns:
        def tenure_bucket(t):
            if t <= 12:
                return '0-12'
            if t <= 24:
                return '13-24'
            if t <= 48:
                return '25-48'
            if t <= 60:
                return '49-60'
            return '61+'
        df['tenure_group'] = df['tenure'].apply(tenure_bucket)

    # services_count: count number of 'Yes' across select service columns
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    present_service_cols = [c for c in service_cols if c in df.columns]
    if present_service_cols:
        # treat 'Yes' as 1, anything else as 0
        df['services_count'] = df[present_service_cols].apply(lambda row: sum(1 for v in row if str(v).strip().lower() == 'yes'), axis=1)

    # has_internet boolean
    if 'InternetService' in df.columns:
        df['has_internet'] = df['InternetService'].apply(lambda x: 0 if str(x).strip().lower() in ['no', 'none', 'nan'] else 1)

    # monthly_to_total_ratio (guard divide by zero)
    if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
        df['monthly_total_ratio'] = df.apply(lambda r: r['MonthlyCharges'] / r['TotalCharges'] if r['TotalCharges'] and r['TotalCharges'] != 0 else 0, axis=1)

    # Convert Churn to binary 1/0 (if present)
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # --- 6️⃣ Drop columns not needed (customerID and any exact duplicates of engineered fields) ---
    drop_cols = ['customerID'] if 'customerID' in df.columns else []
    # drop any implicit redundant columns used only for raw tracking
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # --- 7️⃣ Optional: reorder columns to have target at the end ---
    if 'Churn' in df.columns:
        cols = [c for c in df.columns if c != 'Churn'] + ['Churn']
        df = df[cols]

    # --- 8️⃣ Save transformed data ---
    staged_path = os.path.join(staged_dir, "telco_transformed.csv")
    df.to_csv(staged_path, index=False)
    print(f"✅ Data transformed and saved at: {staged_path}")
    return staged_path


if __name__ == "__main__":
    # run transform after extract if used as standalone
    from extract import extract_data
    raw_path = extract_data()
    transform_data(raw_path)
#!/usr/bin/env python3
