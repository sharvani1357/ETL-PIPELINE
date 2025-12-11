import os
import pandas as pd
def extract_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(data_dir, exist_ok=True)
    telco_source = os.path.join(base_dir, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if not os.path.exists(telco_source):
        raise FileNotFoundError(f"❌ Telco dataset not found at: {telco_source}\n"
                                f"➡ Please place the CSV inside your project folder.")
    df = pd.read_csv(telco_source)
    raw_path = os.path.join(data_dir, "telco_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"✅ Data extracted and saved at: {raw_path}")
    return raw_path

if __name__ == "__main__":
    extract_data()
