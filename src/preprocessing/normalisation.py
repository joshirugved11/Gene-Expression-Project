import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SELECTED_FEATURES_PATH = r"data\processed\selected_features.csv"
NORMALIZED_DATA_PATH = r"data\processed\normalized_data.csv"
NORMALIZATION_METHOD = "minmax"

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler() if NORMALIZATION_METHOD == "minmax" else StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

if __name__ == "__main__":
    df = pd.read_csv(SELECTED_FEATURES_PATH)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")  # Add this line
    df_normalized = normalize_data(df)
    df_normalized.to_csv(NORMALIZED_DATA_PATH, index=False)
    print(f"Normalized data saved to {NORMALIZED_DATA_PATH}")
