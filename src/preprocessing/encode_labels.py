import pandas as pd
from sklearn.preprocessing import LabelEncoder

LABEL_COLUMNS = ["column1", "column2", "column3"]
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"

def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    encoder = LabelEncoder()
    for column in LABEL_COLUMNS:
        if column in df.columns:
            df[column] = encoder.fit_transform(df[column])
    return df

if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df_encoded = encode_labels(df)
    df_encoded.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Encoded labels saved to {PROCESSED_DATA_PATH}")
