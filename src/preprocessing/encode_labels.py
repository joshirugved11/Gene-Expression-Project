import pandas as pd
from sklearn.preprocessing import LabelEncoder

LABEL_COLUMNS = ["PLEKHN1", "TTLL10", "HES4"]
PROCESSED_DATA_PATH = r"data\raw\snRNA_seq_data.csv"

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
