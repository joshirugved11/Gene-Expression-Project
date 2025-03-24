from preprocessing.encode_labels import encode_labels
from preprocessing.feature_selection import select_features
from preprocessing.normalisation import normalize_data
import pandas as pd

# Define file paths
RAW_DATA_PATH = "data/raw/dataset.csv"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
SELECTED_FEATURES_PATH = "data/processed/selected_features.csv"
NORMALIZED_DATA_PATH = "data/processed/normalized_data.csv"

if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv(RAW_DATA_PATH)

    print("Encoding labels...")
    df_encoded = encode_labels(df)
    df_encoded.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Selecting features...")
    df_selected = select_features(df_encoded)
    df_selected.to_csv(SELECTED_FEATURES_PATH, index=False)

    print("Normalizing data...")
    df_normalized = normalize_data(df_selected)
    df_normalized.to_csv(NORMALIZED_DATA_PATH, index=False)

    print(f"Preprocessing completed! Normalized data saved at {NORMALIZED_DATA_PATH}")
