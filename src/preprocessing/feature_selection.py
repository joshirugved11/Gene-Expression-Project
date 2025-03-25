import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

PROCESSED_DATA_PATH = r"data\processed\processed_data.csv"
SELECTED_FEATURES_PATH = r"data\processed\selected_features.csv"
TARGET_COLUMN = "S100B"
NUM_FEATURES = 10

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    selector = SelectKBest(score_func=f_classif, k=NUM_FEATURES)
    X_selected = selector.fit_transform(X, y)

    selected_features = df.columns[selector.get_support()]
    return pd.DataFrame(X_selected, columns=selected_features)

if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df_selected = select_features(df)
    df_selected.to_csv(SELECTED_FEATURES_PATH, index=False)
    print(f"Feature selection saved to {SELECTED_FEATURES_PATH}")
