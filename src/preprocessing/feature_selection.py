import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold

PROCESSED_DATA_PATH = r"data\raw\snRNA_seq_data.csv"
SELECTED_FEATURES_PATH = r"data\processed\selected_features.csv"
TARGET_COLUMN = "S100B"
NUM_FEATURES = 10

def remove_constant_features(df):
    selector = VarianceThreshold(threshold=0.0)  # Removes constant features
    df_reduced = df.iloc[:, selector.fit(df).get_support()]
    return df_reduced

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[TARGET_COLUMN])  # Exclude target column
    y = df[TARGET_COLUMN]

    selector = SelectKBest(score_func=f_classif, k=min(NUM_FEATURES, X.shape[1]))  # Ensure k is not greater than available features
    X_selected = selector.fit_transform(X, y)

    support_mask = selector.get_support()
    
    if len(support_mask) != X.shape[1]:  # Compare only with feature columns
        raise ValueError(f"Feature selection mask size {len(support_mask)} does not match feature column size {X.shape[1]}")
    
    selected_features = X.columns[support_mask]
    return pd.DataFrame(X_selected, columns=selected_features)

if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")  # Drop unnamed column if present
    df = remove_constant_features(df)  # Remove constant features
    df_selected = select_features(df)
    df_selected[TARGET_COLUMN] = df[TARGET_COLUMN].values  # Add target column back
    df_selected.to_csv(SELECTED_FEATURES_PATH, index=False)
    print(f"Feature selection saved to {SELECTED_FEATURES_PATH}")
