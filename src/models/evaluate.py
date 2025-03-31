import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

torch.serialization.add_safe_globals([])  # Allow safe loading if needed

# Load test data
TEST_DATA_PATH = r"data\processed\processed_data.csv"  # Ensure you have test data
USE_ML_MODEL = True
USE_DL_MODEL = True

MODEL_PATHS = {
    "ml": r"saved_models\ml_model.pkl",
    "dl": r"saved_models\dl_model.pth"
}

def load_models():
    ml_model, dl_model = None, None

    if USE_ML_MODEL:
        with open(MODEL_PATHS["ml"], "rb") as file:
            ml_model = pickle.load(file)
        print("ML model loaded.")

    if USE_DL_MODEL:
        dl_model = torch.load(MODEL_PATHS["dl"], weights_only=False)
        dl_model.eval()
        print("DL model loaded.")

    return ml_model, dl_model

def evaluate_models():
    # Load test data
    if not TEST_DATA_PATH:
        raise FileNotFoundError("Test dataset not found!")

    test_data = pd.read_csv(TEST_DATA_PATH)
    X_test, y_test = test_data.drop(columns=['S100B']), test_data['S100B']

    ml_model, dl_model = load_models()

    # Evaluate ML model
    if USE_ML_MODEL and ml_model:
        y_pred_ml = ml_model.predict(X_test)
        
        mae_ml = mean_absolute_error(y_test, y_pred_ml)
        mse_ml = mean_squared_error(y_test, y_pred_ml)
        r2_ml = r2_score(y_test, y_pred_ml)
        
        print(f"ML Model - MAE: {mae_ml:.4f}, MSE: {mse_ml:.4f}, R2 Score: {r2_ml:.4f}")

    # Evaluate DL model
    if USE_DL_MODEL and dl_model:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).to(device)

        with torch.no_grad():
            y_pred_dl = dl_model(X_tensor).cpu().numpy()
        
        mae_dl = mean_absolute_error(y_test, y_pred_dl)
        mse_dl = mean_squared_error(y_test, y_pred_dl)
        r2_dl = r2_score(y_test, y_pred_dl)
        
        print(f"DL Model - MAE: {mae_dl:.4f}, MSE: {mse_dl:.4f}, R2 Score: {r2_dl:.4f}")

if __name__ == "__main__":
    evaluate_models()