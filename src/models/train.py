import pickle
import torch
import random
import numpy as np
import pandas as pd
import os
import sys

# Ensure proper module resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import ML and DL training functions
from models.evaluate import USE_ML_MODEL, USE_DL_MODEL
from models.ml_models import train_ml_model
from models.dl_models import train_dl_model  

# Define constants
SEED = 42  # Fixed seed for reproducibility
EPOCHS = 20  # Define EPOCHS explicitly
BATCH_SIZE = 32  # Define batch size explicitly
MODEL_DIR = "saved_models/"  # Directory to save models
DATA_PATH = r"data\processed\processed_data.csv"  # Directly specifying the CSV file path

# Ensure consistency in randomness
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For multi-GPU setups
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Ensures consistency

def train():
    # Ensure the processed data file exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed data file not found: {DATA_PATH}")

    # Load processed data
    data = pd.read_csv(DATA_PATH)
    X, y = data.drop(columns=['S100B']), data['S100B']

    # Ensure the model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Train ML model
    if USE_ML_MODEL:
        ml_model = train_ml_model(X, y, model_type="random_forest")
        ml_model_path = os.path.join(MODEL_DIR, "ml_model.pkl")
        with open(ml_model_path, "wb") as file:
            pickle.dump(ml_model, file)
        print(f"Trained and saved ML model at {ml_model_path}")

    # Train DL model
    if USE_DL_MODEL:
        dl_model = train_dl_model(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
        dl_model_path = os.path.join(MODEL_DIR, "dl_model.pth")
        torch.save(dl_model, dl_model_path)
        print(f"Trained and saved DL model at {dl_model_path}")

if __name__ == "__main__":
    train()
