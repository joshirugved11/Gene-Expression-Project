import pickle
import torch
import random
import numpy as np
import pandas as pd
from main import PROCESSED_DATA_PATH
from models.evaluate import USE_ML_MODEL, USE_DL_MODEL
from models.ml_models import train_ml_model
from models.dl_models import train_dl_model  # Fixed incorrect import

# Define constants
SEED = 42  # Fixed seed for reproducibility
EPOCHS = 20  # Define EPOCHS explicitly
BATCH_SIZE = 32  # Define batch size explicitly
MODEL_DIR = "saved_models/"

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For multi-GPU setups
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Ensures consistency

def train():
    data = pd.read_csv(PROCESSED_DATA_PATH)
    X, y = data.drop(columns=['S100B']), data['S100B']

    if USE_ML_MODEL:
        ml_model = train_ml_model(X, y, model_type="random_forest", random_state=SEED)  # Set random_state
        with open(MODEL_DIR + "ml_model.pkl", "wb") as file:
            pickle.dump(ml_model, file)
        print("Trained and saved ML model!")

    if USE_DL_MODEL:
        dl_model = train_dl_model(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)  # Fixed variable names
        torch.save(dl_model, MODEL_DIR + "dl_model.h5")
        print("Trained and saved DL model!")

if __name__ == "__main__":
    train()
