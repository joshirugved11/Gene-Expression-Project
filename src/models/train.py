from calendar import EPOCH
import pickle
import torch
import pandas as pd
from main import PROCESSED_DATA_PATH
from models.evaluate import USE_ML_MODEL, USE_DL_MODEL
from models.ml_models import train_ml_model, train_dl_model

MODEL_DIR = "models/"

def train():
    data = pd.read_csv(PROCESSED_DATA_PATH)
    X, y = data.drop(columns=['S100B']), data['S100B']

    if USE_ML_MODEL:
        ml_model = train_ml_model(X, y, model_type="random_forest")
        with open(MODEL_DIR + "ml_model.pkl", "wb") as file:
            pickle.dump(ml_model, file)
        print("Trained and saved ML model!")

    if USE_DL_MODEL:
        dl_model = train_dl_model(X, y, epochs=EPOCH, batch_size=BATCH_SIZE) # type: ignore
        torch.save(dl_model, MODEL_DIR + "dl_model.h5")
        print("Trained and saved DL model!")

if __name__ == "__main__":
    train()
