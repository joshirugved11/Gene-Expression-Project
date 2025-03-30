import pickle
import torch

USE_ML_MODEL = True
USE_DL_MODEL = True

MODEL_PATHS = {
    "ml": "saved_models/ml_model.pkl",
    "dl": "saved_models/dl_model.h5"
}

def load_models():
    ml_model, dl_model = None, None

    if USE_ML_MODEL:
        with open(MODEL_PATHS["ml"], "rb") as file:
            ml_model = pickle.load(file)
        print("ML model loaded.")

    if USE_DL_MODEL:
        dl_model = torch.load(MODEL_PATHS["dl"])
        dl_model.eval()
        print("DL model loaded.")

    return ml_model, dl_model
