from src.models.ml_models import train_ml_model
from src.models.dl_models import train_dl_model
import pandas as pd

# Define training settings
USE_ML_MODEL = True
USE_DL_MODEL = False
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"

EPOCHS = 20
BATCH_SIZE = 32

def train():
    data = pd.read_csv(PROCESSED_DATA_PATH)

    X, y = data.drop(columns=['target']), data['target']

    if USE_ML_MODEL:
        ml_model = train_ml_model(X, y, model_type="random_forest")
        print("Trained ML model!")

    if USE_DL_MODEL:
        dl_model = train_dl_model(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
        print("Trained DL model!")

if __name__ == "__main__":
    train()
