from src.models.ml_models import train_ml_model
import numpy as np

# Define the required constants
TOP_K_FEATURES = 20  # Number of features for the model

def test_train_ml_model():
    X = np.random.rand(100, TOP_K_FEATURES)
    y = np.random.randint(0, 2, 100)

    model = train_ml_model(X, y, model_type="random_forest")
    assert model is not None, "Model training failed"
