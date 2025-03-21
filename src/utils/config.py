class Config:
    # Dataset Paths
    DATA_PATHS = {
        "gene_expression": "data/gene_expression.csv",
        "drug_response": "data/drug_response.csv"
    }
    
    # Model Paths
    MODEL_PATHS = {
        "ml": "models/ml_model.pkl",   # For Machine Learning models
        "dl": "models/dl_model.h5"     # For Deep Learning models
    }

    # Training Parameters
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    # Feature Selection
    TOP_K_FEATURES = 20

    # Model Type Selection
    USE_ML_MODEL = True
    USE_DL_MODEL = True  # You can set this to False if not using DL models

config = Config()