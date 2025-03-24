from sklearn.ensemble import RandomForestClassifier

# Define feature selection parameter
TOP_K_FEATURES = 20

def train_ml_model(X, y, model_type='random_forest'):
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=TOP_K_FEATURES, random_state=42)
    else:
        raise ValueError("Unsupported model type")
    
    model.fit(X, y)
    return model
