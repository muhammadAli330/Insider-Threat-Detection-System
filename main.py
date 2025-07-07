from utils.preprocess import preprocess_data
from models.autoencoder import train_autoencoder
from models.isolation_forest import train_isolation_forest
from models.one_class_svm import train_one_class_svm

def run_system(file_path, model_type):
    X_scaled, y_true, original_df, _ = preprocess_data(file_path)

    if model_type == "isolation_forest":
        preds = train_isolation_forest(X_scaled)
    elif model_type == "one_class_svm":
        preds = train_one_class_svm(X_scaled)
    elif model_type == "autoencoder":
        preds = train_autoencoder(X_scaled)
    else:
        raise ValueError("Invalid model type.")

    original_df['predicted_anomaly'] = preds
    return original_df
