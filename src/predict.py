
import joblib
import pandas as pd

from src.feature_extractor import extract_features


MODEL_PATH = "models/best_model.joblib"


def load_model(model_path=MODEL_PATH):
    """Load the trained modulation classification model."""
    artifact = joblib.load(model_path)

    return artifact


def predict_modulation(signal, model_path=MODEL_PATH):
    """
    Predict the modulation type of a complex-valued signal.
    """

    artifact = load_model(model_path)

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    # Extract all available signal features
    features = extract_features(signal)

    # Select features in the exact order used during training
    feature_values = {
        name: features[name]
        for name in feature_columns
    }

    # Convert features into a DataFrame
    input_data = pd.DataFrame([feature_values])

    # Predict modulation
    prediction = model.predict(input_data)[0]

    return prediction
