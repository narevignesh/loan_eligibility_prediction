import pickle
import pandas as pd


def load_model_and_helpers():
    try:
        with open("model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        with open("scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        with open("label_encoder.pkl", "rb") as label_encoder_file:
            label_encoder = pickle.load(label_encoder_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file is missing: {e.filename}")
    except EOFError as e:
        raise ValueError(f"File is corrupted: {e}")
    return model, scaler, label_encoder
