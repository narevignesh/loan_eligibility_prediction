from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)


logistic_model = pickle.load(open("logistic_model.pkl", "rb"))
random_forest_model = pickle.load(open("random_forest_model.pkl", "rb"))
gradient_boosting_model = pickle.load(open("gradient_boosting_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)
    

    df = pd.get_dummies(df)

 
    missing_cols = set(feature_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0


    df = df[feature_columns]


    scaled_data = scaler.transform(df)

    # Model predictions
    random_forest_pred = random_forest_model.predict(scaled_data)

    # Prepare the response
    response = {
        "LOAN STATUS": label_encoder.inverse_transform(random_forest_pred)[0],
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
