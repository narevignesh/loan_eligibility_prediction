from flask import Flask, request, jsonify
from model import load_model_and_helpers

app = Flask(__name__)

# Load model and helpers
model, scaler, label_encoder = load_model_and_helpers()

@app.route("/")
def home():
    return jsonify({"message": "Loan Eligibility Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not isinstance(data, list):
            return jsonify({"error": "Input should be a list of records"}), 400
        
        # Transform input data
        input_df = pd.DataFrame(data)
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_scaled = scaler.transform(input_encoded)
        
        # Ensure all columns match training data
        prediction = model.predict(input_scaled)
        prediction_decoded = label_encoder.inverse_transform(prediction)

        # Respond with predictions
        return jsonify({"predictions": prediction_decoded.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
