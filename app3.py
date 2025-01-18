from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the models and scalers
svm_model_card = joblib.load('card_model.pkl')  # Cardiovascular model
svm_model_dia = joblib.load('dia_model.pkl')   # Diabetes model
svm_model_asthma = joblib.load('asthma_model.pkl')  # Asthma model

scaler_card = joblib.load('scaler1.pkl')       # Cardiovascular scaler
scaler_dia = joblib.load('scaler.pkl')        # Diabetes scaler
scaler_asthma = joblib.load('scaler2.pkl')     # Asthma scaler

predictions_mapping = {
    0: "Normal",
    1: "Borderline",
    2: "Critical"
}
# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Chronic Disease Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the input JSON
        data = request.json
        disease = data.get('disease')  # Disease type: 'cardiovascular', 'diabetes', 'asthma'
        features = data.get('features')  # Features for prediction

        # Validate input
        if not disease or not features:
            return jsonify({"error": "Missing 'disease' or 'features' in the request."}), 400
        
        # Convert features to numpy array and reshape
        features = np.array(features).reshape(1, -1)

        # Perform prediction based on the disease type
        if disease == 'cardiovascular':
            scaled_features = scaler_card.transform(features)
            prediction = svm_model_card.predict(scaled_features)[0]
        elif disease == 'diabetes':
            scaled_features = scaler_dia.transform(features)
            prediction = svm_model_dia.predict(scaled_features)[0]
        elif disease == 'asthma':
            scaled_features = scaler_asthma.transform(features)
            prediction = svm_model_asthma.predict(scaled_features)[0]
        else:
            return jsonify({"error": "Invalid 'disease' type. Use 'cardiovascular', 'diabetes', or 'asthma'."}), 400
        # Map prediction to label
        prediction_label = predictions_mapping.get(prediction, "Unknown")
        # Return the prediction result
        return jsonify({
            "prediction": prediction_label  # Convert prediction to int for JSON serialization
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)


