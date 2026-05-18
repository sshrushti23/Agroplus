from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Models load karein
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    print(f"Error loading models: {e}")

# Crop Mapping Dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Chickpea", 4: "Kidneybeans", 5: "Pigeonpeas",
    6: "Mothbeans", 7: "Mungbean", 8: "Blackgram", 9: "Lentil", 10: "Pomegranate",
    11: "Banana", 12: "Mango", 13: "Grapes", 14: "Watermelon", 15: "Muskmelon",
    16: "Apple", 17: "Orange", 18: "Papaya", 19: "Coconut", 20: "Cotton",
    21: "Jute", 22: "Coffee"
}

@app.route('/')
def home():
    return "AgroPlus AI Server is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # 7 Features extract karein
        values = [[
            float(data['N']), 
            float(data['P']), 
            float(data['K']),
            float(data['temperature']), 
            float(data['humidity']), 
            float(data['ph']),
            float(data['rainfall'])
        ]]

        # Scaling aur Prediction
        values_scaled = scaler.transform(values)
        prediction_id = model.predict(values_scaled)[0]

        # Convert ID to Name (Numpy type handle karne ke liye .item() use kiya)
        crop_name = crop_dict.get(int(prediction_id), "Unknown Crop")

        return jsonify({'crop': crop_name})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)