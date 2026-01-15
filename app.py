from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and encoder
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

@app.route("/")
def home():
    return "Fertilizer Recommendation System API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    crop_encoded = encoder.transform([data["crop"]])[0]

    input_data = np.array([[ 
        data["N"], 
        data["P"], 
        data["K"], 
        data["temperature"], 
        data["humidity"], 
        data["ph"], 
        crop_encoded
    ]])

    prediction = model.predict(input_data)[0]

    return jsonify({
        "Recommended Fertilizer": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)
