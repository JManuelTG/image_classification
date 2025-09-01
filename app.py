# app.py
from flask import Flask, request, jsonify
from model import load_model, preprocess_image, extract_features, predict_class
import json

app = Flask(__name__)
model, feature_extractor, device, classes = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image_tensor = preprocess_image(file, device)

    predicted_class = predict_class(model, image_tensor, classes)
    return jsonify({'prediction': predicted_class})


@app.route('/extract_features', methods=['POST'])
def extract():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image_tensor = preprocess_image(file, device)

    features = extract_features(feature_extractor, image_tensor)
    return jsonify({'features': features.tolist()})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
