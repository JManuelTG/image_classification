# app.py
from flask import Flask, request, jsonify
from model import load_model, preprocess_image, extract_features, predict_class
import json, numpy as np
from scipy.spatial.distance import cosine, euclidean, cityblock

app = Flask(__name__)
model, feature_extractor, device, classes = load_model()

DB_FILE = "data/features.json"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image_tensor = preprocess_image(file, device)
    predicted_class = predict_class(model, image_tensor, classes)

    return jsonify({'prediction': predicted_class})


@app.route('/extract_features', methods=['POST'])
def extract_only():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image_tensor = preprocess_image(file, device)
    features = extract_features(feature_extractor, image_tensor)

    return jsonify({'features': features.tolist()})


@app.route('/search', methods=['POST'])
def search_similar():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image_tensor = preprocess_image(file, device)
    query_vector = extract_features(feature_extractor, image_tensor).cpu().numpy()

    # cargar base de features
    try:
        with open(DB_FILE, "r") as f:
            feature_database = json.load(f)
    except FileNotFoundError:
        return jsonify({'error': f'Feature database {DB_FILE} not found'}), 500

    image_names = list(feature_database.keys())
    vectors = np.array([feature_database[name] for name in image_names])

    results = []
    for i, vec in enumerate(vectors):
        cos_sim = 1 - cosine(query_vector, vec)
        eucl = euclidean(query_vector, vec)
        manh = cityblock(query_vector, vec)
        results.append({
            "image": image_names[i],
            "cosine_similarity": float(cos_sim),
            "euclidean_distance": float(eucl),
            "manhattan_distance": float(manh)
        })

    # ordenar igual que en search.py
    results.sort(key=lambda x: (-x["cosine_similarity"], x["euclidean_distance"], x["manhattan_distance"]))

    return jsonify({
        "query_image": file.filename,
        "results": results[:10]
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
