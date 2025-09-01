# search.py
import json, requests, numpy as np
from scipy.spatial.distance import cosine, euclidean, cityblock

DB_FILE = "data/features.json"
QUERY_IMG = "images/test6.jpg"

with open(QUERY_IMG, "rb") as img:
    r = requests.post("http://127.0.0.1:5000/extract_features", files={"file": img})
query_vector = np.array(r.json().get("features", []))

with open(DB_FILE, "r") as f:
    feature_database = json.load(f)

image_names = list(feature_database.keys())
vectors = np.array([feature_database[name] for name in image_names])

results = []
for i, vec in enumerate(vectors):
    cos_sim = 1 - cosine(query_vector, vec)
    eucl = euclidean(query_vector, vec)
    manh = cityblock(query_vector, vec)
    results.append((image_names[i], cos_sim, eucl, manh))

results.sort(key=lambda x: (-x[1], x[2], x[3]))

print(f"[INFO] Most similar images to {QUERY_IMG}:")
for name, cos_sim, eucl, manh in results[:10]:
    print(f"--> {name} | Cosine: {cos_sim:.4f} | Euclidean: {eucl:.4f} | Manhattan: {manh:.4f}")
