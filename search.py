# client_search.py
import requests

API_URL = "http://127.0.0.1:5000/search"
QUERY_IMG = "static/images/test6.jpg"

with open(QUERY_IMG, "rb") as img:
    response = requests.post(API_URL, files={"file": img})

if response.status_code == 200:
    data = response.json()
    print(f"[INFO] Most similar images to {data['query_image']}:")
    for r in data["results"]:
        print(f"--> {r['image']} | Cosine: {r['cosine_similarity']:.4f} "
              f"| Euclidean: {r['euclidean_distance']:.4f} "
              f"| Manhattan: {r['manhattan_distance']:.4f}")
else:
    print("Error:", response.json())
