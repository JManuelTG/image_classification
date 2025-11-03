# indexer.py
import os, json, requests

IMAGE_DIR = "static/images/archive"
OUTPUT_FILE = "data/features.json"

def index_images():
    feature_database = {}
    for image_name in os.listdir(IMAGE_DIR):
        path = os.path.join(IMAGE_DIR, image_name)
        with open(path, "rb") as img:
            r = requests.post("http://127.0.0.1:5000/extract_features", files={"file": img})
        if r.status_code == 200:
            feature_database[image_name] = r.json()["features"]
        else:
            print(f"[X] Error with {image_name}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(feature_database, f, indent=4)
    print(f"[OK] Features saved at {OUTPUT_FILE}")


if __name__ == "__main__":
    index_images()
