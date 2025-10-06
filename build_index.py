import os
import pandas as pd
import numpy as np
import tensorflow_hub as hub
from PIL import Image
import requests
from urllib.parse import urlparse
import hashlib

APP_DIR = os.path.dirname(_file_)
DATA_DIR = os.path.join(APP_DIR, "data")
CACHE_DIR = os.path.join(APP_DIR, "cache")
CATALOG_CSV = os.path.join(DATA_DIR, "products.csv")
INDEX_PATH = os.path.join(DATA_DIR, "index.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# MobileNetV2 feature extractor
model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/5")

# ----------------- Utils -----------------
def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]  # shape (1,224,224,3)

def encode_f32(v: np.ndarray) -> str:
    return v.astype(np.float32).tobytes().hex()

def fetch_image(url: str) -> str:
    """Download image to cache and return local path."""
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    ext = os.path.splitext(urlparse(url).path)[1] or ".jpg"
    path = os.path.join(CACHE_DIR, f"{h}{ext}")
    if not os.path.exists(path):
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path

# ----------------- Build index -----------------
def build_index():
    if not os.path.exists(CATALOG_CSV):
        raise FileNotFoundError(f"{CATALOG_CSV} not found! Create your catalog first.")

    cat = pd.read_csv(CATALOG_CSV)
    rows = []

    for _, row in cat.iterrows():
        try:
            path = row["image_url"]
            if path.startswith("http://") or path.startswith("https://"):
                path = fetch_image(path)
            img = preprocess_image(path)
            emb = model(img).numpy().reshape(-1)
            emb /= (np.linalg.norm(emb) + 1e-8)  # L2 normalize

            rows.append({
                "id": row["id"],
                "name": row.get("name", f"Product {row['id']}"),
                "category": row.get("category", ""),
                "price": float(row.get("price", 0.0)),
                "image_src": row["image_url"],
                "emb": encode_f32(emb),
            })

        except Exception as e:
            print(f"⚠ Skipping {row.get('id')} due to error:", e)

    if rows:
        pd.DataFrame(rows).to_csv(INDEX_PATH, index=False)
        print(f"✅ index.csv ready with {len(rows)} entries!")
    else:
        print("⚠ No valid images found; index.csv not created.")

if _name_ == "_main_":
    build_index()