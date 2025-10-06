import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import tensorflow_hub as hub
import traceback

APP_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(APP_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "index.csv")
MODEL_DIR = os.path.join(APP_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "mobilenet_v2_140_224")

app = Flask(__name__)

# ----------------- Helpers -----------------
def preprocess_image(file):
    img = Image.open(file)
    img.thumbnail((1024,1024))   # limit image size
    img = img.convert("RGB").resize((224,224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]

def encode_f32(v: np.ndarray) -> str:
    return v.astype(np.float32).tobytes().hex()

def decode_f32(hexs: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(hexs), dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a)+1e-8)
    b = b / (np.linalg.norm(b)+1e-8)
    return float(np.dot(a, b))

# ----------------- Load model safely -----------------
try:
    if os.path.exists(MODEL_PATH):
        model = hub.load(MODEL_PATH)
    else:
        print("Error: model folder not found. Make sure it is included in the Docker image.")
        model = None
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    traceback.print_exc()
    model = None

# ----------------- Load index safely -----------------
if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"{INDEX_PATH} not found! Make sure index.csv is deployed.")
df_index = pd.read_csv(INDEX_PATH)
df_index["emb"] = df_index["emb"].apply(decode_f32)
emb_size = len(df_index["emb"][0])

# ----------------- Routes -----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/search", methods=["POST"])
def api_search():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded properly"}), 500

        file = request.files.get("file")
        url = request.form.get("url", "").strip()

        if not file and not url:
            return jsonify({"error": "No file or URL provided"}), 400

        if file:
            img = preprocess_image(file)
        else:
            from io import BytesIO
            import requests
            try:
                r = requests.get(
                    url, 
                    timeout=10, 
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                r.raise_for_status()

                # Ensure the response is an image
                if "image" not in r.headers.get("Content-Type", ""):
                    return jsonify({"error": "Provided URL did not return an image"}), 400

                img = preprocess_image(BytesIO(r.content))

            except Exception as e:
                return jsonify({"error": f"Failed to fetch/process image from URL: {e}"}), 400

        # ---- compute embedding ----
        q_emb = model(img).numpy().reshape(-1)
        q_emb /= (np.linalg.norm(q_emb)+1e-8)

        results = []
        for _, row in df_index.iterrows():
            score = cosine_sim(q_emb, row["emb"])
            results.append({
                "id": int(row["id"]),
                "name": row["name"],
                "category": row.get("category",""),
                "price": float(row.get("price",0.0)),
                "image_src": row["image_src"],
                "score": score
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return jsonify({"results": results[:10]})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
