# 🔍 Image Similarity Search Engine  
  
A **deep learning–powered** visual search engine that lets you upload an image (or paste a URL) and instantly find visually similar products from a catalog 🛍️. It uses **MobileNetV2 embeddings + cosine similarity** to rank catalog items most similar to the query image. 📦.  
 
---

## 🌐 Live Demo  
https://photo-analyzer-al39.onrender.com/ 
<!-- Example placeholders for deployment -->
<!-- [Frontend Live](#) | [Backend API](#) -->

---

# 🧭 Approach  

### Catalog Preparation & Indexing
- Process all images and metadata from `products.csv`
- Each image is passed through **MobileNetV2 (pretrained on ImageNet)** to generate a *1280-dimensional embedding**.
- Embeddings are normalized and saved to `index.csv` for fast lookup during runtime.
### Query Processing
- When a user uploads an image or provides a URL, the same embedding process is applied.
- The query image is converted into a `1280-D vector`.
### Similarity Computation
- **Cosine similarity** is calculated between the query embedding and catalog embeddings.
- This measures how visually similar the images are in feature space.
### Result Retrieval & Ranking
- Catalog embeddings are sorted by similarity score.
- Top-N most similar products are selected.
- Output includes product name, category, price, image, and similarity score.
### Optimizations
- Images and embeddings are cached for faster queries.
- CSV-based indexing is lightweight and efficient for small to medium catalogs.

---

## ✨ Features  

- 📂 **Catalog Indexing** from a `products.csv` (id, name, category, price, image URL)  
- 📷 **Query by Image Upload or URL** (supports remote & local images)  
- 🧠 **Deep Learning Embeddings** using pretrained MobileNetV2 (ImageNet)  
- 📊 **Cosine Similarity Search** across catalog embeddings  
- 🎛  Adjustable **Similarity Threshold** (default 70%)  
- ⚡ **Fast Caching** of images & embeddings  
- 🎨 Simple, responsive **Flask + HTML UI**  

---

## 🗂 Project Structure  

```bash
photo-analyzer/
├─ app.py                 # Flask backend (routes, embedding, search)
├─ build_index.py         # Generates index.csv from products.csv
├─ download_model.py      # Downloads & saves MobileNetV2 locally
├─ data/
│   ├─ products.csv       # Catalog metadata (id, name, price, img_url)
│   └─ index.csv          # Precomputed embeddings
├─ models/
│   └─ mobilenet_v2_140_224/   # Saved TensorFlow Hub model
├─ static/
│   ├─ images/            # Local product images
│   └─ styles.css         # Custom styles
├─ templates/
│   └─ index.html         # Web frontend (UI)
├─ requirements.txt       # Python dependencies
├─ Dockerfile             # Docker container setup
├─ render.yaml            # Render deployment config
└─ runtime.txt            # Python runtime version
```
# 🔎 How It Works  

## 1️⃣ Indexing  
- Reads `products.csv`  
- Downloads or loads product images  
- Extracts **1280-D** embeddings using pretrained **MobileNetV2**
- Saves them in `index.csv`  

## 2️⃣ Query  
- User uploads an image or provides a URL
- Preprocess → extract embedding vector
- Compute cosine similarity against all catalog embeddings
- Sort matches by similarity score
- Display name, category, price, image, similarity score

## 3️⃣ Results  
- Filters matches ≥ threshold (default 70%)  
- Sorts by similarity score  
- Displays **name, category, price, image, similarity score**  

---

# 🛠 Tech Stack  

- **Backend** → Flask, Pandas, NumPy, Pillow, Requests
- **Model** → Pretrained MobileNetV2 (ImageNet features)  
- **Frontend** → HTML, CSS, Bootstrap (Flask templates)  
- **Storage** → CSV-based lightweight indexing (products.csv, index.csv)  
- **Deployment** → Flask (Docker + Render)  

---

# 🚀 Getting Started (Local)  

## 1️⃣ Clone the repo  
```bash
git clone https://github.com/Ritesh15102/Visual-Product-Matcher-Build.git
cd Visual-Product-Matcher-Build
```
## 2️⃣ Setup environment
``` bash
python -m venv .venv
# Activate
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```
### 📌 requirements.txt
``` txt
setuptools>=65.0.0
wheel
Flask
numpy
pandas
Pillow
requests
tensorflow==2.16.1
tensorflow_hub==0.16.1
gunicorn
```
## 3️⃣ Prepare catalog data
- Create/edit backend/data/products.csv like:
``` csv
id,name,category,image_url,price
1,Product 1 (Tops),Tops,static/images/item001.jpg,9.99
2,Product 2 (Shoes),Shoes,static/images/item002.jpg,15.19
```
## 4️⃣ Run backend
``` bash
python app.py
```
### App runs at → http://127.0.0.1:5000

### Go to browser → upload an image / paste URL → see results 🚀

### 📡 API Reference
`🔸 POST /search`
### 📌 Form-Data Params
- `file` → uploaded image *(optional)*
- `image_url` → string *(optional)*
---

### 📤 Response Example
``` json
{
  "results": [
    {
      "id": 1,
      "name": "Product 1 (Tops)",
      "category": "Tops",
      "price": 9.99,
      "image_src": "static/images/item001.jpg",
      "score": 0.9231
    }
  ]
}
```
## 📊 Dataset Insights
- Unique Categories: 5 (Tops, Shoes, Bags, Home, Accessories)
- Unique Products: 50 (Product 1 → Product 50)
- Unique Prices: 50 (range: 9.99 → 64.79, all distinct)
- ✅ Balanced catalog with diverse items & unique prices.

## ☁ Deployment
- ✅ Run locally via Flask
- ✅ Deploy with Docker + Render
- ✅ Pre-download model with `download_model.py` for faster startup

## 📸 Screenshots
<img width="1056" height="484" alt="image" src="https://github.com/user-attachments/assets/52187dcd-1a12-482d-81bd-8ce25d11bb9f" />
<img width="1079" height="950" alt="image" src="https://github.com/user-attachments/assets/07b4aa93-2a15-4fcf-83c4-98b3174ca331" />




