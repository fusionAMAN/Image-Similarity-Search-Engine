import tensorflow_hub as hub
import tensorflow as tf
import os

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "mobilenet_v2_140_224")

os.makedirs(MODEL_DIR, exist_ok=True)

# Download the TF Hub model
model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/5")

# Save it using tf.saved_model
tf.saved_model.save(model, MODEL_PATH)

print(f"Model saved at {MODEL_PATH}")