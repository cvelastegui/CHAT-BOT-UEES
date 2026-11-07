from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models"
MAX_LEN = 18

_model = None
_tokenizer = None
_encoder = None

def ensure_model():
    required = [
        MODEL_DIR / "intent_model.keras",
        MODEL_DIR / "tokenizer.pkl",
        MODEL_DIR / "label_encoder.pkl",
    ]
    if not all(p.exists() for p in required):
        from train import train_model
        train_model(verbose=0)

def load_components():
    global _model, _tokenizer, _encoder
    if _model is None:
        ensure_model()
        _model = tf.keras.models.load_model(MODEL_DIR / "intent_model.keras")
        with open(MODEL_DIR / "tokenizer.pkl", "rb") as f:
            _tokenizer = pickle.load(f)
        with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
            _encoder = pickle.load(f)
    return _model, _tokenizer, _encoder

def predict_intent(text: str):
    model, tokenizer, encoder = load_components()
    seq = tokenizer.texts_to_sequences([text.lower().strip()])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=MAX_LEN, padding="post", truncating="post"
    )
    probs = model.predict(padded, verbose=0)[0]
    idx = int(np.argmax(probs))
    return encoder.inverse_transform([idx])[0], float(probs[idx])
