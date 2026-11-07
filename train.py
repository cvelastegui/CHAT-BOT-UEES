from pathlib import Path
import json
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "intents.json"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

MAX_WORDS = 2000
MAX_LEN = 20
EPOCHS = 120
BATCH_SIZE = 8
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

def load_dataset():
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    texts, labels = [], []
    for item in data["intents"]:
        for pattern in item["patterns"]:
            texts.append(pattern.lower().strip())
            labels.append(item["tag"])
    return np.array(texts), np.array(labels), data

def build_model(vocab_size, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=MAX_LEN),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_model(verbose=1):
    texts, labels, raw_data = load_dataset()

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>"
    )
    tokenizer.fit_on_texts(texts)
    seq = tokenizer.texts_to_sequences(texts)
    X = tf.keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=MAX_LEN, padding="post", truncating="post"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
    model = build_model(vocab_size, len(encoder.classes_))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=verbose,
        callbacks=callbacks
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    report = classification_report(
        y_test, pred,
        labels=list(range(len(encoder.classes_))),
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(y_test, pred).tolist()

    model.save(MODEL_DIR / "intent_model.keras")
    with open(MODEL_DIR / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": cm,
        "classes": encoder.classes_.tolist(),
        "epochs_trained": len(history.history["loss"])
    }
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nAccuracy de prueba: {accuracy:.4f}")
    print(f"Épocas ejecutadas: {len(history.history['loss'])}")
    print("\nModelo guardado en /models")
    return metrics

if __name__ == "__main__":
    train_model(verbose=1)
