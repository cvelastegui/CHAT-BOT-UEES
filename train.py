from pathlib import Path
import json
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "intents.json"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

MAX_WORDS = 4000
MAX_LEN = 18
EPOCHS = 180
BATCH_SIZE = 16
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
    return np.array(texts), np.array(labels)

def build_model(vocab_size, num_classes):
    inputs = tf.keras.Input(shape=(MAX_LEN,))
    x = tf.keras.layers.Embedding(vocab_size, 64)(inputs)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.10)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_model(verbose=1):
    texts, labels = load_dataset()

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>",
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    )
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    X = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=MAX_LEN, padding="post", truncating="post"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
    model = build_model(vocab_size, len(encoder.classes_))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=25,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=verbose,
        callbacks=callbacks
    )

    pred_probs = model.predict(X_test, verbose=0)
    pred = np.argmax(pred_probs, axis=1)

    report = classification_report(
        y_test, pred,
        labels=list(range(len(encoder.classes_))),
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classes": encoder.classes_.tolist(),
        "epochs_trained": len(history.history["loss"]),
        "dataset_size": int(len(texts))
    }

    model.save(MODEL_DIR / "intent_model.keras")
    with open(MODEL_DIR / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\n=== RESULTADOS ===")
    print(f"Dataset: {len(texts)} frases")
    print(f"Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"Precision: {metrics['macro_precision']:.4f}")
    print(f"Recall:    {metrics['macro_recall']:.4f}")
    print(f"F1 macro:  {metrics['macro_f1']:.4f}")
    print(f"Epocas:    {metrics['epochs_trained']}")
    return metrics

if __name__ == "__main__":
    train_model()
