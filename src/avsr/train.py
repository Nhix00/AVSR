"""Training pipeline: load data, train all models, evaluate, save."""
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from avsr.config import cfg
from avsr.models import build_audio_model, build_fusion_model, build_video_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(file_path: str) -> tuple:
    """
    Load a dataset NPZ file.

    Args:
        file_path: Path to the NPZ file.

    Returns:
        Tuple (X_audio, X_video, Y_labels_categorical, Y_metadata).
    """
    logger.info("Loading %s ...", file_path)
    data = np.load(file_path, allow_pickle=True)
    X_audio = data["X_audio"]
    X_video = data["X_video"]
    Y_labels = data["Y_labels"]
    Y_metadata = data["Y_metadata"]
    Y_labels_cat = to_categorical(Y_labels, num_classes=cfg.dataset.num_classes)
    return X_audio, X_video, Y_labels_cat, Y_metadata


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_video(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply Z-score normalisation to video features, ignoring zero-padded frames.

    Args:
        X: Array of shape (N, T, F).
        mean: Per-feature mean of shape (F,).
        std: Per-feature std of shape (F,) — zero entries replaced with 1.

    Returns:
        Normalised copy of X.
    """
    X_norm = np.copy(X)
    valid = np.any(X != 0, axis=-1)
    X_norm[valid] = (X[valid] - mean) / std
    return X_norm


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, X_train, Y_train, X_val, Y_val) -> tuple:
    """
    Compile and train a model with early stopping.

    Args:
        model: Keras model to train.
        X_train: Training features (array or list of arrays for fusion).
        Y_train: One-hot training labels.
        X_val: Validation features.
        Y_val: One-hot validation labels.

    Returns:
        (trained_model, history) tuple.
    """
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=cfg.model.patience,
        restore_best_weights=True,
        verbose=1,
    )
    logger.info("Training %s ...", model.name)
    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=cfg.model.epochs,
        batch_size=cfg.model.batch_size,
        callbacks=[early_stop],
        verbose=1,
    )
    return model, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_conditions(
    model,
    X_test,
    Y_test: np.ndarray,
    metadata: np.ndarray,
    conditions: list,
    is_fusion: bool = False,
) -> dict:
    """
    Evaluate a model on the full test set and per noise/augmentation condition.

    Args:
        model: Trained Keras model.
        X_test: Test features (array or [audio_arr, video_arr] for fusion).
        Y_test: One-hot test labels.
        metadata: String array of condition tags per sample.
        conditions: List of condition tag strings to evaluate individually.
        is_fusion: Set True for multi-input fusion models.

    Returns:
        Dict mapping condition name → accuracy float (or None if missing).
    """
    accuracies = {}
    _, acc = model.evaluate(X_test, Y_test, verbose=0)
    accuracies["Overall"] = acc

    for condition in conditions:
        mask = metadata == condition
        if not np.any(mask):
            accuracies[condition] = None
            continue
        X_cond = [X_test[0][mask], X_test[1][mask]] if is_fusion else X_test[mask]
        _, acc = model.evaluate(X_cond, Y_test[mask], verbose=0)
        accuracies[condition] = acc

    return accuracies


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_learning_curves(history, title: str, filename: str) -> None:
    """Save a loss + accuracy learning curve figure."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    acc_key = "accuracy" if "accuracy" in history.history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"

    axes[0].plot(history.history["loss"], label="Training Loss")
    axes[0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history[acc_key], label="Training Accuracy")
    axes[1].plot(history.history[val_acc_key], label="Validation Accuracy")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    logger.info("Saved %s", filename)


def plot_confusion_matrix(
    model,
    X_test,
    Y_test_cat: np.ndarray,
    class_names: list,
    title: str,
    filename: str,
) -> None:
    """Save a seaborn confusion matrix figure."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    Y_pred = np.argmax(model.predict(X_test), axis=1)
    Y_true = np.argmax(Y_test_cat, axis=1)
    cm = confusion_matrix(Y_true, Y_pred, labels=np.arange(len(class_names))).astype("int")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"va": "center", "ha": "center"},
    )
    plt.title(f"{title}\nConfusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", filename)


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def _print_results_table(audio_res: dict, video_res: dict, fusion_res: dict, conditions: list) -> None:
    w = 115
    fmt = "| {:<18} | {:<9} | {:<7} | {:<11} | {:<11} | {:<11} | {:<11} | {:<9} |"

    def pct(v):
        return f"{v*100:>6.2f}%" if v is not None else "   N/A"

    print("\n" + "=" * w)
    print("GRANULAR EVALUATION RESULTS".center(w))
    print("=" * w)
    header = fmt.format("Model", "Overall", "Clean", "Audio Light", "Audio Heavy", "Video Light", "Video Heavy", "A+V Light")
    print(header)
    print("-" * len(header))
    for name, res in [("Audio-Only LSTM", audio_res), ("Video-Only LSTM", video_res), ("Early Fusion", fusion_res)]:
        print(fmt.format(
            name,
            pct(res.get("Overall")),
            pct(res.get("clean")),
            pct(res.get("audio_light")),
            pct(res.get("audio_heavy")),
            pct(res.get("video_light")),
            pct(res.get("video_heavy")),
            pct(res.get("audio_video_light")),
        ))
    print("=" * w + "\n")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run() -> None:
    """Full training pipeline: load → normalise → build → train → evaluate → save."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print(">>> 1. Loading datasets")
    X_audio_train, X_video_train, Y_train, meta_train = load_data(cfg.dataset.train_file)
    X_audio_val, X_video_val, Y_val, meta_val = load_data(cfg.dataset.val_file)
    X_audio_test, X_video_test, Y_test, meta_test = load_data(cfg.dataset.test_file)

    print(">>> 2. Normalising video features (Z-score, padding-aware)")
    valid_mask = np.any(X_video_train != 0, axis=-1)
    video_mean = X_video_train[valid_mask].mean(axis=0)
    video_std = X_video_train[valid_mask].std(axis=0)
    video_std[video_std == 0] = 1.0

    X_video_train = normalize_video(X_video_train, video_mean, video_std)
    X_video_val = normalize_video(X_video_val, video_mean, video_std)
    X_video_test = normalize_video(X_video_test, video_mean, video_std)

    print(">>> 3. Building models")
    audio_shape = X_audio_train.shape[1:]
    video_shape = X_video_train.shape[1:]
    audio_model = build_audio_model(input_shape=audio_shape)
    video_model = build_video_model(input_shape=video_shape)
    fusion_model = build_fusion_model(audio_shape=audio_shape, video_shape=video_shape)

    print(">>> 4. Training")
    audio_model, audio_hist = train_model(audio_model, X_audio_train, Y_train, X_audio_val, Y_val)
    video_model, video_hist = train_model(video_model, X_video_train, Y_train, X_video_val, Y_val)
    fusion_model, fusion_hist = train_model(
        fusion_model,
        [X_audio_train, X_video_train], Y_train,
        [X_audio_val, X_video_val], Y_val,
    )

    print(">>> 5. Evaluating")
    conditions = ["clean", "audio_light", "audio_heavy", "video_light", "video_heavy", "audio_video_light"]
    audio_res = evaluate_on_conditions(audio_model, X_audio_test, Y_test, meta_test, conditions)
    video_res = evaluate_on_conditions(video_model, X_video_test, Y_test, meta_test, conditions)
    fusion_res = evaluate_on_conditions(
        fusion_model, [X_audio_test, X_video_test], Y_test, meta_test, conditions, is_fusion=True
    )

    _print_results_table(audio_res, video_res, fusion_res, conditions)

    print(">>> 6. Plotting")
    os.makedirs("results", exist_ok=True)
    class_names = [k for k, _ in sorted(
        {k: v for k, v in cfg.dataset.class_map.items() if k != "sì"}.items(),
        key=lambda kv: kv[1]
    )]

    plot_learning_curves(audio_hist, "Audio Model", "results/audio_learning_curves.png")
    plot_confusion_matrix(audio_model, X_audio_test, Y_test, class_names, "Audio Model", "results/audio_confusion_matrix.png")
    plot_learning_curves(video_hist, "Video Model", "results/video_learning_curves.png")
    plot_confusion_matrix(video_model, X_video_test, Y_test, class_names, "Video Model", "results/video_confusion_matrix.png")
    plot_learning_curves(fusion_hist, "Fusion Model", "results/fusion_learning_curves.png")
    plot_confusion_matrix(fusion_model, [X_audio_test, X_video_test], Y_test, class_names, "Fusion Model", "results/fusion_confusion_matrix.png")

    print(">>> 7. Saving models")
    os.makedirs("models", exist_ok=True)
    audio_model.save("models/audio_model.keras")
    video_model.save("models/video_model.keras")
    fusion_model.save("models/fusion_model.keras")
    print("All models saved.")
