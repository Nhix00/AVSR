"""
Real-time AVSR inference demo.

Controls:
    SPACE — push-to-talk: record audio + video and run inference
    Q / ESC — quit
"""
import logging
import os
import sys
import tempfile
import threading
import time
import warnings

import cv2
import numpy as np

try:
    import sounddevice as sd
except ImportError as e:
    raise ImportError("sounddevice not found. Install with: pip install sounddevice") from e

try:
    import tensorflow as tf
except ImportError as e:
    raise ImportError("TensorFlow not found. Install with: pip install tensorflow") from e

try:
    import librosa
except ImportError as e:
    raise ImportError("librosa not found. Install with: pip install librosa") from e

from avsr.config import cfg
from avsr.features import extract_audio_features, extract_video_features
from avsr.preprocessing import pad_with_noise_canvas

logger = logging.getLogger(__name__)

# Derive ordered class name list from cfg (first occurrence per index wins)
_inv: dict = {}
for _word, _idx in cfg.dataset.class_map.items():
    if _idx not in _inv:
        _inv[_idx] = _word
CLASS_NAMES: list = [_inv[i] for i in range(len(_inv))]

_COLOUR_OK = (0, 220, 80)
_COLOUR_REC = (0, 60, 255)
_COLOUR_WAIT = (200, 200, 200)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _pad_or_truncate_deterministic(sequence: np.ndarray, max_frames: int) -> np.ndarray:
    """Pad at end only (deterministic for inference)."""
    if sequence is None or len(sequence) == 0:
        raise ValueError("Received an empty sequence.")
    curr = sequence.shape[0]
    if curr > max_frames:
        return sequence[:max_frames, ...]
    if curr < max_frames:
        return np.pad(
            sequence,
            ((0, max_frames - curr),) + ((0, 0),) * (sequence.ndim - 1),
            mode="constant",
            constant_values=0,
        )
    return sequence


def preprocess_audio(audio_array: np.ndarray) -> np.ndarray:
    """
    Run training-identical audio preprocessing.

    Args:
        audio_array: 1D float32 array at cfg.audio.sample_rate Hz.

    Returns:
        np.ndarray of shape (1, max_frames, n_mfcc) — batch-ready.
    """
    peak = np.max(np.abs(audio_array))
    if peak > 0.001:
        audio_array = (audio_array / peak) * 0.10

    audio_processed = pad_with_noise_canvas(
        audio_array,
        noise_path=None,
        snr_db=50,
        target_length=cfg.dataset.target_length,
    )
    mfccs = extract_audio_features(y=audio_processed, target_sr=cfg.audio.sample_rate)
    mfccs = _pad_or_truncate_deterministic(mfccs, cfg.dataset.max_frames)
    return mfccs[np.newaxis, ...]


def preprocess_video_from_frames(
    frames: list,
    video_norm_stats: dict = None,
) -> np.ndarray:
    """
    Extract geometric mouth features from a list of raw BGR webcam frames.

    Args:
        frames: List of BGR uint8 numpy arrays.
        video_norm_stats: Dict with 'mean' and 'std' arrays of shape (F,), or None.

    Returns:
        np.ndarray of shape (1, max_frames, F) or None if face detection fails.
    """
    if not frames:
        return None

    tmp_path = os.path.join(tempfile.gettempdir(), "avsr_live_buffer.mp4")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), cfg.video.fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feats = extract_video_features(tmp_path, apply_jitter=False, apply_tilt=False)
    except Exception as exc:
        logger.warning("extract_video_features failed: %s", exc)
        return None

    if feats is None or len(feats) == 0:
        return None

    feats = _pad_or_truncate_deterministic(feats, cfg.dataset.max_frames)

    if video_norm_stats is not None:
        mean = video_norm_stats["mean"]
        std = video_norm_stats["std"]
        valid = np.any(feats != 0, axis=-1)
        feats[valid] = (feats[valid] - mean) / std

    return feats[np.newaxis, ...]


# ---------------------------------------------------------------------------
# Video normalisation stats
# ---------------------------------------------------------------------------

def load_video_norm_stats(npz_path: str) -> dict:
    """
    Compute per-feature mean and std from a training NPZ (mirrors train_avsr logic).

    Args:
        npz_path: Path to dataset_train.npz.

    Returns:
        Dict with keys 'mean' and 'std', each of shape (F,).
    """
    logger.info("Computing video normalisation stats from %s ...", npz_path)
    data = np.load(npz_path, allow_pickle=True)
    X_video = data["X_video"]
    valid = np.any(X_video != 0, axis=-1)
    mean = X_video[valid].mean(axis=0)
    std = X_video[valid].std(axis=0)
    std[std == 0] = 1.0
    logger.info("Video stats computed. Feature dim = %d", mean.shape[0])
    return {"mean": mean, "std": std}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(mode: str):
    """
    Load the .keras model for the selected mode from the models/ directory.

    Args:
        mode: 'audio', 'video', or 'fusion'.

    Returns:
        Loaded Keras model.

    Raises:
        ValueError: If mode is invalid.
        FileNotFoundError: If the model file does not exist.
    """
    paths = {
        "audio": os.path.join("models", "audio_model.keras"),
        "video": os.path.join("models", "video_model.keras"),
        "fusion": os.path.join("models", "fusion_model.keras"),
    }
    if mode not in paths:
        raise ValueError(f"Invalid mode '{mode}'. Choose from: {list(paths.keys())}")
    path = paths[mode]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}\nTrain and save models first with: python train.py"
        )
    logger.info("Loading %s model from %s ...", mode, path)
    model = tf.keras.models.load_model(path)
    model.summary(print_fn=lambda s: None)
    logger.info("Model loaded.")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, mode: str, audio_input, video_input) -> tuple:
    """
    Run model prediction and return (class_name, confidence_percent).

    Args:
        model: Loaded Keras model.
        mode: 'audio', 'video', or 'fusion'.
        audio_input: shape (1, max_frames, n_mfcc) or None.
        video_input: shape (1, max_frames, F) or None.

    Returns:
        (word, confidence_percent) tuple.
    """
    if mode == "audio":
        probs = model.predict(audio_input, verbose=0)
    elif mode == "video":
        probs = model.predict(video_input, verbose=0)
    else:
        probs = model.predict([audio_input, video_input], verbose=0)

    idx = int(np.argmax(probs[0]))
    return CLASS_NAMES[idx], float(probs[0][idx]) * 100.0


# ---------------------------------------------------------------------------
# OSD overlay
# ---------------------------------------------------------------------------

def draw_overlay(frame: np.ndarray, state: str, prediction: str, confidence: float, mode: str) -> np.ndarray:
    """Draw a semi-transparent HUD on the live feed frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 90), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"Mode: [{mode.upper()}]", (12, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 100), 1, cv2.LINE_AA)

    if state == "idle":
        colour, line1, line2 = _COLOUR_WAIT, "Press SPACE to record", "Q / ESC to quit"
    elif state == "recording":
        colour, line1, line2 = _COLOUR_REC, "Recording...", "Hold still & speak clearly"
    else:
        colour, line1, line2 = _COLOUR_OK, f"  {prediction.upper()}", f"  Confidence: {confidence:.1f}%"

    cv2.putText(frame, line1, (12, h - 62), cv2.FONT_HERSHEY_DUPLEX, 0.9, colour, 2, cv2.LINE_AA)
    cv2.putText(frame, line2, (12, h - 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, colour, 1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

class Recorder:
    """Synchronised push-to-talk audio recorder using sounddevice."""

    def __init__(self):
        self._audio = None
        self._thread = None

    def start(self) -> threading.Thread:
        """Launch asynchronous audio capture. Returns the thread."""
        n = int(cfg.inference.record_seconds * cfg.audio.sample_rate)
        self._audio = np.zeros(n, dtype=np.float32)
        self._thread = threading.Thread(target=self._record, daemon=True)
        self._thread.start()
        return self._thread

    def _record(self) -> None:
        n = int(cfg.inference.record_seconds * cfg.audio.sample_rate)
        recording = sd.rec(n, samplerate=cfg.audio.sample_rate, channels=1, dtype="float32")
        sd.wait()
        self._audio = recording[:, 0]

    def wait(self) -> None:
        self._thread.join()

    @property
    def audio(self) -> np.ndarray:
        return self._audio


# ---------------------------------------------------------------------------
# Main demo loop
# ---------------------------------------------------------------------------

def _run_demo(mode: str, norm_stats_path: str = None) -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (device 0). Check camera permissions.")

    model = load_model(mode)

    video_norm_stats = None
    if mode in ("video", "fusion"):
        if norm_stats_path and os.path.exists(norm_stats_path):
            video_norm_stats = load_video_norm_stats(norm_stats_path)
        else:
            logger.warning(
                "--norm_stats not provided. Video features will NOT be Z-score normalised. "
                "Pass --norm_stats dataset_train.npz for best accuracy."
            )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    logger.info("Demo ready. Press SPACE to record, Q/ESC to quit.")

    state = "idle"
    prediction = "—"
    confidence = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        if key == ord(" ") and state == "idle":
            state = "recording"
            draw_overlay(frame, state, prediction, confidence, mode)
            cv2.imshow("AVSR Live Demo", frame)
            cv2.waitKey(1)

            recorder = Recorder()
            audio_thread = recorder.start()

            recorded_frames = []
            t_start = time.time()
            while (time.time() - t_start) < cfg.inference.record_seconds:
                ret2, vid_frame = cap.read()
                if ret2:
                    vid_frame = cv2.flip(vid_frame, 1)
                    recorded_frames.append(vid_frame.copy())
                display = vid_frame.copy() if ret2 else frame.copy()
                draw_overlay(display, "recording", prediction, confidence, mode)
                cv2.imshow("AVSR Live Demo", display)
                cv2.waitKey(1)

            audio_thread.join()
            logger.info(
                "Captured %d video frames, %d audio samples.",
                len(recorded_frames), len(recorder.audio),
            )

            try:
                audio_input = None
                video_input = None

                if mode in ("audio", "fusion"):
                    audio_input = preprocess_audio(recorder.audio)
                    logger.info("Audio features shape: %s", audio_input.shape)

                if mode in ("video", "fusion"):
                    video_input = preprocess_video_from_frames(recorded_frames, video_norm_stats)
                    if video_input is None:
                        logger.warning("Face not detected. Please try again facing the camera.")
                        state = "idle"
                        continue
                    logger.info("Video features shape: %s", video_input.shape)

                prediction, confidence = run_inference(model, mode, audio_input, video_input)
                logger.info("Predicted: '%s'  (%.1f%%)", prediction, confidence)
                state = "result"

            except Exception as exc:
                logger.error("Inference failed: %s", exc)
                state = "idle"

        draw_overlay(frame, state, prediction, confidence, mode)
        cv2.imshow("AVSR Live Demo", frame)

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Demo closed.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(mode: str = None, norm_stats: str = None) -> None:
    """
    Start the real-time AVSR demo.

    Args:
        mode: 'audio', 'video', or 'fusion'. Defaults to cfg.inference.mode.
        norm_stats: Path to dataset_train.npz for video Z-score normalisation.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _run_demo(mode or cfg.inference.mode, norm_stats)
