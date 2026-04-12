"""
Real-time AVSR inference demo.

Controls:
    SPACE — push-to-talk: record audio + video and run inference
    Q / ESC — quit
"""
import logging
import os
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

# Project root is three directories above this file (src/avsr/inference.py → project root)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def preprocess_audio(audio_array: np.ndarray, noise_path: str = None, snr_db: float = 50) -> np.ndarray:
    """
    Run training-identical audio preprocessing.

    Args:
        audio_array: 1D float32 array at cfg.audio.sample_rate Hz.
        noise_path: Path to noise file (None = faint Gaussian noise).
        snr_db: Signal-to-noise ratio in dB (50 = essentially clean).

    Returns:
        np.ndarray of shape (1, max_frames, n_mfcc) — batch-ready.
    """
    rms_input = float(np.sqrt(np.mean(audio_array ** 2)))

    # Normalise speech amplitude to match training data level.
    # Training recordings had median RMS ≈ 0.0048.  A different microphone
    # (especially USB/external) records at a much higher level, shifting all
    # MFCC values by 20·log10(ratio) dB and placing features outside the range
    # the model was trained on.  Scaling to the training median ensures the
    # noise canvas is mixed at the same relative level as during training.
    _TARGET_RMS = 0.0055
    if rms_input > 1e-5:
        audio_array = audio_array * (_TARGET_RMS / rms_input)

    logger.info(
        "RAW audio — input_RMS=%.5f  → normalised to %.5f  SNR=%s dB  noise=%s",
        rms_input, _TARGET_RMS, snr_db, noise_path or "Gaussian",
    )
    audio_processed = pad_with_noise_canvas(
        audio_array,
        noise_path=noise_path,
        snr_db=snr_db,
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

    # Subsample captured frames to match the target FPS used during training.
    # Webcam may run at 30 FPS while cfg.video.fps = 15, so without subsampling
    # only the first half of the recording would be used after max_frames truncation.
    target_frames = cfg.dataset.max_frames
    if len(frames) > target_frames:
        indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]

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

    # Log raw (pre-normalization) video stats so we can compare to NPZ
    valid_raw = np.any(feats != 0, axis=-1)
    if valid_raw.any():
        fv = feats[valid_raw]
        logger.info(
            "RAW video features (before norm) — valid_frames=%d  mean=%.4f  std=%.4f  "
            "base_mean=%.4f  base_std=%.4f",
            int(valid_raw.sum()), float(fv.mean()), float(fv.std()),
            float(fv[:, :3].mean()), float(fv[:, :3].std()),
        )

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
    logger.info(
        "Video norm stats — dim=%d  mean(all)=%.4f  std(all)=%.4f  "
        "base_mean=%.4f  base_std=%.4f",
        mean.shape[0], float(mean.mean()), float(std.mean()),
        float(mean[:3].mean()), float(std[:3].mean()),
    )
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
        "audio": os.path.join(_PROJECT_ROOT, "models", "audio_model.keras"),
        "video": os.path.join(_PROJECT_ROOT, "models", "video_model.keras"),
        "fusion": os.path.join(_PROJECT_ROOT, "models", "fusion_model.keras"),
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
    import tensorflow as tf

    # Diagnostic: log feature statistics so we can compare against NPZ stats.
    # NPZ reference (audio_heavy): mean≈-4.79, std≈70.16
    # NPZ reference (clean):        mean≈-43.52, std≈188.57
    # NPZ reference (video norm):   mean≈-0.047, std≈0.638
    if audio_input is not None:
        a = audio_input[0]
        logger.info(
            "AUDIO features — shape=%s  mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
            a.shape, float(a.mean()), float(a.std()), float(a.min()), float(a.max()),
        )
    if video_input is not None:
        v = video_input[0]
        valid = np.any(v != 0, axis=-1)
        v_valid = v[valid]
        if len(v_valid):
            logger.info(
                "VIDEO features — shape=%s  valid_frames=%d  mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
                v.shape, int(valid.sum()), float(v_valid.mean()), float(v_valid.std()),
                float(v_valid.min()), float(v_valid.max()),
            )

    if mode == "audio":
        inputs = tf.constant(audio_input, dtype=tf.float32)
    elif mode == "video":
        inputs = tf.constant(video_input, dtype=tf.float32)
    else:
        # Named dict avoids Keras 3.x ambiguity when passing a list to a
        # multi-input model (list of 2 arrays can be misread as a batch of 2).
        inputs = {
            "audio_input": tf.constant(audio_input, dtype=tf.float32),
            "video_input": tf.constant(video_input, dtype=tf.float32),
        }

    probs = model(inputs, training=False).numpy()
    # Log top-3 predictions for debugging
    top3_idx = np.argsort(probs[0])[::-1][:3]
    logger.info(
        "Top-3: %s",
        "  |  ".join(
            f"{CLASS_NAMES[i]}={probs[0][i]*100:.1f}%" for i in top3_idx
        ),
    )
    idx = int(np.argmax(probs[0]))
    return CLASS_NAMES[idx], float(probs[0][idx]) * 100.0


# ---------------------------------------------------------------------------
# OSD overlay
# ---------------------------------------------------------------------------

def draw_overlay(frame: np.ndarray, state: str, prediction: str, confidence: float,
                 mode: str, active_mode: str = None) -> np.ndarray:
    """Draw a semi-transparent HUD on the live feed frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 90), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    display_mode = mode.upper()
    if active_mode and active_mode != mode:
        display_mode = f"{mode.upper()} -> {active_mode.upper()} (noise)"
    cv2.putText(frame, f"Mode: [{display_mode}]", (12, 28),
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

    def __init__(self, audio_device=None):
        self._audio = None
        self._thread = None
        self._device = audio_device

    def start(self) -> threading.Thread:
        """Launch asynchronous audio capture. Returns the thread."""
        n = int(cfg.inference.record_seconds * cfg.audio.sample_rate)
        self._audio = np.zeros(n, dtype=np.float32)
        self._thread = threading.Thread(target=self._record, daemon=True)
        self._thread.start()
        return self._thread

    def _record(self) -> None:
        n = int(cfg.inference.record_seconds * cfg.audio.sample_rate)
        recording = sd.rec(
            n, samplerate=cfg.audio.sample_rate, channels=1, dtype="float32",
            device=self._device,
        )
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

def _run_demo(mode: str, norm_stats_path: str = None, video_device: int = 0, audio_device: int = None) -> None:
    cap = cv2.VideoCapture(video_device)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera device {video_device}. Check index with --list-devices.")

    model = load_model(mode)

    # For fusion mode, also load the video model so we can fall back to it at heavy
    # noise (SNR < 0 dB).  At audio_heavy (-15 dB) the audio-only accuracy is 45%
    # while video-only achieves 97%, so using video at heavy noise is strictly better.
    video_model_fallback = None
    if mode == "fusion":
        try:
            video_model_fallback = load_model("video")
            logger.info("Video fallback model loaded — will switch to video-only below SNR 0 dB.")
        except FileNotFoundError:
            logger.warning("video_model.keras not found; fusion will run at all noise levels.")

    # SNR threshold below which video model is preferred over fusion
    _FUSION_VIDEO_THRESHOLD_DB = 0

    video_norm_stats = None
    if mode in ("video", "fusion"):
        # Resolve norm_stats_path: try as-is first, then relative to project root
        _nsp = norm_stats_path
        if _nsp and not os.path.exists(_nsp):
            _nsp_abs = os.path.join(_PROJECT_ROOT, _nsp)
            if os.path.exists(_nsp_abs):
                _nsp = _nsp_abs
        if _nsp and os.path.exists(_nsp):
            video_norm_stats = load_video_norm_stats(_nsp)
        else:
            logger.warning(
                "--norm_stats not provided or not found. Video features will NOT be Z-score normalised. "
                "Pass --norm_stats dataset_train.npz for best accuracy."
            )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Noise file from training config — try both CWD-relative and project-root-relative paths
    _noise_cfg = cfg.dataset.noise_file
    noise_file = (
        _noise_cfg if os.path.exists(_noise_cfg)
        else os.path.join(_PROJECT_ROOT, _noise_cfg) if os.path.exists(os.path.join(_PROJECT_ROOT, _noise_cfg))
        else None
    )
    if noise_file is None:
        logger.warning("Noise file '%s' not found — slider will use Gaussian noise.", _noise_cfg)
    else:
        logger.info("Using noise file: %s", noise_file)

    # Trackbar: value 0‥65 → SNR -15‥+50 dB  (matches training range)
    _WINDOW = "AVSR Live Demo"
    _SNR_OFFSET = 15          # snr_db = trackbar - _SNR_OFFSET
    _SNR_CLEAN  = 65          # trackbar value that means "clean" (50 dB)
    cv2.namedWindow(_WINDOW)
    cv2.createTrackbar("Noise SNR (dB)  [-15 .. +50]", _WINDOW, _SNR_CLEAN, 65, lambda _: None)

    logger.info("Demo ready. Press SPACE to record, Q/ESC to quit.")

    state = "idle"
    prediction = "—"
    confidence = 0.0
    last_active_mode = mode   # tracks which model was actually used for the last result

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        if key == ord(" ") and state in ("idle", "result"):
            state = "recording"
            draw_overlay(frame, state, prediction, confidence, mode, last_active_mode)
            cv2.imshow(_WINDOW, frame)
            cv2.waitKey(1)

            recorder = Recorder(audio_device=audio_device)
            audio_thread = recorder.start()

            recorded_frames = []
            t_start = time.time()
            while (time.time() - t_start) < cfg.inference.record_seconds:
                ret2, vid_frame = cap.read()
                if ret2:
                    vid_frame = cv2.flip(vid_frame, 1)
                    recorded_frames.append(vid_frame.copy())
                display = vid_frame.copy() if ret2 else frame.copy()
                draw_overlay(display, "recording", prediction, confidence, mode, last_active_mode)
                cv2.imshow(_WINDOW, display)
                cv2.waitKey(1)

            audio_thread.join()
            logger.info(
                "Captured %d video frames, %d audio samples.",
                len(recorded_frames), len(recorder.audio),
            )

            try:
                audio_input = None
                video_input = None

                tb_val = cv2.getTrackbarPos("Noise SNR (dB)  [-15 .. +50]", _WINDOW)
                snr_db = tb_val - _SNR_OFFSET

                if mode in ("audio", "fusion"):
                    logger.info("Applying noise: SNR = %d dB (trackbar = %d)", snr_db, tb_val)
                    audio_input = preprocess_audio(recorder.audio, noise_path=noise_file, snr_db=snr_db)
                    logger.info("Audio features shape: %s", audio_input.shape)

                if mode in ("video", "fusion"):
                    video_input = preprocess_video_from_frames(recorded_frames, video_norm_stats)
                    if video_input is None:
                        logger.warning("Face not detected. Please try again facing the camera.")
                        state = "idle"
                        continue
                    logger.info("Video features shape: %s", video_input.shape)

                # At heavy noise, the fusion model relies on nearly-useless audio (45%
                # accuracy alone at -15 dB) and only reaches 86%, while video-only
                # achieves 97%.  Fall back to the video model when SNR < threshold.
                active_model = model
                active_mode = mode
                if (mode == "fusion"
                        and video_model_fallback is not None
                        and snr_db < _FUSION_VIDEO_THRESHOLD_DB):
                    active_model = video_model_fallback
                    active_mode = "video"
                    logger.info("Heavy noise (SNR=%d dB): switching to video-only model.", snr_db)

                prediction, confidence = run_inference(active_model, active_mode, audio_input, video_input)
                last_active_mode = active_mode
                logger.info("Predicted: '%s'  (%.1f%%)", prediction, confidence)
                state = "result"

            except Exception as exc:
                import traceback
                logger.error("Inference failed: %s\n%s", exc, traceback.format_exc())
                state = "idle"

        draw_overlay(frame, state, prediction, confidence, mode, last_active_mode)
        cv2.imshow(_WINDOW, frame)

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Demo closed.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def list_devices() -> None:
    """Print available audio input devices and probe video device indices."""
    print("\n=== Audio input devices ===")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            marker = " (default)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {d['name']}{marker}")

    print("\n=== Video devices (probing indices 0–9) ===")
    for idx in range(10):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  [{idx}] available  ({w}x{h})")
            cap.release()
    print()


def run(mode: str = None, norm_stats: str = None, video_device: int = 0, audio_device: int = None) -> None:
    """
    Start the real-time AVSR demo.

    Args:
        mode: 'audio', 'video', or 'fusion'. Defaults to cfg.inference.mode.
        norm_stats: Path to dataset_train.npz for video Z-score normalisation.
        video_device: Camera index (default 0).
        audio_device: Audio input device index (default: system default).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _run_demo(mode or cfg.inference.mode, norm_stats, video_device=video_device, audio_device=audio_device)
