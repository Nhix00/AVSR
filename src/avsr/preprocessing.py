"""Dataset building pipeline: raw recordings → train/val/test NPZ files."""
import glob
import logging
import os

import librosa
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from avsr.config import cfg
from avsr.features import extract_audio_features, extract_video_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio canvas helper
# ---------------------------------------------------------------------------

def pad_with_noise_canvas(
    speech_audio: np.ndarray,
    noise_path,
    snr_db: float,
    target_length: int,
) -> np.ndarray:
    """
    Superimpose speech onto a noise canvas of exactly target_length samples.

    This guarantees MFCC extraction yields the desired number of frames without
    zero-padding artefacts.

    Args:
        speech_audio: Raw speech waveform array.
        noise_path: Path to an environmental noise file, or None for faint Gaussian noise.
        snr_db: Target Signal-to-Noise Ratio in dB.
        target_length: Exact number of output samples required.

    Returns:
        np.ndarray of shape (target_length,).
    """
    eps = 1e-9

    if noise_path is not None and os.path.exists(noise_path):
        noise_audio, _ = librosa.load(noise_path, sr=cfg.audio.sample_rate)
    else:
        noise_audio = np.random.randn(target_length).astype(np.float32)

    while len(noise_audio) < target_length:
        noise_audio = np.concatenate([noise_audio, noise_audio])

    if len(noise_audio) > target_length:
        max_start = len(noise_audio) - target_length
        noise_canvas = noise_audio[np.random.randint(0, max_start + 1) :][: target_length].copy()
    else:
        noise_canvas = noise_audio.copy()

    rms_speech = np.sqrt(np.mean(speech_audio**2)) + eps
    target_rms = rms_speech / (10.0 ** (snr_db / 20.0))
    rms_canvas = np.sqrt(np.mean(noise_canvas**2)) + eps
    scaled_noise = noise_canvas * (target_rms / rms_canvas)

    speech_len = len(speech_audio)
    if speech_len > target_length:
        crop_start = np.random.randint(0, speech_len - target_length + 1)
        scaled_noise += speech_audio[crop_start : crop_start + target_length]
    else:
        max_start = target_length - speech_len
        start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        scaled_noise[start : start + speech_len] += speech_audio

    max_amp = np.max(np.abs(scaled_noise))
    if max_amp > 1.0:
        scaled_noise /= max_amp

    return scaled_noise


# ---------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------

def pad_or_truncate(sequence: np.ndarray, max_frames=None) -> np.ndarray:
    """
    Force the temporal dimension to exactly max_frames by zero-padding or truncating.

    Padding position is chosen randomly (training-style augmentation).

    Args:
        sequence: Array of shape (T, F).
        max_frames: Target length. Defaults to cfg.dataset.max_frames.

    Returns:
        Array of shape (max_frames, F).
    """
    if max_frames is None:
        max_frames = cfg.dataset.max_frames

    if sequence is None or len(sequence) == 0:
        return sequence

    curr = sequence.shape[0]
    if curr > max_frames:
        return sequence[:max_frames, ...]
    if curr < max_frames:
        max_start = max_frames - curr
        start = np.random.randint(0, max_start + 1)
        end = max_frames - curr - start
        pad_width = [(start, end)] + [(0, 0)] * (sequence.ndim - 1)
        return np.pad(sequence, pad_width, mode="constant", constant_values=0)
    return sequence


# ---------------------------------------------------------------------------
# Dataset split helpers
# ---------------------------------------------------------------------------

def _print_split_distribution(split_name: str, samples: list) -> None:
    from collections import Counter

    class_map = cfg.dataset.class_map
    id_to_class = {v: k for k, v in class_map.items()}
    counts = Counter(s["label"] for s in samples)
    print(f"\n{'─'*40}")
    print(f"  {split_name} split  ({len(samples)} raw samples)")
    print(f"{'─'*40}")
    for label_id in sorted(counts):
        name = id_to_class.get(label_id, str(label_id))
        bar = "█" * counts[label_id]
        print(f"  [{label_id:>2}] {name:<10}  {counts[label_id]:>4}  {bar}")
    print(f"{'─'*40}")


def _process_partition(samples: list, output_path: str) -> None:
    """Extract features and augment for the given sample list, saving to output_path."""
    max_frames = cfg.dataset.max_frames
    target_length = cfg.dataset.target_length
    noise_file = cfg.dataset.noise_file

    X_audio, X_video, Y_labels, Y_metadata, Y_groups = [], [], [], [], []

    augmentation_configs = [
        # (tag, noise_path, snr_db, apply_jitter, apply_tilt)
        ("clean",             None,       50,   False, False),
        ("audio_light",       noise_file,  5,   False, False),
        ("audio_heavy",       noise_file, -15,  False, False),
        ("video_light",       None,       50,   True,  False),
        ("video_heavy",       None,       50,   False, True),
        ("audio_video_light", noise_file,  5,   True,  False),
    ]

    for sample in tqdm(samples, desc=f"Processing {os.path.basename(output_path)}"):
        try:
            y_clean, _ = librosa.load(sample["audio_path"], sr=cfg.audio.sample_rate)
        except Exception as exc:
            logger.warning("Failed to load audio %s: %s", sample["audio_path"], exc)
            continue

        for tag, noise_path, snr_db, apply_jitter, apply_tilt in augmentation_configs:
            try:
                a = pad_with_noise_canvas(y_clean, noise_path, snr_db, target_length)
                v = extract_video_features(sample["video_path"], apply_jitter, apply_tilt)
                if v is None or len(v) == 0:
                    continue
                X_audio.append(extract_audio_features(y=a, target_sr=cfg.audio.sample_rate))
                X_video.append(pad_or_truncate(v, max_frames))
                Y_labels.append(sample["label"])
                Y_metadata.append(tag)
                Y_groups.append(sample["group"])
            except Exception as exc:
                logger.warning("Augmentation '%s' failed for %s: %s", tag, sample["video_path"], exc)

    np.savez_compressed(
        output_path,
        X_audio=np.stack(X_audio, axis=0),
        X_video=np.stack(X_video, axis=0),
        Y_labels=np.array(Y_labels, dtype=np.int32),
        Y_metadata=np.array(Y_metadata, dtype=object),
        Y_groups=np.array(Y_groups, dtype=object),
    )
    logger.info("%s → %d samples saved.", output_path, len(Y_labels))
    print(f"[{output_path}] → {len(Y_labels)} samples saved.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_dataset(raw_dir=None) -> None:
    """
    Scan raw_dir for audio/video pairs, perform stratified group splits, extract
    features with augmentation, and save train/val/test NPZ files.

    Args:
        raw_dir: Path to the raw dataset directory. Defaults to cfg.dataset.raw_dir.
    """
    if raw_dir is None:
        raw_dir = cfg.dataset.raw_dir

    noise_file = cfg.dataset.noise_file
    if not os.path.exists(noise_file):
        raise FileNotFoundError(f"Babble noise file not found: {noise_file}")

    class_map = cfg.dataset.class_map
    all_samples = []

    for class_name in os.listdir(raw_dir):
        class_dir = os.path.join(raw_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        if class_name.lower() not in class_map:
            continue

        label_id = class_map[class_name.lower()]
        audio_dir = os.path.join(class_dir, "audio")
        video_dir = os.path.join(class_dir, "video")
        if not os.path.exists(audio_dir) or not os.path.exists(video_dir):
            continue

        for audio_path in glob.glob(os.path.join(audio_dir, "*.wav")):
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            video_path = next(
                (
                    os.path.join(video_dir, base_name + ext)
                    for ext in [".avi", ".mp4", ".mkv"]
                    if os.path.exists(os.path.join(video_dir, base_name + ext))
                ),
                None,
            )
            if video_path:
                all_samples.append(
                    {
                        "audio_path": audio_path,
                        "video_path": video_path,
                        "label": label_id,
                        "group": base_name,
                    }
                )

    print(f"Found {len(all_samples)} raw samples. Splitting with stratification...")

    X_dummy = list(range(len(all_samples)))
    y = [s["label"] for s in all_samples]
    groups = [s["group"] for s in all_samples]

    sgkf1 = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
    train_val_idx, test_idx = next(sgkf1.split(X_dummy, y, groups))

    train_val = [all_samples[i] for i in train_val_idx]
    test_samples = [all_samples[i] for i in test_idx]

    sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    y_tv = [s["label"] for s in train_val]
    groups_tv = [s["group"] for s in train_val]
    train_idx, val_idx = next(sgkf2.split(list(range(len(train_val))), y_tv, groups_tv))

    train_samples = [train_val[i] for i in train_idx]
    val_samples = [train_val[i] for i in val_idx]

    _print_split_distribution("TRAIN", train_samples)
    _print_split_distribution("VAL", val_samples)
    _print_split_distribution("TEST", test_samples)

    _process_partition(train_samples, cfg.dataset.train_file)
    _process_partition(val_samples, cfg.dataset.val_file)
    _process_partition(test_samples, cfg.dataset.test_file)
