"""Audio and video feature extraction using MediaPipe and Librosa."""
import logging
import math
import os
import urllib.request
import warnings

import cv2
import librosa
import mediapipe as mp
import numpy as np

from avsr.augmentations import add_noise, spatial_jittering
from avsr.config import cfg

logger = logging.getLogger(__name__)

# MediaPipe Face Mesh landmark indices
_OUTER_LIP_TOP = 0
_OUTER_LIP_BOTTOM = 17
_INNER_LIP_TOP = 13
_INNER_LIP_BOTTOM = 14
_MOUTH_LEFT = 61
_MOUTH_RIGHT = 291
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263

_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_FACE_LANDMARKER_PATH = os.path.join("models", "face_landmarker.task")


def _get_face_landmarker_path() -> str:
    """Return path to face_landmarker.task, downloading it if absent."""
    if not os.path.exists(_FACE_LANDMARKER_PATH):
        os.makedirs("models", exist_ok=True)
        logger.info("Downloading MediaPipe face landmarker model...")
        urllib.request.urlretrieve(_FACE_LANDMARKER_URL, _FACE_LANDMARKER_PATH)
        logger.info("Downloaded face_landmarker.task")
    return _FACE_LANDMARKER_PATH


def _euclidean(p1, p2) -> float:
    """Euclidean distance between two 2D points (MediaPipe landmarks or numpy arrays)."""
    if hasattr(p1, "x"):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def extract_video_features(
    video_path: str,
    apply_jitter: bool = False,
    apply_tilt: bool = False,
) -> np.ndarray:
    """
    Extract geometric mouth features from a video using MediaPipe Face Mesh.

    Output features per frame: [inner_aperture, outer_aperture, mouth_width]
    (normalized by inter-eye distance), plus their first- and second-order
    temporal derivatives — giving shape (num_frames, 9).

    Args:
        video_path: Path to the video file.
        apply_jitter: Apply spatial jittering augmentation to landmarks.
        apply_tilt: Apply head-tilt rotation augmentation to landmarks.

    Returns:
        np.ndarray of shape (num_frames, 9).

    Raises:
        FileNotFoundError: If the video file does not exist.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_get_face_landmarker_path()),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg.video.fps
    features_list = []

    if apply_tilt:
        angle_deg = np.random.uniform(-5.0, 5.0)
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[c, -s], [s, c]])

    last_valid = np.zeros(3)
    frame_idx = 0

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as face_landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000.0 / fps)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.face_landmarks:
                lm = results.face_landmarks[0]
                pts = np.array(
                    [
                        [lm[_OUTER_LIP_TOP].x, lm[_OUTER_LIP_TOP].y],
                        [lm[_OUTER_LIP_BOTTOM].x, lm[_OUTER_LIP_BOTTOM].y],
                        [lm[_INNER_LIP_TOP].x, lm[_INNER_LIP_TOP].y],
                        [lm[_INNER_LIP_BOTTOM].x, lm[_INNER_LIP_BOTTOM].y],
                        [lm[_MOUTH_LEFT].x, lm[_MOUTH_LEFT].y],
                        [lm[_MOUTH_RIGHT].x, lm[_MOUTH_RIGHT].y],
                        [lm[_LEFT_EYE_OUTER].x, lm[_LEFT_EYE_OUTER].y],
                        [lm[_RIGHT_EYE_OUTER].x, lm[_RIGHT_EYE_OUTER].y],
                    ]
                )

                if apply_jitter:
                    pts = spatial_jittering(pts, mean=0.0, std=0.01)
                if apply_tilt:
                    centroid = np.mean(pts, axis=0)
                    pts = np.dot(pts - centroid, rotation_matrix.T) + centroid

                inner = _euclidean(pts[2], pts[3])
                outer = _euclidean(pts[0], pts[1])
                width = _euclidean(pts[4], pts[5])
                norm_dist = _euclidean(pts[6], pts[7])

                if norm_dist > 0:
                    frame_feat = np.array([inner / norm_dist, outer / norm_dist, width / norm_dist])
                else:
                    frame_feat = last_valid.copy()

                last_valid = frame_feat
            else:
                frame_feat = last_valid.copy()

            features_list.append(frame_feat)
            frame_idx += 1

    cap.release()

    features_array = np.array(features_list)  # (num_frames, 3)

    if len(features_array) > 4:
        delta1 = librosa.feature.delta(features_array, order=1, axis=0, width=3)
        delta2 = librosa.feature.delta(features_array, order=2, axis=0, width=3)
        return np.hstack((features_array, delta1, delta2))  # (num_frames, 9)

    return np.pad(features_array, ((0, 0), (0, 6)), mode="constant")


def extract_audio_features(
    audio_path: str = None,
    y: np.ndarray = None,
    target_fps: int = None,
    target_sr: int = None,
    num_frames: int = None,
    apply_noise_snr: float = None,
) -> np.ndarray:
    """
    Extract MFCCs from an audio file or pre-loaded array.

    The hop_length is calculated dynamically to synchronize with video FPS.

    Args:
        audio_path: Path to the audio file (used if y is None).
        y: Pre-loaded audio array (takes precedence over audio_path).
        target_fps: Video FPS for hop_length synchronisation. Defaults to cfg.video.fps.
        target_sr: Target sampling rate. Defaults to cfg.audio.sample_rate.
        num_frames: If set, output is padded/truncated to this many frames.
        apply_noise_snr: If set, adds white noise at this SNR (dB) before extraction.

    Returns:
        np.ndarray of shape (num_frames, n_mfcc) or (T, n_mfcc) if num_frames is None.

    Raises:
        FileNotFoundError: If audio_path is given but does not exist.
    """
    sr = target_sr or cfg.audio.sample_rate
    fps = target_fps or cfg.video.fps
    n_mfcc = cfg.audio.n_mfcc

    if y is None:
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        y, _ = librosa.load(audio_path, sr=sr)

    if apply_noise_snr is not None:
        noise = np.random.randn(len(y)).astype(np.float32)
        y = add_noise(y, noise, apply_noise_snr)

    hop_length = int(sr // fps)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length).T

    if num_frames is not None:
        if len(mfccs) > num_frames:
            mfccs = mfccs[:num_frames]
        elif len(mfccs) < num_frames:
            mfccs = np.pad(mfccs, ((0, num_frames - len(mfccs)), (0, 0)), mode="constant")

    return mfccs
