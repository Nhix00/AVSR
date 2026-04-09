"""Audio and video augmentation utilities."""
import numpy as np


def add_noise(audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Mix clean audio with a background noise signal at a specific SNR in decibels.

    Args:
        audio: 1D array of the clean audio signal.
        noise: 1D array of the background noise signal.
        snr_db: Target Signal-to-Noise Ratio (SNR) in dB.

    Returns:
        Noisy audio signal preserving the original length.
    """
    audio_len = len(audio)
    noise_len = len(noise)

    if noise_len < audio_len:
        repeats = int(np.ceil(audio_len / noise_len))
        noise = np.tile(noise, repeats)
        noise_len = len(noise)

    start_idx = np.random.randint(0, noise_len - audio_len + 1)
    noise = noise[start_idx : start_idx + audio_len]

    audio_rms = np.sqrt(np.mean(audio**2) + 1e-8)
    noise_rms = np.sqrt(np.mean(noise**2) + 1e-8)

    if audio_rms <= 1e-4 or noise_rms <= 1e-4:
        return audio.copy()

    target_noise_rms = audio_rms / (10 ** (snr_db / 20))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    noisy_audio = audio + scaled_noise
    return np.clip(noisy_audio, -1.0, 1.0)


def random_gain(audio: np.ndarray, min_gain: float = 0.8, max_gain: float = 1.2) -> np.ndarray:
    """
    Randomly scale the amplitude of the audio signal.

    Args:
        audio: 1D array of the audio signal.
        min_gain: Minimum gain factor.
        max_gain: Maximum gain factor.

    Returns:
        Amplitude-shifted audio signal.
    """
    return audio * np.random.uniform(min_gain, max_gain)


def spatial_jittering(landmarks: np.ndarray, mean: float = 0.0, std: float = 0.01) -> np.ndarray:
    """
    Add Gaussian noise to landmark coordinates to simulate tracking imperfections.

    Args:
        landmarks: Array of coordinates (any shape).
        mean: Mean of the Gaussian distribution.
        std: Standard deviation of the Gaussian distribution.

    Returns:
        Jittered coordinates preserving original shape.
    """
    return landmarks + np.random.normal(loc=mean, scale=std, size=landmarks.shape)


def head_tilt(landmarks: np.ndarray, min_angle: float = -5.0, max_angle: float = 5.0) -> np.ndarray:
    """
    Apply a small 2D rotation to (x, y) landmark coordinates.

    Args:
        landmarks: Array of shape (..., 2).
        min_angle: Minimum rotation angle in degrees.
        max_angle: Maximum rotation angle in degrees.

    Returns:
        Rotated coordinates preserving original shape.
    """
    angle = np.random.uniform(min_angle, max_angle)
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]])

    original_shape = landmarks.shape
    landmarks_2d = landmarks.reshape(-1, 2)
    centroid = np.mean(landmarks_2d, axis=0)
    centered = landmarks_2d - centroid
    rotated = np.dot(centered, rotation_matrix.T) + centroid
    return rotated.reshape(original_shape)
