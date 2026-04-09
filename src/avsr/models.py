"""Keras model factories for audio-only, video-only, and fusion AVSR."""
import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Dense,
    Dropout,
    Input,
    LSTM,
    Masking,
)
from tensorflow.keras.models import Model

from avsr.config import cfg


def build_audio_model(
    input_shape: tuple = (30, 13),
    num_classes: int = None,
) -> Model:
    """
    Build the audio-only LSTM baseline.

    Args:
        input_shape: (timesteps, n_mfcc).
        num_classes: Output classes. Defaults to cfg.dataset.num_classes.

    Returns:
        Compiled-ready Keras Model.
    """
    if num_classes is None:
        num_classes = cfg.dataset.num_classes

    inputs = Input(shape=input_shape, name="audio_input")
    x = BatchNormalization()(inputs)
    x = LSTM(cfg.model.lstm_units)(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation="softmax", name="audio_output")(x)
    return Model(inputs=inputs, outputs=outputs, name="Audio_Only_Baseline")


def build_video_model(
    input_shape: tuple = (30, 9),
    num_classes: int = None,
) -> Model:
    """
    Build the video-only bidirectional LSTM baseline.

    Uses Masking to ignore zero-padded frames.

    Args:
        input_shape: (timesteps, feature_dim).
        num_classes: Output classes. Defaults to cfg.dataset.num_classes.

    Returns:
        Compiled-ready Keras Model.
    """
    if num_classes is None:
        num_classes = cfg.dataset.num_classes

    inputs = Input(shape=input_shape, name="video_input")
    x = Masking(mask_value=0.0)(inputs)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(cfg.model.lstm_units))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax", name="video_output")(x)
    return Model(inputs=inputs, outputs=outputs, name="Video_Only_Baseline")


def build_fusion_model(
    audio_shape: tuple = (30, 13),
    video_shape: tuple = (30, 9),
    num_classes: int = None,
) -> Model:
    """
    Build the early-fusion multimodal LSTM model.

    Concatenates audio and video LSTM branch outputs before the classifier.

    Args:
        audio_shape: (timesteps, n_mfcc).
        video_shape: (timesteps, video_feature_dim).
        num_classes: Output classes. Defaults to cfg.dataset.num_classes.

    Returns:
        Compiled-ready Keras Model.
    """
    if num_classes is None:
        num_classes = cfg.dataset.num_classes

    audio_input = Input(shape=audio_shape, name="audio_input")
    x_audio = BatchNormalization()(audio_input)
    audio_out = LSTM(cfg.model.lstm_units)(x_audio)

    video_input = Input(shape=video_shape, name="video_input")
    x_video = Masking(mask_value=0.0)(video_input)
    x_video = BatchNormalization()(x_video)
    x_video = Bidirectional(LSTM(cfg.model.lstm_units))(x_video)
    video_out = Dropout(0.5)(x_video)

    merged = Concatenate()([audio_out, video_out])
    x = Dense(cfg.model.lstm_units, activation="relu")(merged)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation="softmax", name="fusion_output")(x)
    return Model(inputs=[audio_input, video_input], outputs=outputs, name="Multimodal_Fusion")
