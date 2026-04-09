"""
Evaluate pre-trained audio, video, and fusion models on the test set.

Prints the granular condition-accuracy table and saves confusion matrix plots.

Run: python scripts/evaluate.py [--norm_stats dataset_train.npz]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import numpy as np
import tensorflow as tf

from avsr.config import cfg
from avsr.train import (
    load_data,
    normalize_video,
    evaluate_on_conditions,
    plot_confusion_matrix,
    _print_results_table,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse():
    parser = argparse.ArgumentParser(description="Evaluate trained AVSR models.")
    parser.add_argument(
        "--norm_stats",
        default=cfg.dataset.train_file,
        metavar="PATH",
        help=f"NPZ used to compute video Z-score stats (default: {cfg.dataset.train_file}).",
    )
    return parser.parse_args()


def main():
    args = _parse()

    print(">>> Loading test data")
    X_audio_test, X_video_test, Y_test, meta_test = load_data(cfg.dataset.test_file)

    print(">>> Computing video normalisation stats from", args.norm_stats)
    X_audio_train, X_video_train, _, _ = load_data(args.norm_stats)
    valid = np.any(X_video_train != 0, axis=-1)
    video_mean = X_video_train[valid].mean(axis=0)
    video_std = X_video_train[valid].std(axis=0)
    video_std[video_std == 0] = 1.0
    X_video_test = normalize_video(X_video_test, video_mean, video_std)

    print(">>> Loading models")
    audio_model = tf.keras.models.load_model("models/audio_model.keras")
    video_model = tf.keras.models.load_model("models/video_model.keras")
    fusion_model = tf.keras.models.load_model("models/fusion_model.keras")

    conditions = ["clean", "audio_light", "audio_heavy", "video_light", "video_heavy", "audio_video_light"]

    print(">>> Evaluating")
    audio_res = evaluate_on_conditions(audio_model, X_audio_test, Y_test, meta_test, conditions)
    video_res = evaluate_on_conditions(video_model, X_video_test, Y_test, meta_test, conditions)
    fusion_res = evaluate_on_conditions(
        fusion_model, [X_audio_test, X_video_test], Y_test, meta_test, conditions, is_fusion=True
    )

    _print_results_table(audio_res, video_res, fusion_res, conditions)

    print(">>> Saving confusion matrices to results/")
    class_names = [k for k, _ in sorted(
        {k: v for k, v in cfg.dataset.class_map.items() if k != "sì"}.items(),
        key=lambda kv: kv[1],
    )]
    plot_confusion_matrix(audio_model, X_audio_test, Y_test, class_names,
                          "Audio Model", "results/eval_audio_confusion_matrix.png")
    plot_confusion_matrix(video_model, X_video_test, Y_test, class_names,
                          "Video Model", "results/eval_video_confusion_matrix.png")
    plot_confusion_matrix(fusion_model, [X_audio_test, X_video_test], Y_test, class_names,
                          "Fusion Model", "results/eval_fusion_confusion_matrix.png")
    print("Done.")


if __name__ == "__main__":
    main()
