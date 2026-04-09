# AVSR — Audio-Visual Keyword Spotting

**Course:** Multimodal Interaction · Sapienza University of Rome
**Type:** Individual Academic Project
**Branch:** `refactor/delivery`

> A multimodal system for real-time keyword spotting that fuses audio (MFCC)
> and lip-motion (MediaPipe landmarks) streams using LSTM-based early fusion.
> The core research question: does visual lip information compensate for audio
> degradation in noisy environments?

Speech recognition degrades sharply under acoustic noise. Lip movement, however, is
entirely noise-invariant. This project implements and evaluates three models — audio-only,
video-only, and an early-fusion multimodal model — to quantify the robustness gain
achieved by combining both modalities.

---

## System Architecture

Three models were trained and compared on the same dataset and test conditions:

| Model | Input | Architecture | Output |
|---|---|---|---|
| Audio-Only LSTM | MFCCs `(30, 13)` | BatchNorm → LSTM(64) → Dropout | Softmax(10) |
| Video-Only BiLSTM | Lip landmarks `(30, 9)` | Masking → BatchNorm → BiLSTM(64) → Dropout | Softmax(10) |
| Early Fusion | Audio + Video | Two branches → Concatenate → Dense(64) → Dropout | Softmax(10) |

### Pipeline

```
Webcam + Microphone
       │
       ▼
  Preprocessing
  ├── Audio: 44.1 kHz WAV → 16 kHz → 13 MFCCs × 30 timesteps
  └── Video: 640×480 AVI → MediaPipe FaceMesh → 40 lip landmarks × 30 frames
       │
       ▼
  Feature Extraction  (src/avsr/features.py)
       │
       ▼
  Model Inference
  ├── Audio branch  →─┐
  └── Video branch  →─┴─ Concatenate → Classifier
       │
       ▼
  Predicted Label + Per-modality Confidence
```

Architecture diagrams: [`results/arch_audio.png`](results/arch_audio.png) · [`results/arch_video.png`](results/arch_video.png) · [`results/arch_fusion.png`](results/arch_fusion.png)

---

## Dataset

- **Vocabulary:** 10 Italian keywords — *Avvia, Stop, Sopra, Sotto, Sinistra, Destra, Apri, Chiudi, Sì, No*
- **Size:** 200 samples (20 per keyword), 2 seconds each, custom-recorded
- **Audio:** 44.1 kHz mono WAV, downsampled to 16 kHz for MFCC extraction (13 coefficients)
- **Video:** 640×480 @ 15 FPS AVI, processed through MediaPipe FaceMesh — 3 geometric lip measurements (inner aperture, outer aperture, width) + Δ + ΔΔ = 9 features per frame
- **Split:** Two-stage Stratified Group K-Fold (n_splits=7 for test, n_splits=5 for val), stored as `.npz` files

### Noise Augmentation Conditions

Six conditions were generated to evaluate noise robustness:

| Condition | Description |
|---|---|
| `clean` | No augmentation |
| `audio_light` | Low-level café babble noise added to audio |
| `audio_heavy` | High-level café babble noise added to audio |
| `video_light` | Mild spatial jitter applied to lip landmarks |
| `video_heavy` | Strong spatial jitter applied to lip landmarks |
| `audio_video_light` | Light noise on both modalities simultaneously |

Capture was performed via `data_collection.py`, which uses multithreaded AV recording with
millisecond-level audio-video alignment (buffer flushing at start, post-hoc trimming/padding).

---

## Results

Evaluated on the held-out test set across all six noise conditions:

| Model | Overall | Clean | Audio Light | Audio Heavy | Video Light | Video Heavy | A+V Light |
|---|---|---|---|---|---|---|---|
| Audio-Only LSTM | 90.23% | 100.00% | 100.00% | 44.83% | 100.00% | 100.00% | 96.55% |
| Video-Only BiLSTM | 77.01% | 96.55% | 96.55% | 96.55% | 41.38% | 96.55% | 34.48% |
| **Early Fusion** | **96.55%** | **100.00%** | **100.00%** | **86.21%** | **96.55%** | **100.00%** | **96.55%** |

Under heavy audio noise, the audio-only model collapses from 100% to **44.83%** (−55 pp).
The fusion model recovers to **86.21%** (+41 pp over audio-only), confirming that visual
lip information effectively compensates for acoustic degradation.

The video-only model is unaffected by audio noise but degrades significantly under visual
perturbation (41.38% under `video_heavy`), highlighting complementary failure modes that
fusion resolves.

Confusion matrices: [`results/audio_confusion_matrix.png`](results/audio_confusion_matrix.png) · [`results/video_confusion_matrix.png`](results/video_confusion_matrix.png) · [`results/fusion_confusion_matrix.png`](results/fusion_confusion_matrix.png)

---

## Data & Models

The dataset, preprocessed `.npz` files, noise file, and trained models are available on Google Drive:

**[MI Project — Drive folder](https://drive.google.com/drive/folders/1VsOBelcmCs962XBYAYxhioPDbDpqKcPc?usp=sharing)**

| File / Folder | Purpose |
|---|---|
| `dataset/` | Raw WAV + MP4 recordings (required for `preprocess.py`) |
| `cafe_sound.wav` | Babble noise used during augmentation |
| `dataset_train/val/test.npz` | Preprocessed features (skip steps 2–3 below) |
| `models/*.keras` | Trained model weights (skip steps 2–4 below) |

Place downloaded files in the project root before running any command.

---

## Quickstart

### 1. Install dependencies

```bash
bash install_dependencies.sh
```

Or manually:

```bash
pip install -r requirements.txt
```

### 2. Preprocess raw dataset

```bash
python preprocess.py
```

Reads `dataset/` → outputs `dataset_train.npz`, `dataset_val.npz`, `dataset_test.npz`.

### 3. Train all three models

```bash
python train.py
```

Trains audio-only, video-only, and fusion models. Saves weights to `models/` and plots to `results/`.

### 4. Run the real-time demo

```bash
python demo.py
```

Opens webcam and microphone. Press space to record a 2-second keyword. Displays predicted label and per-modality confidence. Inference mode is configurable in `config.yaml` (`audio`, `video`, or `fusion`).

### 5. Evaluate on test set

```bash
python scripts/evaluate.py
```

Prints the full condition-accuracy comparison table and saves confusion matrices to `results/`.

---

## Project Structure

```
MI_project/
├── config.yaml              # all hyperparameters, paths, and class map
├── train.py                 # entrypoint → avsr.train.run()
├── demo.py                  # entrypoint → avsr.inference.run()
├── preprocess.py            # entrypoint → avsr.preprocessing.build_dataset()
├── data_collection.py       # synchronized AV recording tool
├── requirements.txt
├── src/
│   └── avsr/
│       ├── config.py        # loads config.yaml, exposes cfg singleton
│       ├── features.py      # MFCC + MediaPipe lip landmark extraction
│       ├── augmentations.py # noise and spatial augmentation utilities
│       ├── preprocessing.py # raw dataset → .npz pipeline
│       ├── models.py        # Keras model factories (audio, video, fusion)
│       ├── train.py         # training loop, evaluation, plotting
│       └── inference.py     # real-time push-to-talk demo loop
├── scripts/
│   ├── evaluate.py          # full condition-accuracy evaluation table
│   └── inspect_dataset.py   # dataset shape and class balance check
├── models/                  # saved .keras model weights
├── results/                 # confusion matrices, learning curves, arch diagrams
└── report/                  # LaTeX source and compiled PDF report
```

---

## Configuration

All tuneable parameters are centralized in [`config.yaml`](config.yaml):

```yaml
audio:
  sample_rate: 16000
  n_mfcc: 13

video:
  fps: 15

model:
  lstm_units: 64
  epochs: 100
  batch_size: 32
  patience: 15

inference:
  mode: "fusion"        # audio | video | fusion
  record_seconds: 2.0
```

---

## Tech Stack

| Component | Library |
|---|---|
| Deep Learning | TensorFlow 2.13 / Keras |
| Audio Features | Librosa |
| Lip Landmarks | MediaPipe FaceMesh |
| Video Capture | OpenCV |
| Real-time Audio | PyAudio |
| Data | NumPy, Pandas |
