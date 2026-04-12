"""
Generates a 3-panel MFCC pipeline figure for the presentation slide 7.
Output: report/assets/mfcc_pipeline.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display

AUDIO_PATH = "dataset/Avvia/audio/Avvia_001_20260220_130742.wav"
OUTPUT_PATH = "report/assets/mfcc_pipeline.png"

N_MFCC = 13
N_FRAMES = 30
SR_TARGET = 16000

# --- Load and resample ---
y, sr = librosa.load(AUDIO_PATH, sr=SR_TARGET)

# --- Mel spectrogram (for middle panel) ---
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# --- MFCCs (13 coefficients, padded/trimmed to 30 frames) ---
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
# Pad or trim to exactly 30 frames
if mfccs.shape[1] < N_FRAMES:
    mfccs = np.pad(mfccs, ((0, 0), (0, N_FRAMES - mfccs.shape[1])), mode="constant")
else:
    mfccs = mfccs[:, :N_FRAMES]

# --- Time axis for waveform ---
times = np.linspace(0, len(y) / sr, len(y))

# ---- Plot ----------------------------------------------------------------
fig = plt.figure(figsize=(12, 3.2))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

DARK_RED = "#8B1A2B"
CMAP_MAIN = "magma"

# Panel 1 — Waveform
ax1 = fig.add_subplot(gs[0])
ax1.plot(times * 1000, y, color=DARK_RED, linewidth=0.7)
ax1.set_title("1  Raw Waveform", fontsize=11, fontweight="bold", pad=6)
ax1.set_xlabel("Time (ms)", fontsize=9)
ax1.set_ylabel("Amplitude", fontsize=9)
ax1.set_xlim(0, times[-1] * 1000)
ax1.tick_params(labelsize=8)
ax1.spines[["top", "right"]].set_visible(False)

# Panel 2 — Mel Filterbank (log-mel spectrogram)
ax2 = fig.add_subplot(gs[1])
img2 = librosa.display.specshow(
    mel_spec_db, sr=sr, x_axis="time", y_axis="mel",
    fmax=8000, ax=ax2, cmap=CMAP_MAIN
)
ax2.set_title("2  Mel Filterbank", fontsize=11, fontweight="bold", pad=6)
ax2.set_xlabel("Time (s)", fontsize=9)
ax2.set_ylabel("Frequency (Hz)", fontsize=9)
ax2.tick_params(labelsize=8)
fig.colorbar(img2, ax=ax2, format="%+2.0f dB", pad=0.02).ax.tick_params(labelsize=7)

# Panel 3 — MFCC heatmap  (30 × 13)
ax3 = fig.add_subplot(gs[2])
img3 = ax3.imshow(
    mfccs, aspect="auto", origin="lower",
    cmap=CMAP_MAIN, interpolation="nearest"
)
ax3.set_title(r"3  MFCCs  $(30 \times 13)$", fontsize=11, fontweight="bold", pad=6)
ax3.set_xlabel("Frame index", fontsize=9)
ax3.set_ylabel("MFCC coefficient", fontsize=9)
ax3.set_xticks(np.arange(0, N_FRAMES + 1, 5))
ax3.set_yticks(np.arange(N_MFCC))
ax3.set_yticklabels([str(i) for i in range(N_MFCC)], fontsize=7)
ax3.tick_params(labelsize=8)
fig.colorbar(img3, ax=ax3, pad=0.02).ax.tick_params(labelsize=7)

# Arrows between panels
fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")  # dummy save to fix positions
for ax_src, ax_dst in [(ax1, ax2), (ax2, ax3)]:
    x_src = ax_src.get_position().x1
    x_dst = ax_dst.get_position().x0
    ym = (ax_src.get_position().y0 + ax_src.get_position().y1) / 2
    fig.add_artist(
        plt.matplotlib.patches.FancyArrowPatch(
            (x_src + 0.005, ym), (x_dst - 0.005, ym),
            transform=fig.transFigure,
            arrowstyle="->", color="gray", lw=1.5,
            mutation_scale=14,
        )
    )

fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUTPUT_PATH}")
