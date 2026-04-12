"""
Slide 11 — Bias #2: Padding Leakage
Left:  MFCC heatmap with zero-padding (silence visible at end)
Right: MFCC heatmap with Continuous Noise Canvas (random offset, no silence)
Output: report/assets/padding_leakage.png
"""

import numpy as np
import librosa
import librosa.feature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

AUDIO_PATH  = "dataset/Avvia/audio/Avvia_001_20260220_130742.wav"
OUTPUT_PATH = "report/assets/padding_leakage.png"

N_MFCC    = 13
N_FRAMES  = 30
SR        = 16000
NOISE_STD = 0.008
RNG       = np.random.default_rng(7)

# ── load audio ───────────────────────────────────────────────────────────────
y, sr = librosa.load(AUDIO_PATH, sr=SR)

def to_mfcc(signal):
    m = librosa.feature.mfcc(y=signal, sr=SR, n_mfcc=N_MFCC)
    if m.shape[1] < N_FRAMES:
        m = np.pad(m, ((0, 0), (0, N_FRAMES - m.shape[1])), mode="constant")
    else:
        m = m[:, :N_FRAMES]
    return m

# ── Panel A: zero-padded ──────────────────────────────────────────────────────
samples_per_frame = int(sr / 100)   # ~160 samples @ 16 kHz / 100 fps
speech_frames = int(np.ceil(len(y) / samples_per_frame))
speech_frames = min(speech_frames, N_FRAMES)

y_padded = np.zeros(N_FRAMES * samples_per_frame)
y_padded[:len(y)] = y[:N_FRAMES * samples_per_frame]
mfcc_padded = to_mfcc(y_padded)

# ── Panel B: noise canvas with random offset ──────────────────────────────────
total_samples = N_FRAMES * samples_per_frame
noise_canvas  = RNG.normal(0, NOISE_STD, total_samples).astype(np.float32)
max_offset    = total_samples - len(y)
offset        = RNG.integers(0, max(1, max_offset // 2))   # left-biased for clarity
end           = min(offset + len(y), total_samples)
noise_canvas[offset:end] += y[:end - offset]
mfcc_noisy = to_mfcc(noise_canvas)

# ── plot ─────────────────────────────────────────────────────────────────────
CMAP   = "magma"
vmin   = min(mfcc_padded.min(), mfcc_noisy.min())
vmax   = max(mfcc_padded.max(), mfcc_noisy.max())

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"wspace": 0.35})

titles = [
    "Zero-Padding  ✗\n(LSTM can count silent frames)",
    "Continuous Noise Canvas  ✓\n(duration informationally invisible)",
]
title_colors = ["#C44E52", "#2A7A4F"]
data = [mfcc_padded, mfcc_noisy]

for ax, d, title, tc in zip(axes, data, titles, title_colors):
    im = ax.imshow(d, aspect="auto", origin="lower",
                   cmap=CMAP, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlabel("Frame index", fontsize=9)
    ax.set_ylabel("MFCC coeff.", fontsize=9)
    ax.set_xticks(np.arange(0, N_FRAMES + 1, 5))
    ax.set_yticks(np.arange(0, N_MFCC, 3))
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=tc, pad=7)
    fig.colorbar(im, ax=ax, pad=0.02).ax.tick_params(labelsize=7)

# Annotate zero-pad region on left panel
pad_start = speech_frames
axes[0].axvline(x=pad_start - 0.5, color="#F5C518", lw=2, linestyle="--")
axes[0].text(pad_start + 0.3, N_MFCC / 2, "← silence\n   (zeros)",
             color="#F5C518", fontsize=8, va="center", fontweight="bold")

# Annotate speech region on right panel
axes[1].axvline(x=offset / samples_per_frame - 0.5, color="#7FD88A", lw=2, ls="--")
axes[1].axvline(x=min(offset / samples_per_frame + speech_frames, N_FRAMES) - 0.5,
                color="#7FD88A", lw=2, ls="--")
mid = offset / samples_per_frame + speech_frames / 2
axes[1].text(mid, N_MFCC - 1.5, "speech at\nrandom offset",
             color="#7FD88A", fontsize=8, ha="center", fontweight="bold")

fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUTPUT_PATH}")
