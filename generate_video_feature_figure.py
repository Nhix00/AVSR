"""
Generates a 2-panel video feature extraction figure for presentation slide 8.
Left:  face frame with MediaPipe landmarks + annotated geometric measures
Right: time-series of the 9 kinematic features (raw, Δ, ΔΔ)
Output: report/assets/video_pipeline.png
"""

import sys
import cv2
import numpy as np
import mediapipe as mp
import librosa.feature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

VIDEO_PATH  = "dataset/Avvia/video/Avvia_001_20260220_130742.avi"
OUTPUT_PATH = "report/assets/video_pipeline.png"

# ── landmark indices (same as src/avsr/features.py) ──────────────────────────
LM = {
    "inner_top":    13,
    "inner_bot":    14,
    "outer_top":     0,
    "outer_bot":    17,
    "lip_left":     61,
    "lip_right":   291,
    "eye_left":     33,
    "eye_right":   263,
}

DARK_RED  = "#8B1A2B"
COL_RAW   = "#2E86AB"   # blue   – raw
COL_D1    = "#E84855"   # red    – delta
COL_D2    = "#3BB273"   # green  – delta-delta

LABELS_RAW = ["Inner aperture", "Outer aperture", "Lip width"]

# ── 1. Extract features from all frames ──────────────────────────────────────
mp_face = mp.solutions.face_mesh
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    sys.exit(f"Cannot open {VIDEO_PATH}")

rows, cols = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_feats = []
best_frame  = None          # frame with largest mouth aperture (most expressive)
best_score  = -1
best_lm     = None

with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                      refine_landmarks=True, min_detection_confidence=0.5) as fm:
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            frame_feats.append([0.0, 0.0, 0.0])
            continue
        lm = res.multi_face_landmarks[0].landmark

        def pt(idx):
            return np.array([lm[idx].x * cols, lm[idx].y * rows])

        eye_d  = np.linalg.norm(pt(LM["eye_left"]) - pt(LM["eye_right"])) + 1e-6
        inner  = np.linalg.norm(pt(LM["inner_top"]) - pt(LM["inner_bot"])) / eye_d
        outer  = np.linalg.norm(pt(LM["outer_top"]) - pt(LM["outer_bot"])) / eye_d
        width  = np.linalg.norm(pt(LM["lip_left"])  - pt(LM["lip_right"])) / eye_d
        frame_feats.append([inner, outer, width])

        score = inner + outer
        if score > best_score:
            best_score = score
            best_frame = rgb.copy()
            best_lm    = lm

cap.release()

feats  = np.array(frame_feats)                        # (T, 3)
d1     = librosa.feature.delta(feats, order=1, axis=0, width=3)
d2     = librosa.feature.delta(feats, order=2, axis=0, width=3)
all9   = np.hstack([feats, d1, d2])                   # (T, 9)
T      = len(feats)

# ── 2. Draw annotations on best frame ────────────────────────────────────────
def px(idx):
    return (int(best_lm[idx].x * cols), int(best_lm[idx].y * rows))

# Draw subtle full mesh
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles
ann = best_frame.copy()

# Draw face mesh connections in light gray
mp_draw.draw_landmarks(
    ann,
    type("FakeLM", (), {"landmark": best_lm})(),
    mp.solutions.face_mesh.FACEMESH_TESSELATION,
    landmark_drawing_spec=None,
    connection_drawing_spec=mp_draw.DrawingSpec(color=(180, 180, 180), thickness=1),
)

# Highlight lip contour
mp_draw.draw_landmarks(
    ann,
    type("FakeLM", (), {"landmark": best_lm})(),
    mp.solutions.face_mesh.FACEMESH_LIPS,
    landmark_drawing_spec=None,
    connection_drawing_spec=mp_draw.DrawingSpec(color=(220, 80, 50), thickness=2),
)

# ── draw the 3 measurement lines ──────────────────────────────────────────────
def draw_measure(img, idx_a, idx_b, color, thickness=2):
    cv2.line(img, px(idx_a), px(idx_b), color, thickness)
    # small circles at endpoints
    for idx in (idx_a, idx_b):
        cv2.circle(img, px(idx), 4, color, -1)

draw_measure(ann, LM["inner_top"], LM["inner_bot"], (255, 200,  50), 3)   # yellow – inner
draw_measure(ann, LM["outer_top"], LM["outer_bot"], ( 50, 200, 255), 3)   # cyan   – outer
draw_measure(ann, LM["lip_left"],  LM["lip_right"],  (50, 220, 100), 3)   # green  – width

# ── 3. Plot ──────────────────────────────────────────────────────────────────
fig, (ax_face, ax_ts) = plt.subplots(1, 2, figsize=(12, 3.8),
                                      gridspec_kw={"width_ratios": [1, 1.6]})

# Left – annotated face
ax_face.imshow(ann)
ax_face.axis("off")
ax_face.set_title("Geometric Measures on Face Mesh", fontsize=11, fontweight="bold", pad=8)
leg_face = [
    mpatches.Patch(color=np.array([255, 200, 50]) / 255, label="Inner aperture"),
    mpatches.Patch(color=np.array([ 50, 200, 255]) / 255, label="Outer aperture"),
    mpatches.Patch(color=np.array([ 50, 220, 100]) / 255, label="Lip width"),
]
ax_face.legend(handles=leg_face, loc="lower left", fontsize=8,
               framealpha=0.75, handlelength=1.2)

# Right – time series (3 groups: raw, Δ, ΔΔ)
t = np.arange(T)
name_map = ["Inner ap.", "Outer ap.", "Lip width"]
ls_raw = "-";  ls_d1 = "--";  ls_d2 = ":"
colors3 = [COL_RAW, "#E07B39", "#9B5DE5"]   # one colour per feature

for i in range(3):
    ax_ts.plot(t, all9[:, i],     color=colors3[i], lw=1.6, ls=ls_raw)
    ax_ts.plot(t, all9[:, 3 + i], color=colors3[i], lw=1.2, ls=ls_d1, alpha=0.8)
    ax_ts.plot(t, all9[:, 6 + i], color=colors3[i], lw=1.0, ls=ls_d2, alpha=0.6)

ax_ts.set_title(r"9 Kinematic Features per Frame  $(30 \times 9)$",
                fontsize=11, fontweight="bold", pad=8)
ax_ts.set_xlabel("Frame index", fontsize=9)
ax_ts.set_ylabel("Normalised value", fontsize=9)
ax_ts.tick_params(labelsize=8)
ax_ts.spines[["top", "right"]].set_visible(False)

legend_elems = (
    [mpatches.Patch(color=c, label=n) for c, n in zip(colors3, name_map)] +
    [Line2D([0], [0], color="gray", lw=1.6, ls=ls_raw,  label="raw"),
     Line2D([0], [0], color="gray", lw=1.2, ls=ls_d1,   label="Δ  velocity"),
     Line2D([0], [0], color="gray", lw=1.0, ls=ls_d2,   label="ΔΔ accel.")]
)
ax_ts.legend(handles=legend_elems, fontsize=8, ncol=2,
             loc="upper right", framealpha=0.75, handlelength=1.6)

plt.tight_layout(pad=1.2)
fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUTPUT_PATH}")
