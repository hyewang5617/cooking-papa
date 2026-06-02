"""
White silhouette hand avatar drawn from MediaPipe landmarks.
Open hand → white/blue. Gripped → green tint.
"""
import cv2
import numpy as np

# Finger segments: each sub-list is a chain of landmark indices
_FINGER_SEGS = [
    [0, 1, 2, 3, 4],    # thumb
    [5, 6, 7, 8],       # index
    [9, 10, 11, 12],    # middle
    [13, 14, 15, 16],   # ring
    [17, 18, 19, 20],   # pinky
]
# Palm outline landmark indices
_PALM_IDX = [0, 1, 2, 5, 9, 13, 17]

# Fingertip indices (for highlighting when open)
_TIP_IDX = [4, 8, 12, 16, 20]


def draw_all(frame, hand_states, scale=1.0):
    """Draw a white silhouette avatar for every detected hand."""
    for hs in hand_states:
        if hs.detected and hs.landmarks:
            _draw_one(frame, hs, scale=scale)


def _draw_one(frame, hand_state, scale=1.0):
    lms     = hand_state.landmarks
    gripped = hand_state.gripped
    H, W    = frame.shape[:2]

    # Map landmarks to full display pixels
    pts_full = [(int(lm.x * W), int(lm.y * H)) for lm in lms]

    # Scale the drawing around the hand's centroid
    if scale != 1.0:
        cx = sum(p[0] for p in pts_full) / len(pts_full)
        cy = sum(p[1] for p in pts_full) / len(pts_full)
        pts = [(int(cx + (p[0] - cx) * scale),
                int(cy + (p[1] - cy) * scale)) for p in pts_full]
    else:
        pts = pts_full

    # Colors
    if gripped:
        fill    = (160, 255, 160)
        outline = (30,  100,  30)
    else:
        fill    = (235, 235, 255)
        outline = (60,  60,   90)

    # Scaled stroke / radius sizes
    s_out  = max(2, int(20 * scale))
    s_fill = max(1, int(13 * scale))
    s_tip  = max(2, int(9  * scale))
    s_kn   = max(1, int(6  * scale))
    pad    = max(6, int(35 * scale))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(W, max(xs) + pad)
    y2 = min(H, max(ys) + pad)
    if x2 <= x1 or y2 <= y1:
        return

    rw, rh = x2 - x1, y2 - y1
    canvas = np.zeros((rh, rw, 4), dtype=np.uint8)

    lp = [(p[0] - x1, p[1] - y1) for p in pts]
    oc = (*outline, 230)
    fc = (*fill,    210)

    # ── 1) Dark outline ───────────────────────────────────────
    for seg in _FINGER_SEGS:
        for i in range(len(seg) - 1):
            cv2.line(canvas, lp[seg[i]], lp[seg[i+1]], oc, s_out)
    palm = np.array([lp[i] for i in _PALM_IDX])
    cv2.fillPoly(canvas, [palm], oc)

    # ── 2) Light fill ─────────────────────────────────────────
    for seg in _FINGER_SEGS:
        for i in range(len(seg) - 1):
            cv2.line(canvas, lp[seg[i]], lp[seg[i+1]], fc, s_fill)
    cv2.fillPoly(canvas, [palm], fc)

    # ── 3) Fingertip circles ──────────────────────────────────
    for t in _TIP_IDX:
        cv2.circle(canvas, lp[t], s_tip, fc, -1)
        cv2.circle(canvas, lp[t], s_tip, oc, max(1, s_out // 5))

    # ── 4) Knuckle joints ────────────────────────────────────
    for k in [1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]:
        cv2.circle(canvas, lp[k], s_kn, fc, -1)

    # ── 5) Alpha-blend onto frame ────────────────────────────
    roi = frame[y1:y2, x1:x2]
    a   = canvas[:, :, 3:4].astype(np.float32) / 255.0
    roi[:] = (canvas[:, :, :3].astype(np.float32) * a +
              roi.astype(np.float32) * (1.0 - a)).astype(np.uint8)
