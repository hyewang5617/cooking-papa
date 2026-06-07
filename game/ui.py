import cv2
import math as _ui_math
import numpy as np

FONT = cv2.FONT_HERSHEY_DUPLEX
COLOR_PRIMARY = (30, 190, 255)
COLOR_SUCCESS = (60, 220, 60)
COLOR_DANGER  = (60, 60, 220)
COLOR_WHITE   = (255, 255, 255)
COLOR_GREY    = (150, 150, 150)


def draw_text(frame, text, pos, scale=1.0, color=COLOR_WHITE, thickness=2):
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1), FONT, scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness)


def draw_text_centered(frame, text, y, scale=1.0, color=COLOR_WHITE, thickness=2):
    size = cv2.getTextSize(text, FONT, scale, thickness)[0]
    x = (frame.shape[1] - size[0]) // 2
    draw_text(frame, text, (x, y), scale, color, thickness)


def draw_progress_bar(frame, x, y, w, h, progress, color=COLOR_SUCCESS):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
    fill = int(w * max(0.0, min(1.0, progress)))
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_WHITE, 1)


def draw_panel(frame, x, y, w, h, alpha=0.65, color=(15, 15, 15)):
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    bg = np.empty_like(roi)
    bg[:] = color
    frame[y1:y2, x1:x2] = cv2.addWeighted(bg, alpha, roi, 1 - alpha, 0)


def dim(frame, alpha=0.5):
    overlay = np.zeros_like(frame)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_reveal_burst(frame, cx, cy, t):
    """Animated blue starburst for the dish reveal screen."""
    # Bright background glow circle
    ovl = frame.copy()
    cv2.circle(ovl, (cx, cy), 440, (252, 250, 245), -1)
    cv2.addWeighted(frame, 0.28, ovl, 0.72, 0, frame)

    rot   = (t * 22) % 360
    DARK  = (230, 110, 15)   # bright blue  (BGR)
    LIGHT = (255, 205, 130)  # sky blue      (BGR)
    PALE  = (255, 240, 210)  # pale blue     (BGR)

    # 20 alternating long/short radiating lines
    for k in range(20):
        ang = _ui_math.radians(18 * k + rot)
        if k % 2 == 0:
            r1, r2, thick, col = 148, 375, 5, DARK
        else:
            r1, r2, thick, col = 190, 272, 3, LIGHT
        cv2.line(frame,
                 (int(cx + r1 * _ui_math.cos(ang)), int(cy + r1 * _ui_math.sin(ang))),
                 (int(cx + r2 * _ui_math.cos(ang)), int(cy + r2 * _ui_math.sin(ang))),
                 col, thick, cv2.LINE_AA)

    # 10 extra short gap-lines between the main ones
    for k in range(10):
        ang = _ui_math.radians(36 * k + rot + 18)
        cv2.line(frame,
                 (int(cx + 225 * _ui_math.cos(ang)), int(cy + 225 * _ui_math.sin(ang))),
                 (int(cx + 315 * _ui_math.cos(ang)), int(cy + 315 * _ui_math.sin(ang))),
                 DARK, 2, cv2.LINE_AA)

    # 4-pointed sparkle diamonds at fixed positions
    for sx, sy, sr, ph in [
        (cx - 296, cy - 196, 23, 0.0), (cx + 312, cy - 178, 27, 0.6),
        (cx - 316, cy + 162, 19, 1.2), (cx + 284, cy + 200, 25, 1.8),
        (cx - 162, cy - 300, 16, 2.4), (cx + 218, cy - 282, 18, 0.9),
        (cx + 348, cy +  66, 21, 0.4), (cx - 242, cy + 260, 14, 1.6),
    ]:
        pulse = 0.76 + 0.24 * _ui_math.sin(t * 5 + ph)
        pr = max(4, int(sr * pulse))
        pts = []
        for j in range(4):
            a = _ui_math.pi / 2 * j + ph
            pts.append([int(sx + pr * _ui_math.cos(a)), int(sy + pr * _ui_math.sin(a))])
            ia = a + _ui_math.pi / 4
            pts.append([int(sx + max(2, pr // 6) * _ui_math.cos(ia)),
                        int(sy + max(2, pr // 6) * _ui_math.sin(ia))])
        arr = np.array(pts, np.int32)
        cv2.fillPoly(frame, [arr], PALE)
        cv2.polylines(frame, [arr], True, DARK, 1, cv2.LINE_AA)

    # Small + crosses
    for cxs, cys, cr in [(cx + 368, cy - 120, 11),
                          (cx - 358, cy +  34,  9),
                          (cx + 100, cy - 348, 10)]:
        cv2.line(frame, (cxs - cr, cys), (cxs + cr, cys), LIGHT, 2, cv2.LINE_AA)
        cv2.line(frame, (cxs, cys - cr), (cxs, cys + cr), LIGHT, 2, cv2.LINE_AA)

    # Small hollow circles
    for ocx, ocy, ocr in [(cx - 385, cy - 56, 8),
                            (cx + 360, cy + 104, 6),
                            (cx - 130, cy + 315, 7)]:
        cv2.circle(frame, (ocx, ocy), ocr, LIGHT, 2, cv2.LINE_AA)
