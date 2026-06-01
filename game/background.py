import cv2
import numpy as np

_CACHE = {}


def get_kitchen_bg(w=1280, h=720):
    """Return a cached Cooking-Mama-style kitchen background (BGR)."""
    key = (w, h)
    if key not in _CACHE:
        _CACHE[key] = _make(w, h)
    return _CACHE[key]


def _make(w, h):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    CY = 340  # counter line (must match _COUNTER_Y in cooking_scene.py)

    # ── Ceiling strip ─────────────────────────────────────────────────────────
    cv2.rectangle(img, (0, 0), (w, 26), (192, 218, 232), -1)

    # ── Tiled wall ────────────────────────────────────────────────────────────
    img[26:CY, :] = (228, 236, 240)

    TW, TH = 86, 70
    TILE  = (246, 250, 252)
    GROUT = (188, 196, 202)
    for ry in range(26, CY, TH):
        for cx in range(0, w, TW):
            t1 = (cx + 3, ry + 3)
            t2 = (min(cx + TW - 3, w - 1), min(ry + TH - 3, CY - 1))
            if t2[0] > t1[0] and t2[1] > t1[1]:
                cv2.rectangle(img, t1, t2, TILE, -1)
    for ry in range(26, CY, TH):
        cv2.line(img, (0, ry), (w, ry), GROUT, 2)
    for cx in range(TW, w, TW):
        cv2.line(img, (cx, 26), (cx, CY), GROUT, 2)

    # ── Counter ledge (wood-tone) ─────────────────────────────────────────────
    cv2.rectangle(img, (0, CY - 20), (w, CY + 6),  (148, 120, 90),  -1)
    cv2.line     (img, (0, CY - 20), (w, CY - 20), (188, 160, 128),  3)   # shine
    cv2.rectangle(img, (0, CY + 6),  (w, CY + 20), (112, 86,  64),  -1)
    cv2.line     (img, (0, CY + 20), (w, CY + 20), (72,  54,  38),   2)   # shadow

    # ── Checkered tablecloth surface ──────────────────────────────────────────
    surf_y = CY + 20
    TS     = 64
    COL_A  = (208, 168, 204)   # mauve-pink  (BGR)
    COL_B  = (240, 214, 240)   # light pink
    GRID   = (182, 142, 178)   # grid line

    for ty in range(surf_y, h, TS):
        for tx in range(0, w, TS):
            col = COL_A if ((tx // TS + (ty - surf_y) // TS) % 2 == 0) else COL_B
            cv2.rectangle(img, (tx, ty), (min(tx + TS, w), min(ty + TS, h)), col, -1)
    for ty in range(surf_y, h, TS):
        cv2.line(img, (0, ty), (w, ty), GRID, 1)
    for tx in range(0, w, TS):
        cv2.line(img, (tx, surf_y), (tx, h), GRID, 1)

    return img
