import cv2
import math
import numpy as np

_cache = {}


def get_knife(size=130):
    key = ('knife', size)
    if key not in _cache:
        _cache[key] = _make_knife(size)
    return _cache[key]


def get_spatula(size=150):
    key = ('spatula', size)
    if key not in _cache:
        _cache[key] = _make_spatula(size)
    return _cache[key]


def get_bowl(w=260, h=140):
    key = ('bowl', w, h)
    if key not in _cache:
        _cache[key] = _make_bowl(w, h)
    return _cache[key]


def get_carrot_whole(size=120):
    key = ('carrot_whole', size)
    if key not in _cache:
        _cache[key] = _make_carrot_whole(size)
    return _cache[key]


def get_carrot_half_left(size=120):
    key = ('carrot_half_l', size)
    if key not in _cache:
        _cache[key] = _make_carrot_half(size, left=True)
    return _cache[key]


def get_carrot_half_right(size=120):
    key = ('carrot_half_r', size)
    if key not in _cache:
        _cache[key] = _make_carrot_half(size, left=False)
    return _cache[key]


def get_cucumber_whole(size=120):
    key = ('cucumber_whole', size)
    if key not in _cache:
        _cache[key] = _make_cucumber_whole(size)
    return _cache[key]


def get_cucumber_half_left(size=120):
    key = ('cucumber_half_l', size)
    if key not in _cache:
        _cache[key] = _make_cucumber_half(size, left=True)
    return _cache[key]


def get_cucumber_half_right(size=120):
    key = ('cucumber_half_r', size)
    if key not in _cache:
        _cache[key] = _make_cucumber_half(size, left=False)
    return _cache[key]


def get_pepper_whole(size=120):
    key = ('pepper_whole', size)
    if key not in _cache:
        _cache[key] = _make_pepper_whole(size)
    return _cache[key]


def get_pepper_half_left(size=120):
    key = ('pepper_half_l', size)
    if key not in _cache:
        _cache[key] = _make_pepper_half(size, left=True)
    return _cache[key]


def get_pepper_half_right(size=120):
    key = ('pepper_half_r', size)
    if key not in _cache:
        _cache[key] = _make_pepper_half(size, left=False)
    return _cache[key]


# ── Sprite loading ────────────────────────────────────────────────────────────

def _make_knife(size):
    """Chef's knife drawn programmatically, blade pointing straight up."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    cx  = size // 2

    tip_y    = int(size * 0.03)
    root_y   = int(size * 0.60)
    guard_y2 = int(size * 0.68)
    hndl_y2  = int(size * 0.96)
    bl       = cx - int(size * 0.07)   # blade left x
    br       = cx + int(size * 0.11)   # blade right x (spine side)
    hx1      = cx - int(size * 0.09)
    hx2      = cx + int(size * 0.09)

    # ── blade (steel silver) ──────────────────────────────────────────────────
    blade = np.array([[cx, tip_y], [bl, root_y], [br, root_y]])
    cv2.fillPoly(img, [blade], (200, 208, 220, 255))
    # left shiny edge
    cv2.line(img, (cx, tip_y + 2), (bl, root_y), (240, 248, 255, 210), 2)
    # right spine (slightly darker)
    spine = np.array([[cx, tip_y + 3], [br, root_y],
                      [br - 3, root_y], [cx - 2, tip_y + 6]])
    cv2.fillPoly(img, [spine], (158, 165, 178, 255))

    # ── guard / bolster ───────────────────────────────────────────────────────
    cv2.rectangle(img, (bl - 3, root_y), (br + 3, guard_y2), (125, 132, 145, 255), -1)
    cv2.rectangle(img, (bl - 3, root_y), (br + 3, guard_y2), (85, 90, 100, 255), 2)

    # ── handle (dark wood) ────────────────────────────────────────────────────
    cv2.rectangle(img, (hx1, guard_y2), (hx2, hndl_y2), (48, 78, 112, 255), -1)
    mid_x = (hx1 + hx2) // 2
    # three rivets
    for ry in [guard_y2 + int(size * 0.06),
               guard_y2 + int(size * 0.14),
               guard_y2 + int(size * 0.22)]:
        r = max(2, int(size * 0.028))
        cv2.circle(img, (mid_x, ry), r, (82, 112, 148, 255), -1)
        cv2.circle(img, (mid_x, ry), r, (38, 58, 88,  255), 1)
    # wood grain lines
    for i in range(5):
        ly = guard_y2 + int(size * (0.04 + i * 0.06))
        cv2.line(img, (hx1 + 2, ly), (hx2 - 2, ly), (36, 60, 88, 130), 1)
    cv2.rectangle(img, (hx1, guard_y2), (hx2, hndl_y2), (28, 48, 78, 255), 2)

    return img


def _pepper_poly(cx, cy, r):
    """Generate bell-pepper outline: 4-lobe polygon using cosine modulation."""
    pts = []
    for deg in range(0, 360, 2):
        a  = math.radians(deg)
        rr = r * (1.0 + 0.12 * math.cos(4 * a + math.pi / 4))
        pts.append((int(cx + math.cos(a) * rr), int(cy + math.sin(a) * rr)))
    return np.array(pts, dtype=np.int32)


def _make_pepper_whole(size):
    """Whole red bell pepper, top-down view."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    cx  = size // 2
    cy  = int(size * 0.54)
    r   = int(size * 0.33)

    DARK  = (10,  14, 148, 255)
    RED   = (35,  50, 220, 255)
    LITE  = (58,  80, 238, 255)
    GRN   = (30, 140,  45, 255)
    DKGRN = (15,  80,  25, 255)

    # Dark outline (slightly inflated polygon)
    cv2.fillPoly(img, [_pepper_poly(cx, cy, r + 3)], DARK)
    # Main red fill
    cv2.fillPoly(img, [_pepper_poly(cx, cy, r)],     RED)
    # Brighter highlight on upper portion (3D curve illusion)
    cv2.fillPoly(img, [_pepper_poly(cx, cy - int(r * 0.18), int(r * 0.68))], LITE)

    # Stem
    sw   = max(4, int(size * 0.08))
    syb  = cy - r + int(r * 0.05)
    syt  = syb - int(size * 0.17)
    cv2.rectangle(img, (cx - sw // 2, syt), (cx + sw // 2, syb), GRN,   -1)
    cv2.rectangle(img, (cx - sw // 2, syt), (cx + sw // 2, syb), DKGRN,  2)

    # Calyx (5 small petals around stem base)
    for i in range(5):
        a  = math.radians(i * 72)
        lx = int(cx + math.cos(a) * r * 0.44)
        ly = int(syb + math.sin(a) * r * 0.22)
        cv2.line(img, (cx, syb), (lx, ly), GRN, 2)

    # Gloss highlight
    cv2.ellipse(img,
                (cx - int(r * 0.27), cy - int(r * 0.30)),
                (int(r * 0.23), int(r * 0.14)), -25, 0, 360,
                (130, 130, 255, 140), -1)
    return img


def _make_pepper_half(size, left=True):
    """Half of a bell pepper after a vertical (left / right) cut."""
    img  = _make_pepper_whole(size)
    cx   = size // 2
    cy   = int(size * 0.54)
    r    = int(size * 0.33)

    # Mask out the unwanted half
    if left:
        img[:, cx + 1:, 3] = 0
    else:
        img[:, :cx,     3] = 0

    # ── Cut face: thin strip at the cut edge showing interior ────────────────
    face_ry = int(r * 0.80)
    face_t  = 10

    # Red pepper wall
    cv2.rectangle(img,
                  (cx - face_t // 2,     cy - face_ry),
                  (cx + face_t // 2,     cy + face_ry),
                  (10, 14, 148, 255), -1)
    # Cream interior cavity
    cv2.rectangle(img,
                  (cx - face_t // 2 + 2, cy - int(face_ry * 0.86)),
                  (cx + face_t // 2 - 2, cy + int(face_ry * 0.86)),
                  (185, 245, 245, 255), -1)
    # Seeds (3 white dots)
    for i in range(3):
        sy_s = int(cy + (i - 1) * face_ry * 0.36)
        cv2.circle(img, (cx, sy_s), max(2, int(r * 0.057)), (215, 245, 255, 255), -1)

    return img


def _make_carrot_whole(size):
    """Whole carrot, top-down view (orange teardrop with green leaves)."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    cx  = size // 2
    cy  = int(size * 0.50)
    rx  = int(size * 0.22)   # narrower
    ry  = int(size * 0.39)   # taller

    DARK = (0,  75, 158, 255)
    ORG  = (0, 155, 252, 255)
    LITE = (20, 200, 255, 255)
    GRN  = (25, 138,  40, 255)
    DKGN = (12,  78,  22, 255)

    # Outline (inflated ellipse)
    cv2.ellipse(img, (cx, cy), (rx + 3, ry + 3), 0, 0, 360, DARK, -1)
    # Main orange body
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, ORG, -1)
    # Highlight
    cv2.ellipse(img, (cx - int(rx * 0.35), cy - int(ry * 0.22)),
                (int(rx * 0.45), int(ry * 0.35)), 0, 0, 360, LITE, -1)
    # Tapered tip at bottom
    tip = np.array([
        [cx - int(rx * 0.50), cy + int(ry * 0.62)],
        [cx + int(rx * 0.50), cy + int(ry * 0.62)],
        [cx,                  cy + ry + int(size * 0.10)],
    ], dtype=np.int32)
    cv2.fillPoly(img, [tip], ORG)
    cv2.polylines(img, [tip], True, DARK, 2)
    # Texture rings
    for yf in [0.20, 0.46, 0.70]:
        ring_y  = int(cy - ry + 2 * ry * yf)
        ring_rx = max(2, int(rx * (1.0 - abs(yf - 0.45) * 0.30)))
        cv2.ellipse(img, (cx, ring_y), (ring_rx, max(2, int(ry * 0.04))),
                    0, 0, 360, DARK, 1)
    # Stem
    sy_base = cy - ry + int(ry * 0.06)
    sw = max(3, int(size * 0.06))
    cv2.rectangle(img, (cx - sw // 2, sy_base - int(size * 0.14)),
                  (cx + sw // 2, sy_base), GRN, -1)
    cv2.rectangle(img, (cx - sw // 2, sy_base - int(size * 0.14)),
                  (cx + sw // 2, sy_base), DKGN, 1)
    # Feathery leaves
    for i in range(5):
        a   = math.radians(-105 + i * 26)
        lx2 = int(cx + math.cos(a) * rx * 1.9)
        ly2 = int(sy_base - int(size * 0.06) + math.sin(a) * ry * 0.38)
        cv2.line(img, (cx, sy_base - int(size * 0.03)), (lx2, ly2), GRN, 2)
    # Gloss
    cv2.ellipse(img, (cx - int(rx * 0.30), cy - int(ry * 0.28)),
                (int(rx * 0.20), int(ry * 0.13)), -20, 0, 360, (80, 220, 255, 130), -1)
    return img


def _make_carrot_half(size, left=True):
    """Half carrot after vertical cut, interior (orange rings) visible."""
    img    = _make_carrot_whole(size)
    cx     = size // 2
    cy     = int(size * 0.50)
    ry     = int(size * 0.39)

    if left:
        img[:, cx + 1:, 3] = 0
    else:
        img[:, :cx, 3] = 0

    face_ry = int(ry * 0.84)
    face_t  = 9
    # Dark orange wall
    cv2.rectangle(img, (cx - face_t // 2, cy - face_ry),
                  (cx + face_t // 2, cy + face_ry), (0, 80, 165, 255), -1)
    # Lighter orange flesh
    cv2.rectangle(img, (cx - face_t // 2 + 2, cy - int(face_ry * 0.87)),
                  (cx + face_t // 2 - 2, cy + int(face_ry * 0.87)), (20, 195, 255, 255), -1)
    # Inner ring hint
    cv2.ellipse(img, (cx, cy), (face_t // 2 - 1, int(face_ry * 0.28)),
                0, 0, 360, (0, 90, 175, 255), 1)
    return img


def _make_cucumber_whole(size):
    """Whole cucumber, top-down view (dark green oval with light stripes)."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    cx  = size // 2
    cy  = int(size * 0.50)
    rx  = int(size * 0.23)
    ry  = int(size * 0.40)

    DARK = (12,  55, 15, 255)
    GRN  = (22, 108, 28, 255)
    LITE = (70, 175, 55, 255)

    # Outline
    cv2.ellipse(img, (cx, cy), (rx + 3, ry + 3), 0, 0, 360, DARK, -1)
    # Body
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, GRN, -1)
    # Light stripes (3 vertical)
    for sx in [-int(rx * 0.50), 0, int(rx * 0.50)]:
        cv2.ellipse(img, (cx + sx, cy), (max(2, int(rx * 0.13)), int(ry * 0.84)),
                    0, 0, 360, LITE, -1)
    # End bumps
    for ey in [cy - ry, cy + ry]:
        cv2.circle(img, (cx, ey), int(rx * 0.42), GRN,  -1)
        cv2.circle(img, (cx, ey), int(rx * 0.25), LITE, -1)
    cv2.ellipse(img, (cx, cy), (rx + 3, ry + 3), 0, 0, 360, DARK, 2)
    # Gloss
    cv2.ellipse(img, (cx - int(rx * 0.32), cy - int(ry * 0.24)),
                (int(rx * 0.30), int(ry * 0.18)), -20, 0, 360, (150, 220, 140, 110), -1)
    return img


def _make_cucumber_half(size, left=True):
    """Half cucumber after vertical cut, light green flesh + seeds visible."""
    img    = _make_cucumber_whole(size)
    cx     = size // 2
    cy     = int(size * 0.50)
    ry     = int(size * 0.40)

    if left:
        img[:, cx + 1:, 3] = 0
    else:
        img[:, :cx, 3] = 0

    face_ry = int(ry * 0.84)
    face_t  = 9
    # Dark green outer ring
    cv2.rectangle(img, (cx - face_t // 2, cy - face_ry),
                  (cx + face_t // 2, cy + face_ry), (12, 55, 15, 255), -1)
    # Light green flesh
    cv2.rectangle(img, (cx - face_t // 2 + 2, cy - int(face_ry * 0.87)),
                  (cx + face_t // 2 - 2, cy + int(face_ry * 0.87)), (120, 208, 88, 255), -1)
    # Seed dots
    for i in range(4):
        sy_s = int(cy - face_ry * 0.52 + i * face_ry * 0.35)
        cv2.circle(img, (cx, sy_s), max(1, int(size * 0.024)), (210, 250, 195, 255), -1)
    return img


def _make_spatula(size):
    """Wooden paddle spatula drawn programmatically."""
    W, H = size // 3, size
    img = np.zeros((H, W, 4), dtype=np.uint8)

    hw = max(W // 3, 3)
    cx = W // 2

    # Handle (dark wood brown)
    cv2.rectangle(img, (cx - hw, H // 3), (cx + hw, H - 6), (40, 80, 130, 255), -1)
    cv2.rectangle(img, (cx - hw, H // 3), (cx + hw, H - 6), (20, 50, 90, 255), 2)
    for i in range(3):
        gy = H // 3 + 10 + i * 18
        cv2.line(img, (cx - hw + 2, gy), (cx + hw - 2, gy), (30, 60, 100, 180), 1)

    # Paddle head (light wood)
    cv2.rectangle(img, (2, 4), (W - 3, H // 3 + 4), (80, 140, 200, 255), -1)
    cv2.rectangle(img, (2, 4), (W - 3, H // 3 + 4), (50, 100, 150, 255), 2)

    # Expand to square canvas
    canvas = np.zeros((H, H, 4), dtype=np.uint8)
    xo = (H - W) // 2
    canvas[:, xo:xo + W] = img
    return cv2.resize(canvas, (size, size))


def _make_bowl(W, H):
    """Side-view pot/bowl drawn programmatically."""
    canvas = np.zeros((H + 20, W, 4), dtype=np.uint8)
    cx = W // 2

    body = np.array([[20, H - 10], [W - 20, H - 10], [W - 50, 20], [50, 20]])
    cv2.fillPoly(canvas, [body], (165, 168, 175, 255))
    cv2.polylines(canvas, [body], True, (110, 112, 118, 255), 3)

    # Rim ellipse
    cv2.ellipse(canvas, (cx, 20), (cx - 45, 14), 0, 0, 360, (190, 192, 198, 255), -1)
    cv2.ellipse(canvas, (cx, 20), (cx - 45, 14), 0, 0, 360, (120, 122, 128, 255), 2)

    # Inner rim
    cv2.ellipse(canvas, (cx, 20), (cx - 65, 9), 0, 0, 360, (140, 142, 150, 255), -1)

    # Highlight
    cv2.line(canvas, (60, H // 2), (W - 60, H // 2 - 8), (220, 222, 228, 180), 3)

    return canvas


# ── Rendering ─────────────────────────────────────────────────────────────────

def overlay(frame, sprite_bgra, cx, cy, size=None, angle=0.0):
    """Alpha-blend a BGRA sprite centered at pixel (cx, cy) onto a BGR frame."""
    if sprite_bgra is None:
        return
    s = sprite_bgra.copy()

    if size is not None:
        oh, ow = s.shape[:2]
        ratio = size / max(ow, oh)
        nw, nh = max(1, int(ow * ratio)), max(1, int(oh * ratio))
        s = cv2.resize(s, (nw, nh))

    if angle != 0.0:
        sh, sw = s.shape[:2]
        M = cv2.getRotationMatrix2D((sw / 2, sh / 2), angle, 1.0)
        s = cv2.warpAffine(s, M, (sw, sh), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    sh, sw = s.shape[:2]
    fh, fw = frame.shape[:2]
    x1, y1 = cx - sw // 2, cy - sh // 2
    x2, y2 = x1 + sw, y1 + sh

    bx1, by1 = max(0, x1), max(0, y1)
    bx2, by2 = min(fw, x2), min(fh, y2)
    if bx2 <= bx1 or by2 <= by1:
        return

    sx1, sy1 = bx1 - x1, by1 - y1
    sx2, sy2 = sx1 + (bx2 - bx1), sy1 + (by2 - by1)

    roi = frame[by1:by2, bx1:bx2]
    spr = s[sy1:sy2, sx1:sx2]
    a   = spr[:, :, 3:4].astype(np.float32) / 255.0
    roi[:] = (spr[:, :, :3].astype(np.float32) * a +
              roi.astype(np.float32) * (1.0 - a)).astype(np.uint8)
