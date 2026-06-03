import cv2
import math
import time
import threading
import urllib.request
import os
from dataclasses import dataclass, field

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_PATH = 'hand_landmarker.task'
MODEL_URL  = (
    'https://storage.googleapis.com/mediapipe-models/'
    'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
)

PROC_W, PROC_H = 640, 360
GRIP_THRESHOLD = 0.17
PALM_REAL_M    = 0.085
Z_REF_M        = 0.55
UNITY_SCALE    = 9.0

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


@dataclass
class HandState:
    detected:   bool  = False
    x:          float = 0.0   # world units, right = positive
    y:          float = 0.0   # world units, up = positive
    z:          float = 0.0
    vx:         float = 0.0
    vy:         float = 0.0
    gripped:    bool  = False
    grip_dist:  float = 0.0
    screen_x:   int   = 0     # pixel on 1280x720 display
    screen_y:   int   = 0
    landmarks:  list  = field(default_factory=list)  # NormalizedLandmark list for avatar


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print('Downloading hand landmark model (~8 MB)...')
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print('Download complete.')


def _to_world(nx, ny):
    fx = PROC_W * 0.87
    cx, cy = PROC_W / 2.0, PROC_H / 2.0
    x_n = (nx * PROC_W - cx) / fx
    y_n = (ny * PROC_H - cy) / fx
    return (round(float(max(-8.0, min(8.0,  x_n * UNITY_SCALE))), 4),
            round(float(max(-4.5, min(4.5, -y_n * UNITY_SCALE))), 4))


def _calc_depth(lms):
    fx  = PROC_W * 0.87
    w0, m9 = lms[0], lms[9]
    d_px = math.sqrt((w0.x - m9.x)**2 * PROC_W**2 + (w0.y - m9.y)**2 * PROC_H**2)
    if d_px < 1:
        return 0.0
    Z_m = fx * PALM_REAL_M / d_px
    return round(float(max(-4.0, min(4.0, (Z_m - Z_REF_M) * 6.0))), 4)


def _calc_grip(lms):
    TIP_IDS = [8, 12, 16, 20]
    PIP_IDS = [6, 10, 14, 18]
    curl = sum(1 for t, p in zip(TIP_IDS, PIP_IDS) if lms[t].y > lms[p].y)
    palm_x = sum(lms[i].x for i in [0, 5, 9, 13, 17]) / 5
    palm_y = sum(lms[i].y for i in [0, 5, 9, 13, 17]) / 5
    avg_dist = sum(
        math.sqrt((lms[t].x - palm_x)**2 + (lms[t].y - palm_y)**2) for t in TIP_IDS
    ) / 4
    return (curl >= 3) and (avg_dist < GRIP_THRESHOLD), round(avg_dist, 4)


def _world_to_screen(wx, wy, W=1280, H=720):
    sx = int((wx / 8.0 + 1.0) * W / 2)
    sy = int((-wy / 4.5 + 1.0) * H / 2)
    return sx, sy


class _SharedResult:
    def __init__(self):
        self._result = None
        self._new    = False
        self._lock   = threading.Lock()

    def set(self, result):
        with self._lock:
            self._result = result
            self._new    = True

    def take(self):
        with self._lock:
            if self._new:
                self._new = False
                return self._result
            return None

    def peek_landmarks(self):
        """Return list of landmark lists for all detected hands (thread-safe peek)."""
        with self._lock:
            if self._result and self._result.hand_landmarks:
                return list(self._result.hand_landmarks)
            return []


class _VelocityTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix    = np.eye(4, dtype=np.float32)
        self.kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.kf.processNoiseCov     = np.diag([1e-4, 1e-4, 1e-2, 1e-2]).astype(np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5e-2
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self._ready  = False
        self._prev_t = None

    def update(self, x, y):
        now = time.time()
        if not self._ready:
            self.kf.statePost = np.array([[x], [y], [0.], [0.]], dtype=np.float32)
            self._ready, self._prev_t = True, now
            return 0.0, 0.0
        dt = max(now - self._prev_t, 1e-4)
        self._prev_t = now
        self.kf.transitionMatrix[0, 2] = dt
        self.kf.transitionMatrix[1, 3] = dt
        self.kf.predict()
        c = self.kf.correct(np.array([[x], [y]], dtype=np.float32))
        return round(float(c[2, 0]), 4), round(float(c[3, 0]), 4)

    def reset(self):
        self._ready = False


class HandTracker:
    def __init__(self):
        _ensure_model()
        self._shared   = _SharedResult()
        self._trackers = [_VelocityTracker(), _VelocityTracker()]
        self._states: list = []   # List[HandState], length 0-2
        self._last_ts  = 0

        def _on_result(result, output_image, timestamp_ms):
            self._shared.set(result)

        base = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=_on_result,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, frame):
        """Send frame to MediaPipe async; update internal hand states if result ready."""
        small  = cv2.resize(frame, (PROC_W, PROC_H))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        ts = max(int(time.time() * 1000), self._last_ts + 1)
        self._last_ts = ts
        self._landmarker.detect_async(mp_img, ts)

        result = self._shared.take()
        if result is None:
            return

        new_states = []
        n = len(result.hand_landmarks)

        for i, lms in enumerate(result.hand_landmarks):
            tracker = self._trackers[i] if i < len(self._trackers) else self._trackers[-1]
            gripped, g_dist = _calc_grip(lms)

            # World / velocity: wrist + middle-MCP midpoint (stable palm reference)
            px = (lms[0].x + lms[9].x) / 2
            py = (lms[0].y + lms[9].y) / 2
            wx, wy = _to_world(px, py)
            vx, vy = tracker.update(wx, wy)
            wz     = _calc_depth(lms)

            # Screen position: centroid of ALL 21 landmarks, direct linear mapping.
            # This matches exactly where hand_avatar.py centres the drawn avatar,
            # so grab / chop detection fires where the player sees the mini hand.
            avg_nx = sum(lm.x for lm in lms) / len(lms)
            avg_ny = sum(lm.y for lm in lms) / len(lms)
            sx = int(avg_nx * 1280)
            sy = int(avg_ny * 720)

            new_states.append(
                HandState(True, wx, wy, wz, vx, vy, gripped, g_dist, sx, sy, list(lms))
            )

        # Reset trackers for hands that disappeared
        for i in range(n, len(self._trackers)):
            self._trackers[i].reset()

        self._states = new_states

    def get_hand_states(self) -> list:
        """Return list of all currently tracked HandState objects (0-2 items)."""
        return self._states

    def get_hand_state(self) -> HandState:
        """Return first detected hand, or empty HandState (backward compat)."""
        return self._states[0] if self._states else HandState()

    def draw(self, frame, result=None):
        """Draw skeletons for all detected hands (called only when avatar is NOT used)."""
        all_lms = self._shared.peek_landmarks()
        if not all_lms:
            return
        h, w = frame.shape[:2]
        for idx, lms in enumerate(all_lms):
            # Color by grip state
            gripped = self._states[idx].gripped if idx < len(self._states) else False
            color   = (0, 80, 255) if gripped else (255, 200, 0)
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
            for a, b in CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], color, 2)
            for pt in pts:
                cv2.circle(frame, pt, 4, (0, 230, 120), -1)
