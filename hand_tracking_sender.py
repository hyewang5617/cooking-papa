"""
hand_tracking_sender.py
─────────────────────────────────────────────────────────
실행 순서:
  1. 웹캠 열기
  2. MediaPipe로 손 랜드마크 21개 검출
  3. 손 중심 좌표 / pinch 여부 / 속도 계산
  4. JSON으로 직렬화 → UDP로 Unity(5052포트)에 전송
  5. 디버그 창에 시각화 (ESC로 종료)
"""

import cv2
import json
import math
import socket
import time
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# ── 설정 ────────────────────────────────────────────────
UDP_IP   = "127.0.0.1"
UDP_PORT = 5052

# Unity 좌표계에 맞게 스케일링할 범위
UNITY_X_RANGE = (-8.0, 8.0)
UNITY_Y_RANGE = (-4.5, 4.5)

# Pinch 판정 임계값 (엄지-검지 끝 거리, 정규화 좌표 기준)
PINCH_THRESHOLD = 0.06

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ── 모델 다운로드 ────────────────────────────────────────
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~8 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")

# ── 좌표 변환 : 웹캠(0~1) → Unity 좌표계 ───────────────
def to_unity(nx, ny):
    """
    MediaPipe 정규화 좌표(0~1)를 Unity 월드 좌표로 변환.
    x는 좌우 반전(웹캠 미러 보정 이미 flip됨), y는 위아래 반전.
    """
    ux = nx * (UNITY_X_RANGE[1] - UNITY_X_RANGE[0]) + UNITY_X_RANGE[0]
    uy = (1.0 - ny) * (UNITY_Y_RANGE[1] - UNITY_Y_RANGE[0]) + UNITY_Y_RANGE[0]
    return round(ux, 4), round(uy, 4)

# ── Pinch 계산 (엄지 끝 #4 ↔ 검지 끝 #8) ────────────────
def calc_pinch(landmarks):
    thumb  = landmarks[4]
    index  = landmarks[8]
    dist = math.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2)
    return dist < PINCH_THRESHOLD, round(dist, 4)

# ── 손 중심 (wrist #0 + middle MCP #9 의 중점) ──────────
def calc_palm_center(landmarks):
    wx = (landmarks[0].x + landmarks[9].x) / 2
    wy = (landmarks[0].y + landmarks[9].y) / 2
    return wx, wy

# ── 좌표 스무딩 (EMA 저역통과 필터) ─────────────────────
class PositionSmoother:
    def __init__(self, alpha=0.5):
        # alpha: 0에 가까울수록 부드럽고 느림, 1에 가까울수록 즉각적
        self.alpha = alpha
        self._x = self._y = None

    def update(self, x, y):
        if self._x is None:
            self._x, self._y = x, y
        else:
            self._x = self.alpha * x + (1 - self.alpha) * self._x
            self._y = self.alpha * y + (1 - self.alpha) * self._y
        return round(self._x, 4), round(self._y, 4)

# ── 속도 계산 ────────────────────────────────────────────
class VelocityTracker:
    def __init__(self):
        self._prev_pos  = None
        self._prev_time = None

    def update(self, x, y):
        now = time.time()
        vx = vy = 0.0
        if self._prev_pos and self._prev_time:
            dt = now - self._prev_time
            if dt > 0:
                vx = (x - self._prev_pos[0]) / dt
                vy = (y - self._prev_pos[1]) / dt
        self._prev_pos  = (x, y)
        self._prev_time = now
        return round(vx, 4), round(vy, 4)

# ── 시각화 ───────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

def draw_hand(frame, landmarks, pinched):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    color = (0, 80, 255) if pinched else (255, 200, 0)
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 230, 120), -1)
    # 엄지-검지 연결선 강조
    cv2.line(frame, pts[4], pts[8], (0, 255, 255), 3)

# ── 메인 ────────────────────────────────────────────────
def main():
    ensure_model()

    # MediaPipe HandLandmarker 초기화
    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    # UDP 소켓
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 웹캠
    cap = None
    for idx in range(3):
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            ret, _ = c.read()
            if ret:
                cap = c
                print(f"Camera index {idx}")
                break
            c.release()
    if cap is None:
        print("Error: No camera found.")
        return

    velocity = VelocityTracker()
    smoother = PositionSmoother(alpha=0.5)

    cv2.namedWindow("Hand Tracking Sender", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1280, 720))

        # 손 검출
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        payload = {"detected": False}

        if result.hand_landmarks:
            lms = result.hand_landmarks[0]
            pinched, pinch_dist = calc_pinch(lms)
            px, py = calc_palm_center(lms)
            ux, uy = to_unity(px, py)
            ux, uy = smoother.update(ux, uy)   # EMA 스무딩
            vx, vy = velocity.update(ux, uy)

            payload = {
                "detected":   True,
                "x":          ux,          # Unity 월드 X
                "y":          uy,          # Unity 월드 Y
                "vx":         vx,          # X 속도 (Unity 단위/초)
                "vy":         vy,          # Y 속도
                "pinched":    pinched,     # True/False
                "pinch_dist": pinch_dist,  # 엄지-검지 거리
            }

            draw_hand(frame, lms, pinched)

            # 화면에 데이터 표시
            label = f"({'PINCH' if pinched else 'open'})  x={ux:.2f}  y={uy:.2f}  v=({vx:.1f},{vy:.1f})"
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_DUPLEX,
                        0.8, (0, 255, 200), 2)
        else:
            cv2.putText(frame, "No hand detected", (20, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 80, 255), 2)

        # UDP 전송
        data = json.dumps(payload).encode("utf-8")
        sock.sendto(data, (UDP_IP, UDP_PORT))

        cv2.imshow("Hand Tracking Sender", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    sock.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
