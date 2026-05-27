import cv2
import json
import math
import socket
import time
import threading
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

UDP_IP   = "127.0.0.1"
UDP_PORT = 5052

GRIP_THRESHOLD = 0.17   # AND 조건 보조용 거리 임계값
PALM_REAL_M     = 0.085
Z_REF_M         = 0.55
UNITY_SCALE     = 9.0

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~8 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")

# ── 좌표 변환 ────────────────────────────────────────────
def to_unity_xy(nx, ny, fw, fh):
    fx = fw * 0.87
    cx, cy = fw / 2.0, fh / 2.0
    x_n = (nx * fw - cx) / fx
    y_n = (ny * fh - cy) / fx
    return (round(float(max(-8.0, min(8.0,  x_n * UNITY_SCALE))), 4),
            round(float(max(-4.5, min(4.5, -y_n * UNITY_SCALE))), 4))

def calc_depth_z(landmarks, fw, fh):
    fx = fw * 0.87
    w0, m9 = landmarks[0], landmarks[9]
    d_px = math.sqrt((w0.x * fw - m9.x * fw)**2 + (w0.y * fh - m9.y * fh)**2)
    if d_px < 1:
        return 0.0
    Z_m = fx * PALM_REAL_M / d_px
    return round(float(max(-4.0, min(4.0, (Z_m - Z_REF_M) * 6.0))), 4)

def calc_grip(landmarks):
    # ── 방법 1: tip.y > pip.y → 손가락 굽힘 (레포3 방식, 스케일 불변)
    # normalized coords에서 y는 아래로 증가. 끝이 PIP보다 아래 = 굽힘
    TIP_IDS = [8, 12, 16, 20]
    PIP_IDS = [6, 10, 14, 18]
    curl = sum(1 for t, p in zip(TIP_IDS, PIP_IDS)
               if landmarks[t].y > landmarks[p].y)   # 굽혀진 손가락 수

    # ── 방법 2: 손끝 ~ 손바닥 평균 거리 (보조)
    palm_x = (landmarks[0].x + landmarks[5].x + landmarks[9].x +
              landmarks[13].x + landmarks[17].x) / 5
    palm_y = (landmarks[0].y + landmarks[5].y + landmarks[9].y +
              landmarks[13].y + landmarks[17].y) / 5
    avg_dist = sum(math.sqrt((landmarks[t].x - palm_x)**2 + (landmarks[t].y - palm_y)**2)
                   for t in TIP_IDS) / 4

    # 4개 모두 굽힘(curl==4) AND 거리 조건 → 확실한 grip
    # 3개 굽힘 + 거리 조건 → 부분 grip도 허용 (칼 잡을 때 새끼손가락 안 굽히는 경우)
    is_grip = (curl >= 3) and (avg_dist < GRIP_THRESHOLD)
    return is_grip, round(avg_dist, 4)

def calc_palm_center(landmarks):
    return ((landmarks[0].x + landmarks[9].x) / 2,
            (landmarks[0].y + landmarks[9].y) / 2)

# ── 시각화 ───────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

def draw_hand(frame, landmarks, pinched):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    color = (0, 80, 255) if pinched else (255, 200, 0)
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 230, 120), -1)
    cv2.line(frame, pts[4], pts[8], (0, 255, 255), 3)

# ── 속도 계산 ────────────────────────────────────────────
class VelocityTracker:
    def __init__(self):
        self._prev = None
        self._t    = None

    def update(self, x, y):
        now = time.time()
        vx = vy = 0.0
        if self._prev and self._t:
            dt = now - self._t
            if dt > 0:
                vx = (x - self._prev[0]) / dt
                vy = (y - self._prev[1]) / dt
        self._prev, self._t = (x, y), now
        return round(vx, 4), round(vy, 4)

    def reset(self):
        self._prev = None
        self._t    = None

# ── LIVE_STREAM 콜백 결과 공유 (스레드 안전) ─────────────
class SharedResult:
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

# ── 60fps 전송용 손 상태 공유 ─────────────────────────────
# MediaPipe(30fps)가 ground truth를 갱신하고,
# 전송 스레드(60fps)가 velocity로 예측한 좌표를 Unity에 보냄
class HandState:
    def __init__(self):
        self._lock      = threading.Lock()
        self.detected   = False
        self.x = self.y = self.z = 0.0
        self.vx = self.vy        = 0.0
        self.pinched             = False
        self.pinch_dist          = 0.0
        self._t                  = 0.0

    def update(self, detected, x=0.0, y=0.0, z=0.0,
               vx=0.0, vy=0.0, pinched=False, pinch_dist=0.0):
        with self._lock:
            self.detected   = detected
            self.x, self.y, self.z = x, y, z
            self.vx, self.vy       = vx, vy
            self.pinched           = pinched
            self.pinch_dist        = pinch_dist
            self._t                = time.time()

    def payload(self):
        with self._lock:
            if not self.detected:
                return {"detected": False}
            dt = time.time() - self._t
            return {
                "detected": True,
                "x":  round(self.x + self.vx * dt, 4),
                "y":  round(self.y + self.vy * dt, 4),
                "z":  self.z,
                "vx": self.vx, "vy": self.vy,
                "pinched":    self.pinched,
                "pinch_dist": self.pinch_dist,
            }

def _send_loop(sock, hand_state, stop_evt):
    """60fps로 Unity에 UDP 전송. MediaPipe 갱신 사이를 velocity로 예측."""
    interval = 1.0 / 60
    next_t   = time.time()
    while not stop_evt.is_set():
        sock.sendto(json.dumps(hand_state.payload()).encode(), (UDP_IP, UDP_PORT))
        next_t += interval
        wait = next_t - time.time()
        if wait > 0:
            time.sleep(wait)

def main():
    ensure_model()

    shared = SharedResult()

    # LIVE_STREAM 모드: MediaPipe가 별도 스레드에서 비동기 처리
    # 결과는 콜백으로 즉시 전달 → 카메라 읽기가 블로킹되지 않음
    def on_result(result, output_image, timestamp_ms):
        shared.set(result)

    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=on_result,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    sock      = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    hand_state = HandState()
    stop_evt   = threading.Event()
    sender     = threading.Thread(target=_send_loop, args=(sock, hand_state, stop_evt),
                                  daemon=True)
    sender.start()

    cap = None
    for idx in range(3):
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            ret, _ = c.read()
            if ret:
                cap = c
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 60)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 최신 프레임만 유지
                print(f"Camera index {idx}  {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
                      f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}"
                      f"  FPS={cap.get(cv2.CAP_PROP_FPS):.0f}")
                break
            c.release()
    if cap is None:
        print("Error: No camera found.")
        return

    tracker = VelocityTracker()
    PROC_W, PROC_H = 640, 360
    fps_timer, fps_count, fps_display = time.time(), 0, 0
    # detect_async requires strictly increasing timestamps
    _last_ts_ms = 0

    cv2.namedWindow("Hand Tracking Sender", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        small = cv2.resize(frame, (PROC_W, PROC_H))

        # 비동기 전송: 결과를 기다리지 않고 바로 다음 프레임으로
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB),
        )
        ts_ms = max(int(time.time() * 1000), _last_ts_ms + 1)
        _last_ts_ms = ts_ms
        landmarker.detect_async(mp_image, ts_ms)

        # MediaPipe 새 결과 있을 때만 처리 + UDP 전송
        result  = shared.take()
        display = cv2.resize(frame, (1280, 720))

        if result is not None:
            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                pinched, pinch_dist = calc_grip(lms)
                px, py = calc_palm_center(lms)
                ux, uy = to_unity_xy(px, py, PROC_W, PROC_H)
                vx, vy = tracker.update(ux, uy)
                uz     = calc_depth_z(lms, PROC_W, PROC_H)
                hand_state.update(True, ux, uy, uz, vx, vy, pinched, pinch_dist)
                draw_hand(display, lms, pinched)
                label = f"({'GRIP' if pinched else 'open'})  x={ux:.1f}  y={uy:.1f}  z={uz:.1f}  FPS:{fps_display}"
                cv2.putText(display, label, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 200), 2)
            else:
                tracker.reset()
                hand_state.update(False)
                cv2.putText(display, f"No hand  FPS:{fps_display}", (20, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 80, 255), 2)
        else:
            cv2.putText(display, f"FPS:{fps_display}", (20, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (180, 180, 180), 1)

        fps_count += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_count
            fps_count   = 0
            fps_timer   = time.time()

        cv2.imshow("Hand Tracking Sender", display)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    stop_evt.set()
    cap.release()
    sock.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
