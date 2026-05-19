import cv2
import time
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_PATH = 'hand_landmarker.task'
MODEL_URL  = (
    'https://storage.googleapis.com/mediapipe-models/'
    'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print('Downloading hand landmark model (~8 MB)...')
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print('Download complete.')


class HandTracker:
    def __init__(self):
        _ensure_model()
        base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, frame):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        return self._landmarker.detect_for_video(mp_image, int(time.time() * 1000))

    def get_position(self, result, frame_shape, landmark_id=8):
        if not result.hand_landmarks:
            return None
        h, w = frame_shape[:2]
        lm = result.hand_landmarks[0][landmark_id]
        return (int(lm.x * w), int(lm.y * h))

    def draw(self, frame, result):
        if not result.hand_landmarks:
            return frame
        h, w = frame.shape[:2]
        pts = [
            (int(lm.x * w), int(lm.y * h))
            for lm in result.hand_landmarks[0]
        ]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 200, 0), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 230, 120), -1)
        return frame
