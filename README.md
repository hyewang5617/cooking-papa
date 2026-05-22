# AR Cooking Mama

웹캠 기반 손 추적으로 Unity 3D 조리도구를 직접 잡고 움직이는 쿠킹마마 스타일 미니게임

> 컴퓨터비전 수업 텀프로젝트

---

## 데모

<!-- 완성 후 GIF / 스크린샷 추가 -->
*Coming soon*

---

## 프로젝트 소개

기존 키보드·마우스 대신 **실제 손 움직임**을 입력으로 사용합니다.  
Python이 웹캠 영상에서 손을 추적하고, Unity가 그 데이터를 받아 3D 조리도구를 제어합니다.

| 미션 | 동작 | 판정 |
|------|------|------|
| 칼질 (Cutting) | 식칼을 잡고 위아래로 빠르게 움직임 | Y 속도 방향 전환 횟수 |
| 휘핑 (Whisking) | 휘핑기를 잡고 원형으로 돌림 | 360° 회전 누적 횟수 |

---

## 시스템 구조

```
[웹캠]
  │
  ▼
[Python: hand_tracking_sender.py]
  │  MediaPipe HandLandmarker
  │  - 손 중심 좌표 (Unity 좌표계로 스케일링)
  │  - pinch 여부 (엄지-검지 거리)
  │  - 손 속도 (vx, vy)
  │
  │  UDP JSON (port 5052)
  │
  ▼
[Unity: AR-cooking-mama]
  ├─ UDPReceiver.cs      ← JSON 수신·파싱
  ├─ ToolController.cs   ← 도구 이동 / pinch로 잡기
  ├─ CuttingGameManager  ← 칼질 미션 판정
  ├─ WhiskingGameManager ← 휘핑 미션 판정
  ├─ ScoreManager        ← 점수·콤보 계산
  └─ RankingManager      ← rankings.json 저장
```

---

## 사용 기술

| 분류 | 기술 |
|------|------|
| 언어 | Python 3.12, C# |
| 비전 | OpenCV, MediaPipe Hands |
| 게임 엔진 | Unity 6 |
| 통신 | UDP Socket (127.0.0.1:5052) |
| 데이터 | JSON |

---

## 실행 방법

### 사전 준비

```bash
pip install opencv-python mediapipe numpy
```

### 1. Python 송신기 실행

```bash
python hand_tracking_sender.py
```

첫 실행 시 `hand_landmarker.task` (~8 MB) 를 자동 다운로드합니다.  
웹캠 창이 열리고 손이 인식되면 터미널에 좌표가 출력됩니다.

### 2. Unity 수신기 실행

1. Unity에서 `AR-cooking-mama` 프로젝트를 열기
2. `UDPReceiver` 스크립트가 붙은 오브젝트 확인 (port: 5052)
3. Play 버튼 클릭

> Python과 Unity는 **같은 PC에서** 실행한다고 가정합니다.

### 조작법

| 동작 | 효과 |
|------|------|
| 손을 도구 근처로 가져가기 | 도구 하이라이트 |
| 엄지 + 검지 붙이기 (pinch) | 도구 잡기 |
| 잡은 상태에서 위아래 빠르게 | 칼질 판정 |
| 잡은 상태에서 원형으로 | 휘핑 판정 |
| ESC / Q | 종료 |

---

## Unity 씬 구성

```
Main Scene
├── [Camera] Main Camera
│     position: (0, 0, -10)
├── [Empty] GameManager
│     ├── UDPReceiver.cs
│     ├── ScoreManager.cs
│     └── RankingManager.cs
├── [Cube] Knife          ← ToolController (type: Knife)
├── [Cylinder] Whisk      ← ToolController (type: Whisk)
├── [Cube] Ingredient     ← 재료 오브젝트
├── [Sphere] Bowl         ← 그릇 오브젝트
└── [Canvas] UI
      ├── Text_Score
      ├── Text_Combo
      ├── Text_Timer
      └── Text_Ranking
```

> 외부 에셋 없이 Unity 기본 Primitive(Cube, Cylinder, Sphere)로 테스트 가능합니다.

---

## UDP 데이터 형식

Python → Unity 방향, 매 프레임 전송

```json
{
  "detected":   true,
  "x":          2.34,
  "y":         -1.12,
  "vx":        -0.5,
  "vy":         3.2,
  "pinched":    false,
  "pinch_dist": 0.08
}
```

| 필드 | 설명 |
|------|------|
| `x`, `y` | Unity 월드 좌표 (X: −8~8, Y: −4.5~4.5) |
| `vx`, `vy` | Unity 단위/초 속도 |
| `pinched` | 엄지-검지 거리 < 0.06 이면 true |

---

## 한계점

- 조명이 어두우면 손 인식률 저하
- 단일 손만 지원 (멀티핸드 미구현)
- Unity 씬에 실제 3D 에셋 없이 Primitive 사용
- 네트워크 지연 (로컬 UDP라 실사용 영향 없음)
- 손이 화면 밖으로 나가면 마지막 좌표 유지

---

## 참고자료

- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [OpenCV Python Docs](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Unity UDP Socket 통신](https://docs.unity3d.com/ScriptReference/Network.html)
