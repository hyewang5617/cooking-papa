import cv2
import math
import numpy as np
import time

from .hand_tracker import HandTracker
from .hand_avatar   import draw_all as draw_avatars
from .score_manager import ScoreManager
from .cooking_scene import CookingScene
from .background    import get_kitchen_bg
from .ui import (draw_panel, draw_progress_bar, draw_text, draw_text_centered,
                 dim, COLOR_PRIMARY, COLOR_SUCCESS, COLOR_DANGER,
                 COLOR_WHITE, COLOR_GREY, FONT)

SEQUENCE = [CookingScene]
POINTS   = {'cut': 10, 'stir': 15, 'flip': 20, 'grab': 0}
BONUS    = {CookingScene: 300}


def _grade(score):
    if score >= 900: return 'S'
    if score >= 650: return 'A'
    if score >= 400: return 'B'
    if score >= 200: return 'C'
    return 'D'


# ── Tutorial illustration functions ──────────────────────────────────────────

def _tut_draw_grip(frame, cx, cy):
    """Open hand (left) → fist / green (right)."""
    lx, rx = cx - 160, cx + 160
    FONT_D = cv2.FONT_HERSHEY_DUPLEX

    # Open hand — white
    cv2.ellipse(frame, (lx, cy + 22), (30, 32), 0, 0, 360, (200, 200, 235), -1)
    for fx, fw in [(-25, 11), (-9, 12), (9, 12), (25, 11)]:
        cv2.ellipse(frame, (lx + fx, cy - 28), (fw // 2, 25),
                    0, 0, 360, (200, 200, 235), -1)
        cv2.ellipse(frame, (lx + fx, cy - 28), (fw // 2, 25),
                    0, 0, 360, (110, 110, 150), 1)
    cv2.ellipse(frame, (lx - 40, cy + 7), (9, 19), -35, 0, 360, (200, 200, 235), -1)
    cv2.putText(frame, 'OPEN', (lx - 24, cy + 90), FONT_D, 0.65, (180, 180, 210), 1)

    # Arrow
    cv2.arrowedLine(frame, (lx + 58, cy), (rx - 58, cy),
                    (220, 200, 50), 3, tipLength=0.3)

    # Closed fist — green
    pts = np.array([
        [rx - 34, cy + 27], [rx - 40, cy + 4], [rx - 34, cy - 21],
        [rx, cy - 29], [rx + 34, cy - 21], [rx + 40, cy + 4], [rx + 34, cy + 27]
    ])
    cv2.fillPoly(frame, [pts], (70, 210, 70))
    cv2.polylines(frame, [pts], True, (30, 120, 30), 2)
    for kx, ky in [(rx - 20, cy - 27), (rx - 2, cy - 29),
                   (rx + 16, cy - 27), (rx + 34, cy - 21)]:
        cv2.ellipse(frame, (kx, ky), (8, 7), 0, 0, 360, (70, 210, 70), -1)
        cv2.ellipse(frame, (kx, ky), (8, 7), 0, 0, 360, (30, 120, 30), 1)
    cv2.ellipse(frame, (rx + 44, cy + 7), (11, 17), 25, 0, 360, (70, 210, 70), -1)
    cv2.putText(frame, 'GRIP = GREEN', (rx - 66, cy + 90), FONT_D, 0.65, (70, 210, 70), 1)


def _tut_draw_grab(frame, cx, cy):
    """Fist (left, green) approaching a glowing tool (right)."""
    kx, ky = cx + 95, cy - 15
    fx, fy = cx - 95, cy

    # Tool icon (knife shape)
    blade = np.array([[kx - 5, ky - 60], [kx + 5, ky - 60],
                      [kx + 8, ky + 5],  [kx - 8, ky + 5]])
    cv2.fillPoly(frame, [blade], (190, 190, 210))
    cv2.polylines(frame, [blade], True, (130, 130, 160), 2)
    cv2.rectangle(frame, (kx - 8, ky + 5), (kx + 8, ky + 28), (100, 70, 45), -1)
    cv2.rectangle(frame, (kx - 8, ky + 5), (kx + 8, ky + 28), (70, 45, 25), 2)
    # Pulsing glow ring
    r = int(40 + 10 * math.sin(time.time() * 5))
    cv2.circle(frame, (kx, ky - 15), r, (0, 200, 255), 2, cv2.LINE_AA)

    # Fist icon (green, gripping)
    fpts = np.array([
        [fx - 28, fy + 22], [fx - 34, fy + 4], [fx - 28, fy - 18],
        [fx, fy - 24],      [fx + 28, fy - 18], [fx + 34, fy + 4], [fx + 28, fy + 22]
    ])
    cv2.fillPoly(frame, [fpts], (70, 210, 70))
    cv2.polylines(frame, [fpts], True, (30, 120, 30), 2)

    # Arrow: fist → tool
    cv2.arrowedLine(frame, (fx + 44, fy), (kx - 48, ky - 15),
                    (255, 200, 50), 3, tipLength=0.3)


def _tut_draw_chop(frame, cx, cy):
    """Vertical double-arrow."""
    cv2.arrowedLine(frame, (cx, cy + 70), (cx, cy - 80),
                    (0, 240, 190), 7, tipLength=0.18)
    cv2.arrowedLine(frame, (cx, cy - 80), (cx, cy + 70),
                    (0, 200, 160), 4, tipLength=0.18)
    cv2.putText(frame, 'UP  &  DOWN', (cx - 90, cy + 112),
                cv2.FONT_HERSHEY_DUPLEX, 0.82, (0, 240, 190), 2)


def _tut_draw_stir(frame, cx, cy):
    """Clockwise circular arrow."""
    r   = 72
    pts = [(int(cx + r * math.cos(math.radians(a))),
            int(cy + r * math.sin(math.radians(a)))) for a in range(362)]
    for i in range(len(pts) - 1):
        t = i / len(pts)
        cv2.line(frame, pts[i], pts[i + 1],
                 (int(20 + 60 * t), int(175 + 80 * t), int(100 + 80 * t)), 4, cv2.LINE_AA)
    a0 = math.radians(345)
    a1 = math.radians(361)
    p0 = (int(cx + r * math.cos(a0)), int(cy + r * math.sin(a0)))
    p1 = (int(cx + r * math.cos(a1)), int(cy + r * math.sin(a1)))
    cv2.arrowedLine(frame, p0, p1, (80, 255, 180), 4, tipLength=0.55)
    cv2.putText(frame, '3  full  rounds', (cx - 88, cy + 110),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (80, 255, 180), 2)


def _tut_draw_flip(frame, cx, cy):
    """Triple upward arrows (emphasis)."""
    for ox, alpha in [(-24, 0.35), (0, 1.0), (24, 0.35)]:
        c = int(255 * alpha)
        cv2.arrowedLine(frame, (cx + ox, cy + 76), (cx + ox, cy - 76),
                        (60, c, int(120 * alpha)), max(2, int(7 * alpha)),
                        tipLength=0.22)
    cv2.putText(frame, 'FLICK  UP  FAST!', (cx - 112, cy + 112),
                cv2.FONT_HERSHEY_DUPLEX, 0.82, (60, 255, 120), 2)


# Each entry: (title, body_lines, illustration_fn_or_None)
_TUT_PAGES = [
    ('Your Virtual Hand',
     ['Your real hand appears on screen as a WHITE hand.',
      'Move your hand in front of the camera to try it!'],
     None),

    ('Make a FIST  =  Grip',
     ['Curl all 4 fingers tightly into your palm.',
      'The hand turns GREEN — that means you are gripping!'],
     _tut_draw_grip),

    ('Step 1  —  Grab the Knife',
     ['Move your GREEN FIST close to the GLOWING knife.',
      'Keep gripping — the knife will follow your hand!'],
     _tut_draw_grab),

    ('Step 1  —  Chop!',
     ['Swing your hand sharply UP then DOWN.',
      'Repeat this 5 times to chop the tomato!'],
     _tut_draw_chop),

    ('Step 2  —  Grab the Spatula',
     ['After chopping, the spatula appears.',
      'Grab it the same way — FIST near the spatula!'],
     _tut_draw_grab),

    ('Step 2  —  Stir!',
     ['Move your hand in full CIRCLES around the pot.',
      'Complete 3 full rotations to finish stirring!'],
     _tut_draw_stir),

    ('Step 3  —  Grab the Pan',
     ['Find the pan on the RIGHT side of the screen.',
      'Move your FIST close to it to grab it!'],
     _tut_draw_grab),

    ('Step 3  —  Flip!',
     ['Flick your hand sharply UPWARD to flip the food.',
      'Do this 2 times to finish cooking!'],
     _tut_draw_flip),

    ("You're Ready to Cook!",
     ['Good luck!   Press SPACE to start!'],
     None),
]


class GameManager:
    def __init__(self):
        self.tracker     = HandTracker()
        self.score       = ScoreManager()
        self.state       = 'MENU'
        self.game_idx    = 0
        self.game        = None
        self.hand_states = []

        self._countdown_start  = 0.0
        self._result_start     = 0.0
        self._game_start_score = 0
        self._last_success     = False
        self._last_game_score  = 0
        self._player_name      = ''
        self._tutorial_idx     = 0

    # ──────────────────────────────────────────────────────────── public ──────

    def update(self, frame):
        self.tracker.process(frame)          # hand tracking uses real camera frame
        self.hand_states = self.tracker.get_hand_states()

        h, w = frame.shape[:2]
        bg = get_kitchen_bg(w, h).copy()     # kitchen background for display

        output = {
            'MENU':       self._menu,
            'TUTORIAL':   self._tutorial,
            'COUNTDOWN':  self._countdown,
            'PLAYING':    self._playing,
            'RESULT':     self._result,
            'NAME_INPUT': self._name_input,
            'GAME_OVER':  self._game_over,
            'RANKING':    self._ranking,
        }.get(self.state, lambda f: f)(bg)

        return output

    def handle_key(self, key):
        if self.state == 'MENU' and key == ord(' '):
            self._tutorial_idx = 0
            self.state = 'TUTORIAL'

        elif self.state == 'MENU' and key == 13:   # Enter = skip tutorial
            self._begin_countdown()

        elif self.state == 'TUTORIAL':
            if key == ord(' '):
                self._tutorial_idx += 1
                if self._tutorial_idx >= len(_TUT_PAGES):
                    self._tutorial_idx = 0
                    self._begin_countdown()
            elif key == 13:              # Enter = skip to game
                self._tutorial_idx = 0
                self._begin_countdown()
            elif key == 27:             # ESC = back to menu
                self._tutorial_idx = 0
                self.state = 'MENU'

        elif self.state == 'NAME_INPUT':
            if key == 13 and self._player_name:
                self.score.save(self._player_name)
                self.state = 'GAME_OVER'
            elif key == 8:
                self._player_name = self._player_name[:-1]
            elif 32 <= key <= 126 and len(self._player_name) < 12:
                self._player_name += chr(key)

        elif self.state == 'GAME_OVER':
            if key == ord(' '):
                self.state = 'RANKING'
            elif key == ord('r'):
                self._reset()

        elif self.state == 'RANKING':
            if key in (ord(' '), ord('r')):
                self._reset()

    # ─────────────────────────────────────────────────────────── private ─────

    def _begin_countdown(self):
        self._countdown_start = time.time()
        self.state = 'COUNTDOWN'

    def _launch_game(self):
        cls       = SEQUENCE[self.game_idx]
        self.game = cls()
        self.game.start()
        self._game_start_score = self.score.total_score
        self.state = 'PLAYING'

    def _reset(self):
        self.state        = 'MENU'
        self.game_idx     = 0
        self.game         = None
        self._player_name = ''
        self.score.reset()

    # ─────────────────────────────────────────────────────────── states ──────

    def _menu(self, frame):
        h, w = frame.shape[:2]
        dim(frame, 0.5)

        draw_text_centered(frame, 'AR  Cooking  Mama',
                           h // 5, scale=1.8, color=COLOR_PRIMARY, thickness=3)
        draw_text_centered(frame, 'Webcam + Hand Simulator',
                           h // 5 + 65, scale=0.9, color=COLOR_WHITE)

        if int(time.time() * 2) % 2 == 0:
            draw_text_centered(frame, 'Press  SPACE  to learn how to play',
                               h * 4 // 5, scale=1.1, color=COLOR_SUCCESS, thickness=2)
        draw_text_centered(frame, 'Press  ENTER  to skip tutorial and start',
                           h * 4 // 5 + 50, scale=0.75, color=COLOR_GREY)
        return frame

    def _tutorial(self, frame):
        h, w = frame.shape[:2]
        dim(frame, 0.52)

        title, body, draw_fn = _TUT_PAGES[self._tutorial_idx]
        total = len(_TUT_PAGES)
        last  = self._tutorial_idx == total - 1

        # ── Page indicator dots ───────────────────────────────────────────────
        dot_cx = w // 2 - (total * 22) // 2
        for i in range(total):
            filled = (i == self._tutorial_idx)
            color  = COLOR_PRIMARY if filled else COLOR_GREY
            cv2.circle(frame, (dot_cx + i * 22, 40), 6, color, -1 if filled else 2)

        # ── Title ─────────────────────────────────────────────────────────────
        draw_text_centered(frame, title,
                           h // 6 + 10, scale=1.5, color=COLOR_PRIMARY, thickness=3)

        # ── Illustration ──────────────────────────────────────────────────────
        if draw_fn:
            draw_fn(frame, w // 2, h // 2 - 10)
        else:
            # Page 1: hint arrow toward where the hand avatar will appear
            if self._tutorial_idx == 0:
                draw_text_centered(frame, '(your hand appears here)',
                                   h // 2 + 10, scale=0.85, color=(160, 160, 190))
                for ox in [-12, 0, 12]:
                    cv2.arrowedLine(frame,
                                    (w // 2 + ox, h // 2 + 55),
                                    (w // 2 + ox, h // 2 + 100),
                                    (160, 160, 190), 2, tipLength=0.4)

        # ── Body text ─────────────────────────────────────────────────────────
        text_y = h * 3 // 4
        for i, line in enumerate(body):
            draw_text_centered(frame, line, text_y + i * 44,
                               scale=0.85, color=COLOR_WHITE)

        # ── Navigation hint ───────────────────────────────────────────────────
        hint = 'SPACE: Start Cooking!' if last else 'SPACE: Next  →'
        if int(time.time() * 2) % 2 == 0:
            draw_text_centered(frame, hint, h - 55,
                               scale=1.0, color=COLOR_SUCCESS, thickness=2)
        if not last:
            draw_text_centered(frame, 'ENTER: skip tutorial   ESC: back to menu',
                               h - 22, scale=0.6, color=COLOR_GREY)

        return frame

    def _countdown(self, frame):
        h, w    = frame.shape[:2]
        elapsed = time.time() - self._countdown_start
        dim(frame, 0.5)

        if self.game_idx < len(SEQUENCE):
            cls = SEQUENCE[self.game_idx]
            draw_text_centered(frame, cls.name,        h // 4,
                               scale=1.8, color=COLOR_PRIMARY, thickness=3)
            draw_text_centered(frame, cls.instruction, h // 4 + 70,
                               scale=0.85, color=COLOR_WHITE)

        remaining = 3 - int(elapsed)
        if remaining > 0:
            draw_text_centered(frame, str(remaining), h // 2 + 40,
                               scale=6.0, color=(0, 255, 255), thickness=10)
        else:
            draw_text_centered(frame, 'GO!', h // 2 + 40,
                               scale=6.0, color=COLOR_SUCCESS, thickness=10)
            if elapsed >= 4.0:
                self._launch_game()
        return frame

    def _playing(self, frame):
        events = self.game.update(self.hand_states)

        for ev in events:
            self.score.add_score(POINTS.get(ev, 10))

        frame = self.game.draw(frame, self.hand_states)
        self._hud(frame)

        if self.game.check_done():
            self._last_success    = self.game.succeeded
            if self._last_success:
                base   = BONUS.get(type(self.game), 100)
                time_b = int(self.game.time_remaining * 3)
                self.score.add_bonus(base + time_b)
            self._last_game_score = self.score.total_score - self._game_start_score
            self.score.record_game(self.game.name, self._last_game_score)
            self.game_idx += 1
            self._result_start = time.time()
            self.state = 'RESULT'
        return frame

    def _result(self, frame):
        h, w    = frame.shape[:2]
        elapsed = time.time() - self._result_start
        dim(frame, 0.55)

        label = 'SUCCESS!' if self._last_success else 'TIME UP!'
        color = COLOR_SUCCESS if self._last_success else COLOR_DANGER
        draw_text_centered(frame, label,                       h // 3,
                           scale=3.0, color=color, thickness=5)
        draw_text_centered(frame, f'+{self._last_game_score}', h // 2,
                           scale=2.0, color=COLOR_PRIMARY, thickness=3)
        draw_text_centered(frame, f'Total: {self.score.total_score}',
                           h // 2 + 70, scale=1.1, color=COLOR_WHITE)

        if elapsed > 3.0:
            if self.game_idx >= len(SEQUENCE):
                self.state = 'NAME_INPUT'
            else:
                self._begin_countdown()
        return frame

    def _name_input(self, frame):
        h, w = frame.shape[:2]
        dim(frame, 0.7)

        draw_text_centered(frame, 'All Clear!',
                           h // 5,      scale=2.5, color=COLOR_PRIMARY, thickness=4)
        draw_text_centered(frame, f'Final Score: {self.score.total_score}',
                           h // 3,      scale=1.6, color=COLOR_WHITE, thickness=2)
        draw_text_centered(frame, 'Enter your name:',
                           h // 2 - 20, scale=1.0, color=COLOR_GREY)

        bw, bh = 420, 62
        bx, by = (w - bw) // 2, h // 2 + 20
        draw_panel(frame, bx, by, bw, bh, alpha=0.7, color=(20, 20, 20))
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), COLOR_PRIMARY, 2)
        cursor = '_' if int(time.time() * 2) % 2 == 0 else ' '
        cv2.putText(frame, self._player_name + cursor, (bx + 14, by + 44),
                    FONT, 1.3, COLOR_WHITE, 2)
        draw_text_centered(frame, 'Press Enter to save', h * 4 // 5,
                           scale=0.9, color=COLOR_GREY)
        return frame

    def _game_over(self, frame):
        h, w = frame.shape[:2]
        dim(frame, 0.7)

        draw_text_centered(frame, 'GAME  OVER',
                           h // 6, scale=2.8, color=COLOR_PRIMARY, thickness=4)
        draw_text_centered(frame, f'Final Score: {self.score.total_score}',
                           h // 3, scale=1.8, color=COLOR_WHITE, thickness=3)

        for i, gs in enumerate(self.score.game_scores):
            draw_text_centered(frame, f"{gs['game']}:  +{gs['score']}",
                               h // 2 + i * 50, scale=0.9, color=COLOR_GREY)

        draw_text_centered(frame, f'Grade:  {_grade(self.score.total_score)}',
                           h * 3 // 4, scale=1.6, color=COLOR_SUCCESS, thickness=2)
        draw_text_centered(frame, 'SPACE: Ranking     R: Retry',
                           h * 5 // 6, scale=0.9, color=COLOR_GREY)
        return frame

    def _ranking(self, frame):
        h, w = frame.shape[:2]
        dim(frame, 0.8)

        draw_text_centered(frame, 'RANKING', 70, scale=2.2, color=COLOR_PRIMARY, thickness=3)

        rank_colors = [COLOR_PRIMARY, (180, 180, 220), (0, 165, 210)]
        for i, entry in enumerate(self.score.rankings[:8]):
            color = rank_colors[i] if i < 3 else COLOR_WHITE
            text  = f"  {i+1}.  {entry['name']:<12}  {entry['score']:>6}    {entry['date']}"
            draw_text_centered(frame, text, 145 + i * 58, scale=0.85, color=color)

        draw_text_centered(frame, 'SPACE / R: Play again', h - 55,
                           scale=1.0, color=COLOR_GREY)
        return frame

    def _hud(self, frame):
        h, w = frame.shape[:2]
        draw_panel(frame, 0, 0, w, 75)

        draw_text(frame, self.game.name, (18, 50), scale=1.1,
                  color=COLOR_PRIMARY, thickness=2)

        if not self.game.timer_started:
            t_str = 'GRAB the knife!'
            t_col = (0, 200, 255)
        else:
            t = self.game.time_remaining
            t_str = f'TIME  {int(t) + 1:02d}'
            t_col = COLOR_DANGER if t < 15 else COLOR_WHITE
        sz = cv2.getTextSize(t_str, FONT, 1.0, 2)[0]
        draw_text(frame, t_str, (w - sz[0] - 18, 50), scale=1.0, color=t_col, thickness=2)

        s_str = f'Score: {self.score.total_score}'
        ssz   = cv2.getTextSize(s_str, FONT, 0.85, 2)[0]
        draw_text(frame, s_str, ((w - ssz[0]) // 2, 50), scale=0.85, color=COLOR_WHITE)

        draw_progress_bar(frame, 15, h - 28, w - 30, 14, self.game.time_ratio,
                          color=COLOR_DANGER if (self.game.timer_started and
                                                  self.game.time_remaining < 15)
                                            else COLOR_SUCCESS)

        pt = self.game.progress_text
        if pt:
            draw_text(frame, pt, (18, 110), scale=0.85, color=COLOR_WHITE)
