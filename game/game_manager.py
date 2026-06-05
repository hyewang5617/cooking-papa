import cv2
import math
import numpy as np
import time

from .hand_tracker import HandTracker
from .hand_avatar   import draw_all as draw_avatars
from .score_manager import ScoreManager
from .cooking_scene  import CookingScene
from .pancake_scene  import PancakeScene
from .background     import get_kitchen_bg
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
    """Counter-clockwise circular arrow."""
    r   = 72
    pts = [(int(cx + r * math.cos(math.radians(a))),
            int(cy + r * math.sin(math.radians(a)))) for a in range(362)]
    for i in range(len(pts) - 1):
        t = i / len(pts)
        cv2.line(frame, pts[i], pts[i + 1],
                 (int(20 + 60 * t), int(175 + 80 * t), int(100 + 80 * t)), 4, cv2.LINE_AA)
    # Clockwise: arrowhead near 360° pointing downward (from 345° → 361°)
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


def _tut_draw_knife(frame, cx, cy):
    """Simple cartoon knife for tutorial interact page."""
    # Blade
    blade = np.array([
        (cx, cy - 70), (cx + 14, cy - 50),
        (cx + 10, cy + 20), (cx - 10, cy + 20),
        (cx - 14, cy - 50),
    ], np.int32)
    cv2.fillPoly(frame, [blade], (170, 175, 180))
    cv2.polylines(frame, [blade], True, (220, 225, 230), 2)
    # Handle
    cv2.rectangle(frame, (cx - 12, cy + 20), (cx + 12, cy + 70), (80, 55, 35), -1)
    cv2.rectangle(frame, (cx - 12, cy + 20), (cx + 12, cy + 70), (110, 80, 50), 2)
    # Guard
    cv2.rectangle(frame, (cx - 18, cy + 16), (cx + 18, cy + 26), (145, 150, 155), -1)


# Each entry: (title, body_lines, page_tag)
# page_tag is used by the renderer for special interactive pages
_TUT_PAGES = [
    ('Show your hand to the camera!',
     ['Your hand appears on screen as a white avatar.',
      'TIP: Face your PALM toward the camera for best tracking.'],
     'show_hand'),

    ('Move your hand left and right!',
     ['The on-screen hand follows your real hand.',
      'Try waving slowly in front of the camera.'],
     'wave'),

    ('Make a FIST  =  Grip!',
     ['Curl 4 fingers toward your palm to grip.',
      'TIP: No need to squeeze hard -- a light fist works!'],
     'grip'),

    ('Grip to grab tools or press buttons!',
     ['Try grabbing the knife on the left, or pressing the button on the right!'],
     'interact'),

    ("Are you ready?",
     ["Let's start cooking!"],
     'ready'),
]


class GameManager:
    def __init__(self):
        self.tracker     = HandTracker()
        self.score       = ScoreManager()
        self.state       = 'TUTORIAL'
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
        self._debug_jump_idx   = -1
        self._tut_knife_grabbed = False
        self._tut_btn_held      = 0
        self._tut_knife_pos     = [280, 360]
        # Menu button grab state
        self._menu_btn_hold    = [0, 0, 0]
        self._MENU_HOLD_FRAMES = 22
        # Stage select grab state
        self._stage_btn_hold   = [0, 0]
        self._selected_scene   = CookingScene   # default
        self._last_category    = 'steak'        # 'steak' or 'pancake'

    # ──────────────────────────────────────────────────────────── public ──────

    def update(self, frame):
        self.tracker.process(frame)          # hand tracking uses real camera frame
        self.hand_states = self.tracker.get_hand_states()
        # Switch num_hands based on current stage requirement
        if self.game and hasattr(self.game, 'required_hands'):
            req = self.game.required_hands
            if req != self.tracker.num_hands:
                self.tracker.set_num_hands(req)

        h, w = frame.shape[:2]
        bg = get_kitchen_bg(w, h).copy()     # kitchen background for display

        output = {
            'MENU':         self._menu,
            'STAGE_SELECT': self._stage_select,
            'TUTORIAL':     self._tutorial,
            'COUNTDOWN':    self._countdown,
            'PLAYING':    self._playing,
            'RESULT':     self._result,
            'NAME_INPUT': self._name_input,
            'GAME_OVER':  self._game_over,
            'RANKING':    self._ranking,
        }.get(self.state, lambda f: f)(bg)

        draw_avatars(output, self.hand_states, scale=0.5)

        return output

    def handle_key(self, key):
        # ── DEBUG: number keys 1-9 jump to specific phase (MENU or PLAYING) ──
        if key in range(ord('1'), ord('9') + 1):
            idx = key - ord('1')
            if self.state in ('MENU', 'STAGE_SELECT'):
                self._begin_countdown()
                self._debug_jump_idx = idx
            elif self.state == 'PLAYING' and self.game and hasattr(self.game, 'jump_to_phase'):
                phases = self.game._DEBUG_PHASES
                if idx < len(phases):
                    self.game.jump_to_phase(phases[idx])
            return

        if self.state == 'MENU' and key == ord(' '):
            self._tutorial_idx = 0
            self.state = 'TUTORIAL'

        elif self.state == 'MENU' and key == 13:   # Enter → stage select
            self.state = 'STAGE_SELECT'
            self._stage_btn_hold = [0, 0]

        elif self.state == 'STAGE_SELECT':
            if key == 27 or key == ord('q'):   # ESC → back to menu
                self.state = 'MENU'
            elif key == 13:                    # Enter → steak (only stage available)
                self._begin_countdown()

        elif self.state == 'TUTORIAL':
            if key == ord(' '):
                self._tutorial_idx += 1
                if self._tutorial_idx >= len(_TUT_PAGES):
                    self._tutorial_idx = 0
                    self.state = 'MENU'   # tutorial done → title
            elif key == 13:              # Enter = skip to title
                self._tutorial_idx = 0
                self.state = 'MENU'
            elif key == 27:             # ESC = title (if came from menu)
                self._tutorial_idx = 0
                self.state = 'MENU'

        elif self.state == 'NAME_INPUT':
            if key == 13 and self._player_name:
                self.score.save(self._player_name, self._last_category)
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
        cls       = getattr(self, '_selected_scene', CookingScene)
        self._last_category = 'pancake' if cls is PancakeScene else 'steak'
        self.game = cls()
        self.game.start()
        self._game_start_score = self.score.total_score
        self.state = 'PLAYING'
        # Debug: jump to specific phase if requested
        if self._debug_jump_idx >= 0:
            phases = self.game._DEBUG_PHASES
            if self._debug_jump_idx < len(phases):
                self.game.jump_to_phase(phases[self._debug_jump_idx])
            self._debug_jump_idx = -1

    def _reset(self):
        self.state        = 'MENU'
        self.game_idx     = 0
        self.game         = None
        self._player_name = ''
        self.score.reset()

    # ─────────────────────────────────────────────────────────── states ──────

    def _menu(self, frame):
        import math as _math
        h, w = frame.shape[:2]
        t = time.time()

        # ── Title badge ───────────────────────────────────────────────────────
        badge_w, badge_h = 640, 160
        bx = (w - badge_w) // 2
        by = h // 2 - 220
        cv2.rectangle(frame, (bx + 7, by + 7), (bx + badge_w + 7, by + badge_h + 7),
                      (20, 18, 16), -1)
        for i in range(badge_h):
            frac = i / badge_h
            cv2.line(frame, (bx, by + i), (bx + badge_w, by + i),
                     (int(20 + 10*frac), int(180 + 60*frac), int(255 - 30*frac)), 1)
        cv2.rectangle(frame, (bx, by), (bx + badge_w, by + badge_h), (0, 160, 255), 7)
        cv2.rectangle(frame, (bx+3, by+3), (bx+badge_w-3, by+badge_h-3), (0,220,255), 2)

        # Title text — centered correctly
        draw_text_centered(frame, 'Cooking  Papa',
                           by + 100, scale=2.4, color=(255,255,255), thickness=6)
        draw_text_centered(frame, 'Cooking  Papa',
                           by + 100, scale=2.4, color=(0, 80, 220), thickness=2)

        # ── Strawberry (left of badge) ────────────────────────────────────────
        _draw_strawberry(frame, bx - 85, by + badge_h // 2, size=75)

        # ── Broccoli (right of badge) ─────────────────────────────────────────
        _draw_broccoli(frame, bx + badge_w + 85, by + badge_h // 2, size=75)

        # ── Three buttons side by side ────────────────────────────────────────
        btn_labels = ['START', 'TUTORIAL', 'EXIT']
        btn_colors = [(36, 170, 56), (36, 130, 210), (55, 55, 175)]
        btn_w, btn_h = 210, 68
        gap      = 28
        total_w  = 3 * btn_w + 2 * gap
        btn_y    = h // 2 + 30
        btns_x0  = (w - total_w) // 2

        hand_sx, hand_sy, hand_gripped = -1, -1, False
        if self.hand_states:
            hs = self.hand_states[0]
            if hs.detected:
                hand_sx, hand_sy, hand_gripped = hs.screen_x, hs.screen_y, hs.gripped

        activated = -1
        for bi in range(3):
            bx_btn = btns_x0 + bi * (btn_w + gap)
            cx_btn = bx_btn + btn_w // 2
            cy_btn = btn_y + btn_h // 2
            over   = (bx_btn <= hand_sx <= bx_btn + btn_w and
                      btn_y  <= hand_sy <= btn_y  + btn_h and hand_gripped)
            if over:
                self._menu_btn_hold[bi] += 1
            else:
                self._menu_btn_hold[bi] = max(0, self._menu_btn_hold[bi] - 2)
            hold_frac = self._menu_btn_hold[bi] / self._MENU_HOLD_FRAMES
            if hold_frac >= 1.0:
                activated = bi
                self._menu_btn_hold = [0, 0, 0]

            pulse = int(5 * abs(_math.sin(t * 4 + bi))) if over else 0
            col   = btn_colors[bi]
            draw_col = tuple(min(255, c + 55) for c in col) if over else col
            # Shadow
            cv2.rectangle(frame, (bx_btn+5, btn_y+5), (bx_btn+btn_w+5, btn_y+btn_h+5),
                          (20,18,16), -1)
            cv2.rectangle(frame, (bx_btn-pulse, btn_y-pulse),
                          (bx_btn+btn_w+pulse, btn_y+btn_h+pulse), draw_col, -1)
            cv2.rectangle(frame, (bx_btn, btn_y), (bx_btn+btn_w, btn_y+btn_h),
                          (255,255,255), 2)
            # Button label — centered within button box
            lbl = btn_labels[bi]
            (tw, th), _ = cv2.getTextSize(lbl, FONT, 1.0, 2)
            cv2.putText(frame, lbl, (cx_btn - tw//2, cy_btn + th//2),
                        FONT, 1.0, (255,255,255), 2, cv2.LINE_AA)
            # Hold progress bar
            if hold_frac > 0:
                bar_w = int((btn_w - 14) * hold_frac)
                cv2.rectangle(frame, (bx_btn+7, btn_y+btn_h-10),
                              (bx_btn+7+bar_w, btn_y+btn_h-4), (255,255,255), -1)

        if activated == 0:
            self.state = 'STAGE_SELECT'
            self._stage_btn_hold = [0, 0]
        elif activated == 1:
            self._tutorial_idx = 0
            self.state = 'TUTORIAL'
        elif activated == 2:
            import sys; sys.exit(0)

        draw_text_centered(frame, 'Grip  over  a  button  to  select',
                           btn_y + btn_h + 44, scale=0.68,
                           color=(200, 200, 220), thickness=1)

        # Debug shortcut legend (bottom of screen)
        from .cooking_scene import CookingScene as _CS
        phases = _CS._DEBUG_PHASES
        hint = '  '.join(f'{i+1}:{p.replace("_"," ")}' for i, p in enumerate(phases))
        draw_panel(frame, 0, h - 40, w, 40, alpha=0.75)
        draw_text_centered(frame, f'DEBUG:  {hint}',
                           h - 18, scale=0.42, color=(120, 120, 140), thickness=1)
        return frame

    def _stage_select(self, frame):
        import math as _math
        h, w = frame.shape[:2]
        t    = time.time()

        # Dark overlay
        frame[:] = (22, 20, 18)

        # Title
        draw_text_centered(frame, 'Select  Stage', 60, scale=1.1,
                           color=COLOR_PRIMARY, thickness=2)

        # Card definitions: (label, subtitle, locked)
        stages = [
            ('Pancakes',       'Play Now!',  False),
            ('Salisbury Steak', 'Play Now!', False),
        ]
        card_w, card_h = 480, 420
        gap            = 60
        total_w        = 2 * card_w + gap
        x0             = (w - total_w) // 2
        card_y         = h // 2 - card_h // 2

        hand_sx, hand_sy, hand_gripped = -1, -1, False
        if self.hand_states:
            hs = self.hand_states[0]
            if hs.detected:
                hand_sx, hand_sy, hand_gripped = hs.screen_x, hs.screen_y, hs.gripped

        activated = -1
        for bi, (label, sub, locked) in enumerate(stages):
            cx_c = x0 + bi * (card_w + gap) + card_w // 2
            cy_c = h // 2

            over = (not locked and
                    x0 + bi*(card_w+gap) <= hand_sx <= x0 + bi*(card_w+gap) + card_w and
                    card_y <= hand_sy <= card_y + card_h and hand_gripped)

            if over:
                self._stage_btn_hold[bi] = min(self._stage_btn_hold[bi] + 1, self._MENU_HOLD_FRAMES + 1)
            else:
                self._stage_btn_hold[bi] = max(0, self._stage_btn_hold[bi] - 2)

            hold_frac = self._stage_btn_hold[bi] / self._MENU_HOLD_FRAMES
            if hold_frac >= 1.0 and not locked:
                activated = bi

            # Card background
            pulse = int(8 * abs(_math.sin(t * 4))) if over else 0
            base_col = (45, 42, 38) if not locked else (30, 28, 26)
            border_col = COLOR_PRIMARY if over else ((100, 100, 110) if locked else (80, 85, 95))
            bx = x0 + bi * (card_w + gap)
            cv2.rectangle(frame, (bx - pulse, card_y - pulse),
                          (bx + card_w + pulse, card_y + card_h + pulse), base_col, -1)
            cv2.rectangle(frame, (bx - pulse, card_y - pulse),
                          (bx + card_w + pulse, card_y + card_h + pulse), border_col, 3)

            # Stage icon
            icon_cy = card_y + card_h // 2 - 40
            if bi == 0:   # Pancake icon
                _draw_pancake_icon(frame, cx_c, icon_cy)
            else:         # Steak icon
                _draw_steak_icon(frame, cx_c, icon_cy)

            # Label below icon (both cards)
            if locked:
                (tw, th), _ = cv2.getTextSize(label, FONT, 0.90, 2)
                cv2.putText(frame, label, (cx_c - tw//2, card_y + card_h - 88),
                            FONT, 0.90, (140, 140, 160), 2, cv2.LINE_AA)
            if not locked:
                # Label centered within the card
                for txt, scale, col, y_off in [
                    (label, 0.90, COLOR_WHITE,   card_y + card_h - 88),
                    (sub,   0.60, COLOR_SUCCESS,  card_y + card_h - 52),
                ]:
                    (tw, th), _ = cv2.getTextSize(txt, FONT, scale, 2)
                    cv2.putText(frame, txt, (cx_c - tw//2, y_off),
                                FONT, scale, col, 2, cv2.LINE_AA)
                # Hold bar
                if hold_frac > 0:
                    bar_w = int((card_w - 40) * hold_frac)
                    cv2.rectangle(frame,
                                  (bx + 20, card_y + card_h - 28),
                                  (bx + 20 + bar_w, card_y + card_h - 16),
                                  COLOR_PRIMARY, -1)

        if activated == 0:   # Pancake selected
            self._selected_scene = PancakeScene
            self._begin_countdown()
        elif activated == 1:   # Steak selected
            self._selected_scene = CookingScene
            self._begin_countdown()

        draw_text_centered(frame, 'Grip  over  a  card  to  select  |  ESC: back',
                           h - 28, scale=0.55, color=COLOR_GREY, thickness=1)
        return frame

    def _tutorial(self, frame):
        import math as _m
        h, w = frame.shape[:2]
        t    = time.time()

        # Black background
        frame[:] = (18, 16, 14)

        title, body, tag = _TUT_PAGES[self._tutorial_idx]
        total = len(_TUT_PAGES)
        last  = (self._tutorial_idx == total - 1)

        # Hand state
        hand_sx, hand_sy, hand_gripped = -1, -1, False
        if self.hand_states:
            hs = self.hand_states[0]
            if hs.detected:
                hand_sx, hand_sy, hand_gripped = hs.screen_x, hs.screen_y, hs.gripped

        # ── Page dots ────────────────────────────────────────────────────────
        dot_gap = 28
        dot_x0  = w // 2 - (total - 1) * dot_gap // 2
        for i in range(total):
            col = COLOR_PRIMARY if i == self._tutorial_idx else (80, 80, 90)
            cv2.circle(frame, (dot_x0 + i * dot_gap, 32), 7, col, -1 if i == self._tutorial_idx else 2)

        # ── Title ─────────────────────────────────────────────────────────────
        draw_text_centered(frame, title, 80, scale=0.95, color=COLOR_PRIMARY, thickness=1)

        # ── Illustration area ─────────────────────────────────────────────────
        ill_cy = h // 2 - 20

        if tag == 'show_hand':
            # Single arrow pointing down to where hand will appear
            ax = w // 2
            cv2.arrowedLine(frame, (ax, ill_cy - 70), (ax, ill_cy + 50),
                            (120, 120, 150), 3, tipLength=0.25)
            draw_text_centered(frame, 'Your hand appears here',
                               ill_cy + 90, scale=0.65, color=(140, 140, 170))

        elif tag == 'wave':
            # Left-right arrow
            ax = w // 2
            cv2.arrowedLine(frame, (ax + 120, ill_cy), (ax - 120, ill_cy),
                            (80, 200, 255), 4, tipLength=0.18)
            cv2.arrowedLine(frame, (ax - 120, ill_cy), (ax + 120, ill_cy),
                            (80, 200, 255), 4, tipLength=0.18)
            draw_text_centered(frame, 'Move your hand left and right!',
                               ill_cy + 60, scale=0.65, color=(80, 200, 255))

        elif tag == 'grip':
            # Open hand → fist illustration
            _tut_draw_grip(frame, w // 2, ill_cy)

        elif tag == 'interact':
            # Interactive: knife on left, button on right
            knife_cx, knife_cy = 320, ill_cy
            btn_cx,   btn_cy   = 960, ill_cy
            btn_r = 55

            # Knife
            kx = int(self._tut_knife_pos[0])
            ky = int(self._tut_knife_pos[1])
            _tut_draw_knife(frame, kx, ky)

            # Grab ring on knife if not held
            if not self._tut_knife_grabbed:
                kr = int(52 + 10 * abs(_m.sin(t * 5)))
                cv2.circle(frame, (kx, ky), kr, (0, 200, 255), 2)
                draw_text_centered(frame, 'GRAB', ky - kr - 18, scale=0.55, color=(0, 200, 255))

            # Button
            pulse = int(8 * abs(_m.sin(t * 4))) if not self._tut_btn_held else 0
            bcol  = (40, 200, 80) if self._tut_btn_held >= 20 else (40, 130, 210)
            cv2.circle(frame, (btn_cx, btn_cy), btn_r + pulse, bcol, -1)
            cv2.circle(frame, (btn_cx, btn_cy), btn_r + pulse, (255, 255, 255), 3)
            lbl2 = 'OK!' if self._tut_btn_held >= 20 else 'PRESS'
            (tw2, th2), _ = cv2.getTextSize(lbl2, FONT, 0.9, 2)
            cv2.putText(frame, lbl2, (btn_cx - tw2//2, btn_cy + th2//2),
                        FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            # Grab ring on button if not pressed
            if self._tut_btn_held < 20:
                kr2 = int(btn_r + 16 + 8 * abs(_m.sin(t * 5)))
                cv2.circle(frame, (btn_cx, btn_cy), kr2, (0, 200, 255), 2)
                draw_text_centered(frame, 'GRAB', btn_cy - kr2 - 18,
                                   scale=0.55, color=(0, 200, 255))

            # Progress bar under button
            if 0 < self._tut_btn_held < 20:
                bw2 = int((btn_r * 2) * self._tut_btn_held / 20)
                cv2.rectangle(frame, (btn_cx - btn_r, btn_cy + btn_r + 10),
                              (btn_cx - btn_r + bw2, btn_cy + btn_r + 18),
                              (40, 200, 80), -1)

            # Update interact state
            if hand_gripped:
                # Knife grab
                if not self._tut_knife_grabbed:
                    if (hand_sx - kx)**2 + (hand_sy - ky)**2 < 70**2:
                        self._tut_knife_grabbed = True
                else:
                    self._tut_knife_pos = [hand_sx, hand_sy]

                # Button hold
                if (hand_sx - btn_cx)**2 + (hand_sy - btn_cy)**2 < (btn_r + 30)**2:
                    self._tut_btn_held = min(self._tut_btn_held + 1, 22)
                else:
                    self._tut_btn_held = max(0, self._tut_btn_held - 1)
            else:
                if self._tut_knife_grabbed:
                    self._tut_knife_grabbed = False
                self._tut_btn_held = max(0, self._tut_btn_held - 2)

            draw_text_centered(frame, 'Grip the knife on the LEFT  or  press the button on the RIGHT!',
                               ill_cy + 120, scale=0.62, color=(200, 200, 210))

        elif tag == 'ready':
            draw_text_centered(frame, "Are you ready?", ill_cy - 30,
                               scale=1.6, color=(80, 255, 160), thickness=3)
            draw_text_centered(frame, "Let's start cooking!", ill_cy + 50,
                               scale=1.1, color=COLOR_WHITE, thickness=2)

        # ── Body text ─────────────────────────────────────────────────────────
        text_y = h * 3 // 4 + 10
        for i, line in enumerate(body):
            draw_text_centered(frame, line, text_y + i * 38,
                               scale=0.62, color=(200, 200, 210), thickness=1)

        # ── Nav hint ──────────────────────────────────────────────────────────
        nav = 'SPACE: Title!' if last else 'SPACE: Next  ->'
        if int(t * 2) % 2 == 0:
            draw_text_centered(frame, nav, h - 52, scale=0.78,
                               color=COLOR_SUCCESS, thickness=1)
        draw_text_centered(frame, 'ENTER: skip tutorial   ESC: title',
                           h - 24, scale=0.52, color=COLOR_GREY)

        # Reset interact state when leaving page
        if tag != 'interact':
            self._tut_knife_grabbed = False
            self._tut_btn_held      = 0
            self._tut_knife_pos     = [320, h // 2 - 20]

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

        draw_text_centered(frame, 'RANKING', 62, scale=2.2, color=COLOR_PRIMARY, thickness=3)

        rank_colors  = [COLOR_PRIMARY, (180, 180, 220), (0, 165, 210)]
        col_centers  = [w // 4, w * 3 // 4]
        categories   = ['steak',           'pancake']
        col_headers  = ['Salisbury  Steak', 'Pancakes']

        # Vertical divider
        cv2.line(frame, (w // 2, 105), (w // 2, h - 75), (70, 72, 82), 2)

        for ci, (cat, header) in enumerate(zip(categories, col_headers)):
            cx_col   = col_centers[ci]
            is_last  = (cat == self._last_category)
            hcol     = COLOR_SUCCESS if is_last else COLOR_PRIMARY

            # Column header + underline
            (tw, _), _ = cv2.getTextSize(header, FONT, 0.88, 2)
            cv2.putText(frame, header, (cx_col - tw // 2, 120),
                        FONT, 0.88, hcol, 2, cv2.LINE_AA)
            cv2.line(frame, (cx_col - 175, 130), (cx_col + 175, 130), hcol, 2)

            entries = self.score.get_rankings(cat)[:6]
            if not entries:
                (ew, _), _ = cv2.getTextSize('No records yet', FONT, 0.65, 1)
                cv2.putText(frame, 'No records yet', (cx_col - ew // 2, 220),
                            FONT, 0.65, COLOR_GREY, 1, cv2.LINE_AA)
                continue

            for i, entry in enumerate(entries):
                color = rank_colors[i] if i < 3 else COLOR_WHITE
                y     = 178 + i * 68

                # Rank
                rank_txt = f'{i + 1}.'
                cv2.putText(frame, rank_txt, (cx_col - 185, y),
                            FONT, 0.80, color, 2, cv2.LINE_AA)
                # Name
                cv2.putText(frame, entry['name'][:12], (cx_col - 148, y),
                            FONT, 0.80, COLOR_WHITE, 2, cv2.LINE_AA)
                # Score (right-aligned within column)
                sc_txt = str(entry['score'])
                (sw, _), _ = cv2.getTextSize(sc_txt, FONT, 0.80, 2)
                cv2.putText(frame, sc_txt, (cx_col + 185 - sw, y),
                            FONT, 0.80, color, 2, cv2.LINE_AA)

        draw_text_centered(frame, 'SPACE / R: Play again', h - 45,
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


# ── Title screen decorations ─────────────────────────────────────────────────

def _draw_strawberry(frame, cx, cy, size=75):
    """Cartoon strawberry."""
    import math as _m
    s = size
    # Body (red teardrop)
    body_pts = np.array([
        [cx,           cy - s],
        [cx + int(s*0.72), cy - int(s*0.3)],
        [cx + int(s*0.55), cy + int(s*0.5)],
        [cx,           cy + s],
        [cx - int(s*0.55), cy + int(s*0.5)],
        [cx - int(s*0.72), cy - int(s*0.3)],
    ], np.int32)
    cv2.fillPoly(frame, [body_pts + np.array([2,3])], (22, 30, 100))  # shadow
    cv2.fillPoly(frame, [body_pts], (35, 50, 220))   # red BGR
    cv2.fillPoly(frame, [body_pts], (45, 65, 235))
    cv2.polylines(frame, [body_pts], True, (20, 30, 160), 3)
    # Seeds
    rng = np.random.default_rng(seed=3)
    for _ in range(14):
        sx = cx + int(rng.integers(-int(s*0.45), int(s*0.45)))
        sy = cy + int(rng.integers(-int(s*0.6), int(s*0.7)))
        cv2.ellipse(frame, (sx, sy), (3, 4), 0, 0, 360, (20, 180, 230), -1)
    # Leaves (green)
    for ang in [-30, 0, 30]:
        a = _m.radians(ang - 90)
        lx = cx + int(_m.cos(a) * s * 0.35)
        ly = cy - s + int(_m.sin(a) * s * 0.35)
        leaf_pts = np.array([
            [cx, cy - s],
            [lx - int(s*0.18), ly - int(s*0.25)],
            [lx, ly - int(s*0.5)],
            [lx + int(s*0.18), ly - int(s*0.25)],
        ], np.int32)
        cv2.fillPoly(frame, [leaf_pts], (28, 140, 50))
        cv2.polylines(frame, [leaf_pts], True, (18, 100, 35), 2)


def _draw_broccoli(frame, cx, cy, size=75):
    """Cartoon broccoli."""
    s = size
    # Stem
    stem_w = int(s * 0.22)
    cv2.rectangle(frame, (cx - stem_w, cy + int(s*0.1)),
                  (cx + stem_w, cy + s), (28, 100, 45), -1)
    cv2.rectangle(frame, (cx - stem_w, cy + int(s*0.1)),
                  (cx + stem_w, cy + s), (18, 75, 30), 2)
    # Main head clusters
    clusters = [(0, -int(s*0.35), int(s*0.52)),
                (-int(s*0.38), -int(s*0.12), int(s*0.36)),
                ( int(s*0.38), -int(s*0.12), int(s*0.36)),
                (-int(s*0.22),  int(s*0.08), int(s*0.28)),
                ( int(s*0.22),  int(s*0.08), int(s*0.28))]
    for dx, dy, r in clusters:
        cv2.circle(frame, (cx+dx+2, cy+dy+3), r, (18, 75, 30), -1)
        cv2.circle(frame, (cx+dx, cy+dy), r, (42, 155, 60), -1)
        # Darker bumps on top
        for bx_off, by_off in [(-r//3,-r//3),(0,-r//2),(r//3,-r//3)]:
            cv2.circle(frame, (cx+dx+bx_off, cy+dy+by_off), r//4, (30, 125, 48), -1)
    # Outline
    cv2.ellipse(frame, (cx, cy - int(s*0.18)), (int(s*0.72), int(s*0.62)),
                0, 0, 360, (18, 90, 35), 3)


def _draw_steak_icon(frame, cx, cy, size=90):
    """Simple steak shape for stage select card."""
    s = size
    pts = np.array([
        [cx-s,    cy+int(s*0.2)],  [cx-int(s*0.75), cy-int(s*0.6)],
        [cx-int(s*0.15), cy-int(s*0.75)], [cx+int(s*0.45), cy-int(s*0.65)],
        [cx+s,    cy-int(s*0.15)], [cx+int(s*1.05), cy+int(s*0.3)],
        [cx+int(s*0.5),  cy+int(s*0.65)], [cx-int(s*0.25), cy+int(s*0.7)],
        [cx-int(s*0.8),  cy+int(s*0.5)],
    ], np.int32)
    cv2.fillPoly(frame, [pts + np.array([3,4])], (18,15,12))
    cv2.fillPoly(frame, [pts], (65, 88, 168))
    # Grill marks
    for gx in range(cx-int(s*0.7), cx+int(s*0.9), int(s*0.4)):
        cv2.line(frame, (gx-int(s*0.18), cy+int(s*0.3)),
                 (gx+int(s*0.18), cy-int(s*0.3)), (35, 55, 110), 4)
    cv2.polylines(frame, [pts], True, (40, 60, 120), 3)


def _draw_pancake_icon(frame, cx, cy, size=80):
    """Simple cute pancake stack."""
    s = size
    for i in range(3):
        oy = cy + int(s*0.3) - i * int(s*0.45)
        rx, ry = int(s*0.95), int(s*0.28)
        col = (50 - i*3, 148 - i*5, 215 - i*5)
        cv2.ellipse(frame, (cx+3, oy+5), (rx, ry), 0, 0, 360, (15,12,10), -1)
        cv2.ellipse(frame, (cx, oy), (rx, ry), 0, 0, 360, col, -1)
        cv2.ellipse(frame, (cx-int(rx*0.3), oy-int(ry*0.3)),
                    (int(rx*0.4), int(ry*0.4)), 0, 0, 360,
                    tuple(min(255,c+30) for c in col), -1)
        cv2.ellipse(frame, (cx, oy), (rx, ry), 0, 0, 360,
                    tuple(max(0,c-30) for c in col), 2)
    # Butter on top
    top_y = cy + int(s*0.3) - 2*int(s*0.45) - int(s*0.08)
    cv2.rectangle(frame, (cx-int(s*0.18), top_y-int(s*0.1)),
                  (cx+int(s*0.18), top_y+int(s*0.1)), (72, 220, 248), -1)
    cv2.rectangle(frame, (cx-int(s*0.18), top_y-int(s*0.1)),
                  (cx+int(s*0.18), top_y+int(s*0.1)), (48, 185, 218), 2)

