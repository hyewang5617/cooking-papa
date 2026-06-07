import cv2
import math
import time
import numpy as np
from . import audio
from .minigames.base import BaseMiniGame

# ── Mixer constants ───────────────────────────────────────────────────────────
_MX_CX       = 640
_MX_CY       = 360
_MX_BOWL_RX  = 195
_MX_BOWL_RY  = 155
_MX_LIGHT_INTERVAL = 1.5    # seconds between light changes
_MX_GRAB_R   = 140
_MX_INGREDIENTS = [
    'Flour', 'Milk', 'Sugar',
    'Baking Powder', 'Butter', 'Meringue',
]
_MX_ING_COLORS = [          # BGR colors for each ingredient icon
    (240, 245, 250),         # flour - white
    (235, 240, 248),         # milk - cream white
    (200, 228, 248),         # sugar - light white
    (205, 220, 242),         # baking powder - off-white
    (65,  210, 248),         # butter - pale yellow
    (250, 252, 255),         # meringue - pure white
]
_MX_ING_X    = 1080          # ingredient spawn x
_MX_ING_Y    = 300           # ingredient spawn y

# ── Cook Pancake constants ────────────────────────────────────────────────────
_PC_PAN_CX      = 640
_PC_PAN_CY      = 390
_PC_PAN_R       = 175
_PC_PAN_AREA_R  = 140
_PC_PAN_HX      = 640               # handle bottom center x
_PC_PAN_HY      = 390 + 175 + 65   # handle bottom y
_PC_GAUGE_SPEED = 0.0038       # fill per frame (oscillates back and forth)
_PC_GREEN_MIN   = 0.50         # green zone start
_PC_GREEN_MAX   = 0.78         # green zone end → red zone after this
_PC_FLIP_PX     = 65           # upward px to detect flip
_PC_PLATE_X     = 160          # plate resting x (LEFT side)
_PC_PLATE_Y     = 440          # plate resting y
_PC_PLATE_GRAB_R = 80

# ── Stack Pancake constants ───────────────────────────────────────────────────
_ST_GOAL       = 3       # pancakes to stack
_ST_PLATE_CX   = 640
_ST_PLATE_Y    = 575     # plate center y
_ST_PRX        = 115     # pancake rx (half-width)
_ST_PRY        = 28      # pancake ry (perspective)
_ST_THICK      = 22      # pancake side thickness
_ST_SWING_Y    = 100     # y of swinging pancake
_ST_AMPLITUDE  = 260     # oscillation amplitude px
_ST_BASE_SPEED = 1.4     # oscillation rad/s (increases per stack)
_ST_TOLERANCE  = 0.40    # 40% of pancake width
_ST_FALL_SPEED = 14      # px per frame while falling

# ── Syrup constants ───────────────────────────────────────────────────────────
_SY_CX        = 640
_SY_CY        = 360
_SY_PR        = 200           # pancake radius
_SY_BUTTER_W  = 90
_SY_BUTTER_H  = 45
_SY_COLOR     = (30, 120, 200)   # maple syrup BGR (amber)
_SY_THICKNESS = 5
_SY_GOAL_LEN  = 3200          # total stroke px to complete

# ── Egg constants ─────────────────────────────────────────────────────────────
_E_EGGS      = 2
_E_CX        = 640          # shell center x (fixed)
_E_CY        = 300          # shell center y (fixed)
_E_RX        = 68           # shell half-width
_E_RY        = 80           # shell half-height
_E_GRAB_R    = 95           # grab detection radius
_E_MIN_ROT   = 0.65         # min CCW radians for white to pour (success eligible)
_E_YOLK_ROT  = 1.05         # yolk starts peeking past rim
_E_FAIL_ROT  = 1.45         # yolk falls out → fail
_E_BOWL_CX   = 640
_E_BOWL_CY   = 540


def _shadow_text(frame, text, x, y, scale, color, thickness, center=False):
    if center:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
        x = x - tw // 2
    cv2.putText(frame, text, (x+2, y+2), cv2.FONT_HERSHEY_DUPLEX,
                scale, (10, 10, 10), thickness+1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _rot(px, py, cx, cy, a):
    """Rotate point (px,py) around (cx,cy) by angle a."""
    dx, dy = px - cx, py - cy
    return (int(cx + dx*math.cos(a) - dy*math.sin(a)),
            int(cy + dx*math.sin(a) + dy*math.cos(a)))


# ── Intermission "title card" screens between major stages ───────────────────
_PINTRO_DURATION = 2.4
_PINTRO_TEXT = {
    'pintro_egg':  ('Yolk Time!',     'Crack the eggs and separate the yolks'),
    'pintro_mix':  ('Mixing Time!',   'Combine the batter ingredients'),
    'pintro_cook': ('Cooking Time!',  'Pour the batter and flip the pancakes'),
    'pintro_deco': ('Decorate Time!', 'Add butter and drizzle the syrup'),
}
_PINTRO_NEXT = {
    'pintro_egg':  'egg_separate',
    'pintro_mix':  'mixer',
    'pintro_cook': 'cook_pancake',
    'pintro_deco': 'stack_pancake',
}


class PancakeScene(BaseMiniGame):
    name        = 'Pancakes'
    instruction = 'Make pancakes!'
    duration    = 130.0
    grab_phase  = True

    _DEBUG_PHASES = ['egg_separate', 'mixer', 'cook_pancake', 'stack_pancake', 'syrup', 'reveal']

    def __init__(self):
        super().__init__()
        self._phase     = 'pintro_egg'
        self._reveal_t0 = 0.0
        self._inter_t0  = 0.0
        self._init_egg()
        self._init_mixer()
        self._init_cook_pancake()
        self._init_stack_pancake()
        self._init_syrup()

    def jump_to_phase(self, phase):
        self._phase = phase
        self._begin_timer()
        if phase == 'cook_pancake':
            self._init_cook_pancake()
        elif phase == 'mixer':
            self._init_mixer()
        elif phase == 'stack_pancake':
            self._init_stack_pancake()
        elif phase == 'syrup':
            self._init_syrup()
        elif phase == 'reveal':
            self._reveal_t0 = 0.0
        elif phase in _PINTRO_NEXT:
            self._inter_t0 = 0.0

    def _init_egg(self):
        self._e_current   = 0
        self._e_done      = [False] * _E_EGGS
        self._e_grabbed   = False
        self._e_prev_ang  = None
        self._e_total_ccw = 0.0    # total CCW radians rotated
        self._e_gauge     = 0.0    # 0→1 fill gauge
        self._e_fail_anim = 0
        self._e_succ_anim = 0
        self._e_white_in_bowl = 0.0

    @property
    def succeeded(self):
        return self._phase == 'complete'

    @property
    def progress_text(self):
        return f'Eggs: {sum(self._e_done)} / {_E_EGGS}'

    @property
    def required_hands(self):
        return 1

    def check_done(self):
        return self._phase == 'complete'

    # ── update ────────────────────────────────────────────────────────────────

    def update(self, hands):
        if not self.timer_started:
            self._begin_timer()

        audio.loop_cooking(self._phase == 'cook_pancake')
        if self._phase == 'mixer':
            _whisk_speed = {'green': 1.0, 'orange': 1.5, 'red': 2.0}.get(self._mx_light, 1.0)
            audio.loop_whisk(True, speed=_whisk_speed)
        else:
            audio.loop_whisk(False)
        audio.loop_syrup(self._phase == 'syrup' and self._sy_drawing)

        if self._phase in _PINTRO_NEXT:
            if self._inter_t0 == 0.0:
                self._inter_t0 = time.time()
            elif time.time() - self._inter_t0 > _PINTRO_DURATION:
                nxt = _PINTRO_NEXT[self._phase]
                self._inter_t0 = 0.0
                self._phase    = nxt
                if nxt == 'mixer':
                    self._init_mixer()
                elif nxt == 'cook_pancake':
                    self._init_cook_pancake()
                elif nxt == 'stack_pancake':
                    self._init_stack_pancake()
                elif nxt == 'syrup':
                    self._init_syrup()
            return []

        if self._phase == 'egg_separate':
            self._update_egg(hands)
        elif self._phase == 'mixer':
            self._update_mixer(hands)
        elif self._phase == 'cook_pancake':
            self._update_cook_pancake(hands)
        elif self._phase == 'stack_pancake':
            self._update_stack_pancake(hands)
        elif self._phase == 'syrup':
            self._update_syrup(hands)
        elif self._phase == 'reveal':
            if self._reveal_t0 == 0.0:
                self._reveal_t0 = time.time()
            elif time.time() - self._reveal_t0 > 5.0:
                self._phase = 'complete'
        return []

    def _update_egg(self, hands):
        # Fail animation: yolk falling, then reset
        if self._e_fail_anim > 0:
            self._e_fail_anim -= 1
            if self._e_fail_anim == 0:
                self._e_grabbed   = False
                self._e_prev_ang  = None
                self._e_total_ccw = 0.0
            return

        # Success animation: show briefly, then move to next egg
        if self._e_succ_anim > 0:
            self._e_succ_anim -= 1
            if self._e_succ_anim == 0:
                # Advance to next egg
                next_e = next((i for i, d in enumerate(self._e_done) if not d), _E_EGGS)
                self._e_current   = next_e
                self._e_grabbed   = False
                self._e_prev_ang  = None
                self._e_total_ccw = 0.0
                if all(self._e_done):
                    self._phase = 'pintro_mix'
                    self._inter_t0 = 0.0
            return

        for hand in hands:
            if not hand.detected:
                continue

            hx = float(hand.screen_x)
            hy = float(hand.screen_y)

            if not self._e_grabbed:
                if hand.gripped:
                    dx = hx - _E_CX
                    dy = hy - _E_CY
                    if dx*dx + dy*dy < _E_GRAB_R**2:
                        self._e_grabbed  = True
                        self._e_prev_ang = math.atan2(dy, dx)
                continue

            if not hand.gripped:
                # Released before gauge full — gauge drains
                self._e_gauge     = max(0.0, self._e_gauge - 0.03)
                self._e_grabbed   = False
                self._e_prev_ang  = None
                break

            # Track CCW rotation around shell center
            dx = hx - _E_CX
            dy = hy - _E_CY
            if math.sqrt(dx*dx + dy*dy) < 20:
                continue
            angle = math.atan2(dy, dx)
            if self._e_prev_ang is not None:
                delta = angle - self._e_prev_ang
                if delta >  math.pi: delta -= 2 * math.pi
                if delta < -math.pi: delta += 2 * math.pi
                self._e_total_ccw = max(0.0, self._e_total_ccw - delta)

            self._e_prev_ang = angle

            # Fail check
            if self._e_total_ccw >= _E_FAIL_ROT:
                self._e_gauge     = 0.0
                self._e_fail_anim = 40
                audio.play_fail()
                self._e_grabbed   = False
                self._e_prev_ang  = None
                self._e_total_ccw = 0.0
                break

            # Gauge fills faster the more the egg is tilted (min threshold = 0.3 rad)
            if self._e_total_ccw >= 0.3:
                fill_rate = (self._e_total_ccw ** 2) * 0.006
                self._e_gauge = min(1.0, self._e_gauge + fill_rate)
            else:
                self._e_gauge = max(0.0, self._e_gauge - 0.01)

            # Success when gauge full
            if self._e_gauge >= 1.0:
                self._e_done[self._e_current] = True
                self._e_white_in_bowl = min(1.0, self._e_white_in_bowl + 0.5)
                self._e_gauge     = 0.0
                self._e_succ_anim = 30
                audio.play_success()
                break

            break

    # ── draw ──────────────────────────────────────────────────────────────────

    def draw(self, frame, hands):
        t  = time.time()
        h, w = frame.shape[:2]
        if self._phase in _PINTRO_NEXT:
            self._draw_intermission(frame)
            return frame
        if self._phase == 'mixer':
            return self._draw_mixer(frame)
        if self._phase == 'cook_pancake':
            return self._draw_cook_pancake(frame)
        if self._phase == 'stack_pancake':
            return self._draw_stack_pancake(frame)
        if self._phase == 'syrup':
            return self._draw_syrup(frame)
        if self._phase == 'reveal':
            self._draw_reveal(frame)
            return frame

        # ── Bowl ──────────────────────────────────────────────────────────────
        bcx, bcy = _E_BOWL_CX, _E_BOWL_CY
        cv2.ellipse(frame, (bcx+5, bcy+20), (180, 26), 0, 0, 360, (20,18,16), -1)
        cv2.ellipse(frame, (bcx, bcy), (180, 108), 0, 0, 180, (215, 222, 230), -1)
        if self._e_white_in_bowl > 0:
            wry = int(60 * self._e_white_in_bowl)
            wrx = int(160 * self._e_white_in_bowl)
            cv2.ellipse(frame, (bcx, bcy - 8), (wrx, wry), 0, 0, 360, (244, 248, 252), -1)
        cv2.ellipse(frame, (bcx, bcy), (180, 108), 0, 180, 360, (215, 222, 230), -1)
        cv2.ellipse(frame, (bcx, bcy), (180, 108), 0, 0, 360, (172, 180, 190), 4)

        # ── Completed eggs (small broken shells to the side) ──────────────────
        for idx in range(_E_EGGS):
            if self._e_done[idx]:
                sx = 160 + idx * 100
                self._draw_shell_small(frame, sx, 120)

        # ── Current egg ───────────────────────────────────────────────────────
        if self._e_current < _E_EGGS and not self._e_done[self._e_current]:
            tilt = self._e_total_ccw  # CCW tilt angle

            if self._e_fail_anim > 0:
                # Fail: keep shell tilted at fail angle, yolk falling
                self._draw_egg_shell(frame, _E_CX, _E_CY, _E_FAIL_ROT)
                # Falling yolk
                fall_t = (40 - self._e_fail_anim) / 40.0
                yolk_y = int(_E_CY + 60 + fall_t * 200)
                yolk_x = int(_E_CX - 30)
                cv2.circle(frame, (yolk_x+3, yolk_y+4), 28, (14,12,10), -1)
                cv2.circle(frame, (yolk_x, yolk_y), 26, (28, 168, 242), -1)
                cv2.circle(frame, (yolk_x-7, yolk_y-7), 10, (55, 195, 255), -1)
                _shadow_text(frame, 'Yolk dropped!  Try again',
                             w//2, _E_CY - _E_RY - 40, 0.70,
                             (80, 120, 255), 1, center=True)
            elif self._e_succ_anim > 0:
                # Success: shell tilted to pour angle, flash
                self._draw_egg_shell(frame, _E_CX, _E_CY, _E_MIN_ROT + 0.1)
                _shadow_text(frame, 'Perfect!', w//2, _E_CY - _E_RY - 40,
                             1.0, (80, 240, 140), 2, center=True)
            else:
                self._draw_egg_shell(frame, _E_CX, _E_CY, tilt)

                # Grab ring if not grabbed
                if not self._e_grabbed:
                    ring_r = int(62 + 10*abs(math.sin(t*5)))
                    cv2.circle(frame, (_E_CX, _E_CY), ring_r, (0, 200, 255), 2, cv2.LINE_AA)
                    _shadow_text(frame, 'GRAB', _E_CX, _E_CY - ring_r - 14,
                                 0.58, (0, 200, 255), 1, center=True)
                    # CCW arrow hint
                    for ai in range(60, 300, 45):
                        a1 = math.radians(ai - 90)
                        a2 = math.radians(ai - 30 - 90)
                        p1 = (int(_E_CX + (ring_r+20)*math.cos(a1)),
                              int(_E_CY + (ring_r+20)*math.sin(a1)))
                        p2 = (int(_E_CX + (ring_r+20)*math.cos(a2)),
                              int(_E_CY + (ring_r+20)*math.sin(a2)))
                        cv2.arrowedLine(frame, p1, p2, (80, 220, 255), 2, tipLength=0.45)

                # White pouring stream
                if tilt >= 0.2:
                    self._draw_pour_stream(frame, tilt)

                # ── Gauge bar ─────────────────────────────────────────────────
                if self._e_grabbed and tilt >= 0.3:
                    bar_x  = _E_CX + _E_RX + 28
                    bar_h  = 120
                    bar_y0 = _E_CY - bar_h // 2
                    # Background
                    cv2.rectangle(frame, (bar_x, bar_y0), (bar_x+18, bar_y0+bar_h),
                                  (40, 40, 55), -1)
                    cv2.rectangle(frame, (bar_x, bar_y0), (bar_x+18, bar_y0+bar_h),
                                  (80, 85, 100), 2)
                    # Fill (bottom up)
                    filled_h = int(bar_h * self._e_gauge)
                    if filled_h > 0:
                        g = self._e_gauge
                        gcol = (int(40*(1-g)), int(200*g + 80*(1-g)), int(80*(1-g) + 40*g))
                        cv2.rectangle(frame,
                                      (bar_x+2, bar_y0 + bar_h - filled_h),
                                      (bar_x+16, bar_y0 + bar_h - 2),
                                      gcol, -1)
                    # Pulse at top when near full
                    if self._e_gauge > 0.85:
                        pulse = int(6*abs(math.sin(time.time()*10)))
                        cv2.rectangle(frame,
                                      (bar_x-pulse, bar_y0-pulse),
                                      (bar_x+18+pulse, bar_y0+bar_h+pulse),
                                      (80, 240, 140), 2)

                # Danger indicator
                if tilt >= _E_YOLK_ROT:
                    danger_frac = (tilt - _E_YOLK_ROT) / (_E_FAIL_ROT - _E_YOLK_ROT)
                    pulse = int(12*abs(math.sin(t*8)))
                    dcol  = (0, int(220*(1-danger_frac)), int(220*danger_frac))
                    _shadow_text(frame, 'STOP!', _E_CX, _E_CY - _E_RY - 45 - pulse,
                                 0.85, dcol, 2, center=True)

        # ── HUD ───────────────────────────────────────────────────────────────
        _eg_lbl = f'Eggs: {sum(self._e_done)} / {_E_EGGS}'
        _eg_w   = cv2.getTextSize(_eg_lbl, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0][0]
        _shadow_text(frame, _eg_lbl, w - 26 - _eg_w, 100, 1.0, (255, 230, 100), 2)
        if not self._e_grabbed and self._e_fail_anim == 0 and self._e_succ_anim == 0:
            _shadow_text(frame, 'Grip then rotate  COUNTER-CLOCKWISE  slowly',
                         w//2, h-35, 0.60, (200, 210, 230), 1, center=True)
        return frame

    # ── drawing helpers ───────────────────────────────────────────────────────

    def _draw_egg_shell(self, frame, cx, cy, tilt):
        """Draw half egg shell tilted CCW by `tilt` radians, with yolk inside."""
        rx, ry   = _E_RX, _E_RY
        n_jags   = 8
        yolk_r   = 26

        # ── Shell polygon (rotated D-shape) ───────────────────────────────────
        pts = []
        # Bottom arc: deg 0..180 in unrotated frame (cup bottom)
        for deg in range(0, 181, 7):
            a  = math.radians(deg)
            lx = cx + rx * math.cos(a)
            ly = cy + ry * math.sin(a)
            pts.append(_rot(lx, ly, cx, cy, -tilt))   # negative = CCW rotation

        # Jagged rim (top edge in unrotated frame = y=cy)
        for i in range(n_jags, -1, -1):
            frac = i / n_jags
            jx   = cx - rx + frac * 2 * rx
            jy   = cy + (12 if i % 2 == 0 else -12)
            pts.append(_rot(jx, jy, cx, cy, -tilt))

        arr = np.array(pts, np.int32)
        cv2.fillPoly(frame, [arr + np.array([3,3])], (18,16,14))
        cv2.fillPoly(frame, [arr], (238, 242, 248))

        # Inner shell (slightly grey)
        inner = []
        for deg in range(0, 181, 12):
            a  = math.radians(deg)
            lx = cx + (rx-10) * math.cos(a)
            ly = cy + (ry-8)  * math.sin(a)
            inner.append(_rot(lx, ly, cx, cy, -tilt))
        inner_arr = np.array(inner, np.int32)
        cv2.polylines(frame, [inner_arr], False, (210, 215, 222), 3)

        cv2.polylines(frame, [arr], True, (175, 182, 192), 3)

        # ── Yolk: sits at bottom of shell, slides with tilt (gravity) ─────────
        # "Bottom" of shell in world coords = cx + sin(-tilt)*(ry-yolk_r),
        #                                     cy + cos(-tilt)*(ry-yolk_r)  ... approx
        # Gravity pulls yolk toward the lowest point inside the shell
        yolk_wx = int(cx - math.sin(tilt) * (ry - yolk_r - 4))
        yolk_wy = int(cy + math.cos(tilt) * (ry - yolk_r - 4))

        # Only draw yolk if it hasn't fallen past the rim
        yolk_vis_frac = max(0.0,
            (tilt - _E_YOLK_ROT) / (_E_FAIL_ROT - _E_YOLK_ROT)) if tilt > _E_YOLK_ROT else 0.0

        cv2.circle(frame, (yolk_wx+3, yolk_wy+3), yolk_r+2, (16,14,12), -1)
        col_b = int(28  + yolk_vis_frac * 60)
        col_g = int(168 - yolk_vis_frac * 80)
        col_r = 242
        cv2.circle(frame, (yolk_wx, yolk_wy), yolk_r, (col_b, col_g, col_r), -1)
        cv2.circle(frame, (yolk_wx-7, yolk_wy-7), 9, (55, 200, 255), -1)

        # Danger flash when yolk near rim
        if yolk_vis_frac > 0.1:
            t_now = time.time()
            if int(t_now * 8) % 2 == 0:
                cv2.circle(frame, (yolk_wx, yolk_wy), yolk_r+4,
                           (0, int(200*(1-yolk_vis_frac)), int(200*yolk_vis_frac)), 2)

    def _draw_pour_stream(self, frame, tilt):
        """Draw egg white pouring from the tilted rim toward the bowl."""
        # The rim (opening) of the tilted shell
        # At tilt=0, rim is at y=_E_CY (horizontal line)
        # The lowest point of the rim when tilted:
        rim_low_x = int(_E_CX - math.cos(tilt - math.pi/2) * _E_RX)
        rim_low_y = int(_E_CY + math.sin(tilt) * _E_RX * 0.5 + _E_RY * 0.2)

        pour_frac = min(1.0, (tilt - 0.2) / (_E_MIN_ROT - 0.2))
        stream_w  = max(1, int(14 * pour_frac))
        stream_end_y = int(rim_low_y + (_E_BOWL_CY - rim_low_y) * pour_frac)

        cv2.line(frame, (rim_low_x, rim_low_y), (rim_low_x, stream_end_y),
                 (240, 244, 250), stream_w, cv2.LINE_AA)
        if pour_frac > 0.5:
            cv2.circle(frame, (rim_low_x, stream_end_y), stream_w, (244, 248, 252), -1)

    def _draw_shell_small(self, frame, cx, cy):
        """Tiny empty broken shell for completed egg indicator."""
        rx, ry = 28, 32
        pts = []
        for deg in range(0, 181, 12):
            a = math.radians(deg)
            pts.append([int(cx + rx*math.cos(a)), int(cy + ry*math.sin(a))])
        for i in range(6, -1, -1):
            frac = i / 6
            pts.append([int(cx - rx + frac*2*rx), cy + (6 if i%2==0 else -6)])
        arr = np.array(pts, np.int32)
        cv2.fillPoly(frame, [arr], (225, 230, 238))
        cv2.polylines(frame, [arr], True, (170, 178, 188), 2)
        cv2.circle(frame, (cx, cy + 8), 4, (80, 220, 130), -1)

    # ── Cook Pancake phase ───────────────────────────────────────────────────

    def _init_cook_pancake(self):
        self._pc_gauge      = [0.0, 0.0]   # side 0 and side 1 fill
        self._pc_gauge_dir  = [1,   1]     # oscillation direction (+1 / -1)
        self._pc_side       = 0
        self._pc_subphase   = 'cook'
        self._pc_flip_anim   = 0
        self._pc_flip_count  = 0   # total flips done (0→serve after 2nd flip)
        self._pc_pan_grabbed = False
        self._pc_flip_prev_y = None
        self._pc_plate_grabbed = False
        self._pc_plate_pos   = [float(_PC_PLATE_X), float(_PC_PLATE_Y)]

    def _update_cook_pancake(self, hands):
        sp = self._pc_subphase

        # ── Flip animation ────────────────────────────────────────────────────
        if sp == 'flip_anim':
            self._pc_flip_anim -= 1
            if self._pc_flip_anim == 20:
                self._pc_side = 1 - self._pc_side
            if self._pc_flip_anim <= 0:
                self._pc_flip_count += 1
                self._pc_pan_grabbed  = False
                self._pc_flip_prev_y  = None
                if self._pc_flip_count >= 2:
                    self._pc_subphase = 'serve'
                else:
                    self._pc_subphase = 'cook'
            return

        in_green = _PC_GREEN_MIN <= self._pc_gauge[self._pc_side] <= _PC_GREEN_MAX

        # ── Cook: flip when gauge is in green zone ────────────────────────────
        if sp == 'cook' and in_green:
            self._handle_flip(hands)

        # ── Serve phase ───────────────────────────────────────────────────────
        elif sp == 'serve':
            for hand in hands:
                if not hand.detected:
                    continue
                hx, hy = float(hand.screen_x), float(hand.screen_y)
                if not self._pc_plate_grabbed:
                    if hand.gripped:
                        dx = hx - self._pc_plate_pos[0]
                        dy = hy - self._pc_plate_pos[1]
                        if dx*dx + dy*dy < _PC_PLATE_GRAB_R**2:
                            self._pc_plate_grabbed = True
                    continue
                self._pc_plate_pos[0] = hx
                self._pc_plate_pos[1] = hy
                if not hand.gripped:
                    self._pc_plate_grabbed = False
                dx = self._pc_plate_pos[0] - _PC_PAN_CX
                dy = self._pc_plate_pos[1] - _PC_PAN_CY
                if dx*dx + dy*dy < (_PC_PAN_R * 0.85)**2:
                    self._phase    = 'pintro_deco'
                    self._inter_t0 = 0.0
                break

        # ── Oscillating gauge — frozen in serve ───────────────────────────────
        if sp == 'cook':
            s = self._pc_side
            self._pc_gauge[s] += _PC_GAUGE_SPEED * self._pc_gauge_dir[s]
            if self._pc_gauge[s] >= 1.0:
                self._pc_gauge[s] = 1.0; self._pc_gauge_dir[s] = -1
            elif self._pc_gauge[s] <= 0.0:
                self._pc_gauge[s] = 0.0; self._pc_gauge_dir[s] = 1

    def _handle_flip(self, hands):
        for hand in hands:
            if not hand.detected or not hand.gripped:
                if self._pc_pan_grabbed:
                    self._pc_pan_grabbed = False
                    self._pc_flip_prev_y = None
                continue
            hx, hy = float(hand.screen_x), float(hand.screen_y)
            if not self._pc_pan_grabbed:
                dx = hx - _PC_PAN_HX
                dy = hy - _PC_PAN_HY
                if dx*dx + dy*dy < 90**2:
                    self._pc_pan_grabbed = True
                continue
            sy = hand.screen_y
            if self._pc_flip_prev_y is not None:
                if self._pc_flip_prev_y - sy >= _PC_FLIP_PX:
                    self._pc_subphase  = 'flip_anim'
                    self._pc_flip_anim = 40
                    audio.play_success()
                    return
            self._pc_flip_prev_y = sy
            break

    def _draw_cook_pancake(self, frame):
        t  = time.time()
        h, w = frame.shape[:2]
        cx, cy = _PC_PAN_CX, _PC_PAN_CY
        sp = self._pc_subphase

        # ── Stove base ────────────────────────────────────────────────────────
        cv2.ellipse(frame, (cx, cy + _PC_PAN_R + 18), (_PC_PAN_R+45, 38),
                    0, 0, 360, (48, 44, 40), -1)

        # ── Flames ────────────────────────────────────────────────────────────
        fl_span = int(_PC_PAN_R * 1.55)
        for fi in range(9):
            fx  = cx - fl_span//2 + fi * (fl_span//8)
            flk = int(14 * math.sin(t * 9.5 + fi * 1.3))
            fy0 = cy + _PC_PAN_R + 5
            cv2.ellipse(frame, (fx, fy0), (7, 16+flk//2), 0, 0, 360, (190,95,18), -1)
            pts_f = np.array([[fx-11,fy0],[fx,fy0-44-flk],[fx+11,fy0]], np.int32)
            cv2.fillPoly(frame, [pts_f], (0, int(148+flk*3), 255))

        # ── Pan body ──────────────────────────────────────────────────────────
        cv2.circle(frame, (cx+4, cy+4), _PC_PAN_R, (15,12,10), -1)
        cv2.circle(frame, (cx, cy), _PC_PAN_R, (42, 38, 34), -1)
        cv2.circle(frame, (cx, cy), _PC_PAN_R, (22, 18, 14), 8)
        cook_heat = self._pc_gauge[self._pc_side]
        sv = int(138 - cook_heat * 22)
        cv2.circle(frame, (cx, cy), _PC_PAN_AREA_R, (sv, sv+2, sv+4), -1)
        cv2.circle(frame, (cx, cy), _PC_PAN_AREA_R, (55, 50, 45), 2)

        # ── Pan handle (BOTTOM) ───────────────────────────────────────────────
        hy1 = cy + _PC_PAN_R - 10
        hy2 = _PC_PAN_HY + 28
        cv2.rectangle(frame, (cx-24, hy1), (cx+24, hy2), (30, 26, 22), -1)
        cv2.rectangle(frame, (cx-24, hy1), (cx+24, hy2), (14, 11, 9), 3)
        for gy in range(hy1+20, hy2-10, 22):
            cv2.line(frame, (cx-20, gy), (cx+20, gy), (42, 38, 34), 3)

        # ── Pancake ───────────────────────────────────────────────────────────
        flip_frac  = self._pc_flip_anim / 40.0
        air_height = int(120 * math.sin(math.pi * (1-flip_frac))) if self._pc_flip_anim > 0 else 0
        cos_scale  = abs(math.cos(math.pi * (1-flip_frac))) if self._pc_flip_anim > 0 else 1.0
        p_cy       = cy - air_height

        pr   = int(_PC_PAN_AREA_R * 0.80)
        pry  = int(pr * 0.30)    # perspective oval
        prxA = max(4, int(pr * cos_scale))

        # Cook color
        g0, g1 = self._pc_gauge[0], self._pc_gauge[1]
        cf = (g0 if self._pc_side==0 else g1) / _PC_GREEN_MAX
        # Raw cream → golden brown
        top_b = int(48  + cf * 20)
        top_g = int(168 - cf * 8)
        top_r = int(215 - cf * 5)
        top_col = (top_b, top_g, top_r)   # BGR
        side_col = (max(0,top_b-25), max(0,top_g-35), max(0,top_r-30))

        # Side edge of thick pancake (before top face)
        side_h = int(pry * 0.9)
        cv2.ellipse(frame, (cx, p_cy + side_h), (prxA, int(pry*0.5)),
                    0, 0, 360, side_col, -1)

        # Top face
        cv2.ellipse(frame, (cx+3, p_cy+4), (prxA+3, pry+3), 0, 0, 360, (15,12,10), -1)
        cv2.ellipse(frame, (cx, p_cy), (prxA, pry), 0, 0, 360, top_col, -1)
        # Bubble holes (texture)
        if cos_scale > 0.25:
            rng2 = np.random.default_rng(seed=42)
            for _ in range(10):
                bx2 = cx + int(rng2.integers(-int(pr*0.55), int(pr*0.55)))
                by2 = p_cy + int(rng2.integers(-int(pry*0.55), int(pry*0.55)))
                brad = int(rng2.integers(4, 9))
                cv2.circle(frame, (bx2, by2), brad, side_col, -1)
        cv2.ellipse(frame, (cx, p_cy), (prxA, pry), 0, 0, 360,
                    tuple(max(0,c-40) for c in top_col), 3)

        # ── Grab ring on handle (flip prompt) ────────────────────────────────
        if (sp == 'cook'
                and self._pc_gauge[self._pc_side] >= _PC_GREEN_MIN
                and not self._pc_pan_grabbed
                and self._pc_flip_anim == 0):
            rr = int(52 + 8*abs(math.sin(t*5)))
            cv2.circle(frame, (_PC_PAN_HX, _PC_PAN_HY), rr, (0,200,255), 2, cv2.LINE_AA)
            _shadow_text(frame, 'GRAB & Flick UP!', _PC_PAN_HX,
                         _PC_PAN_HY + rr + 20, 0.62, (0,200,255), 1, center=True)
            cv2.arrowedLine(frame, (_PC_PAN_HX + 80, _PC_PAN_HY),
                            (_PC_PAN_HX + 80, _PC_PAN_HY - 70),
                            (0,200,255), 3, tipLength=0.35)

        # ── Plate (serve phase) ───────────────────────────────────────────────
        if sp == 'serve':
            px = int(self._pc_plate_pos[0])
            py = int(self._pc_plate_pos[1])
            cv2.ellipse(frame, (px+5, py+10), (110, 35), 0, 0, 360, (15,12,10), -1)
            cv2.ellipse(frame, (px, py+2), (118, 38), 0, 0, 360, (38, 38, 155), -1)
            cv2.ellipse(frame, (px, py), (110, 34), 0, 0, 360, (242, 246, 252), -1)
            cv2.ellipse(frame, (px, py), (98, 28), 0, 0, 360, (228, 232, 240), -1)
            cv2.ellipse(frame, (px, py), (110, 34), 0, 0, 360, (175, 180, 192), 3)
            if not self._pc_plate_grabbed:
                rr = int(55 + 8*abs(math.sin(t*5)))
                cv2.circle(frame, (px, py), rr, (80,220,140), 2, cv2.LINE_AA)
                _shadow_text(frame, 'GRAB PLATE!', px, py - rr - 14,
                             0.62, (80,220,140), 1, center=True)
            cv2.arrowedLine(frame, (px+60, py), (px+200, py),
                            (80,220,140), 3, tipLength=0.28)

        # ── Gauges (right center, large and clear) ────────────────────────────
        gx0 = w - 320
        for idx, gy in enumerate([h//2 - 70, h//2 + 10]):
            in_g  = _PC_GREEN_MIN <= self._pc_gauge[idx] <= _PC_GREEN_MAX
            face  = ':)' if in_g else ':('
            fcol  = (50,220,80) if in_g else (50,80,220)
            cv2.circle(frame, (gx0 - 30, gy+22), 24, fcol, -1)
            cv2.circle(frame, (gx0 - 30, gy+22), 24, (255,255,255), 2)
            _shadow_text(frame, face, gx0-30, gy+29, 0.50, (255,255,255), 1, center=True)
            self._draw_gauge_bar(frame, gx0, gy, self._pc_gauge[idx], f'Side {idx+1}')

        # ── Instruction ───────────────────────────────────────────────────────
        msgs = {
            'cook':      'Cook until GREEN zone!  Then FLIP' if self._pc_side==0 else 'Cook side 2 to GREEN!',
            'flip_anim': 'Flipping!',
            'serve':     'Grab the plate → bring to pan!',
        }
        _shadow_text(frame, msgs.get(sp,''), w//2, h-32,
                     0.62, (200,210,230), 1, center=True)
        return frame

    def _draw_gauge_bar(self, frame, bx, by, value, label):
        bw, bh = 280, 44
        # Background
        cv2.rectangle(frame, (bx-2, by-2), (bx+bw+2, by+bh+2), (18,16,14), -1)
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (30,28,24), -1)
        # Zone colors: red → green → red (bounce)
        zones = [
            (0.0,          _PC_GREEN_MIN,  (38, 48, 210)),   # left red
            (_PC_GREEN_MIN, _PC_GREEN_MAX, (45, 210, 75)),    # green (perfect)
            (_PC_GREEN_MAX, 1.0,           (30, 35, 195)),    # right red (burnt)
        ]
        for zs, ze, col in zones:
            x1 = bx + int(zs * bw) + 1
            x2 = bx + int(ze * bw) - 1
            if x2 > x1:
                cv2.rectangle(frame, (x1, by+3), (x2, by+bh-3), col, -1)
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (65,62,55), 2)
        # Moving indicator pin (white bar)
        fill_x = bx + int(value * bw)
        cv2.rectangle(frame, (fill_x-6, by-6), (fill_x+6, by+bh+6), (255,255,255), -1)
        cv2.rectangle(frame, (fill_x-6, by-6), (fill_x+6, by+bh+6), (80,85,100), 2)
        # Label
        _shadow_text(frame, label, bx, by - 14, 0.52, (200,210,225), 1)

    # ── Stack Pancake phase ──────────────────────────────────────────────────

    def _init_stack_pancake(self):
        self._st_count      = 0             # stacked so far
        self._st_stack_x    = [_ST_PLATE_CX]  # x center of each layer
        self._st_swing_x    = float(_ST_PLATE_CX)
        self._st_swing_t    = 0.0           # phase offset (for oscillation)
        self._st_dropping   = False
        self._st_drop_x     = float(_ST_PLATE_CX)
        self._st_drop_y     = float(_ST_SWING_Y)
        self._st_fail_anim  = 0
        self._st_succ_anim  = 0
        self._st_grip_prev  = False         # previous grip state

    def _update_stack_pancake(self, hands):
        if self._st_fail_anim > 0:
            self._st_fail_anim -= 1
        if self._st_succ_anim > 0:
            self._st_succ_anim -= 1

        speed = _ST_BASE_SPEED * (1.0 + self._st_count * 0.45)

        if not self._st_dropping:
            # Oscillate
            self._st_swing_t += speed / 30.0
            self._st_swing_x = _ST_PLATE_CX + _ST_AMPLITUDE * math.sin(self._st_swing_t)

            # Drop on grip press (rising edge: not gripped → gripped)
            gripped_now = any(h.detected and h.gripped for h in hands)
            if gripped_now and not self._st_grip_prev:
                self._st_dropping = True
                self._st_drop_x   = self._st_swing_x
                self._st_drop_y   = float(_ST_SWING_Y)
            self._st_grip_prev = gripped_now
        else:
            # Fall
            self._st_drop_y += _ST_FALL_SPEED
            target_y = _ST_PLATE_Y - self._st_count * (_ST_THICK + _ST_PRY * 2)

            if self._st_drop_y >= target_y:
                self._st_drop_y = float(target_y)
                # Check accuracy
                target_x = self._st_stack_x[-1]
                offset   = abs(self._st_drop_x - target_x)
                allowed  = _ST_PRX * _ST_TOLERANCE * 2   # 40% of full width
                if offset <= allowed:
                    # Success
                    self._st_stack_x.append(int(self._st_drop_x))
                    self._st_count    += 1
                    self._st_succ_anim = 22
                    if self._st_count >= _ST_GOAL:
                        self._phase = 'syrup'
                        self._init_syrup()
                        audio.play_stage_clear()
                    else:
                        audio.play_success()
                else:
                    self._st_fail_anim = 30
                    audio.play_fail()
                self._st_dropping  = False
                self._st_grip_prev = False

    def _draw_stack_pancake(self, frame):
        t  = time.time()
        h, w = frame.shape[:2]

        # ── Plate ────────────────────────────────────────────────────────────
        pcx, pcy = _ST_PLATE_CX, _ST_PLATE_Y + 30
        cv2.ellipse(frame, (pcx+5, pcy+10), (145, 46), 0, 0, 360, (15,12,10), -1)
        cv2.ellipse(frame, (pcx, pcy+2), (152, 50), 0, 0, 360, (38,38,150), -1)
        cv2.ellipse(frame, (pcx, pcy), (142, 44), 0, 0, 360, (242,246,252), -1)
        cv2.ellipse(frame, (pcx, pcy), (128, 36), 0, 0, 360, (228,232,242), -1)
        cv2.ellipse(frame, (pcx, pcy), (142, 44), 0, 0, 360, (172,178,192), 3)

        # ── Stacked pancakes ──────────────────────────────────────────────────
        for idx in range(self._st_count):
            sx   = self._st_stack_x[idx + 1]
            sy   = _ST_PLATE_Y - idx * (_ST_THICK + _ST_PRY * 2)
            cook = 0.7
            tc   = cook
            col  = (int(48+tc*20), int(168-tc*8), int(215-tc*5))
            side = tuple(max(0,c-30) for c in col)
            # Side
            cv2.rectangle(frame, (sx-_ST_PRX, sy), (sx+_ST_PRX, sy+_ST_THICK),
                          side, -1)
            cv2.ellipse(frame, (sx, sy+_ST_THICK), (_ST_PRX, _ST_PRY),
                        0, 0, 360, side, -1)
            # Top
            cv2.ellipse(frame, (sx+2, sy+2), (_ST_PRX+2, _ST_PRY+2),
                        0, 0, 360, (15,12,10), -1)
            cv2.ellipse(frame, (sx, sy), (_ST_PRX, _ST_PRY), 0, 0, 360, col, -1)
            cv2.ellipse(frame, (sx, sy), (_ST_PRX, _ST_PRY),
                        0, 0, 360, tuple(max(0,c-40) for c in col), 2)

        # ── Target shadow on top of stack ─────────────────────────────────────
        tgt_x = self._st_stack_x[-1]
        tgt_y = _ST_PLATE_Y - self._st_count * (_ST_THICK + _ST_PRY * 2)
        allow = int(_ST_PRX * _ST_TOLERANCE * 2)
        cv2.ellipse(frame, (tgt_x, tgt_y), (allow, _ST_PRY//2),
                    0, 0, 360, (80, 220, 120), 2)   # green target zone

        # ── Swinging / falling pancake ────────────────────────────────────────
        if self._st_dropping:
            dx = int(self._st_drop_x)
            dy = int(self._st_drop_y)
        else:
            dx = int(self._st_swing_x)
            dy = _ST_SWING_Y

        col_s  = (48, 168, 215)
        side_s = (25, 138, 185)
        cv2.rectangle(frame, (dx-_ST_PRX, dy), (dx+_ST_PRX, dy+_ST_THICK), side_s, -1)
        cv2.ellipse(frame, (dx, dy+_ST_THICK), (_ST_PRX, _ST_PRY), 0, 0, 360, side_s, -1)
        cv2.ellipse(frame, (dx+2, dy+2), (_ST_PRX+2, _ST_PRY+2), 0, 0, 360, (15,12,10), -1)
        cv2.ellipse(frame, (dx, dy), (_ST_PRX, _ST_PRY), 0, 0, 360, col_s, -1)
        cv2.ellipse(frame, (dx, dy), (_ST_PRX, _ST_PRY),
                    0, 0, 360, tuple(max(0,c-40) for c in col_s), 2)

        # Accuracy indicator (distance from target)
        if not self._st_dropping:
            offset = abs(self._st_swing_x - tgt_x)
            in_zone = offset <= _ST_PRX * _ST_TOLERANCE * 2
            ind_col = (50, 220, 80) if in_zone else (50, 80, 220)
            pulse   = int(8*abs(math.sin(t*8))) if in_zone else 0
            cv2.line(frame, (dx, dy + _ST_PRY + 5),
                     (tgt_x, tgt_y - _ST_PRY - 5), ind_col, 2, cv2.LINE_AA)
            if in_zone:
                _shadow_text(frame, 'DROP!', dx, dy - 28,
                             0.80, (50,230,80), 2, center=True)

        # Fail / success flash
        if self._st_fail_anim > 0 and self._st_fail_anim % 5 < 3:
            cv2.rectangle(frame, (0,0), (w,h), (0,0,200), 10)
            _shadow_text(frame, 'Too far!  Try again', w//2, h//2,
                         0.90, (80,120,255), 2, center=True)
        if self._st_succ_anim > 0:
            _shadow_text(frame, 'Nice!', w//2, h//2,
                         1.2, (50,230,80), 3, center=True)

        # ── HUD ───────────────────────────────────────────────────────────────
        _st_lbl = f'{self._st_count} / {_ST_GOAL}  stacked'
        _st_w   = cv2.getTextSize(_st_lbl, cv2.FONT_HERSHEY_DUPLEX, 0.90, 2)[0][0]
        _shadow_text(frame, _st_lbl, w - 26 - _st_w, 100, 0.90, (255,230,100), 2)
        _shadow_text(frame, 'GRIP to drop!  Aim for the GREEN circle',
                     w//2, h-32, 0.60, (200,210,228), 1, center=True)
        return frame

    # ── Syrup phase ───────────────────────────────────────────────────────────

    def _init_syrup(self):
        self._sy_strokes   = []       # list of strokes; each stroke = list of (x,y)
        self._sy_drawing   = False
        self._sy_prev_pos  = None
        self._sy_total_len = 0.0

    def _update_syrup(self, hands):
        for hand in hands:
            if not hand.detected:
                if self._sy_drawing:
                    self._sy_drawing  = False
                    self._sy_prev_pos = None
                continue
            hx, hy = float(hand.screen_x), float(hand.screen_y)
            if hand.gripped:
                if not self._sy_drawing:
                    self._sy_drawing = True
                    self._sy_strokes.append([(hx, hy)])
                    self._sy_prev_pos = (hx, hy)
                else:
                    if self._sy_prev_pos is not None:
                        dx = hx - self._sy_prev_pos[0]
                        dy = hy - self._sy_prev_pos[1]
                        self._sy_total_len += math.sqrt(dx*dx + dy*dy)
                    self._sy_strokes[-1].append((hx, hy))
                    self._sy_prev_pos = (hx, hy)
            else:
                if self._sy_drawing:
                    self._sy_drawing  = False
                    self._sy_prev_pos = None
            break

        if self._sy_total_len >= _SY_GOAL_LEN:
            self._phase = 'reveal'
            self._reveal_t0 = 0.0
            audio.play_tada()

    def _draw_syrup(self, frame):
        t = time.time()
        h, w = frame.shape[:2]
        cx, cy = _SY_CX, _SY_CY

        # ── Pancake ────────────────────────────────────────────────────────────
        cv2.circle(frame, (cx+5, cy+6), _SY_PR+5, (15, 12, 10), -1)
        cv2.circle(frame, (cx, cy), _SY_PR, (62, 162, 211), -1)
        cv2.circle(frame, (cx, cy), _SY_PR, (42, 138, 188), 5)

        # ── Butter ─────────────────────────────────────────────────────────────
        bx1 = cx - _SY_BUTTER_W // 2
        by1 = cy - _SY_BUTTER_H // 2
        bx2 = cx + _SY_BUTTER_W // 2
        by2 = cy + _SY_BUTTER_H // 2
        cv2.rectangle(frame, (bx1+3, by1+3), (bx2+3, by2+3), (15, 12, 10), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (120, 240, 255), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (90, 210, 235), 2)

        # ── Zigzag guide (blinks until drawing starts) ────────────────────────
        if self._sy_total_len == 0 and int(t * 3) % 2 == 0:
            n_rows  = 7
            r_inner = _SY_PR * 0.82
            y_start = -int(r_inner)
            angle   = -math.pi / 4  # 45° CW → right end points upward
            pts = []
            for i in range(n_rows + 1):
                ly     = y_start + i * int(r_inner * 2 / n_rows)
                dy_abs = abs(ly)
                mx     = int(math.sqrt(max(0, r_inner**2 - dy_abs**2)))
                lx     = -mx if i % 2 == 0 else mx
                # Rotate around pancake center
                rx = int(cx + lx * math.cos(angle) - ly * math.sin(angle))
                ry = int(cy + lx * math.sin(angle) + ly * math.cos(angle))
                # Clip to circle
                ddx, ddy = rx - cx, ry - cy
                dist = math.sqrt(ddx**2 + ddy**2)
                if dist > r_inner:
                    s  = r_inner / dist
                    rx = int(cx + ddx * s)
                    ry = int(cy + ddy * s)
                pts.append((rx, ry))
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], _SY_COLOR, 2, cv2.LINE_AA)
            _shadow_text(frame, 'GRIP & DRIZZLE!', cx, cy - _SY_PR - 18,
                         0.70, (0, 200, 255), 1, center=True)

        # ── Syrup strokes ──────────────────────────────────────────────────────
        for stroke in self._sy_strokes:
            for i in range(1, len(stroke)):
                p1 = (int(stroke[i-1][0]), int(stroke[i-1][1]))
                p2 = (int(stroke[i][0]),   int(stroke[i][1]))
                cv2.line(frame, p1, p2, _SY_COLOR, _SY_THICKNESS, cv2.LINE_AA)

        # ── Progress bar ───────────────────────────────────────────────────────
        prog  = min(1.0, self._sy_total_len / _SY_GOAL_LEN)
        bar_w = 300
        bx    = w//2 - bar_w//2
        bar_y = h - 58
        cv2.rectangle(frame, (bx, bar_y), (bx+bar_w, bar_y+18), (40, 38, 35), -1)
        cv2.rectangle(frame, (bx, bar_y), (bx+int(prog*bar_w), bar_y+18), _SY_COLOR, -1)
        cv2.rectangle(frame, (bx, bar_y), (bx+bar_w, bar_y+18), (80, 78, 72), 2)
        _shadow_text(frame, 'Drizzle maple syrup!', w//2, h - 72,
                     0.65, (200, 210, 230), 1, center=True)
        return frame

    # ── Mixer phase ───────────────────────────────────────────────────────────

    def _init_mixer(self):
        import random as _rnd
        self._mx_added      = 0           # how many ingredients added
        self._mx_grabbed    = False
        self._mx_ing_pos    = [float(_MX_ING_X), float(_MX_ING_Y)]
        self._mx_light      = 'green'
        self._mx_light_t    = time.time()
        self._mx_spin       = 0.0         # mixer blade angle
        self._mx_spin_speed = 0.0         # current spin speed
        self._mx_fail_flash = 0           # frames of red flash (wrong color)

    def _update_mixer(self, hands):
        t = time.time()

        # Traffic light cycling
        if t - self._mx_light_t >= _MX_LIGHT_INTERVAL:
            import random as _rnd
            self._mx_light   = _rnd.choice(['green', 'orange', 'red',
                                             'green', 'orange', 'red', 'green'])
            self._mx_light_t = t

        # Spin speed based on traffic light: red=fast, orange=medium, green=slow
        target_speed = {'red': 0.28, 'orange': 0.14, 'green': 0.04}[self._mx_light]
        self._mx_spin_speed += (target_speed - self._mx_spin_speed) * 0.08
        self._mx_spin += self._mx_spin_speed

        if self._mx_fail_flash > 0:
            self._mx_fail_flash -= 1

        for hand in hands:
            if not hand.detected:
                continue
            hx, hy = float(hand.screen_x), float(hand.screen_y)

            if not self._mx_grabbed:
                if hand.gripped:
                    dx = hx - self._mx_ing_pos[0]
                    dy = hy - self._mx_ing_pos[1]
                    if dx*dx + dy*dy < _MX_GRAB_R**2:
                        self._mx_grabbed = True
                continue

            # Grabbed: ingredient follows hand
            self._mx_ing_pos = [hx, hy]

            if not hand.gripped:
                # Released: check if over bowl and light is green
                dx = hx - _MX_CX
                dy = hy - _MX_CY
                over_bowl = (dx*dx / (_MX_BOWL_RX**2) + dy*dy / (_MX_BOWL_RY**2)) <= 1.2
                if over_bowl:
                    if self._mx_light == 'green':
                        self._mx_added += 1
                        if self._mx_added >= len(_MX_INGREDIENTS):
                            self._phase = 'pintro_cook'
                            self._inter_t0 = 0.0
                    else:
                        self._mx_fail_flash = 18
                # Reset ingredient position
                self._mx_ing_pos  = [float(_MX_ING_X), float(_MX_ING_Y)]
                self._mx_grabbed  = False
            break

    def _draw_mixer(self, frame):
        t  = time.time()
        h, w = frame.shape[:2]
        cx, cy = _MX_CX, _MX_CY
        n  = self._mx_added           # number of ingredients added
        frac = n / len(_MX_INGREDIENTS)   # 0→1

        # ── Bowl shadow + body ────────────────────────────────────────────────
        cv2.ellipse(frame, (cx+8, cy+12), (_MX_BOWL_RX, _MX_BOWL_RY),
                    0, 0, 360, (20,18,16), -1)
        cv2.ellipse(frame, (cx, cy), (_MX_BOWL_RX, _MX_BOWL_RY),
                    0, 0, 360, (205, 212, 225), -1)

        # ── Bowl contents ─────────────────────────────────────────────────────
        # Start: small dark yellow yolk, progress to full light creamy batter
        content_ry = int(_MX_BOWL_RY * (0.28 + frac * 0.58))
        content_rx = int(_MX_BOWL_RX * (0.45 + frac * 0.48))
        # Color: egg yolk yellow → creamy white (BGR)
        cr = int(30  + frac * 188)   # B: 30 → 218
        cg = int(168 + frac * 58)    # G: 168 → 226
        cb = int(240 + frac * 6)     # R: 240 → 246
        content_col = (cr, cg, cb)
        cv2.ellipse(frame, (cx, cy + int(_MX_BOWL_RY * 0.25)),
                    (content_rx, content_ry), 0, 0, 360, content_col, -1)
        # Lighter highlight on surface
        hl_col = tuple(min(255, c+30) for c in content_col)
        cv2.ellipse(frame, (cx - content_rx//4, cy + int(_MX_BOWL_RY * 0.15)),
                    (content_rx//2, content_ry//3), 0, 0, 360, hl_col, -1)

        # Spinning effect (swirl lines when spinning)
        if self._mx_spin_speed > 0.05:
            for i in range(4):
                a  = self._mx_spin + i * math.pi/2
                sx = int(cx + content_rx * 0.5 * math.cos(a))
                sy = int(cy + content_ry * 0.4 * math.sin(a) + _MX_BOWL_RY * 0.25)
                ex = int(cx + content_rx * 0.5 * math.cos(a + math.pi))
                ey = int(cy + content_ry * 0.4 * math.sin(a + math.pi) + _MX_BOWL_RY * 0.25)
                cv2.line(frame, (sx, sy), (ex, ey), hl_col, 3, cv2.LINE_AA)

        # ── Bowl rim (on top of contents) ─────────────────────────────────────
        cv2.ellipse(frame, (cx, cy), (_MX_BOWL_RX, _MX_BOWL_RY),
                    0, 0, 360, (175, 182, 195), 5)
        # Inner rim highlight (top)
        cv2.ellipse(frame, (cx, cy), (_MX_BOWL_RX, _MX_BOWL_RY),
                    0, 200, 340, (235, 240, 248), 3)

        # ── Mixer head (top) ──────────────────────────────────────────────────
        head_y = cy - _MX_BOWL_RY - 30
        cv2.rectangle(frame, (cx-55, head_y-55), (cx+55, head_y+10),
                      (80, 78, 72), -1)
        cv2.rectangle(frame, (cx-55, head_y-55), (cx+55, head_y+10),
                      (110, 108, 100), 3)
        # Shaft down into bowl
        cv2.rectangle(frame, (cx-8, head_y+10), (cx+8, cy - _MX_BOWL_RY//2),
                      (90, 88, 82), -1)
        # Beater blades (spinning)
        blade_cy = cy - _MX_BOWL_RY//3
        for i in range(3):
            a  = self._mx_spin + i * (2*math.pi/3)
            bx = int(cx + 38 * math.cos(a))
            by = int(blade_cy + 20 * math.sin(a))
            cv2.line(frame, (cx, blade_cy), (bx, by), (155, 158, 165), 5)
            cv2.circle(frame, (bx, by), 6, (175, 178, 185), -1)

        # ── Fail flash ────────────────────────────────────────────────────────
        if self._mx_fail_flash > 0 and self._mx_fail_flash % 4 < 2:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 200), 8)
            _shadow_text(frame, 'GREEN only!', w//2, h//2,
                         1.2, (80, 120, 255), 2, center=True)

        # ── Traffic light ─────────────────────────────────────────────────────
        lx, ly = 90, 160
        lights = [('red', (30, 30, 220)), ('orange', (30, 160, 240)),
                  ('green', (50, 210, 70))]
        # Panel
        cv2.rectangle(frame, (lx-35, ly-75), (lx+35, ly+75), (40, 38, 34), -1)
        cv2.rectangle(frame, (lx-35, ly-75), (lx+35, ly+75), (70, 68, 62), 3)
        for i, (name, col) in enumerate(lights):
            ly_c = ly - 50 + i * 50
            active = (self._mx_light == name)
            draw_col = col if active else tuple(c//4 for c in col)
            pulse = int(8*abs(math.sin(t*6))) if active else 0
            cv2.circle(frame, (lx, ly_c), 18+pulse, draw_col, -1)
            if active:
                cv2.circle(frame, (lx, ly_c), 18+pulse, (255,255,255), 2)

        # Green hint
        light_hint = {'green': 'Add ingredient NOW!',
                      'orange': 'Wait...', 'red': 'STOP!'}
        hint_cols  = {'green': (50, 220, 80), 'orange': (30, 160, 240), 'red': (60, 60, 220)}
        _shadow_text(frame, light_hint[self._mx_light],
                     lx, ly + 95, 0.58, hint_cols[self._mx_light], 1, center=True)

        # ── Current ingredient ────────────────────────────────────────────────
        if self._mx_added < len(_MX_INGREDIENTS):
            ing_name = _MX_INGREDIENTS[self._mx_added]
            ing_col  = _MX_ING_COLORS[self._mx_added]
            ix = int(self._mx_ing_pos[0])
            iy = int(self._mx_ing_pos[1])

            # Draw ingredient as a labeled rounded rectangle (2x size)
            cv2.rectangle(frame, (ix-100+6, iy-76+6), (ix+100+6, iy+76+6), (18,16,14), -1)
            cv2.rectangle(frame, (ix-100, iy-76), (ix+100, iy+76), ing_col, -1)
            cv2.rectangle(frame, (ix-100, iy-76), (ix+100, iy+76),
                          tuple(max(0,c-40) for c in ing_col), 3)
            _shadow_text(frame, ing_name, ix, iy-10, 0.75,
                         (30,30,30), 1, center=True)

            # Grab ring if not grabbed
            if not self._mx_grabbed:
                ring_r = int(104 + 16*abs(math.sin(t*5)))
                cv2.circle(frame, (ix, iy), ring_r, (0,200,255), 2, cv2.LINE_AA)
                _shadow_text(frame, 'GRAB', ix, iy - ring_r - 12,
                             0.825, (0,200,255), 1, center=True)

        # ── Progress + HUD ────────────────────────────────────────────────────
        _mx_lbl = f'{self._mx_added} / {len(_MX_INGREDIENTS)}  added'
        _mx_w   = cv2.getTextSize(_mx_lbl, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)[0][0]
        _shadow_text(frame, _mx_lbl, w - 26 - _mx_w, 100, 0.85, (255, 230, 100), 2)
        _shadow_text(frame, 'Add ingredients only when GREEN',
                     w//2, h-32, 0.58, (190, 200, 215), 1, center=True)
        return frame

    def _draw_intermission(self, frame):
        from .game_manager import _draw_cooking_papa
        h, w = frame.shape[:2]

        overlay = np.zeros_like(frame)
        overlay[:] = (20, 18, 16)
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

        title, sub = _PINTRO_TEXT.get(self._phase, ('', ''))
        _draw_cooking_papa(frame, w // 4, h // 2 + 20, size=150)

        tcx = w * 9 // 16
        _shadow_text(frame, title, tcx, h // 2 - 26,
                     1.55, (80, 220, 255), 3, center=True)
        _shadow_text(frame, sub, tcx, h // 2 + 28,
                     0.68, (225, 225, 240), 1, center=True)

        n_dots = 5
        elapsed = 0.0 if self._inter_t0 == 0.0 else time.time() - self._inter_t0
        filled  = int(n_dots * min(1.0, elapsed / _PINTRO_DURATION))
        for di in range(n_dots):
            dx = tcx - (n_dots - 1) * 16 + di * 32
            dc = (80, 220, 160) if di < filled else (90, 90, 100)
            cv2.circle(frame, (dx, h // 2 + 78), 7, dc, -1)
            cv2.circle(frame, (dx, h // 2 + 78), 7, (200, 205, 215), 1)

    def _draw_reveal(self, frame):
        from .ui import draw_reveal_burst
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        t = time.time()

        draw_reveal_burst(frame, cx, cy, t)

        # ── Pancake stack (center) ────────────────────────────────────────────
        pr = 175   # pancake radius
        for layer, yoff in [(2, 28), (1, 12), (0, 0)]:
            shade = (38 + layer*8, 128 + layer*8, 172 + layer*8)
            shade_d = (25 + layer*6, 102 + layer*6, 145 + layer*6)
            cv2.ellipse(frame, (cx+4, cy+yoff+6), (pr, int(pr*0.38)), 0, 0, 360, (18,14,10), -1)
            cv2.ellipse(frame, (cx, cy+yoff), (pr, int(pr*0.38)), 0, 0, 360, shade, -1)
            cv2.ellipse(frame, (cx, cy+yoff), (pr, int(pr*0.38)), 0, 0, 360, shade_d, 4)

        # Butter pat (pale yellow rectangle on top)
        bw, bh2 = 75, 38
        bx2, by2 = cx - bw//2, cy - int(pr*0.25)
        cv2.rectangle(frame, (bx2+3, by2+4), (bx2+bw+3, by2+bh2+4), (18,14,10), -1)
        cv2.rectangle(frame, (bx2, by2), (bx2+bw, by2+bh2), (120, 225, 250), -1)
        cv2.rectangle(frame, (bx2, by2), (bx2+bw, by2+bh2), (90, 185, 210), 3)

        # Syrup drizzle lines (wavy brown)
        SYRUP = (18, 85, 165)   # dark amber BGR
        import random as _rnd
        _rnd.seed(42)
        for _ in range(7):
            sx0 = cx + _rnd.randint(-pr + 20, pr - 20)
            sy0 = cy + _rnd.randint(-int(pr*0.3), int(pr*0.2))
            pts_sy = [(sx0, sy0)]
            for seg in range(5):
                nx2 = pts_sy[-1][0] + _rnd.randint(18, 38)
                ny2 = pts_sy[-1][1] + _rnd.randint(-14, 14)
                pts_sy.append((nx2, ny2))
            for k in range(len(pts_sy)-1):
                cv2.line(frame, pts_sy[k], pts_sy[k+1], SYRUP, 5, cv2.LINE_AA)

        # Cinnamon powder specks
        _rnd.seed(7)
        for _ in range(30):
            spx = cx + _rnd.randint(-pr+10, pr-10)
            spy = cy + _rnd.randint(-int(pr*0.3), int(pr*0.1))
            cv2.circle(frame, (spx, spy), _rnd.randint(1, 3), (22, 58, 105), -1)

        # ── Title text ────────────────────────────────────────────────────────
        _shadow_text(frame, 'Good Job!', cx, cy - 288,
                     1.0, (255, 255, 255), 2, center=True)
        _shadow_text(frame, 'Pancakes!', cx, cy - 230,
                     1.4, (30, 190, 255), 3, center=True)
