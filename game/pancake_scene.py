import cv2
import math
import time
import numpy as np
from .minigames.base import BaseMiniGame

# ── Mixer constants ───────────────────────────────────────────────────────────
_MX_CX       = 640
_MX_CY       = 360
_MX_BOWL_RX  = 195
_MX_BOWL_RY  = 155
_MX_LIGHT_INTERVAL = 1.5    # seconds between light changes
_MX_GRAB_R   = 70
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
_PC_PAN_R       = 235
_PC_PAN_AREA_R  = 188
_PC_PAN_HX      = 640 - 235 - 80   # handle left side x  (= 325)
_PC_PAN_HY      = 390               # handle y (same as pan center)
_PC_GAUGE_SPEED = 0.0032       # fill per frame
_PC_GREEN_MIN   = 0.58         # green zone start
_PC_GREEN_MAX   = 0.88         # green zone end (burn if exceeded)
_PC_FLIP_PX     = 65           # upward px to detect flip
_PC_PLATE_X     = 1000         # plate resting x
_PC_PLATE_Y     = 440          # plate resting y
_PC_PLATE_GRAB_R = 80

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


class PancakeScene(BaseMiniGame):
    name        = 'Pancakes'
    instruction = 'Make pancakes!'
    duration    = 180.0
    grab_phase  = True

    _DEBUG_PHASES = ['egg_separate', 'mixer', 'cook_pancake']

    def __init__(self):
        super().__init__()
        self._phase = 'egg_separate'
        self._init_egg()
        self._init_mixer()
        self._init_cook_pancake()

    def jump_to_phase(self, phase):
        self._phase = phase
        self._begin_timer()
        if phase == 'cook_pancake':
            self._init_cook_pancake()
        elif phase == 'mixer':
            self._init_mixer()

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
        if self._phase == 'egg_separate':
            self._update_egg(hands)
        elif self._phase == 'mixer':
            self._update_mixer(hands)
        elif self._phase == 'cook_pancake':
            self._update_cook_pancake(hands)
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
                    self._phase = 'mixer'
                    self._init_mixer()
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
                break

            break

    # ── draw ──────────────────────────────────────────────────────────────────

    def draw(self, frame, hands):
        t  = time.time()
        h, w = frame.shape[:2]
        if self._phase == 'mixer':
            return self._draw_mixer(frame)
        if self._phase == 'cook_pancake':
            return self._draw_cook_pancake(frame)

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
        _shadow_text(frame, f'Eggs: {sum(self._e_done)} / {_E_EGGS}',
                     w//2, 55, 1.0, (255, 230, 100), 2, center=True)
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
        self._pc_side       = 0            # current cooking side
        self._pc_subphase   = 'cook'       # 'cook' | 'flip_anim' | 'serve'
        self._pc_flip_anim  = 0            # 0=none, 1-40=in air
        self._pc_pan_grabbed = False
        self._pc_flip_prev_y = None
        self._pc_plate_grabbed = False
        self._pc_plate_pos   = [float(_PC_PLATE_X), float(_PC_PLATE_Y)]
        self._pc_pancake_x   = float(_PC_PAN_CX)
        self._pc_pancake_y   = float(_PC_PAN_CY)

    def _update_cook_pancake(self, hands):
        sp = self._pc_subphase

        # ── Flip animation ────────────────────────────────────────────────────
        if sp == 'flip_anim':
            self._pc_flip_anim -= 1
            if self._pc_flip_anim == 20:
                self._pc_side = 1         # switched to side 1
            if self._pc_flip_anim <= 0:
                self._pc_subphase = 'cook'
                self._pc_pan_grabbed = False
                self._pc_flip_prev_y = None
            return

        # ── Fill gauge for current side ───────────────────────────────────────
        if sp == 'cook':
            s = self._pc_side
            if self._pc_gauge[s] < _PC_GREEN_MAX:
                self._pc_gauge[s] = min(_PC_GREEN_MAX,
                                        self._pc_gauge[s] + _PC_GAUGE_SPEED)

            # After gauge[0] reaches green: prompt to flip
            if s == 0 and self._pc_gauge[0] >= _PC_GREEN_MIN:
                self._handle_flip(hands)

            # After gauge[1] reaches green: go to serve
            if s == 1 and self._pc_gauge[1] >= _PC_GREEN_MIN:
                self._pc_subphase = 'serve'

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
                # Pancake reached pan area?
                dx = self._pc_plate_pos[0] - _PC_PAN_CX
                dy = self._pc_plate_pos[1] - _PC_PAN_CY
                if dx*dx + dy*dy < (_PC_PAN_R * 0.85)**2:
                    self._phase = 'complete'
                break

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
                    return
            self._pc_flip_prev_y = sy
            break

    def _draw_cook_pancake(self, frame):
        t  = time.time()
        h, w = frame.shape[:2]
        cx, cy = _PC_PAN_CX, _PC_PAN_CY
        sp = self._pc_subphase

        # ── Stove ring ────────────────────────────────────────────────────────
        cv2.ellipse(frame, (cx+5, cy+8), (_PC_PAN_R+38, _PC_PAN_R+38),
                    0, 0, 360, (18,15,12), -1)
        cv2.ellipse(frame, (cx, cy), (_PC_PAN_R+38, _PC_PAN_R+38),
                    0, 0, 360, (55, 180, 230), -1)   # blue/orange stove
        cv2.ellipse(frame, (cx, cy), (_PC_PAN_R+24, _PC_PAN_R+24),
                    0, 0, 360, (40, 40, 200), -1)    # inner ring (red-orange)
        # Flame glow
        cv2.ellipse(frame, (cx, cy), (_PC_PAN_R+12, _PC_PAN_R+12),
                    0, 0, 360, (18, 108, 220), -1)

        # ── Pan body ──────────────────────────────────────────────────────────
        cv2.circle(frame, (cx+4, cy+4), _PC_PAN_R, (15,12,10), -1)
        cv2.circle(frame, (cx, cy), _PC_PAN_R, (42, 38, 34), -1)
        cv2.circle(frame, (cx, cy), _PC_PAN_R, (22, 18, 14), 8)
        # Cooking surface
        cook_heat = self._pc_gauge[self._pc_side]
        sv = int(138 - cook_heat * 22)
        cv2.circle(frame, (cx, cy), _PC_PAN_AREA_R, (sv, sv+2, sv+4), -1)
        cv2.circle(frame, (cx, cy), _PC_PAN_AREA_R, (55, 50, 45), 2)

        # ── Pan handle (LEFT side) ────────────────────────────────────────────
        hx1 = cx - _PC_PAN_R + 12
        hx2 = cx - _PC_PAN_R - 150
        cv2.rectangle(frame, (hx2, cy-22), (hx1, cy+22), (28, 24, 20), -1)
        cv2.rectangle(frame, (hx2, cy-22), (hx1, cy+22), (12, 10, 8),  3)
        # Grip texture
        for gx in range(hx2+20, hx1-10, 22):
            cv2.line(frame, (gx, cy-18), (gx, cy+18), (40, 36, 32), 3)

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
        if (sp == 'cook' and self._pc_side == 0
                and self._pc_gauge[0] >= _PC_GREEN_MIN
                and not self._pc_pan_grabbed
                and self._pc_flip_anim == 0):
            rr = int(52 + 8*abs(math.sin(t*5)))
            cv2.circle(frame, (_PC_PAN_HX, _PC_PAN_HY), rr, (0,200,255), 2, cv2.LINE_AA)
            _shadow_text(frame, 'GRAB & Flick UP!', _PC_PAN_HX,
                         _PC_PAN_HY - rr - 14, 0.62, (0,200,255), 1, center=True)
            # Up arrow
            cv2.arrowedLine(frame, (_PC_PAN_HX, _PC_PAN_HY + 40),
                            (_PC_PAN_HX, _PC_PAN_HY - 50),
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
            cv2.arrowedLine(frame, (px-60, py), (px-180, py),
                            (80,220,140), 3, tipLength=0.28)

        # ── Gauges (top right, with face emoji) ───────────────────────────────
        gx0 = w - 340
        for idx, gy in enumerate([30, 85]):
            done  = self._pc_gauge[idx] >= _PC_GREEN_MIN
            face  = ':)' if done else ':('
            fcol  = (50,220,80) if done else (50,80,220)
            # face circle
            cv2.circle(frame, (gx0 - 28, gy+20), 22, fcol, -1)
            cv2.circle(frame, (gx0 - 28, gy+20), 22, (255,255,255), 2)
            _shadow_text(frame, face, gx0-28, gy+27, 0.48, (255,255,255), 1, center=True)
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
        bw, bh = 300, 38
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (25,22,18), -1)
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (60,58,52), 2)
        zones = [
            (0.0,        _PC_GREEN_MIN,  (40, 55, 210)),   # red
            (_PC_GREEN_MIN, _PC_GREEN_MAX, (50, 210, 80)),  # green
        ]
        for zs, ze, col in zones:
            x1 = bx + int(zs * bw) + 2
            x2 = bx + int(ze * bw) - 1
            if x2 > x1:
                cv2.rectangle(frame, (x1, by+3), (x2, by+bh-3), col, -1)
        # Moving indicator pin
        fill_x = bx + int(value * bw)
        cv2.rectangle(frame, (fill_x-5, by-5), (fill_x+5, by+bh+5), (255,255,255), -1)
        cv2.rectangle(frame, (fill_x-5, by-5), (fill_x+5, by+bh+5), (70,75,90), 2)

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
                            self._phase = 'cook_pancake'
                            self._init_cook_pancake()
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

            # Draw ingredient as a labeled rounded rectangle
            cv2.rectangle(frame, (ix-50+3, iy-38+3), (ix+50+3, iy+38+3), (18,16,14), -1)
            cv2.rectangle(frame, (ix-50, iy-38), (ix+50, iy+38), ing_col, -1)
            cv2.rectangle(frame, (ix-50, iy-38), (ix+50, iy+38),
                          tuple(max(0,c-40) for c in ing_col), 3)
            _shadow_text(frame, ing_name, ix, iy-10, 0.50,
                         (30,30,30), 1, center=True)

            # Grab ring if not grabbed
            if not self._mx_grabbed:
                ring_r = int(52 + 8*abs(math.sin(t*5)))
                cv2.circle(frame, (ix, iy), ring_r, (0,200,255), 2, cv2.LINE_AA)
                _shadow_text(frame, 'GRAB', ix, iy - ring_r - 12,
                             0.55, (0,200,255), 1, center=True)

        # ── Progress + HUD ────────────────────────────────────────────────────
        _shadow_text(frame, f'{self._mx_added} / {len(_MX_INGREDIENTS)}  added',
                     w//2, 45, 0.85, (255, 230, 100), 2, center=True)
        _shadow_text(frame, 'Add ingredients only when GREEN',
                     w//2, h-32, 0.58, (190, 200, 215), 1, center=True)
        return frame
