import cv2
import math
import time
import numpy as np
from .minigames.base import BaseMiniGame

# ── Constants ─────────────────────────────────────────────────────────────────
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
    instruction = 'Rotate CCW to pour out the egg white!'
    duration    = 180.0
    grab_phase  = True

    def __init__(self):
        super().__init__()
        self._phase = 'egg_separate'
        self._init_egg()

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
                    self._phase = 'complete'
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
