"""
Sequential cooking scene — one stage at a time.
  Stage 1: grab knife → chop (5x)
  Stage 2: grab spatula → stir (3x)
  Stage 3: grab pan → flip (2x)
"""
import cv2
import math
import time
import numpy as np

from .minigames.base import BaseMiniGame
from .sprites import (get_knife, get_spatula, get_bowl, overlay,
                      get_carrot_whole,   get_carrot_half_left,   get_carrot_half_right,
                      get_cucumber_whole, get_cucumber_half_left, get_cucumber_half_right,
                      get_pepper_whole,   get_pepper_half_left,   get_pepper_half_right)

# ── Task targets ──────────────────────────────────────────────────────────────
CHOP_TARGET  = 5
STIR_TARGET  = 3
FLIP_TARGET  = 2

CHOP_PIXELS  = 55    # screen-pixel travel needed per chop stroke
CHOP_RANGE   = 90    # px — knife center must be within this of the target vegetable
FLIP_PIXELS  = 18    # screen-pixel upward jump per frame to trigger flip
STIR_MIN_R   = 0.40  # world-unit min radius for stir detection
GRAB_RADIUS  = 130   # px proximity to grab

# ── Stage 1 – Cutting  (objects in center of screen, y ≈ 400-440) ─────────────
S1_BOARD_X  = 310
S1_BOARD_Y  = 358
S1_BOARD_W  = 660
S1_BOARD_H  = 140
S1_KNIFE_X  = 430
S1_KNIFE_Y  = 415
S1_VEG_Y    = 415
S1_VEG_XS   = [580, 700, 820]   # carrot, cucumber, pepper x-positions
# Chops each vegetable needs: total = CHOP_TARGET (5)
S1_VEG_CHOP_OFFSET = [0, 2, 4]  # chop index when this veg starts being cut
S1_VEG_CHOP_MAX    = [2, 2, 1]  # max visible cuts per vegetable

# ── Stage 2 – Stirring ────────────────────────────────────────────────────────
S2_POT_X   = 640
S2_POT_Y   = 390
S2_POT_WX  = 0.0     # world X matching screen_x 640
S2_POT_WY  = -0.375  # world Y matching screen_y 390
S2_SPA_X   = 460
S2_SPA_Y   = 330

# ── Stage 3 – Flipping ────────────────────────────────────────────────────────
S3_PAN_X    = 640
S3_PAN_Y    = 390
S3_STOVE_CX = 640
S3_STOVE_CY = 440

_COUNTER_Y  = 340    # kitchen counter line (fixed, not % of height)

# ── Phase hints ───────────────────────────────────────────────────────────────
_PHASE_HINTS = {
    'grab_knife':       ('GRAB the knife!',        (0, 200, 255)),
    'chopping':         ('Chop  UP  and  DOWN!',   (0, 255, 180)),
    'pepper_splitting': ('',                        (0, 255, 100)),
    'grab_spatula':     ('GRAB the spatula!',       (0, 200, 255)),
    'stirring':         ('Stir in CIRCLES!',        (0, 255, 180)),
    'grab_pan':         ('GRAB the pan!',           (0, 200, 255)),
    'flipping':         ('Flick hand UP quickly!',  (50, 200, 255)),
    'complete':         ('ALL DONE!',               (0, 255, 100)),
}


class CookingScene(BaseMiniGame):
    name        = 'Cooking!'
    instruction = 'Grab tools and cook the meal!'
    duration    = 120.0
    grab_phase  = True

    def __init__(self):
        super().__init__()
        self._phase     = 'grab_knife'
        self._held_tool = None
        self._held_by   = -1

        self.chops = 0
        self.stirs = 0
        self.flips = 0

        # chop detection: screen-coordinate based
        self._chop_ref_y = None  # screen_y at last direction change
        self._chop_dir   = None  # 'up' | 'down' | None

        # flip detection: screen-delta based
        self._flip_prev_y = None

        self._cooldown  = 0
        self._prev_ang  = None
        self._total_ang = 0.0

        self._knife_pos   = [S1_KNIFE_X, S1_KNIFE_Y]
        self._spatula_pos = [S2_SPA_X,   S2_SPA_Y]
        self._pan_pos     = [S3_PAN_X,   S3_PAN_Y]

        self._flash     = 0
        self._flash_lbl = ''
        self._flash_col = (255, 255, 255)
        self._trail     = []
        self._anim_pan  = 0

        self._knife_spr   = get_knife(size=130)
        self._spatula_spr = get_spatula(size=120)
        self._bowl_spr    = get_bowl(w=260, h=130)

        self._veg_sprs  = [
            get_carrot_whole(size=120),
            get_cucumber_whole(size=120),
            get_pepper_whole(size=120),
        ]
        self._veg_half_l = [
            get_carrot_half_left(size=120),
            get_cucumber_half_left(size=120),
            get_pepper_half_left(size=120),
        ]
        self._veg_half_r = [
            get_carrot_half_right(size=120),
            get_cucumber_half_right(size=120),
            get_pepper_half_right(size=120),
        ]
        # Per-vegetable split-animation start times (None = not yet cut)
        self._split_ts       = [None, None, None]
        self._pepper_split_t = None   # kept for legacy reference in update()

    # ── BaseMiniGame interface ────────────────────────────────────────────────

    @property
    def succeeded(self):
        return self._phase == 'complete'

    @property
    def progress_text(self):
        p = self._phase
        if p == 'grab_knife':       return 'Step 1/3  —  Grab the knife!'
        if p == 'chopping':         return f'Step 1/3  —  Chop  {self.chops}/{CHOP_TARGET}'
        if p == 'pepper_splitting': return 'Step 1/3  —  Nice cut!'
        if p == 'grab_spatula':     return 'Step 2/3  —  Grab the spatula!'
        if p == 'stirring':         return f'Step 2/3  —  Stir  {self.stirs}/{STIR_TARGET}'
        if p == 'grab_pan':         return 'Step 3/3  —  Grab the pan!'
        if p == 'flipping':         return f'Step 3/3  —  Flip  {self.flips}/{FLIP_TARGET}'
        return 'ALL DONE!'

    def update(self, hands):
        events = []
        if self._flash    > 0: self._flash    -= 1
        if self._anim_pan > 0: self._anim_pan -= 1

        active = self._get_active_hand(hands)
        if active:
            self._trail.append((active.screen_x, active.screen_y))
        else:
            self._trail.clear()
        if len(self._trail) > 35:
            self._trail.pop(0)

        p = self._phase
        if p == 'grab_knife':
            events += self._try_grab(hands, self._knife_pos, 'knife', 'chopping')

        elif p == 'chopping':
            if active is None:
                self._try_reattach(hands, self._knife_pos, 'knife')
                active = self._get_active_hand(hands)
            self._sync_tool_pos(active)
            if active:
                events += self._do_chop(active)
            else:
                self._chop_ref_y = None
                self._chop_dir   = None

        elif p == 'pepper_splitting':
            # Wait for the split animation to finish, then move to stage 2
            if self._pepper_split_t is not None and time.time() - self._pepper_split_t > 0.55:
                self._phase = 'grab_spatula'

        elif p == 'grab_spatula':
            events += self._try_grab(hands, self._spatula_pos, 'spatula', 'stirring')

        elif p == 'stirring':
            if active is None:
                self._try_reattach(hands, self._spatula_pos, 'spatula')
                active = self._get_active_hand(hands)
            self._sync_tool_pos(active)
            if active:
                events += self._do_stir(active)
            else:
                self._prev_ang = None

        elif p == 'grab_pan':
            events += self._try_grab(hands, self._pan_pos, 'pan', 'flipping')

        elif p == 'flipping':
            if active is None:
                self._try_reattach(hands, self._pan_pos, 'pan')
                active = self._get_active_hand(hands)
            self._sync_tool_pos(active)
            if active:
                events += self._do_flip(active)
            else:
                self._flip_prev_y = None

        return events

    # ── Phase helpers ─────────────────────────────────────────────────────────

    def _try_reattach(self, hands, tool_pos, tool_name):
        """Re-attach a dropped tool at its current position without changing the phase."""
        for i, hand in enumerate(hands):
            if not hand.detected or not hand.gripped:
                continue
            dx = hand.screen_x - tool_pos[0]
            dy = hand.screen_y - tool_pos[1]
            if dx * dx + dy * dy < GRAB_RADIUS ** 2:
                self._held_tool = tool_name
                self._held_by   = i
                return

    def _try_grab(self, hands, rest_pos, tool, next_phase):
        for i, hand in enumerate(hands):
            if not hand.detected or not hand.gripped:
                continue
            dx = hand.screen_x - rest_pos[0]
            dy = hand.screen_y - rest_pos[1]
            if dx * dx + dy * dy < GRAB_RADIUS ** 2:
                self._held_tool  = tool
                self._held_by    = i
                self._phase      = next_phase
                self._begin_timer()
                self._chop_ref_y = None
                self._chop_dir   = None
                self._flip_prev_y = None
                self._cooldown   = 0
                self._prev_ang   = None
                self._total_ang  = 0.0
                self._flash_event('GRAB!', (0, 220, 255), 8)
                return ['grab']
        return []

    def _get_active_hand(self, hands):
        if self._held_by < 0:
            return None
        if self._held_by >= len(hands):
            self._held_by   = -1
            self._held_tool = None
            return None
        hand = hands[self._held_by]
        if not hand.detected or not hand.gripped:
            self._held_by   = -1
            self._held_tool = None
            return None
        return hand

    def _sync_tool_pos(self, hand):
        if hand is None:
            return
        if   self._held_tool == 'knife':   self._knife_pos   = [hand.screen_x, hand.screen_y]
        elif self._held_tool == 'spatula': self._spatula_pos = [hand.screen_x, hand.screen_y]
        elif self._held_tool == 'pan':     self._pan_pos     = [hand.screen_x, hand.screen_y]

    def _do_chop(self, hand):
        """Screen-pixel based chop detection — only counts when knife is over target veg."""
        if self._cooldown > 0:
            self._cooldown -= 1
            return []

        # Determine which vegetable is the current target
        if self.chops < S1_VEG_CHOP_OFFSET[1]:    vi = 0  # carrot
        elif self.chops < S1_VEG_CHOP_OFFSET[2]:  vi = 1  # cucumber
        else:                                       vi = 2  # pepper

        # Knife (= hand) must be within CHOP_RANGE of the target vegetable's x
        if abs(hand.screen_x - S1_VEG_XS[vi]) > CHOP_RANGE:
            # Reset tracking so direction history doesn't carry over
            self._chop_ref_y = None
            self._chop_dir   = None
            return []

        sy = hand.screen_y

        # Initialise reference on first call (or after re-entering zone)
        if self._chop_ref_y is None:
            self._chop_ref_y = sy
            return []

        delta = sy - self._chop_ref_y  # +ve = hand moved DOWN on screen

        # Establish direction after first significant move
        if self._chop_dir is None:
            if delta >= CHOP_PIXELS:
                self._chop_dir   = 'down'
                self._chop_ref_y = sy
            elif delta <= -CHOP_PIXELS:
                self._chop_dir   = 'up'
                self._chop_ref_y = sy
            return []

        # Count a chop on each reversal of sufficient amplitude
        counted = False
        if self._chop_dir == 'down' and delta <= -CHOP_PIXELS:
            self._chop_dir   = 'up'
            self._chop_ref_y = sy
            counted = True
        elif self._chop_dir == 'up' and delta >= CHOP_PIXELS:
            self._chop_dir   = 'down'
            self._chop_ref_y = sy
            counted = True

        if counted:
            self.chops    += 1
            self._cooldown = 6
            self._flash_event('CHOP!', (0, 255, 200), 12)

            # Trigger split animation for each vegetable on its last chop
            _split_triggers = [
                S1_VEG_CHOP_OFFSET[i] + S1_VEG_CHOP_MAX[i]
                for i in range(3)
            ]  # [2, 4, 5]
            for vi, trigger in enumerate(_split_triggers):
                if self.chops == trigger and self._split_ts[vi] is None:
                    self._split_ts[vi] = time.time()

            if self.chops >= CHOP_TARGET:
                self._held_tool      = None
                self._held_by        = -1
                self._phase          = 'pepper_splitting'
                self._knife_pos      = [S1_KNIFE_X, S1_KNIFE_Y]
                self._chop_ref_y     = None
                self._chop_dir       = None
                self._pepper_split_t = self._split_ts[2]  # sync with unified list
            return ['cut']
        return []

    def _do_stir(self, hand):
        dx = hand.x - S2_POT_WX
        dy = hand.y - S2_POT_WY
        if math.sqrt(dx * dx + dy * dy) < STIR_MIN_R:
            return []
        angle = math.atan2(dy, dx)
        if self._prev_ang is not None:
            delta = angle - self._prev_ang
            if delta >  math.pi: delta -= 2 * math.pi
            if delta < -math.pi: delta += 2 * math.pi
            self._total_ang += delta
            new_stirs = int(abs(self._total_ang) / (2 * math.pi))
            if new_stirs > self.stirs:
                self.stirs = new_stirs
                self._flash_event('STIR!', (0, 230, 180), 14)
                if self.stirs >= STIR_TARGET:
                    self._held_tool   = None
                    self._held_by     = -1
                    self._phase       = 'grab_pan'
                    self._prev_ang    = None
                    self._spatula_pos = [S2_SPA_X, S2_SPA_Y]
                    return ['stir']
        self._prev_ang = angle
        return []

    def _do_flip(self, hand):
        """Screen-delta based flip — detects a sharp upward flick."""
        if self._cooldown > 0:
            self._cooldown -= 1
            self._flip_prev_y = hand.screen_y
            return []

        sy = hand.screen_y

        if self._flip_prev_y is None:
            self._flip_prev_y = sy
            return []

        # Upward movement in screen coords = screen_y decreasing
        up_delta = self._flip_prev_y - sy   # positive = moved up
        self._flip_prev_y = sy

        if up_delta >= FLIP_PIXELS:
            self.flips    += 1
            self._cooldown = 28
            self._anim_pan = 22
            self._flash_event('FLIP!', (255, 200, 50), 18)
            if self.flips >= FLIP_TARGET:
                self._held_tool = None
                self._held_by   = -1
                self._phase     = 'complete'
                self._complete  = True
            return ['flip']
        return []

    def _flash_event(self, label, color, frames):
        self._flash     = frames
        self._flash_lbl = label
        self._flash_col = color

    # ── Drawing: one stage at a time ─────────────────────────────────────────

    def draw(self, frame, hands):
        h, w = frame.shape[:2]
        _draw_counter(frame, w, h)

        p = self._phase
        if p in ('grab_knife', 'chopping', 'pepper_splitting'):
            self._draw_stage1(frame)
        elif p in ('grab_spatula', 'stirring'):
            self._draw_stage2(frame)
        else:
            self._draw_stage3(frame)

        if len(self._trail) > 1:
            for i in range(1, len(self._trail)):
                a = i / len(self._trail)
                cv2.line(frame, self._trail[i-1], self._trail[i],
                         (int(50*a), int(180*a), 255),
                         max(1, int(a * 4)), cv2.LINE_AA)

        self._draw_demo_hand(frame)

        if self._flash > 0:
            _shadow_text(frame, self._flash_lbl, w // 2, h // 3,
                         2.4, self._flash_col, 5, center=True)

        self._draw_hint(frame, w, h)
        return frame

    def _draw_stage1(self, frame):
        bx, by = S1_BOARD_X, S1_BOARD_Y
        bw, bh = S1_BOARD_W, S1_BOARD_H
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (55, 100, 45), -1)
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (35, 70,  25),  3)
        for i in range(1, 8):
            lx = bx + bw * i // 8
            cv2.line(frame, (lx, by+4), (lx, by+bh-4), (45, 85, 35), 1)

        # Draw 3 vegetables — split animation on last chop, cut marks before that
        for vi, (spr, vx) in enumerate(zip(self._veg_sprs, S1_VEG_XS)):
            split_t = self._split_ts[vi]

            if split_t is not None:
                # Split animation: smooth-step ease, halves slide apart
                raw_t  = min((time.time() - split_t) / 0.45, 1.0)
                t      = raw_t * raw_t * (3.0 - 2.0 * raw_t)
                offset = int(t * 52)
                overlay(frame, self._veg_half_l[vi], vx - offset, S1_VEG_Y, size=115)
                overlay(frame, self._veg_half_r[vi], vx + offset, S1_VEG_Y, size=115)
                continue

            overlay(frame, spr, vx, S1_VEG_Y, size=115)

            # Vertical cut marks before the final split
            n_cuts = min(
                max(self.chops - S1_VEG_CHOP_OFFSET[vi], 0),
                S1_VEG_CHOP_MAX[vi] - 1
            )
            for c in range(n_cuts):
                offset_x = int((c - (n_cuts - 1) / 2.0) * 22)
                cut_x    = vx + offset_x
                cv2.line(frame, (cut_x, S1_VEG_Y - 50), (cut_x, S1_VEG_Y + 50),
                         (210, 235, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, (cut_x, S1_VEG_Y - 50), 2, (190, 220, 255), -1)
                cv2.circle(frame, (cut_x, S1_VEG_Y + 50), 2, (190, 220, 255), -1)

            # Pulsing target ring on the current active vegetable during chopping
            if self._phase == 'chopping':
                if self.chops < S1_VEG_CHOP_OFFSET[1]:   tvi = 0
                elif self.chops < S1_VEG_CHOP_OFFSET[2]: tvi = 1
                else:                                      tvi = 2
                if vi == tvi:
                    _grab_ring(frame, [vx, S1_VEG_Y], label='CHOP')

        overlay(frame, self._knife_spr,
                self._knife_pos[0], self._knife_pos[1], size=130)
        if self._phase == 'grab_knife':
            _grab_ring(frame, self._knife_pos)

    def _draw_stage2(self, frame):
        overlay(frame, self._bowl_spr, S2_POT_X, S2_POT_Y)
        for i in range(6):
            bx = S2_POT_X + int((i - 2.5) * 28)
            by = S2_POT_Y - int(abs(math.sin(time.time() * 2.5 + i)) * 14) - 5
            cv2.circle(frame, (bx, by), 6, (170, 185, 255), -1)
        frac    = (abs(self._total_ang) % (2 * math.pi)) / (2 * math.pi)
        arc_end = int(frac * 360)
        if arc_end > 0 and self._phase == 'stirring':
            bsh = self._bowl_spr.shape[0]
            bsw = self._bowl_spr.shape[1]
            rx, ry = bsw // 2 + 18, bsh // 4 + 10
            cv2.ellipse(frame, (S2_POT_X, S2_POT_Y), (rx, ry),
                        0, -90, -90 + arc_end, (0, 220, 180), 4)

        sp_ang = -45 if self._held_tool != 'spatula' else 0
        overlay(frame, self._spatula_spr,
                self._spatula_pos[0], self._spatula_pos[1], size=120, angle=sp_ang)
        if self._phase == 'grab_spatula':
            _grab_ring(frame, self._spatula_pos)

    def _draw_stage3(self, frame):
        cx, cy = S3_STOVE_CX, S3_STOVE_CY
        cv2.ellipse(frame, (cx, cy), (90, 28), 0, 0, 360, (50, 50, 55), -1)
        cv2.ellipse(frame, (cx, cy), (90, 28), 0, 0, 360, (80, 80, 90),  2)
        cv2.ellipse(frame, (cx, cy), (65, 20), 0, 0, 360, (60, 40, 40),  2)

        anim_off = int(math.sin(self._anim_pan / 22 * math.pi) * 70) if self._anim_pan > 0 else 0
        px, py   = self._pan_pos[0], self._pan_pos[1] - anim_off
        cv2.rectangle(frame, (px + 70, py - 9), (px + 150, py + 9), (55, 55, 55), -1)
        cv2.rectangle(frame, (px + 70, py - 9), (px + 150, py + 9), (40, 40, 40),  2)
        cv2.ellipse(frame, (px, py), (80, 24), 0, 0, 360, (75, 75, 80), -1)
        cv2.ellipse(frame, (px, py), (80, 24), 0, 0, 360, (45, 45, 50),  3)
        food_off   = -anim_off // 2 if self._anim_pan > 11 else 0
        food_color = (40, 165, 255) if self._anim_pan > 11 else (30, 140, 230)
        cv2.ellipse(frame, (px, py - 10 + food_off), (38, 14), 0, 0, 360, food_color, -1)

        if self._phase == 'flipping' and self._held_tool == 'pan' and self._cooldown == 0:
            cv2.arrowedLine(frame, (px, py + 40), (px, py - 40),
                            (80, 255, 160), 3, tipLength=0.35)
        if self._phase == 'grab_pan':
            _grab_ring(frame, self._pan_pos)

    def _draw_demo_hand(self, frame):
        """Animated guide hand showing the required motion for the current phase."""
        t = time.time()
        p = self._phase

        if p == 'pepper_splitting':
            return   # let the split animation speak for itself

        if p == 'grab_knife':
            # Hand slides in toward knife; open when far, closes as it arrives
            wave = (math.sin(t * 1.8) + 1) / 2    # 0→1 oscillation
            off  = int(60 * wave)
            grip = 1.0 - wave                      # open=1 when far, fist=0 when near
            _demo_fist(frame, self._knife_pos[0] - 110 + off,
                       self._knife_pos[1] - 20, openness=grip)

        elif p == 'chopping':
            # Fist bouncing up and down — shown to the LEFT of the cutting board
            cy = int(S1_KNIFE_Y - 70 + 80 * math.sin(t * 3.5))
            dx = S1_BOARD_X - 70
            _demo_fist(frame, dx, cy, openness=0.0)
            going_down = math.cos(t * 3.5) < 0
            ay2 = cy + (35 if going_down else -35)
            cv2.arrowedLine(frame, (dx + 55, cy), (dx + 55, ay2),
                            (0, 230, 180), 2, tipLength=0.45)

        elif p == 'grab_spatula':
            wave = (math.sin(t * 1.8) + 1) / 2
            off  = int(60 * wave)
            grip = 1.0 - wave
            _demo_fist(frame, self._spatula_pos[0] - 110 + off,
                       self._spatula_pos[1] - 20, openness=grip)

        elif p == 'stirring':
            # Fist orbiting the pot clockwise
            r  = 115
            cx = int(S2_POT_X + r * math.cos(t * 2.0))
            cy = int(S2_POT_Y + r * math.sin(t * 2.0))
            _demo_fist(frame, cx, cy, openness=0.0)

        elif p == 'grab_pan':
            wave = (math.sin(t * 1.8) + 1) / 2
            off  = int(60 * wave)
            grip = 1.0 - wave
            _demo_fist(frame, self._pan_pos[0] - 110 + off,
                       self._pan_pos[1] - 20, openness=grip)

        elif p == 'flipping':
            # Periodic upward flick — 0.25 s flick, 1.75 s rest
            cycle = (t % 2.0) / 2.0
            if cycle < 0.25:
                off = -int(cycle / 0.25 * 130)
            else:
                off = -int((1.0 - (cycle - 0.25) / 0.75) * 130)
            _demo_fist(frame, S3_PAN_X - 120, S3_PAN_Y + off, openness=0.0)
            if cycle < 0.25:
                cv2.arrowedLine(frame,
                                (S3_PAN_X - 120, S3_PAN_Y + off + 40),
                                (S3_PAN_X - 120, S3_PAN_Y + off - 30),
                                (80, 255, 160), 2, tipLength=0.45)

    def _draw_hint(self, frame, w, h):
        p = self._phase
        text, color = _PHASE_HINTS.get(p, ('', (255, 255, 255)))
        if not text:
            return
        _panel(frame, 0, 82, w, 44)
        count_str = ''
        if p == 'chopping':   count_str = f'  ({self.chops}/{CHOP_TARGET})'
        elif p == 'stirring': count_str = f'  ({self.stirs}/{STIR_TARGET})'
        elif p == 'flipping': count_str = f'  ({self.flips}/{FLIP_TARGET})'
        _shadow_text(frame, text + count_str, w // 2, 116, 0.9, color, 2, center=True)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _draw_counter(frame, w, h):
    pass  # counter and surface are provided by the kitchen background


def _grab_ring(frame, pos, label='GRAB'):
    t = time.time()
    r = int(52 + 14 * math.sin(t * 5))
    cv2.circle(frame, (int(pos[0]), int(pos[1])), r, (0, 200, 255), 2, cv2.LINE_AA)
    _shadow_text(frame, label, int(pos[0]), int(pos[1]) - r - 10,
                 0.55, (0, 200, 255), 1, center=True)


def _panel(frame, x, y, w, h, color=(15, 15, 15), alpha=0.6):
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x),      max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    bg  = np.empty_like(roi)
    bg[:] = color
    frame[y1:y2, x1:x2] = cv2.addWeighted(bg, alpha, roi, 1 - alpha, 0)


def _demo_fist(frame, cx, cy, size=58, openness=0.0):
    """White cartoon-style hand — openness 0.0=fist, 1.0=fingers fully extended."""
    fh, fw = frame.shape[:2]
    pad = size + 36
    x1, y1 = max(0, cx - pad), max(0, cy - pad)
    x2, y2 = min(fw, cx + pad), min(fh, cy + pad)
    if x2 <= x1 or y2 <= y1:
        return

    canvas = np.zeros((y2 - y1, x2 - x1, 4), dtype=np.uint8)
    lx, ly = cx - x1, cy - y1

    WHITE  = (248, 248, 252, 255)
    DARK   = (22,  22,  22,  255)
    CREASE = (155, 155, 172, 220)
    s      = size
    EX     = 5

    py0  = ly + int(s * 0.08)
    py1  = ly + int(s * 0.80)
    pw0  = int(s * 0.76)
    pw1  = int(s * 0.63)
    wr   = int(s * 0.18)
    fy_c = ly - int(s * 0.02)            # fist: knuckle y (near palm top)
    fy_o = ly - int(s * 0.55)            # open: fingertip y (extended upward)
    fy   = int(fy_c + openness * (fy_o - fy_c))
    fr   = int(s * 0.27)
    fsw  = int(fr * 0.82)                # finger shaft half-width
    fxs  = [lx + int(k * s) for k in (-0.51, -0.17, 0.17, 0.51)]
    tx   = lx - int(s * 0.84)
    ty   = ly + int(s * 0.28)
    tax  = int(s * 0.24)
    tay  = int(s * 0.40)

    def _fill(col, ex=0):
        palm = np.array([
            [lx - pw0 - ex, py0 - ex],
            [lx + pw0 + ex, py0 - ex],
            [lx + pw1 + ex, py1 + ex],
            [lx - pw1 - ex, py1 + ex],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [palm], col)
        cv2.ellipse(canvas, (lx, py1 + ex), (pw1 + ex, wr + ex), 0, 0, 180, col, -1)
        if openness > 0.05:
            for fx in fxs:
                cv2.rectangle(canvas,
                              (fx - fsw - ex, fy + int(fr * 0.35) - ex),
                              (fx + fsw + ex, py0 + ex),
                              col, -1)
        for fx in fxs:
            cv2.circle(canvas, (fx, fy - ex // 2), fr + ex, col, -1)
        cv2.ellipse(canvas, (tx - ex, ty), (tax + ex, tay + ex), -20, 0, 360, col, -1)

    _fill(DARK, EX)
    _fill(WHITE, 0)

    # Back-of-hand view: only between-finger separator lines, no palm/knuckle creases
    for i in range(len(fxs) - 1):
        gx = (fxs[i] + fxs[i+1]) // 2
        if openness > 0.3:
            cv2.line(canvas, (gx, fy - fr + 6), (gx, py0 - 4), CREASE, 2)
        else:
            cv2.line(canvas, (gx, fy - fr + 6), (gx, fy + int(s*0.18)), CREASE, 2)

    roi = frame[y1:y2, x1:x2]
    a   = canvas[:, :, 3:4].astype(np.float32) / 255.0
    roi[:] = (canvas[:, :, :3].astype(np.float32) * a +
              roi.astype(np.float32) * (1.0 - a)).astype(np.uint8)


def _shadow_text(frame, text, x, y, scale, color, thickness=2, center=False):
    font = cv2.FONT_HERSHEY_DUPLEX
    if center:
        tw, _ = cv2.getTextSize(text, font, scale, thickness)[0]
        x = x - tw // 2
    cv2.putText(frame, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, (x,     y),     font, scale, color,     thickness)
