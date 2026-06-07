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

from . import audio
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
CHOP_RANGE   = 115   # px — knife center must be within this of the target vegetable
FLIP_PIXELS  = 18    # screen-pixel upward jump per frame to trigger flip
STIR_MIN_R   = 0.40  # world-unit min radius for stir detection
GRAB_RADIUS  = 130   # px proximity to grab

# ── Stage 1 – Cutting  (objects in center of screen, y ≈ 400-440) ─────────────
S1_BOARD_X  = 190
S1_BOARD_Y  = 336
S1_BOARD_W  = 880
S1_BOARD_H  = 252
S1_KNIFE_X  = 278
S1_KNIFE_Y  = 426
S1_VEG_Y    = 426
S1_VEG_XS   = [400, 640, 880]   # carrot, cucumber, pepper x-positions (240 px gap)
S1_VEG_SIZE = 230               # sprite display size (2× original 115)
# Chops each vegetable needs: total = CHOP_TARGET (5)
S1_VEG_CHOP_OFFSET = [0, 2, 4]  # chop index when this veg starts being cut
S1_VEG_CHOP_MAX    = [2, 2, 1]  # max visible cuts per vegetable

# ── Stage 1 NEW: Onion cutting (Cooking Mama style) ──────────────────────────
_ONB_R        = 165     # display radius of whole onion
_ONB_START_Y  = 155     # initial onion y (top, before placing)
_ONB_BOARD_Y  = 415     # y where onion rests on board
_PEEL_NEEDED  = 3       # horizontal swipes to peel
_PEEL_PX      = 80      # px threshold per swipe
_SLICE_NEEDED   = 4       # slice strokes required
_SLICE_PX       = 52      # px threshold per slice
_SLICE_KNIFE_X  = 370     # knife resting x on board (left of onion)
_SLICE_KNIFE_Y  = 415     # knife resting y
_DICE_NEEDED    = 4       # dice cuts required
_DICE_KNIFE_X   = 870     # dice knife rests to the RIGHT of onion
_DICE_KNIFE_Y   = 415     # same height as onion center
_BOWL2_CX        = 200    # bowl center x  (LEFT)
_BOWL2_CY        = 430    # bowl center y
_PIECES_CX       = 800    # initial pieces center x (RIGHT, on cutting board)
_PIECES_CY       = 430    # initial pieces center y
_BOWL_PIECE_COUNT = 22    # number of onion piece objects
_BOWL_PIECE_GOAL  = 14    # pieces needed in bowl to complete
_KNIFE_PUSH_R     = 80    # px — knife push interaction radius
_BOWL_COLLECT_X   = 350   # pieces passing this x line are "in the bowl"

# ── Onion Fry (between Stage 1 and Stage 2) ──────────────────────────────────
ONION_PAN_CX  = 640
ONION_PAN_CY  = 385
ONION_PAN_R   = 236   # outer rim radius (295 * 0.8)
ONION_AREA_R  = 198   # inner cooking surface (248 * 0.8)
ONION_COUNT   = 52
ONION_STIR_NEEDED = 3600.0  # total stir units needed to reach ONION_READY
ONION_READY   = 0.45   # flame button activates at this fraction
ONION_INFL_R  = 125    # hand influence radius (px)
_FLAME_BTN_CX = 105
_FLAME_BTN_CY = 610
_FLAME_BTN_R  = 55

# ── Meat Grinder ─────────────────────────────────────────────────────────────
MEAT_MIX_TARGET  = 5      # crank rotations to complete
_GR_CX           = 680
_GR_CY           = 370
_GR_W            = 340    # body total width
_GR_H            = 220    # body total height
_GR_HANDLE_CX    = _GR_CX + _GR_W // 2 + 30
_GR_HANDLE_CY    = _GR_CY
_GR_HANDLE_ARM   = 88
_GR_NOZZLE_X     = _GR_CX - _GR_W // 2 - 30
_GR_NOZZLE_Y     = _GR_CY + 35
_GR_BOWL_CX      = _GR_NOZZLE_X - 50
_GR_BOWL_CY      = _GR_CY + _GR_H // 2 + 100
_GR_HOPPER_CX    = _GR_CX
_GR_HOPPER_Y     = _GR_CY - _GR_H // 2 - 55
_GR_BEEF_Y0      = 115
_GR_BEEF_DROP_Y  = _GR_HOPPER_Y + 45
_GR_GRAB_R       = 68

# ── Knead Stage ───────────────────────────────────────────────────────────────
_KN_BOWL_CX   = 640
_KN_BOWL_CY   = 385
_KN_BOWL_RX   = 300
_KN_BOWL_RY   = 230
_KN_ONION_X   = 1005   # fried onion pan resting x
_KN_ONION_Y   = 330
_KN_SWIPE_PX  = 85     # pixels needed for a valid swipe
_KN_ARROW_N   = 7
_KN_DIRS      = ('up', 'down', 'left', 'right')

# ── Cook Steak ───────────────────────────────────────────────────────────────
_SK_PAN_CX      = 640
_SK_PAN_CY      = 370
_SK_PAN_R       = 200
_SK_AREA_R      = 160
_SK_PAN_HX      = _SK_PAN_CX                     # handle grab x (bottom)
_SK_PAN_HY      = _SK_PAN_CY + _SK_PAN_R + 65   # handle grab y (bottom)
_SK_FLAME_CX    = 105
_SK_FLAME_CY    = 610
_SK_FLAME_R     = 55
_SK_ACTION_N    = 6
_SK_ACTIONS     = ('shake', 'press', 'flip', 'flame')
_SK_PRESS_HOLD  = 30    # frames to hold for press
_SK_SPAT_REST   = (_SK_PAN_CX + _SK_PAN_R + 80, _SK_PAN_CY - 70)  # spatula rest spot
_SK_SPAT_TIP_OFF = 150  # distance from gripped point to paddle tip (top of spatula)
_SK_SHAKE_PX    = 55    # px per shake stroke
_SK_SHAKE_N     = 3     # strokes needed for shake
_SK_FLIP_PX     = 70    # px upward flick

# ── Toss Meat ────────────────────────────────────────────────────────────────
_TOSS_L_X      = 185    # left catch zone center x
_TOSS_R_X      = 1095   # right catch zone center x
_TOSS_Y        = 370    # ball flight y center
_TOSS_BALL_R   = 72     # ball base radius
_TOSS_ZONE_R   = 140    # catch zone radius
_TOSS_SPEED    = 10     # pixels per frame
_TOSS_SWIPE_PX = 22     # screen px/frame to count as swipe
_TOSS_TARGET   = 8      # catches needed
_TOSS_READY_HOLD = 12   # frames both hands must be visible before start

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
    # ── new Stage 1 ──────────────────────────────────────────────────────────
    'place_onion':  ('Grip  and  pull  DOWN!',      (0, 220, 255)),
    'peel_skin':    ('Grip  and  swipe  LEFT / RIGHT!', (0, 200, 255)),
    'slice_onion':  ('Grip  and  chop  UP  &  DOWN!',   (0, 255, 180)),
    'dice_onion':   ('Grip  and  chop  UP  &  DOWN!',   (0, 255, 180)),
    'bowl_drop':    ('Grip  pieces  →  bowl!',          (255, 200, 50)),
    # ── old Stage 1 (commented out) ────────────────────────────────────────
    # 'grab_knife':  ('GRAB the knife!',  (0, 200, 255)),
    # 'chopping':    ('Chop UP and DOWN!',(0, 255, 180)),
    'pepper_splitting': ('',                            (0, 255, 100)),
    'onion_fry':        ('Stir  then  turn  off  flame!', (0, 230, 160)),
    'meat_mix':         ('Grab  beef  →  drop  in  grinder!', (100, 180, 255)),
    'knead':            ('Knead  the  meat!',              (180, 140, 255)),
    'toss_meat':        ('Swipe  opposite  to  catch!',   (80, 220, 255)),
    'cook_steak':       ('Follow  the  action!',          (80, 200, 255)),
    'grab_spatula':     ('GRAB the spatula!',           (0, 200, 255)),
    'stirring':         ('Stir in CIRCLES!',        (0, 255, 180)),
    'grab_pan':         ('GRAB the pan!',           (0, 200, 255)),
    'flipping':         ('Flick hand UP quickly!',  (50, 200, 255)),
    'complete':         ('ALL DONE!',               (0, 255, 100)),
}

# ── Intermission "title card" screens between major stages ───────────────────
_INTRO_DURATION = 2.4
_INTRO_TEXT = {
    'intro_prep':  ('Prep Time!',    'Wash and chop the ingredients'),
    'intro_fry':   ('Frying Time!',  'Saute the onions until golden'),
    'intro_meat':  ('Meat Time!',    'Grind, mix and shape the patty'),
    'intro_grill': ('Grill Time!',   'Cook the steak to perfection'),
}
_INTRO_NEXT = {
    'intro_prep':  'place_onion',
    'intro_fry':   'onion_fry',
    'intro_meat':  'meat_mix',
    'intro_grill': 'cook_steak',
}


class CookingScene(BaseMiniGame):
    name        = 'Cooking!'
    instruction = 'Grab tools and cook the meal!'
    duration    = 180.0
    grab_phase  = True

    def __init__(self):
        super().__init__()
        self._phase     = 'intro_prep'   # NEW entry point (title card before prep)
        self._inter_t0  = 0.0
        self._held_tool = None
        self._held_by   = -1
        self._reveal_t0 = 0.0

        self.chops = 0
        self.stirs = 0
        self.flips = 0

        # ── New Stage 1 state ─────────────────────────────────────────────
        self._onb_y       = float(_ONB_START_Y)  # onion y during place phase
        self._onb_grabbed = False               # True once player grips the onion
        self._peel_done   = 0
        self._peel_ref_x  = None
        self._peel_dir    = None     # 'left' | 'right' | None
        self._slice_done       = 0
        self._slice_ref_y      = None
        self._slice_dir        = None
        self._slice_knife_pos  = [_SLICE_KNIFE_X, _SLICE_KNIFE_Y]
        self._slice_knife_hand = -1
        self._dice_done        = 0
        self._dice_ref_y       = None
        self._dice_dir         = None
        self._dice_knife_pos   = [_DICE_KNIFE_X, _DICE_KNIFE_Y]
        self._dice_knife_hand  = -1
        # Bowl-drop piece physics
        self._bd_piece_pos  = []    # [[x,y], ...]
        self._bd_piece_vel  = []    # [[vx,vy], ...]
        self._bd_piece_ang  = []    # rotation angle per piece
        self._bd_piece_avel = []    # angular velocity per piece
        self._bd_in_bowl    = []    # bool per piece
        self._bd_prev_knife = None  # previous knife pos for velocity delta
        self._bowl_knife_pos  = [_PIECES_CX + 120, _PIECES_CY - 20]
        self._bowl_knife_hand = -1

        # chop detection: screen-coordinate based (kept for old stage, unused)
        self._chop_ref_y = None
        self._chop_dir   = None

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

        self._knife_spr   = get_knife(size=195)
        self._spatula_spr = get_spatula(size=120)
        self._bowl_spr    = get_bowl(w=260, h=130)

        self._veg_sprs  = [
            get_carrot_whole(size=S1_VEG_SIZE),
            get_cucumber_whole(size=S1_VEG_SIZE),
            get_pepper_whole(size=S1_VEG_SIZE),
        ]
        self._veg_half_l = [
            get_carrot_half_left(size=S1_VEG_SIZE),
            get_cucumber_half_left(size=S1_VEG_SIZE),
            get_pepper_half_left(size=S1_VEG_SIZE),
        ]
        self._veg_half_r = [
            get_carrot_half_right(size=S1_VEG_SIZE),
            get_cucumber_half_right(size=S1_VEG_SIZE),
            get_pepper_half_right(size=S1_VEG_SIZE),
        ]
        # Per-vegetable split-animation start times (None = not yet cut)
        self._split_ts       = [None, None, None]
        self._pepper_split_t = None   # kept for legacy reference in update()

        # Onion fry state
        self._onion_pos   = None   # (N,2) float
        self._onion_vel   = None
        self._onion_ang   = None
        self._onion_avel  = None
        self._onion_cook  = 0.0    # 0→1
        self._flame_on    = True
        self._prev_hand_onion = None
        self._fry_spatula_pos  = [ONION_PAN_CX + ONION_PAN_R + 60, ONION_PAN_CY]
        self._fry_spatula_hand = -1

        # Cook steak state
        self._sk_queue         = []
        self._sk_idx           = 0
        self._sk_cook          = 0.0
        self._sk_flip_side     = 0
        self._sk_pan_grabbed   = False
        self._sk_pan_pos       = [float(_SK_PAN_HX), float(_SK_PAN_HY)]
        self._sk_press_held    = 0
        self._sk_shake_ref     = None
        self._sk_shake_dir     = None
        self._sk_shake_done    = 0
        self._sk_flip_prev_y   = None
        self._sk_action_anim   = 0

        # Toss meat state
        self._toss_ball_x      = float(_TOSS_L_X)
        self._toss_ball_vx     = 0.0
        self._toss_moving      = False
        self._toss_count       = 0
        self._toss_flat        = 0.0
        self._toss_anim        = 0
        self._toss_catch_cd    = 0
        self._toss_prev_sx     = {}
        self._toss_cur_speed   = float(_TOSS_SPEED)
        self._toss_arrive_time = 0.0

        # Knead state
        self._knead_sub           = 'pour_onion'
        self._knead_onion_x       = float(_KN_ONION_X)
        self._knead_onion_grabbed = False
        self._knead_arrows        = []
        self._knead_idx           = 0
        self._knead_ref           = None
        self._knead_prev_ang      = None
        self._knead_rot_total     = 0.0
        self._knead_mix_angle     = 0.0
        self._knead_active        = False
        self._kn_meat_r = self._kn_meat_ang = self._kn_meat_w = None
        self._kn_meat_h = self._kn_meat_rot = self._kn_meat_col = None
        self._kn_onion_r = self._kn_onion_ang = self._kn_onion_w = None
        self._kn_onion_h = self._kn_onion_rot = self._kn_onion_col = None

        # Meat grinder state
        self._meat_sub_phase      = 'load_beef'
        self._meat_beef_y         = float(_GR_BEEF_Y0)
        self._meat_beef_grabbed   = False
        self._meat_handle_ang     = -math.pi / 2
        self._meat_handle_grabbed  = False
        self._meat_handle_no_grip  = 0
        self._meat_prev_ang        = None
        self._meat_total_ang      = 0.0
        self.meat_stirs           = 0
        self._meat_strand_count   = 0

        # Auto-initialize phase-specific state when starting mid-game
        _auto = {
            'bowl_drop':   self._init_bowl_pieces,
            'onion_fry':   self._init_onion,
            'meat_mix':    self._init_meat_mix,
            'knead':       self._init_knead,
            'toss_meat':   self._init_toss_meat,
            'cook_steak':  self._init_cook_steak,
        }
        if self._phase in _auto:
            _auto[self._phase]()

    # ── BaseMiniGame interface ────────────────────────────────────────────────

    @property
    def succeeded(self):
        return self._phase == 'complete'

    def check_done(self):
        if self._phase == 'reveal':
            return False
        return super().check_done()

    @property
    def required_hands(self):
        return 2 if self._phase == 'toss_meat' else 1

    # Debug shortcut: jump to any phase instantly
    # Number keys 1-9 jump straight to these phases (only 9 slots available,
    # so the near-identical onion prep sub-steps are trimmed down to make room
    # for meat_mix / knead / toss_meat).
    _DEBUG_PHASES = [
        'place_onion', 'dice_onion', 'bowl_drop', 'onion_fry',
        'meat_mix', 'knead', 'toss_meat',
        'cook_steak', 'reveal',
    ]

    def jump_to_phase(self, phase):
        self._phase = phase
        self._begin_timer()
        if phase == 'onion_fry':
            self._init_onion()
        elif phase == 'meat_mix':
            self._init_meat_mix()
        elif phase == 'knead':
            self._init_knead()
        elif phase == 'toss_meat':
            self._init_toss_meat()
        elif phase == 'bowl_drop':
            self._init_bowl_pieces()
        elif phase == 'cook_steak':
            self._init_cook_steak()
        elif phase == 'reveal':
            self._reveal_t0 = 0.0
        elif phase in _INTRO_NEXT:
            self._inter_t0 = 0.0

    @property
    def progress_text(self):
        p = self._phase
        if p == 'place_onion':  return 'Step 1  —  Place the onion!'
        if p == 'peel_skin':    return f'Step 1  —  Peel  {self._peel_done}/{_PEEL_NEEDED}'
        if p == 'slice_onion':  return f'Step 1  —  Slice  {self._slice_done}/{_SLICE_NEEDED}'
        if p == 'dice_onion':   return f'Step 1  —  Dice  {self._dice_done}/{_DICE_NEEDED}'
        if p == 'bowl_drop':
            n_in = sum(self._bd_in_bowl) if self._bd_in_bowl else 0
            return f'Step 1  —  Bowl  {n_in}/{_BOWL_PIECE_GOAL}'
        # if p == 'grab_knife': return 'Step 1/3  —  Grab the knife!'    # old
        # if p == 'chopping':   return ...                                # old
        if p == 'pepper_splitting': return 'Step 1/3  —  Nice cut!'
        if p == 'onion_fry':
            if self._onion_cook < ONION_READY:
                return 'Frying onions...  Stir to cook!'
            return 'Grab the flame button!'
        if p == 'meat_mix':         return f'Mix meat  {self.meat_stirs}/{MEAT_MIX_TARGET}'
        if p == 'knead':
            sub = self._knead_sub
            if sub == 'pour_onion': return 'Pour  in  the  onions!'
            if sub == 'arrows':     return f'Knead  {self._knead_idx}/{_KN_ARROW_N}'
            return 'One  full  turn!'
        if p == 'toss_meat':        return f'Toss  {self._toss_count}/{_TOSS_TARGET}'
        if p == 'cook_steak':
            if self._sk_idx < _SK_ACTION_N:
                act = self._sk_queue[self._sk_idx]
                labels = {'shake':'Shake Pan','press':'Press Steak','flip':'Flip Steak','flame':'Flame Button'}
                return f'{labels.get(act,act)}  {self._sk_idx+1}/{_SK_ACTION_N}'
            return 'Done!'
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

        audio.loop_cooking(p in ('cook_steak', 'onion_fry'))
        audio.loop_mixing(p == 'knead' and self._knead_active)
        audio.loop_grinding(p == 'meat_mix' and self._meat_handle_grabbed)
        audio.loop_knife((p == 'dice_onion'  and self._dice_knife_hand  != -1) or
                         (p == 'slice_onion' and self._slice_knife_hand != -1) or
                         (p == 'bowl_drop'   and self._bowl_knife_hand  != -1))

        if p in _INTRO_NEXT:
            if self._inter_t0 == 0.0:
                self._inter_t0 = time.time()
            elif time.time() - self._inter_t0 > _INTRO_DURATION:
                nxt = _INTRO_NEXT[p]
                self._inter_t0 = 0.0
                self._phase    = nxt
                if nxt == 'onion_fry':
                    self._init_onion()
                elif nxt == 'meat_mix':
                    self._init_meat_mix()
                elif nxt == 'cook_steak':
                    self._init_cook_steak()
            return events

        # ── New Stage 1 ───────────────────────────────────────────────────────
        if p == 'place_onion':
            events += self._do_place_onion(hands)

        elif p == 'peel_skin':
            events += self._do_peel(hands)

        elif p == 'slice_onion':
            events += self._do_slice_onion(hands)

        elif p == 'dice_onion':
            events += self._do_dice_onion(hands)

        elif p == 'bowl_drop':
            events += self._do_bowl_drop(hands)

        # ── Old Stage 1 (commented out) ───────────────────────────────────────
        # elif p == 'grab_knife':
        #     events += self._try_grab(hands, self._knife_pos, 'knife', 'chopping')
        # elif p == 'chopping':
        #     ...  (old chop logic)
        # elif p == 'pepper_splitting':
        #     ...  (old split logic)

        elif p == 'onion_fry':
            self._update_onion_fry(hands)

        elif p == 'meat_mix':
            events += self._do_meat_mix(hands)

        elif p == 'knead':
            events += self._do_knead(hands)

        elif p == 'toss_meat':
            events += self._do_toss_meat(hands)

        elif p == 'cook_steak':
            self._do_cook_steak(hands)

        elif p == 'reveal':
            if self._reveal_t0 == 0.0:
                self._reveal_t0 = time.time()
            elif time.time() - self._reveal_t0 > 5.0:
                self._phase    = 'complete'
                self._complete = True

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
        if label == 'GRAB!':
            audio.play_grab()
        elif label == 'DONE!':
            audio.play_stage_clear()
        elif label in ('PEEL!', 'SLICE!', 'DICE!'):
            pass   # 칼질 동안에는 칼질효과음(파일) 루프가 재생되므로 합성음은 생략
        else:
            audio.play_success()

    # ── Drawing: one stage at a time ─────────────────────────────────────────

    def draw(self, frame, hands):
        h, w = frame.shape[:2]
        _draw_counter(frame, w, h)

        p = self._phase
        if p in _INTRO_NEXT:
            self._draw_intermission(frame)
            return frame
        elif p in ('place_onion', 'peel_skin', 'slice_onion', 'dice_onion', 'bowl_drop'):
            self._draw_onion_cut(frame)
        # elif p in ('grab_knife', 'chopping', 'pepper_splitting'):  # old stage 1
        #     self._draw_stage1(frame)
        elif p == 'onion_fry':
            self._draw_onion_fry(frame)
        elif p == 'meat_mix':
            self._draw_meat_mix(frame)
        elif p == 'knead':
            self._draw_knead(frame)
        elif p == 'toss_meat':
            self._draw_toss_meat(frame, hands)
        elif p == 'cook_steak':
            self._draw_cook_steak(frame)
        elif p == 'reveal':
            self._draw_reveal(frame)
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
                offset = int(t * 80)
                overlay(frame, self._veg_half_l[vi], vx - offset, S1_VEG_Y, size=S1_VEG_SIZE)
                overlay(frame, self._veg_half_r[vi], vx + offset, S1_VEG_Y, size=S1_VEG_SIZE)
                continue

            overlay(frame, spr, vx, S1_VEG_Y, size=S1_VEG_SIZE)

            # Vertical cut marks before the final split
            n_cuts = min(
                max(self.chops - S1_VEG_CHOP_OFFSET[vi], 0),
                S1_VEG_CHOP_MAX[vi] - 1
            )
            half = S1_VEG_SIZE // 2 - 10   # line length matches sprite radius
            for c in range(n_cuts):
                offset_x = int((c - (n_cuts - 1) / 2.0) * 42)
                cut_x    = vx + offset_x
                cv2.line(frame, (cut_x, S1_VEG_Y - half), (cut_x, S1_VEG_Y + half),
                         (210, 235, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, (cut_x, S1_VEG_Y - half), 3, (190, 220, 255), -1)
                cv2.circle(frame, (cut_x, S1_VEG_Y + half), 3, (190, 220, 255), -1)

            # Pulsing target ring on the current active vegetable during chopping
            if self._phase == 'chopping':
                if self.chops < S1_VEG_CHOP_OFFSET[1]:   tvi = 0
                elif self.chops < S1_VEG_CHOP_OFFSET[2]: tvi = 1
                else:                                      tvi = 2
                if vi == tvi:
                    _grab_ring(frame, [vx, S1_VEG_Y], label='CHOP')

        overlay(frame, self._knife_spr,
                self._knife_pos[0], self._knife_pos[1], size=195)
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

        if p == 'place_onion':
            # Hand pulls down from above
            cy_d = int(_ONB_START_Y + 200 * ((math.sin(t * 1.4) + 1) / 2))
            _demo_fist(frame, 640, cy_d, openness=0.3)
            cv2.arrowedLine(frame, (640, cy_d - 70), (640, cy_d + 60),
                            (0, 220, 255), 3, tipLength=0.3)
            return
        elif p == 'peel_skin':
            # Hand swipes left-right
            ox = int(120 * math.sin(t * 2.2))
            _demo_fist(frame, 640 + ox, _ONB_BOARD_Y - 20, openness=0.0)
            return
        elif p == 'slice_onion':
            if self._slice_knife_hand == -1:
                # Demo: approach the knife
                wave = (math.sin(t * 1.8) + 1) / 2
                off  = int(60 * wave)
                grip = 1.0 - wave
                _demo_fist(frame,
                           self._slice_knife_pos[0] - 100 + off,
                           self._slice_knife_pos[1] - 20,
                           openness=grip)
            else:
                # Demo: chop up-down with knife over onion
                oy = int(70 * math.sin(t * 3.0))
                _demo_fist(frame, 640 - 210, _ONB_BOARD_Y + oy, openness=0.0)
                cv2.arrowedLine(frame, (640 - 210, _ONB_BOARD_Y + oy - 50),
                                (640 - 210, _ONB_BOARD_Y + oy + 50),
                                (0, 230, 180), 2, tipLength=0.4)
            return
        elif p == 'bowl_drop':
            if self._bowl_knife_hand == -1:
                # Demo: approach the knife near pieces
                wave = (math.sin(t * 1.8) + 1) / 2
                off  = int(60 * wave)
                _demo_fist(frame,
                           self._bowl_knife_pos[0] - 90 + off,
                           self._bowl_knife_pos[1] - 15,
                           openness=1.0 - wave)
            else:
                # Demo: sweep from right (board/pieces) toward left (bowl)
                wave = (math.sin(t * 1.2) + 1) / 2
                dx = int(_PIECES_CX - (_PIECES_CX - _BOWL2_CX) * wave)
                _demo_fist(frame, dx, _PIECES_CY, openness=0.0)
                cv2.arrowedLine(frame, (dx, _PIECES_CY - 10),
                                (dx - 80, _PIECES_CY - 10),
                                (255, 200, 50), 2, tipLength=0.35)
            return

        if p in ('pepper_splitting', 'onion_fry'):
            if p == 'onion_fry':
                if self._fry_spatula_hand == -1:
                    # Show: approach and grab spatula
                    wave = (math.sin(t * 1.8) + 1) / 2
                    off  = int(60 * wave)
                    _demo_fist(frame,
                               self._fry_spatula_pos[0] - 100 + off,
                               self._fry_spatula_pos[1] - 20,
                               openness=1.0 - wave)
                elif self._onion_cook < ONION_READY:
                    # Spatula held — stir in circles
                    r  = 145
                    dx = int(ONION_PAN_CX + r * math.cos(t * 1.6))
                    dy = int(ONION_PAN_CY + r * math.sin(t * 1.6))
                    _demo_fist(frame, dx, dy, openness=0.0)
                # When ONION_READY reached, regardless of spatula state → show flame btn
                if self._onion_cook >= ONION_READY:
                    wave = (math.sin(t * 1.8) + 1) / 2
                    off  = int(70 * wave)
                    _demo_fist(frame, _FLAME_BTN_CX + 120 - off, _FLAME_BTN_CY,
                               openness=1.0 - wave)
            return

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

        elif p == 'meat_mix':
            if self._meat_sub_phase == 'load_beef':
                # Show hand grabbing beef and moving down
                wave = (math.sin(t * 1.6) + 1) / 2
                by   = int(_GR_BEEF_Y0 + (_GR_BEEF_DROP_Y - _GR_BEEF_Y0) * wave)
                _demo_fist(frame, _GR_HOPPER_CX - 80, by, openness=1.0 - wave)
                cv2.arrowedLine(frame, (_GR_HOPPER_CX - 80, by - 40),
                                (_GR_HOPPER_CX - 80, by + 40),
                                (80, 200, 130), 2, tipLength=0.4)
            else:
                # Show hand rotating around handle
                r  = _GR_HANDLE_ARM + 10
                dx = int(_GR_HANDLE_CX + r * math.cos(t * 2.5))
                dy = int(_GR_HANDLE_CY + r * math.sin(t * 2.5))
                _demo_fist(frame, dx, dy, openness=0.0)

        elif p == 'knead':
            sub = self._knead_sub
            if sub == 'pour_onion':
                wave = (math.sin(t * 1.5) + 1) / 2
                ox   = int(_KN_ONION_X - (_KN_ONION_X - _KN_BOWL_CX - 80) * wave)
                _demo_fist(frame, ox - 70, _KN_ONION_Y, openness=1.0 - wave)
            elif sub == 'arrows' and self._knead_idx < _KN_ARROW_N:
                direction = self._knead_arrows[self._knead_idx]
                wave = math.sin(t * 2.5)
                off  = int(60 * wave)
                ddx  = off if direction == 'right' else (-off if direction == 'left' else 0)
                ddy  = off if direction == 'down'  else (-off if direction == 'up'   else 0)
                _demo_fist(frame, _KN_BOWL_CX + ddx, _KN_BOWL_CY + ddy, openness=0.0)
            elif sub == 'rotate':
                r  = int(_KN_BOWL_RX * 0.55)
                dx = int(_KN_BOWL_CX + r * math.cos(t * 2.2))
                dy = int(_KN_BOWL_CY + r * math.sin(t * 2.2))
                _demo_fist(frame, dx, dy, openness=0.0)

        elif p == 'toss_meat':
            # Show both hands: one on each side, swipe at the right time
            going_right = self._toss_ball_vx > 0
            at_right    = self._toss_ball_x >= _TOSS_R_X - 90
            at_left     = self._toss_ball_x <= _TOSS_L_X + 90

            # Left demo hand
            if at_left:
                # Ball arrived: swipe RIGHT (open hand)
                wave = math.sin(t * 8)
                off  = int(50 * wave)
                _demo_fist(frame, _TOSS_L_X + off, _TOSS_Y, openness=1.0)
            else:
                # Waiting: open hand hover
                _demo_fist(frame, _TOSS_L_X, _TOSS_Y + int(12 * math.sin(t * 2)), openness=1.0)

            # Right demo hand
            if at_right:
                # Ball arrived: swipe LEFT (open hand)
                wave = math.sin(t * 8)
                off  = int(50 * wave)
                _demo_fist(frame, _TOSS_R_X - off, _TOSS_Y, openness=1.0)
            else:
                _demo_fist(frame, _TOSS_R_X, _TOSS_Y + int(12 * math.sin(t * 2 + 1)), openness=1.0)

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

    # ── Onion Fry methods ────────────────────────────────────────────────────

    def _init_onion(self):
        rng    = np.random.default_rng(seed=42)
        radii  = np.sqrt(rng.uniform(0.05, 0.82, ONION_COUNT)) * ONION_AREA_R
        angles = rng.uniform(0, 2 * np.pi, ONION_COUNT)
        self._onion_pos  = np.stack([
            ONION_PAN_CX + np.cos(angles) * radii,
            ONION_PAN_CY + np.sin(angles) * radii,
        ], axis=1).astype(float)
        self._onion_vel  = np.zeros((ONION_COUNT, 2))
        self._onion_ang  = rng.uniform(0, 2 * np.pi, ONION_COUNT)
        self._onion_avel = np.zeros(ONION_COUNT)
        self._onion_cook  = 0.0
        self._flame_on    = True
        self._prev_hand_onion = None
        self._fry_spatula_pos  = [ONION_PAN_CX + ONION_PAN_R + 60, ONION_PAN_CY]
        self._fry_spatula_hand = -1

    def _update_onion_fry(self, hands):
        _SPATULA_GRAB_R = 70

        has_spatula = self._fry_spatula_hand != -1

        for i, hand in enumerate(hands):
            if not hand.detected:
                if self._fry_spatula_hand == i:
                    self._fry_spatula_hand = -1
                    has_spatula = False
                continue

            hx = float(hand.screen_x)
            hy = float(hand.screen_y)

            # Flame button: grip over it when ready (spatula not required)
            if hand.gripped and self._onion_cook >= ONION_READY:
                bdx = hx - _FLAME_BTN_CX
                bdy = hy - _FLAME_BTN_CY
                if bdx * bdx + bdy * bdy < (_FLAME_BTN_R + 22) ** 2:
                    self._flame_on = False
                    self._phase    = 'intro_meat'
                    self._inter_t0 = 0.0
                    return

            # Pick up spatula
            if not has_spatula:
                if hand.gripped:
                    sx, sy = self._fry_spatula_pos
                    if (hx - sx)**2 + (hy - sy)**2 < _SPATULA_GRAB_R**2:
                        self._fry_spatula_hand = i
                        has_spatula = True
                continue

            if self._fry_spatula_hand != i:
                continue

            # Drop spatula when not gripped
            if not hand.gripped:
                self._fry_spatula_hand = -1
                has_spatula = False
                self._prev_hand_onion = None
                continue

            # Spatula follows hand
            self._fry_spatula_pos[0] = hx
            self._fry_spatula_pos[1] = hy

            # Stir: impulse based on spatula movement
            if self._prev_hand_onion is not None:
                dhx = (hx - self._prev_hand_onion[0]) * 0.55
                dhy = (hy - self._prev_hand_onion[1]) * 0.55
            else:
                dhx = dhy = 0.0
            self._prev_hand_onion = (hx, hy)

            if abs(dhx) + abs(dhy) > 0.8:
                sx_pos = float(self._fry_spatula_pos[0])
                sy_pos = float(self._fry_spatula_pos[1])
                dx   = self._onion_pos[:, 0] - sx_pos
                dy   = self._onion_pos[:, 1] - sy_pos
                dist = np.sqrt(dx * dx + dy * dy) + 1e-6
                mask = dist < ONION_INFL_R
                if mask.any():
                    fall = 1.0 - dist[mask] / ONION_INFL_R
                    self._onion_vel[mask, 0] += dhx * fall
                    self._onion_vel[mask, 1] += dhy * fall
                    self._onion_avel[mask]   += np.random.uniform(-0.18, 0.18, mask.sum())
                    stir_amount = (abs(dhx) + abs(dhy)) * mask.sum() / ONION_COUNT
                    self._onion_cook = min(self._onion_cook + stir_amount / ONION_STIR_NEEDED, 1.0)

        if not has_spatula:
            self._prev_hand_onion = None

        # Physics: friction + integrate
        self._onion_vel  *= 0.88
        self._onion_avel *= 0.85
        self._onion_pos  += self._onion_vel
        self._onion_ang  += self._onion_avel

        # Boundary: reflect at pan inner wall
        dx   = self._onion_pos[:, 0] - ONION_PAN_CX
        dy   = self._onion_pos[:, 1] - ONION_PAN_CY
        dist = np.sqrt(dx * dx + dy * dy) + 1e-6
        out  = dist > ONION_AREA_R * 0.88
        if out.any():
            nx  = dx[out] / dist[out]
            ny  = dy[out] / dist[out]
            dot = self._onion_vel[out, 0] * nx + self._onion_vel[out, 1] * ny
            self._onion_vel[out, 0] -= dot * nx * 1.5
            self._onion_vel[out, 1] -= dot * ny * 1.5
            self._onion_pos[out, 0]  = ONION_PAN_CX + nx * ONION_AREA_R * 0.84
            self._onion_pos[out, 1]  = ONION_PAN_CY + ny * ONION_AREA_R * 0.84

    def _draw_onion_fry(self, frame):
        t    = time.time()
        cx   = ONION_PAN_CX
        cy   = ONION_PAN_CY
        cook = self._onion_cook

        # ── Stove base ───────────────────────────────────────────────────────
        cv2.ellipse(frame, (cx, cy + ONION_PAN_R + 22),
                    (ONION_PAN_R + 50, 42), 0, 0, 360, (52, 48, 44), -1)

        # ── Pan outer rim ────────────────────────────────────────────────────
        cv2.circle(frame, (cx, cy), ONION_PAN_R, (40, 36, 32), -1)
        cv2.circle(frame, (cx, cy), ONION_PAN_R, (18, 16, 14), 10)

        # Cooking surface — warms slightly as cook progresses
        sv = int(162 - cook * 28)
        cv2.circle(frame, (cx, cy), ONION_AREA_R + 10, (sv, sv + 2, sv + 4), -1)
        cv2.circle(frame, (cx, cy), ONION_AREA_R + 10, (72, 68, 64), 2)

        # ── Pan handle ───────────────────────────────────────────────────────
        hx1 = cx + ONION_PAN_R - 14
        hx2 = cx + ONION_PAN_R + 145
        cv2.rectangle(frame, (hx1, cy - 26), (hx2, cy + 26), (36, 32, 28), -1)
        cv2.rectangle(frame, (hx1, cy - 26), (hx2, cy + 26), (18, 15, 12),  3)

        # ── Flames ───────────────────────────────────────────────────────────
        if self._flame_on:
            flame_span = int(ONION_PAN_R * 1.6)
            for fi in range(8):
                fx   = cx - flame_span // 2 + fi * (flame_span // 7)
                flk  = int(14 * math.sin(t * 9.5 + fi * 1.3))
                fy0  = cy + ONION_PAN_R + 6
                # Blue core
                cv2.ellipse(frame, (fx, fy0),
                            (7, 16 + flk // 2), 0, 0, 360, (190, 95, 18), -1)
                # Orange-yellow outer flame
                pts = np.array([[fx - 11, fy0],
                                [fx,      fy0 - 44 - flk],
                                [fx + 11, fy0]], dtype=np.int32)
                cv2.fillPoly(frame, [pts], (0, int(148 + flk * 3), 255))

        # ── Onion pieces ─────────────────────────────────────────────────────
        for i in range(ONION_COUNT):
            px  = int(self._onion_pos[i, 0])
            py  = int(self._onion_pos[i, 1])
            ang = float(self._onion_ang[i])

            # BGR colour: cream → light golden → golden brown
            if cook < 0.5:
                tc = cook / 0.5
                col = (int(228 - tc * 158), int(248 - tc * 65), int(255 - tc * 52))
            else:
                tc = (cook - 0.5) / 0.5
                col = (int(70  - tc * 48), int(183 - tc * 135), int(203 - tc * 118))

            pw, ph = 17, 12
            ca, sa = math.cos(ang), math.sin(ang)
            corners = np.array([
                [px + int(-pw*ca + ph*sa), py + int(-pw*sa - ph*ca)],
                [px + int( pw*ca + ph*sa), py + int( pw*sa - ph*ca)],
                [px + int( pw*ca - ph*sa), py + int( pw*sa + ph*ca)],
                [px + int(-pw*ca - ph*sa), py + int(-pw*sa + ph*ca)],
            ], dtype=np.int32)

            cv2.fillPoly(frame, [corners + 3], (20, 18, 16))   # shadow
            cv2.fillPoly(frame, [corners], col)
            dark = tuple(max(0, c - 42) for c in col)
            cv2.polylines(frame, [corners], True, dark, 1)

        # ── Flame button ─────────────────────────────────────────────────────
        ready  = cook >= ONION_READY
        pulse  = int(9 * abs(math.sin(t * 5))) if ready else 0
        bcol   = (0, 185, 255) if ready else (55, 55, 72)
        cv2.circle(frame, (_FLAME_BTN_CX, _FLAME_BTN_CY),
                   _FLAME_BTN_R + pulse, bcol, -1)
        cv2.circle(frame, (_FLAME_BTN_CX, _FLAME_BTN_CY),
                   _FLAME_BTN_R + pulse, (255, 255, 255), 2)

        # Flame icon (simple teardrop polygon)
        fpts = np.array([
            [_FLAME_BTN_CX,      _FLAME_BTN_CY - 30],
            [_FLAME_BTN_CX - 16, _FLAME_BTN_CY +  8],
            [_FLAME_BTN_CX - 8,  _FLAME_BTN_CY + 20],
            [_FLAME_BTN_CX + 8,  _FLAME_BTN_CY + 20],
            [_FLAME_BTN_CX + 16, _FLAME_BTN_CY +  8],
        ], dtype=np.int32)
        icol = (255, 255, 255) if ready else (110, 110, 130)
        cv2.fillPoly(frame, [fpts], icol)

        if not ready:
            pct_str = f'{int(cook / ONION_READY * 100)}%'
            _shadow_text(frame, pct_str,
                         ONION_PAN_CX, ONION_PAN_CY,
                         2.8, (255, 230, 80), 4, center=True)
        else:
            # Pulsing grab ring on flame button
            ring_r = int(_FLAME_BTN_R + 20 + 10 * abs(math.sin(t * 5)))
            cv2.circle(frame, (_FLAME_BTN_CX, _FLAME_BTN_CY),
                       ring_r, (0, 200, 255), 3, cv2.LINE_AA)
            _shadow_text(frame, 'GRAB!', _FLAME_BTN_CX,
                         _FLAME_BTN_CY - ring_r - 12,
                         0.65, (0, 200, 255), 1, center=True)

        # ── Spatula ───────────────────────────────────────────────────────────
        sx = int(self._fry_spatula_pos[0])
        sy = int(self._fry_spatula_pos[1])
        _draw_spatula(frame, sx, sy)
        if self._fry_spatula_hand == -1:
            _grab_ring(frame, self._fry_spatula_pos)

    # ── Meat Grinder ─────────────────────────────────────────────────────────

    def _init_meat_mix(self):
        self._meat_sub_phase      = 'load_beef'
        self._meat_beef_y         = float(_GR_BEEF_Y0)
        self._meat_beef_grabbed   = False
        self._meat_handle_ang     = -math.pi / 2
        self._meat_handle_grabbed  = False
        self._meat_handle_no_grip  = 0
        self._meat_prev_ang        = None
        self._meat_total_ang      = 0.0
        self.meat_stirs           = 0
        self._meat_strand_count   = 0

    def _do_meat_mix(self, hands):
        if self._meat_sub_phase == 'load_beef':
            return self._do_meat_load(hands)
        return self._do_meat_grind(hands)

    def _do_meat_load(self, hands):
        for hand in hands:
            if not hand.detected or not hand.gripped:
                if self._meat_beef_grabbed:
                    self._meat_beef_grabbed = False
                continue
            hx, hy = hand.screen_x, hand.screen_y
            if not self._meat_beef_grabbed:
                bx = _GR_HOPPER_CX
                by = int(self._meat_beef_y)
                if (hx - bx)**2 + (hy - by)**2 < _GR_GRAB_R**2:
                    self._meat_beef_grabbed = True
                continue
            # Beef follows hand vertically, clamped to hopper entry
            self._meat_beef_y = float(min(hy, float(_GR_BEEF_DROP_Y)))
            if self._meat_beef_y >= _GR_BEEF_DROP_Y:
                self._meat_beef_grabbed = False
                self._meat_sub_phase    = 'grind'
                self._flash_event('GRIND!', (80, 200, 130), 16)
        return []

    def _do_meat_grind(self, hands):
        _RELEASE_FRAMES = 10   # frames of no-grip before handle drops
        has_grip = any(h.detected and h.gripped for h in hands)
        if self._meat_handle_grabbed:
            if not has_grip:
                self._meat_handle_no_grip += 1
                if self._meat_handle_no_grip >= _RELEASE_FRAMES:
                    self._meat_handle_grabbed = False
                    self._meat_handle_no_grip = 0
                    self._meat_prev_ang = None
            else:
                self._meat_handle_no_grip = 0
        for hand in hands:
            if not hand.detected or not hand.gripped:
                continue
            hx = float(hand.screen_x)
            hy = float(hand.screen_y)
            if not self._meat_handle_grabbed:
                tip_x = int(_GR_HANDLE_CX + math.cos(self._meat_handle_ang) * _GR_HANDLE_ARM)
                tip_y = int(_GR_HANDLE_CY + math.sin(self._meat_handle_ang) * _GR_HANDLE_ARM)
                if (hx - tip_x)**2 + (hy - tip_y)**2 < _GR_GRAB_R**2:
                    self._meat_handle_grabbed = True
                continue
            dx = hx - _GR_HANDLE_CX
            dy = hy - _GR_HANDLE_CY
            if math.sqrt(dx*dx + dy*dy) < 28:
                continue
            angle = math.atan2(dy, dx)
            self._meat_handle_ang = angle
            if self._meat_prev_ang is not None:
                delta = angle - self._meat_prev_ang
                if delta >  math.pi: delta -= 2 * math.pi
                if delta < -math.pi: delta += 2 * math.pi
                self._meat_total_ang += delta
                new_stirs = int(abs(self._meat_total_ang) / (2 * math.pi))
                if new_stirs > self.meat_stirs:
                    self.meat_stirs = new_stirs
                    self._meat_strand_count = self.meat_stirs * 14
                    self._flash_event('GRIND!', (80, 180, 240), 14)
                    if self.meat_stirs >= MEAT_MIX_TARGET:
                        self._meat_prev_ang = None
                        self._phase = 'knead'
                        self._init_knead()
                        return ['mix']
            self._meat_prev_ang = angle
        return []

    def _draw_meat_mix(self, frame):
        t = time.time()
        gx1 = _GR_CX - _GR_W // 2
        gx2 = _GR_CX + _GR_W // 2
        gy1 = _GR_CY - _GR_H // 2
        gy2 = _GR_CY + _GR_H // 2

        # ── Grinder body ─────────────────────────────────────────────────────
        # Shadow
        cv2.rectangle(frame, (gx1 + 9, gy1 + 9), (gx2 + 9, gy2 + 9), (25, 22, 20), -1)
        # Main face
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (82, 78, 74), -1)
        # Top bevel face
        top_pts = np.array([(gx1, gy1), (gx2, gy1),
                             (gx2 - 18, gy1 - 16), (gx1 - 18, gy1 - 16)], np.int32)
        cv2.fillPoly(frame, [top_pts], (118, 114, 110))
        # Right bevel face
        rgt_pts = np.array([(gx2, gy1), (gx2, gy2),
                             (gx2 - 18, gy2 - 16), (gx2 - 18, gy1 - 16)], np.int32)
        cv2.fillPoly(frame, [rgt_pts], (60, 57, 53))
        # Edge highlight
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (145, 140, 135), 3)
        # Control panel buttons
        px1, py1 = gx2 - 72, gy1 + 28
        cv2.rectangle(frame, (px1, py1), (px1 + 48, py1 + 92), (48, 45, 42), -1)
        cv2.rectangle(frame, (px1 + 7, py1 + 10), (px1 + 39, py1 + 32), (40, 200, 100), -1)
        cv2.rectangle(frame, (px1 + 7, py1 + 44), (px1 + 39, py1 + 66), (70, 70, 80), -1)

        # ── Hopper (top tray) ────────────────────────────────────────────────
        hpx1 = _GR_HOPPER_CX - 115
        hpx2 = _GR_HOPPER_CX + 115
        hpy1 = _GR_HOPPER_Y - 44
        hpy2 = _GR_HOPPER_Y + 8
        cv2.rectangle(frame, (hpx1, hpy1), (hpx2, hpy2), (182, 188, 194), -1)
        cv2.rectangle(frame, (hpx1, hpy1), (hpx2, hpy2), (220, 226, 232), 3)
        # Funnel from hopper to body
        fn_pts = np.array([(hpx1 + 28, hpy2), (hpx2 - 28, hpy2),
                            (gx1 + 138, gy1), (gx1 + 75, gy1)], np.int32)
        cv2.fillPoly(frame, [fn_pts], (155, 162, 168))
        cv2.polylines(frame, [fn_pts], True, (195, 200, 208), 2)

        # ── Raw beef in hopper (load_beef sub-phase) ─────────────────────────
        if self._meat_sub_phase == 'load_beef':
            bx = _GR_HOPPER_CX
            by = int(self._meat_beef_y)
            beef_pts = np.array([
                (bx - 52, by - 28), (bx + 52, by - 28),
                (bx + 62, by + 4),  (bx + 44, by + 28),
                (bx - 44, by + 28), (bx - 62, by + 4),
            ], np.int32)
            cv2.fillPoly(frame, [beef_pts + np.array([[3, 4]])], (35, 42, 105))
            cv2.fillPoly(frame, [beef_pts], (68, 82, 188))
            # Marbling streaks
            cv2.ellipse(frame, (bx - 18, by - 6), (22, 8), 15, 0, 360, (115, 128, 220), -1)
            cv2.ellipse(frame, (bx + 22, by + 8), (16, 6), -20, 0, 360, (115, 128, 220), -1)
            cv2.polylines(frame, [beef_pts], True, (48, 58, 148), 2)
            if not self._meat_beef_grabbed:
                _grab_ring(frame, [bx, by])

        # ── Output nozzle (left side) ─────────────────────────────────────────
        nx, ny = _GR_NOZZLE_X, _GR_NOZZLE_Y
        # Cylinder tube
        cv2.rectangle(frame, (nx - 8, ny - 16), (nx + 28, ny + 16), (145, 150, 155), -1)
        cv2.rectangle(frame, (nx - 8, ny - 16), (nx + 28, ny + 16), (185, 190, 195), 2)
        # Die plate (disc with holes)
        cv2.circle(frame, (nx + 28, ny), 34, (170, 176, 182), -1)
        cv2.circle(frame, (nx + 28, ny), 34, (210, 216, 222), 3)
        for hoff in [(-10,-10),(0,-14),(10,-10),(-14,0),(0,0),(14,0),(-10,10),(0,14),(10,10)]:
            cv2.circle(frame, (nx + 28 + hoff[0], ny + hoff[1]), 4, (48, 45, 42), -1)

        # ── Ground meat flow (3~4 thick falling blobs) ───────────────────────
        n_strands = self._meat_strand_count
        if n_strands > 0:
            # 3 thick blob-streams falling from nozzle into bowl
            stream_offsets = [-10, 0, 10]
            for si, ox in enumerate(stream_offsets):
                x0  = nx + ox
                y0  = ny + 14
                y1  = min(_GR_BOWL_CY + 10, y0 + int((n_strands / 70) * (_GR_BOWL_CY - y0 + 10)))
                wave_phase = si * 1.2 - t * 3.0
                seg_count  = max(2, (y1 - y0) // 14)
                pts_list   = []
                for seg in range(seg_count):
                    ys = y0 + seg * ((y1 - y0) // max(seg_count - 1, 1))
                    xs = x0 + int(8 * math.sin(wave_phase + seg * 1.0))
                    pts_list.append((xs, ys))
                if len(pts_list) >= 2:
                    col = (72 + si*8, 88 + si*6, 185 + si*8)
                    cv2.polylines(frame, [np.array(pts_list, np.int32)],
                                  False, col, 10)

        # ── Collection bowl (~2x larger; flat fill so contents stay fully visible)
        bcx, bcy = _GR_BOWL_CX, _GR_BOWL_CY
        brx_o, bry_o = 200, 100
        cv2.ellipse(frame, (bcx + 14, bcy + 36), (brx_o, 46), 0, 0, 360, (22, 20, 18), -1)
        # Single flat fill — nothing drawn over the contents afterward
        cv2.ellipse(frame, (bcx, bcy), (brx_o, bry_o), 0, 0, 360, (208, 214, 222), -1)
        # 3~4 large ellipses filling bowl as meat accumulates, kept inside the rim
        if n_strands > 0:
            rng_b = np.random.default_rng(seed=77)
            n_blobs = min(4, 1 + n_strands // 18)
            for bi in range(n_blobs):
                boff_x = int(rng_b.integers(-90, 91))
                boff_y = int(rng_b.integers(-27, 28))
                brx    = int(rng_b.integers(49, 96))
                bry    = int(rng_b.integers(25, 41))
                bang   = int(rng_b.integers(0, 180))
                bc     = (int(rng_b.integers(65, 90)),
                          int(rng_b.integers(85, 110)),
                          int(rng_b.integers(175, 210)))
                cv2.ellipse(frame, (bcx + boff_x, bcy + boff_y),
                            (brx, bry), bang, 0, 360, bc, -1)
        # Rim outline drawn last as a thin stroke — frames the bowl without hiding contents
        cv2.ellipse(frame, (bcx, bcy), (brx_o, bry_o), 0, 0, 360, (235, 240, 248), 5)

        # ── Crank handle (grind sub-phase) ────────────────────────────────────
        if self._meat_sub_phase == 'grind':
            arm_ang = self._meat_handle_ang
            tip_x = int(_GR_HANDLE_CX + math.cos(arm_ang) * _GR_HANDLE_ARM)
            tip_y = int(_GR_HANDLE_CY + math.sin(arm_ang) * _GR_HANDLE_ARM)
            # Axle
            cv2.circle(frame, (_GR_HANDLE_CX, _GR_HANDLE_CY), 16, (150, 155, 160), -1)
            cv2.circle(frame, (_GR_HANDLE_CX, _GR_HANDLE_CY), 16, (205, 210, 215), 3)
            # Arm bar
            cv2.line(frame, (_GR_HANDLE_CX, _GR_HANDLE_CY), (tip_x, tip_y), (148, 153, 158), 11)
            cv2.line(frame, (_GR_HANDLE_CX, _GR_HANDLE_CY), (tip_x, tip_y), (195, 200, 206), 5)
            # Grip knob (wood-brown)
            cv2.circle(frame, (tip_x, tip_y), 17, (55, 95, 140), -1)
            cv2.circle(frame, (tip_x, tip_y), 17, (80, 120, 165), 3)
            if not self._meat_handle_grabbed:
                _grab_ring(frame, [tip_x, tip_y])
            # Rotation counter
            _shadow_text(frame, f'{self.meat_stirs}/{MEAT_MIX_TARGET}',
                         _GR_HANDLE_CX, _GR_HANDLE_CY + _GR_HANDLE_ARM + 50,
                         1.0, (255, 230, 100), 2, center=True)

    # ── Toss Meat ────────────────────────────────────────────────────────────

    def _init_toss_meat(self):
        self._toss_started     = False
        self._toss_ready_hold  = 0
        self._toss_ball_x      = float(_TOSS_L_X)
        self._toss_ball_vx     = 0.0
        self._toss_moving      = False
        self._toss_count       = 0
        self._toss_flat        = 0.0
        self._toss_anim        = 0
        self._toss_catch_cd    = 0
        self._toss_prev_sx     = {}
        self._toss_cur_speed   = float(_TOSS_SPEED)  # accumulated speed
        self._toss_arrive_time = time.time()          # when ball last stopped

    def _do_toss_meat(self, hands):
        # Wait for both hands to be visible before the throwing actually starts
        if not self._toss_started:
            if sum(1 for h in hands if h.detected) >= 2:
                self._toss_ready_hold += 1
                if self._toss_ready_hold >= _TOSS_READY_HOLD:
                    self._toss_started     = True
                    self._toss_arrive_time = time.time()
            else:
                self._toss_ready_hold = 0
            return []

        if self._toss_catch_cd > 0:
            self._toss_catch_cd -= 1
        if self._toss_anim > 0:
            self._toss_anim -= 1

        _TIMEOUT   = 1.0   # seconds before speed resets
        _SPEED_MUL = 5.0   # bonus px per second of reaction time saved

        # Move ball if in flight
        if self._toss_moving:
            self._toss_ball_x += self._toss_ball_vx
            if self._toss_ball_vx > 0 and self._toss_ball_x >= _TOSS_R_X:
                self._toss_ball_x      = float(_TOSS_R_X)
                self._toss_ball_vx     = 0.0
                self._toss_moving      = False
                self._toss_arrive_time = time.time()
            elif self._toss_ball_vx < 0 and self._toss_ball_x <= _TOSS_L_X:
                self._toss_ball_x      = float(_TOSS_L_X)
                self._toss_ball_vx     = 0.0
                self._toss_moving      = False
                self._toss_arrive_time = time.time()

        # Timeout: speed resets if player didn't throw in time
        if not self._toss_moving:
            elapsed = time.time() - self._toss_arrive_time
            if elapsed >= _TIMEOUT:
                self._toss_cur_speed = float(_TOSS_SPEED)

        # Throw detection
        if not self._toss_moving and self._toss_catch_cd == 0:
            ball_at_left  = self._toss_ball_x <= _TOSS_L_X + 10
            ball_at_right = self._toss_ball_x >= _TOSS_R_X - 10

            for i, hand in enumerate(hands):
                if not hand.detected:
                    self._toss_prev_sx.pop(i, None)
                    continue
                hx   = hand.screen_x
                prev = self._toss_prev_sx.get(i, hx)
                self._toss_prev_sx[i] = hx
                dvx  = hx - prev

                threw = False
                if ball_at_left  and hx < 450 and dvx >  _TOSS_SWIPE_PX:
                    threw = True; direction = 1
                elif ball_at_right and hx > 830 and dvx < -_TOSS_SWIPE_PX:
                    threw = True; direction = -1

                if threw:
                    # Speed based on reaction time
                    reaction = time.time() - self._toss_arrive_time
                    if reaction >= _TIMEOUT:
                        self._toss_cur_speed = float(_TOSS_SPEED)
                    else:
                        bonus = (_TIMEOUT - reaction) * _SPEED_MUL
                        self._toss_cur_speed += bonus

                    self._toss_ball_vx  = direction * self._toss_cur_speed
                    self._toss_moving   = True
                    self._toss_count   += 1
                    self._toss_flat     = min(self._toss_count / _TOSS_TARGET, 1.0)
                    self._toss_anim     = 18
                    self._toss_catch_cd = 22
                    self._flash_event('THROW!', (80, 230, 120), 10)
                    if self._toss_count >= _TOSS_TARGET:
                        self._phase    = 'intro_grill'
                        self._inter_t0 = 0.0
                        return ['toss']
                    break
        else:
            for i, hand in enumerate(hands):
                if hand.detected:
                    self._toss_prev_sx[i] = hand.screen_x

        return []

    def _draw_toss_meat(self, frame, hands=None):
        t  = time.time()
        fh, fw = frame.shape[:2]

        if not self._toss_started:
            _panel(frame, 0, 0, fw, fh, alpha=0.55)
            _shadow_text(frame, 'This needs BOTH HANDS!',
                         fw // 2, fh // 2 - 70, 1.0, (255, 230, 100), 2, center=True)
            _shadow_text(frame, 'Show both hands to the camera to start',
                         fw // 2, fh // 2 - 28, 0.7, (220, 220, 240), 1, center=True)
            n_detected = sum(1 for h in (hands or []) if h.detected)
            bar_x0, bar_y0, bar_w_max = fw // 2 - 120, fh // 2 + 6, 240
            bar_w = int(bar_w_max * min(1.0, self._toss_ready_hold / _TOSS_READY_HOLD))
            cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + 20),
                          (80, 230, 120), -1)
            cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + bar_w_max, bar_y0 + 20),
                          (200, 200, 210), 2)
            _shadow_text(frame, f'Hands detected: {n_detected} / 2',
                         fw // 2, fh // 2 + 60, 0.6, (200, 220, 255), 1, center=True)
            return

        ball_at_left  = self._toss_ball_x <= _TOSS_L_X + 10
        ball_at_right = self._toss_ball_x >= _TOSS_R_X - 10
        in_flight     = self._toss_moving

        # ── Zone circles ─────────────────────────────────────────────────────
        for zx, has_ball, swipe_lbl, arrow_dir in [
            (_TOSS_L_X, ball_at_left,  'Swipe -->',  1),
            (_TOSS_R_X, ball_at_right, 'Swipe <--', -1),
        ]:
            pulse      = int(12 * abs(math.sin(t * 5))) if has_ball else 0
            ring_col   = (80, 220, 255) if has_ball else (55, 75, 100)
            ring_thick = 3 if has_ball else 1
            cv2.circle(frame, (zx, _TOSS_Y), _TOSS_ZONE_R + pulse, ring_col, ring_thick)
            _shadow_text(frame, 'Place hand here', zx, _TOSS_Y + _TOSS_ZONE_R + 28,
                         0.52, ring_col, 1, center=True)
            if has_ball and not in_flight:
                ap = int(10 * abs(math.sin(t * 6)))
                cv2.arrowedLine(frame,
                                (zx - arrow_dir * (55 + ap), _TOSS_Y),
                                (zx + arrow_dir * (55 + ap), _TOSS_Y),
                                (80, 255, 130), 5, tipLength=0.28)
                _shadow_text(frame, swipe_lbl, zx, _TOSS_Y - _TOSS_ZONE_R - 22,
                             0.75, (80, 255, 130), 2, center=True)

        # ── Ball arc position ─────────────────────────────────────────────────
        span   = float(_TOSS_R_X - _TOSS_L_X)
        t_pos  = (self._toss_ball_x - _TOSS_L_X) / span
        arc_h  = 110
        ball_y = int(_TOSS_Y - arc_h * 4 * t_pos * (1 - t_pos))
        bx     = int(self._toss_ball_x)

        # ── Shape: VERTICAL elongation ────────────────────────────────────────
        f      = self._toss_flat
        semi_a = int(max(12, _TOSS_BALL_R * (1 - f * 0.70)))  # narrows horizontally
        semi_b = int(_TOSS_BALL_R * (1 + f * 1.40))            # grows vertically
        if self._toss_anim > 0:
            sq     = self._toss_anim / 18
            semi_a = int(semi_a * (1 - sq * 0.20))
            semi_b = int(semi_b * (1 + sq * 0.30))

        meat_col = (68, 85, 190)
        cv2.ellipse(frame, (bx + 4, ball_y + 5), (semi_a + 3, semi_b + 3),
                    0, 0, 360, (20, 25, 80), -1)
        cv2.ellipse(frame, (bx, ball_y), (semi_a, semi_b), 0, 0, 360, meat_col, -1)
        cv2.ellipse(frame, (bx - semi_a//4, ball_y - semi_b//4),
                    (max(4, semi_a//3), max(4, semi_b//4)), 0, 0, 360, (110, 128, 225), -1)
        cv2.ellipse(frame, (bx, ball_y), (semi_a, semi_b), 0, 0, 360, (45, 55, 140), 2)

        if f > 0.05:
            lbl = 'PATTY!' if f >= 0.99 else f'{int(f*100)}%'
            _shadow_text(frame, lbl, bx, ball_y - semi_b - 28,
                         0.65, (255, 220, 80), 1, center=True)

        # ── Instruction banner (right side, clear of the top HUD) ─────────────
        _toss_instr = 'Move hand toward the OTHER side to throw!'
        _ti_w = cv2.getTextSize(_toss_instr, cv2.FONT_HERSHEY_DUPLEX, 0.70, 1)[0][0]
        _panel(frame, fw - 42 - _ti_w, 84, _ti_w + 32, 36)
        _shadow_text(frame, _toss_instr, fw - 26 - _ti_w, 108, 0.70, (220, 220, 240), 1)
        _shadow_text(frame, f'Throw: {self._toss_count} / {_TOSS_TARGET}',
                     fw // 2, fh - 35, 1.0, (255, 230, 100), 2, center=True)

    # ── Cook Steak ───────────────────────────────────────────────────────────

    def _init_cook_steak(self):
        import random as _rnd
        first4 = list(_SK_ACTIONS)   # 4가지 액션 각 1번
        _rnd.shuffle(first4)
        self._sk_queue        = first4 + ['flip', 'flame']   # 5번째=뒤집기, 6번째=불조절
        self._sk_idx          = 0
        self._sk_cook         = 0.0   # only advances on press/flip
        self._sk_flip_side    = 0
        self._sk_pan_grabbed  = False
        self._sk_press_held   = 0
        self._sk_spat_pos     = [_SK_SPAT_REST[0], _SK_SPAT_REST[1]]
        self._sk_spat_hand    = -1
        self._sk_shake_ref    = None
        self._sk_shake_dir    = None
        self._sk_shake_done   = 0
        self._sk_flip_prev_y  = None
        self._sk_action_anim  = 0
        # Animation states
        self._sk_flip_anim    = 0    # 0=none, 1-40=in air
        self._sk_steak_x_off  = 0.0  # shake x offset
        self._sk_steak_y_off  = 0.0  # flip y offset

    def _do_cook_steak(self, hands):
        # Drive flip animation
        if self._sk_flip_anim > 0:
            self._sk_flip_anim -= 1
            if self._sk_flip_anim == 20:   # midpoint: flip side
                self._sk_flip_side = 1 - self._sk_flip_side
                self._sk_cook = min(1.0, self._sk_cook + 0.4)
            if self._sk_flip_anim == 0:    # animation done: advance
                self._sk_idx += 1
                self._sk_action_anim = 10
                self._sk_pan_grabbed = False
                self._sk_flip_prev_y = None
                self._flash_event('NICE!', (80, 220, 160), 12)
                if self._sk_idx >= _SK_ACTION_N:
                    self._phase = 'reveal'
                    self._reveal_t0 = 0.0
                    audio.play_tada()
            return

        if self._sk_idx >= _SK_ACTION_N:
            return
        if self._sk_action_anim > 0:
            self._sk_action_anim -= 1
            return

        act = self._sk_queue[self._sk_idx]

        def _advance():
            self._sk_idx += 1
            self._sk_action_anim = 18
            self._sk_pan_grabbed  = False
            self._sk_shake_ref    = None
            self._sk_shake_dir    = None
            self._sk_shake_done   = 0
            self._sk_flip_prev_y  = None
            self._sk_press_held   = 0
            self._sk_spat_hand    = -1
            self._sk_spat_pos     = [_SK_SPAT_REST[0], _SK_SPAT_REST[1]]
            self._sk_steak_x_off  = 0.0
            if self._sk_idx >= _SK_ACTION_N:
                self._phase = 'reveal'
                self._reveal_t0 = 0.0
                audio.play_tada()
            else:
                self._flash_event('NICE!', (80, 220, 160), 12)

        if act == 'shake':
            for hand in hands:
                if not hand.detected or not hand.gripped:
                    if self._sk_pan_grabbed:
                        self._sk_pan_grabbed = False
                        self._sk_shake_ref   = None
                        self._sk_shake_dir   = None
                    continue
                if not self._sk_pan_grabbed:
                    dx = hand.screen_x - _SK_PAN_HX
                    dy = hand.screen_y - _SK_PAN_HY
                    if dx*dx + dy*dy < 90**2:
                        self._sk_pan_grabbed = True
                    continue
                sy = hand.screen_y
                # Steak shakes visually with pan
                self._sk_steak_x_off = float(hand.screen_x - _SK_PAN_HX) * 0.3
                if self._sk_shake_ref is None:
                    self._sk_shake_ref = sy; continue
                delta = sy - self._sk_shake_ref
                if self._sk_shake_dir is None:
                    if delta >  _SK_SHAKE_PX: self._sk_shake_dir='down'; self._sk_shake_ref=sy
                    elif delta < -_SK_SHAKE_PX: self._sk_shake_dir='up';  self._sk_shake_ref=sy
                elif self._sk_shake_dir=='down' and delta < -_SK_SHAKE_PX:
                    self._sk_shake_dir='up'; self._sk_shake_ref=sy
                    self._sk_shake_done+=1
                    if self._sk_shake_done >= _SK_SHAKE_N: _advance(); return
                elif self._sk_shake_dir=='up' and delta > _SK_SHAKE_PX:
                    self._sk_shake_dir='down'; self._sk_shake_ref=sy
                    self._sk_shake_done+=1
                    if self._sk_shake_done >= _SK_SHAKE_N: _advance(); return

        elif act == 'press':
            if self._sk_spat_hand == -1:
                # Not yet holding the spatula — look for a grab near its rest spot
                for i, hand in enumerate(hands):
                    if hand.detected and hand.gripped:
                        dx = hand.screen_x - self._sk_spat_pos[0]
                        dy = hand.screen_y - self._sk_spat_pos[1]
                        if dx*dx + dy*dy < 70**2:
                            self._sk_spat_hand = i
                            break
                self._sk_press_held = max(0, self._sk_press_held - 2)
            else:
                hand = hands[self._sk_spat_hand] if self._sk_spat_hand < len(hands) else None
                if hand is None or not hand.detected or not hand.gripped:
                    self._sk_spat_hand = -1
                    self._sk_spat_pos  = [_SK_SPAT_REST[0], _SK_SPAT_REST[1]]
                    self._sk_press_held = max(0, self._sk_press_held - 2)
                else:
                    self._sk_spat_pos[0] = hand.screen_x
                    self._sk_spat_pos[1] = hand.screen_y
                    # Hold check uses the spatula's paddle tip (top of the spatula)
                    tip_x = self._sk_spat_pos[0]
                    tip_y = self._sk_spat_pos[1] - _SK_SPAT_TIP_OFF
                    dy = tip_y - _SK_PAN_CY
                    if abs(tip_x - _SK_PAN_CX) < _SK_AREA_R and -30 < dy < _SK_AREA_R:
                        self._sk_press_held += 1
                        if self._sk_press_held >= _SK_PRESS_HOLD:
                            self._sk_cook = min(1.0, self._sk_cook + 0.35)
                            _advance()
                    else:
                        self._sk_press_held = max(0, self._sk_press_held - 2)

        elif act == 'flip':
            for hand in hands:
                if not hand.detected or not hand.gripped:
                    if self._sk_pan_grabbed:
                        self._sk_pan_grabbed = False
                        self._sk_flip_prev_y = None
                    continue
                if not self._sk_pan_grabbed:
                    dx = hand.screen_x - _SK_PAN_HX
                    dy = hand.screen_y - _SK_PAN_HY
                    if dx*dx + dy*dy < 90**2:
                        self._sk_pan_grabbed = True
                    continue
                sy = hand.screen_y
                if self._sk_flip_prev_y is not None:
                    up_delta = self._sk_flip_prev_y - sy
                    if up_delta >= _SK_FLIP_PX:
                        self._sk_flip_anim = 40   # start flip animation
                        return
                self._sk_flip_prev_y = sy

        elif act == 'flame':
            for hand in hands:
                if hand.detected and hand.gripped:
                    bx = hand.screen_x - _SK_FLAME_CX
                    by = hand.screen_y - _SK_FLAME_CY
                    if bx*bx + by*by < (_SK_FLAME_R + 25)**2:
                        _advance(); return

    def _draw_cook_steak(self, frame):
        t   = time.time()
        cx, cy = _SK_PAN_CX, _SK_PAN_CY
        cook   = self._sk_cook
        fh, fw = frame.shape[:2]
        act    = self._sk_queue[self._sk_idx] if self._sk_idx < _SK_ACTION_N else ''

        # ── Pan y offset (shake / flip) ───────────────────────────────────────
        flip_frac  = self._sk_flip_anim / 40.0
        pan_y_off  = 0
        if act == 'shake' and self._sk_pan_grabbed:
            pan_y_off = int(18 * math.sin(t * 14))
        if self._sk_flip_anim > 0:
            # Pan jumps up on first half, returns second half
            pan_y_off = int(-35 * math.sin(math.pi * flip_frac))

        pcy = cy + pan_y_off   # pan center y (moved)

        # ── Stove base ───────────────────────────────────────────────────────
        cv2.ellipse(frame, (cx, cy + _SK_PAN_R + 18), (_SK_PAN_R + 45, 38),
                    0, 0, 360, (48, 44, 40), -1)

        # ── Pan ───────────────────────────────────────────────────────────────
        cv2.circle(frame, (cx, pcy), _SK_PAN_R, (36, 32, 28), -1)
        cv2.circle(frame, (cx, pcy), _SK_PAN_R, (18, 15, 12), 10)
        sv = int(148 - cook * 20)
        cv2.circle(frame, (cx, pcy), _SK_AREA_R + 8, (sv, sv+2, sv+4), -1)
        cv2.circle(frame, (cx, pcy), _SK_AREA_R + 8, (65, 60, 56), 2)

        # Pan handle (bottom, moves with pan)
        hy1 = pcy + _SK_PAN_R - 10
        hy2 = pcy + _SK_PAN_R + 120
        cv2.rectangle(frame, (cx - 24, hy1), (cx + 24, hy2), (32, 28, 24), -1)
        cv2.rectangle(frame, (cx - 24, hy1), (cx + 24, hy2), (15, 12, 10), 3)

        # ── Flames (fixed to stove) ───────────────────────────────────────────
        fl_span = int(_SK_PAN_R * 1.55)
        for fi in range(8):
            fx  = cx - fl_span//2 + fi * (fl_span//7)
            flk = int(12 * math.sin(t * 9.0 + fi * 1.4))
            fy0 = cy + _SK_PAN_R + 5
            cv2.ellipse(frame, (fx, fy0), (6, 14 + flk//2), 0, 0, 360, (185, 90, 15), -1)
            pts_f = np.array([[fx-10, fy0],[fx, fy0-40-flk],[fx+10, fy0]], np.int32)
            cv2.fillPoly(frame, [pts_f], (0, int(140+flk*3), 245))

        # ── Steak animations ─────────────────────────────────────────────────
        press_frac  = self._sk_press_held / _SK_PRESS_HOLD if act == 'press' else 0.0
        flip_height = int(90 * math.sin(math.pi * (1 - flip_frac))) if self._sk_flip_anim > 0 else 0
        shake_x = int(self._sk_steak_x_off) if act == 'shake' else 0

        # Squish: wider + shorter when pressing
        squish_x = int(25 * press_frac)
        squish_y = int(18 * press_frac)

        # Steak center: rides on pan (pan_y_off) + flip arc
        scx = cx + shake_x
        scy = pcy - flip_height   # steak sits on pan, both move together except flip arc

        # Color: only changes on press/flip — kept light
        col = (85, 100, 200)   # raw: light pink (BGR)
        if cook > 0:
            if cook < 0.5:
                tc  = cook / 0.5
                col = (int(85+tc*5), int(100+tc*15), int(200-tc*35))   # → lighter tan
            else:
                tc  = (cook-0.5)/0.5
                col = (int(90-tc*20), int(115-tc*25), int(165-tc*30))  # → medium brown, stays light

        # Flip mid-point: show cooked side (darker)
        if self._sk_flip_side == 1:
            col = tuple(max(0, c-45) for c in col)

        col2 = tuple(max(0, c-35) for c in col)

        steak_pts = np.array([
            [scx-100-squish_x, scy+18-squish_y],
            [scx-75,           scy-52+squish_y],
            [scx-15,           scy-68+squish_y],
            [scx+45,           scy-62+squish_y],
            [scx+95,           scy-18],
            [scx+108+squish_x, scy+28-squish_y],
            [scx+48+squish_x,  scy+62-squish_y],
            [scx-28,           scy+66-squish_y],
            [scx-82-squish_x,  scy+48],
        ], np.int32)

        # During flip: scale x based on rotation (cos effect)
        if self._sk_flip_anim > 0:
            cos_a = abs(math.cos(math.pi * (1 - flip_frac)))
            steak_pts[:, 0] = scx + ((steak_pts[:, 0] - scx) * cos_a).astype(int)

        cv2.fillPoly(frame, [steak_pts + np.array([4,5])], (18, 15, 12))
        cv2.fillPoly(frame, [steak_pts], col)
        if cook > 0.05:
            for gx in range(scx-72, scx+80, 28):
                cv2.line(frame, (gx-18, scy+30-squish_y),
                         (gx+18, scy-30+squish_y), col2, 4)
        cv2.polylines(frame, [steak_pts], True, col2, 3)

        # ── Pan grab ring (follows pan y) ────────────────────────────────────
        if not self._sk_pan_grabbed and act in ('shake', 'flip') and self._sk_flip_anim == 0:
            _grab_ring(frame, [_SK_PAN_HX, _SK_PAN_HY + pan_y_off])

        # ── Flame button ──────────────────────────────────────────────────────
        show_flame = act == 'flame'
        pulse = int(9 * abs(math.sin(t*5))) if show_flame else 0
        bcol  = (0, 185, 255) if show_flame else (50, 50, 68)
        cv2.circle(frame, (_SK_FLAME_CX, _SK_FLAME_CY), _SK_FLAME_R+pulse, bcol, -1)
        cv2.circle(frame, (_SK_FLAME_CX, _SK_FLAME_CY), _SK_FLAME_R+pulse, (255,255,255), 2)
        fpts = np.array([
            [_SK_FLAME_CX,      _SK_FLAME_CY-28],
            [_SK_FLAME_CX-14,   _SK_FLAME_CY+6],
            [_SK_FLAME_CX-7,    _SK_FLAME_CY+18],
            [_SK_FLAME_CX+7,    _SK_FLAME_CY+18],
            [_SK_FLAME_CX+14,   _SK_FLAME_CY+6],
        ], np.int32)
        cv2.fillPoly(frame, [fpts], (255,255,255) if show_flame else (100,100,120))
        if show_flame:
            ring_r = int(_SK_FLAME_R+18+10*abs(math.sin(t*5)))
            cv2.circle(frame, (_SK_FLAME_CX, _SK_FLAME_CY), ring_r, (0,200,255), 3, cv2.LINE_AA)
            _shadow_text(frame, 'GRAB!', _SK_FLAME_CX, _SK_FLAME_CY-ring_r-12,
                         0.6, (0,200,255), 1, center=True)

        # ── Action instruction (right-center, clear of the top HUD bar) ───────
        if self._sk_idx < _SK_ACTION_N:
            labels = {'shake':'Grab handle,',
                      'press':'Grab the spatula,',
                      'flip': 'Grab handle,',
                      'flame':'Grip the'}
            labels2 = {'shake':'Shake UP & DOWN  x3',
                       'press':'hold it over the steak',
                       'flip': 'Flick UP sharply',
                       'flame':'FLAME button'}
            l1, l2 = labels.get(act, ''), labels2.get(act, '')
            scale = 0.62
            (tw1, th), _ = cv2.getTextSize(l1, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
            (tw2, _),  _ = cv2.getTextSize(l2, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
            box_w = max(tw1, tw2) + 36
            box_h = th * 2 + 34
            tx = fw - 26 - box_w
            ty = fh // 2 - box_h // 2
            _panel(frame, tx, ty, box_w, box_h)
            _shadow_text(frame, l1, tx + 18, ty + th + 8,  scale, (80, 220, 255), 1)
            _shadow_text(frame, l2, tx + 18, ty + th*2 + 24, scale, (80, 220, 255), 1)

        # Demo arrow per action
        if act == 'shake' and not self._sk_pan_grabbed:
            pass   # grab ring already shown
        elif act == 'shake' and self._sk_pan_grabbed:
            for ay in [cy-60, cy, cy+60]:
                cv2.arrowedLine(frame, (cx-200, ay), (cx-200, ay-50),
                                (80,220,255), 2, tipLength=0.4)
                cv2.arrowedLine(frame, (cx-200, ay), (cx-200, ay+50),
                                (80,220,255), 2, tipLength=0.4)
        elif act == 'press':
            # Blinking target ring over the steak — shows where to bring the spatula tip
            blink   = 0.5 + 0.5 * math.sin(t * 6.0)
            ring_r  = int(_SK_AREA_R * 0.78 + 6 * blink)
            ring_th = 2 + int(2 * blink)
            cv2.circle(frame, (cx, cy - 10), ring_r, (80, 220, 255), ring_th, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy - 10), ring_r + 14, (80, 220, 255), 1, cv2.LINE_AA)
            _shadow_text(frame, 'TARGET', cx, cy - 10 - ring_r - 16,
                         0.55, (80, 220, 255), 1, center=True)

            sx, sy = self._sk_spat_pos
            _draw_spatula(frame, sx, sy)
            if self._sk_spat_hand == -1:
                _grab_ring(frame, (sx, sy))
            else:
                _shadow_text(frame, 'Bring the spatula tip into the ring!', cx, cy - _SK_AREA_R - 34,
                             0.55, (80,220,255), 1, center=True)
        elif act == 'flip' and self._sk_pan_grabbed and self._sk_flip_anim == 0:
            cv2.arrowedLine(frame, (cx-180, cy+60), (cx-180, cy-60),
                            (80,220,255), 3, tipLength=0.3)

        # Progress dots
        for di in range(_SK_ACTION_N):
            dx = fw//2 - (_SK_ACTION_N-1)*22 + di*44
            dc = (80,220,140) if di < self._sk_idx else (60,60,80)
            cv2.circle(frame, (dx, fh-28), 10, dc, -1)
            cv2.circle(frame, (dx, fh-28), 10, (180,190,200), 2)

        # Press hold bar
        if act == 'press' and self._sk_press_held > 0:
            bar_w = int((fw-80) * self._sk_press_held / _SK_PRESS_HOLD)
            cv2.rectangle(frame, (40, fh-58), (40+bar_w, fh-46), (80,220,140), -1)
            cv2.rectangle(frame, (40, fh-58), (fw-40, fh-46), (100,110,120), 2)

    # ── Knead Stage ──────────────────────────────────────────────────────────

    def _init_knead(self):
        import random as _rnd
        self._knead_sub           = 'pour_onion'
        self._knead_onion_x       = float(_KN_ONION_X)
        self._knead_onion_grabbed = False
        self._knead_arrows        = [_rnd.choice(_KN_DIRS) for _ in range(_KN_ARROW_N)]
        self._knead_idx           = 0
        self._knead_ref           = None
        self._knead_prev_ang      = None
        self._knead_rot_total     = 0.0
        self._knead_mix_angle     = 0.0
        self._knead_active        = False
        self._kn_onion_spread     = 1.0
        # Precompute meat blob positions (polar, rotated by mix_angle each frame)
        rng = np.random.default_rng(seed=42)
        n = 88
        self._kn_meat_r   = np.sqrt(rng.uniform(0.01, 0.88, n)) * (_KN_BOWL_RX - 28)
        self._kn_meat_ang = rng.uniform(0, 2*math.pi, n)
        self._kn_meat_w   = rng.integers(11, 26, n)
        self._kn_meat_h   = rng.integers(7,  15, n)
        self._kn_meat_rot = rng.uniform(0, math.pi, n)
        # BGR meat colors: warm pink/salmon (R high, G mid, B low)
        self._kn_meat_col = np.stack([
            rng.integers(55, 82,  n),   # B
            rng.integers(72, 112, n),   # G
            rng.integers(155, 210, n),  # R  → RGB pink/salmon
        ], axis=1)
        # Precompute onion blob positions
        rng2 = np.random.default_rng(seed=8)
        m = 32
        self._kn_onion_r   = rng2.uniform(0, 68, m)
        self._kn_onion_ang = rng2.uniform(0, 2*math.pi, m)
        self._kn_onion_w   = (rng2.integers(14, 32, m) * 0.7).astype(int)
        self._kn_onion_h   = (rng2.integers(9,  18, m) * 0.7).astype(int)
        self._kn_onion_rot = rng2.uniform(0, math.pi, m)
        # BGR golden onion colors (R ~200-230, G ~150-185, B ~15-50)
        self._kn_onion_col = np.stack([
            rng2.integers(15, 50,  m),   # B
            rng2.integers(145, 185, m),  # G
            rng2.integers(190, 232, m),  # R  → RGB golden
        ], axis=1)

    _KN_ONION_SPREAD_MAX = 2.6   # cap keeps the scattered cloud inside the bowl wall
    _KN_ONION_SPREAD_INC = 0.27  # 1.5x the original per-success scatter step

    def _kn_onion_center(self):
        mix = self._knead_mix_angle
        ocx = _KN_BOWL_CX + int(58 * math.cos(mix * 0.7 + 0.8))
        ocy = _KN_BOWL_CY + int(42 * math.sin(mix * 0.7 + 0.8) * 0.76) + 10
        return ocx, ocy

    def _kn_onion_scatter_step(self):
        """Called on each successful mix action — nudges the onion further
        from its center (capped so the cloud stays inside the bowl wall)."""
        self._kn_onion_spread = min(self._KN_ONION_SPREAD_MAX,
                                    self._kn_onion_spread + self._KN_ONION_SPREAD_INC)

    def _do_knead(self, hands):
        if self._knead_sub == 'pour_onion':
            self._knead_active = False
            return self._do_knead_pour(hands)
        self._knead_active = any(hand.detected and hand.gripped for hand in hands)
        if self._knead_sub == 'arrows':
            return self._do_knead_arrows(hands)
        else:
            return self._do_knead_rotate(hands)

    def _do_knead_pour(self, hands):
        for hand in hands:
            if not hand.detected or not hand.gripped:
                if self._knead_onion_grabbed:
                    self._knead_onion_grabbed = False
                continue
            hx = float(hand.screen_x)
            hy = float(hand.screen_y)
            if not self._knead_onion_grabbed:
                if (hx - self._knead_onion_x)**2 + (hy - _KN_ONION_Y)**2 < 130**2:
                    self._knead_onion_grabbed = True
                continue
            self._knead_onion_x = max(float(_KN_BOWL_CX + 50), min(hx, float(_KN_ONION_X)))
            if self._knead_onion_x <= _KN_BOWL_CX + 80:
                self._knead_onion_grabbed = False
                self._knead_sub = 'arrows'
                self._flash_event('NICE!', (80, 220, 140), 16)
        return []

    def _do_knead_arrows(self, hands):
        for hand in hands:
            if not hand.detected or not hand.gripped:
                self._knead_ref = None
                continue
            hx, hy = hand.screen_x, hand.screen_y
            if self._knead_ref is None:
                self._knead_ref = (hx, hy)
                continue
            dx = hx - self._knead_ref[0]
            dy = hy - self._knead_ref[1]
            target = self._knead_arrows[self._knead_idx]
            matched = False
            if target == 'up'    and dy < -_KN_SWIPE_PX and abs(dy) > abs(dx):
                matched = True
            elif target == 'down'  and dy >  _KN_SWIPE_PX and abs(dy) > abs(dx):
                matched = True
            elif target == 'left'  and dx < -_KN_SWIPE_PX and abs(dx) > abs(dy):
                matched = True
            elif target == 'right' and dx >  _KN_SWIPE_PX and abs(dx) > abs(dy):
                matched = True
            if matched:
                self._knead_idx += 1
                self._knead_ref = None
                # Rotate mixture visually in swipe direction
                if target in ('right', 'down'):
                    self._knead_mix_angle += 0.45
                else:
                    self._knead_mix_angle -= 0.45
                self._flash_event('NICE!', (100, 220, 255), 13)
                self._kn_onion_scatter_step()
                if self._knead_idx >= _KN_ARROW_N:
                    self._knead_sub       = 'rotate'
                    self._knead_rot_total = 0.0
                    self._knead_prev_ang  = None
        return []

    def _do_knead_rotate(self, hands):
        for hand in hands:
            if not hand.detected or not hand.gripped:
                self._knead_prev_ang = None
                continue
            hx = hand.screen_x - _KN_BOWL_CX
            hy = hand.screen_y - _KN_BOWL_CY
            if math.sqrt(hx*hx + hy*hy) < 55:
                continue
            angle = math.atan2(hy, hx)
            if self._knead_prev_ang is not None:
                delta = angle - self._knead_prev_ang
                if delta >  math.pi: delta -= 2 * math.pi
                if delta < -math.pi: delta += 2 * math.pi
                if delta > 0:   # clockwise only
                    self._knead_rot_total  += delta
                    self._knead_mix_angle  += delta * 0.8
                if self._knead_rot_total >= 2 * math.pi:
                    self._knead_prev_ang = None
                    self._phase = 'toss_meat'
                    self._init_toss_meat()
                    return ['knead']
            self._knead_prev_ang = angle
        return []

    def _draw_knead(self, frame):
        t   = time.time()
        cx, cy = _KN_BOWL_CX, _KN_BOWL_CY
        rx, ry = _KN_BOWL_RX, _KN_BOWL_RY
        mix = self._knead_mix_angle
        sub = self._knead_sub

        # ── Bowl: draw outer ring only (no full fill that would cover meat) ───
        cv2.ellipse(frame, (cx + 10, cy + 24), (rx - 12, 30), 0, 0, 360, (22, 20, 18), -1)
        # Outer bowl wall (ring between outer and inner ellipse)
        cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, (230, 215, 185), -1)
        # Inner bowl surface (base for food)
        cv2.ellipse(frame, (cx, cy), (rx - 20, ry - 15), 0, 0, 360, (215, 200, 168), -1)

        # ── Meat fill: base + 2 large ellipses rotating with mix_angle ────────
        # Base meat fill covering full interior (scaled to 0.6x so the bowl reads bigger)
        _MEAT_SCALE = 0.6
        cv2.ellipse(frame, (cx, cy + 8),
                    (int((rx - 28) * _MEAT_SCALE), int((ry - 20) * _MEAT_SCALE)),
                    0, 0, 360, (68, 82, 188), -1)
        # Two large overlapping blobs with slight offset (rotate with mix)
        for base_ang, r_frac, bw, bh, col in [
            (0.0, 0.28, int((rx - 60) * _MEAT_SCALE), int((ry - 45) * _MEAT_SCALE), (55, 70, 172)),
            (2.6, 0.22, int((rx - 80) * _MEAT_SCALE), int((ry - 55) * _MEAT_SCALE), (78, 95, 200)),
        ]:
            a  = base_ang + mix
            r  = r_frac * (rx - 20)
            fx = int(cx + math.cos(a) * r)
            fy = int(cy + math.sin(a) * r * 0.76 + 8)
            cv2.ellipse(frame, (fx, fy), (bw, bh), int(math.degrees(a)), 0, 360, col, -1)

        # ── Golden onion blob (appears after pour, orbits & mixes) ───────────
        # Each pass of the hand through it nudges _kn_onion_spread up, so the
        # pieces gradually scatter outward from their shared center — but the
        # per-particle radius (_kn_onion_r, max 68) and _KN_ONION_SPREAD_MAX
        # are chosen so the scattered cloud never crosses the bowl wall.
        if sub in ('arrows', 'rotate'):
            ocx, ocy = self._kn_onion_center()
            spread = self._kn_onion_spread
            for i in range(len(self._kn_onion_r)):
                a   = float(self._kn_onion_ang[i]) + mix * 1.1
                r   = float(self._kn_onion_r[i]) * spread
                fx  = int(ocx + math.cos(a) * r)
                fy  = int(ocy + math.sin(a) * r * 0.82)
                bw  = int(self._kn_onion_w[i])
                bh  = int(self._kn_onion_h[i])
                br  = int(math.degrees(float(self._kn_onion_rot[i]) + mix))
                col = tuple(int(c) for c in self._kn_onion_col[i])
                cv2.ellipse(frame, (fx, fy), (bw, bh), br, 0, 360, col, -1)

        # ── Bowl rim drawn LAST as thin outline only (no thick cover) ─────────
        # Clip any overflow by redrawing the outer wall ring on top
        cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, (230, 215, 185), 22)
        cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, (252, 245, 230), 3)
        cv2.ellipse(frame, (cx, cy), (rx - 20, ry - 15), 0, 0, 360, (215, 200, 168), 3)

        # ── Pour onion sub-phase (pan enlarged so it reads clearly) ───────────
        if sub == 'pour_onion':
            ox = int(self._knead_onion_x)
            oy = _KN_ONION_Y
            # Pan shadow
            cv2.ellipse(frame, (ox + 14, oy + 60), (140, 44), 0, 0, 360, (28, 24, 20), -1)
            # Pan outer rim
            cv2.ellipse(frame, (ox, oy), (148, 110), 0, 0, 360, (58, 54, 50), -1)
            # Pan interior (dark iron surface)
            cv2.ellipse(frame, (ox, oy), (120, 86), 0, 0, 360, (42, 52, 58), -1)
            # Fried onion fill (golden, covers pan interior)
            rng2 = np.random.default_rng(seed=5)
            for _ in range(16):
                fx = int(ox + rng2.integers(-92, 93))
                fy = int(oy + rng2.integers(-58, 59))
                w2 = int(rng2.integers(20, 40))
                h2 = int(rng2.integers(10, 22))
                # Golden onion BGR: (15-45, 140-180, 185-225)
                gc = (int(rng2.integers(15, 45)),
                      int(rng2.integers(140, 180)),
                      int(rng2.integers(185, 225)))
                cv2.ellipse(frame, (fx, fy), (w2, h2),
                            int(rng2.integers(0, 180)), 0, 360, gc, -1)
            # Pan handle
            cv2.rectangle(frame, (ox + 140, oy - 16), (ox + 196, oy + 16), (50, 46, 42), -1)
            cv2.rectangle(frame, (ox + 140, oy - 16), (ox + 196, oy + 16), (78, 74, 68), 2)
            cv2.ellipse(frame, (ox, oy), (148, 110), 0, 0, 360, (82, 78, 74), 6)
            if not self._knead_onion_grabbed:
                _grab_ring(frame, [ox, oy])
            cv2.arrowedLine(frame, (ox - 48, oy), (ox - 260, oy),
                            (80, 220, 140), 4, tipLength=0.28)

        # ── Arrow mini-game ───────────────────────────────────────────────────
        elif sub == 'arrows':
            if self._knead_idx < _KN_ARROW_N:
                direction = self._knead_arrows[self._knead_idx]
                pulse = int(8 * abs(math.sin(t * 5)))
                _draw_dir_arrow(frame, cx, cy - 20, direction, 68 + pulse, (80, 220, 255))
                for i in range(_KN_ARROW_N):
                    dot_x = cx - (_KN_ARROW_N - 1) * 18 + i * 36
                    col = (80, 220, 140) if i < self._knead_idx else (80, 80, 100)
                    cv2.circle(frame, (dot_x, cy + ry + 36), 11, col, -1)
                    cv2.circle(frame, (dot_x, cy + ry + 36), 11, (210, 218, 228), 2)

        # ── Final clockwise rotation ──────────────────────────────────────────
        elif sub == 'rotate':
            prog = min(self._knead_rot_total / (2 * math.pi), 1.0)
            arc  = int(prog * 360)
            cv2.ellipse(frame, (cx, cy), (rx - 6, ry - 6),
                        -90, 0, arc, (100, 240, 140), 6)
            for ai in range(0, 300, 30):
                a1 = math.radians(ai - 90)
                a2 = math.radians(ai + 18 - 90)
                p1 = (int(cx + (rx + 20) * math.cos(a1)), int(cy + (ry + 20) * math.sin(a1)))
                p2 = (int(cx + (rx + 20) * math.cos(a2)), int(cy + (ry + 20) * math.sin(a2)))
                cv2.line(frame, p1, p2, (255, 220, 80), 3)
            _shadow_text(frame, 'TURN!', cx, cy, 1.4, (255, 230, 80), 3, center=True)

    # ── New Stage 1: Onion cutting ────────────────────────────────────────────

    def _do_place_onion(self, hands):
        for hand in hands:
            if not hand.detected or not hand.gripped:
                if not self._onb_grabbed:
                    continue
                else:
                    self._onb_grabbed = False   # dropped
                    continue
            if not self.timer_started:
                self._begin_timer()
            # Grab the onion: hand must be near it
            if not self._onb_grabbed:
                dx = hand.screen_x - 640
                dy = hand.screen_y - int(self._onb_y)
                if dx*dx + dy*dy < (_ONB_R + 55)**2:
                    self._onb_grabbed = True
                    self._flash_event('GRAB!', (0, 220, 255), 8)
                continue
            # Onion follows hand downward once grabbed
            self._onb_y = min(float(hand.screen_y), float(_ONB_BOARD_Y))
            if self._onb_y >= _ONB_BOARD_Y:
                self._phase       = 'peel_skin'
                self._onb_grabbed = False
                self._flash_event('PLACED!', (0, 255, 180), 14)
                return ['grab']
        return []

    def _do_peel(self, hands):
        for hand in hands:
            if not hand.detected or not hand.gripped:
                self._peel_ref_x = None
                self._peel_dir   = None
                continue
            sx = hand.screen_x
            if self._peel_ref_x is None:
                self._peel_ref_x = sx
                continue
            delta = sx - self._peel_ref_x
            if self._peel_dir is None:
                if abs(delta) >= _PEEL_PX:
                    self._peel_dir   = 'right' if delta > 0 else 'left'
                    self._peel_ref_x = sx
            elif self._peel_dir == 'right' and delta <= -_PEEL_PX:
                self._peel_dir   = 'left'
                self._peel_ref_x = sx
                self._peel_done += 1
                self._flash_event('PEEL!', (0, 210, 255), 12)
                if self._peel_done >= _PEEL_NEEDED:
                    self._phase = 'slice_onion'
                return ['cut']
            elif self._peel_dir == 'left' and delta >= _PEEL_PX:
                self._peel_dir   = 'right'
                self._peel_ref_x = sx
                self._peel_done += 1
                self._flash_event('PEEL!', (0, 210, 255), 12)
                if self._peel_done >= _PEEL_NEEDED:
                    self._phase = 'slice_onion'
                return ['cut']
        return []

    def _do_slice_onion(self, hands):
        if self._cooldown > 0:
            self._cooldown -= 1
            return []

        # ── Step A: grab the knife first ─────────────────────────────────────
        if self._slice_knife_hand == -1:
            for i, hand in enumerate(hands):
                if not hand.detected or not hand.gripped:
                    continue
                dx = hand.screen_x - self._slice_knife_pos[0]
                dy = hand.screen_y - self._slice_knife_pos[1]
                if dx * dx + dy * dy < GRAB_RADIUS ** 2:
                    self._slice_knife_hand = i
                    self._slice_ref_y      = None
                    self._slice_dir        = None
                    self._flash_event('GRAB!', (0, 220, 255), 8)
                    return ['grab']
            return []

        # ── Step B: verify hand is still gripping ────────────────────────────
        if self._slice_knife_hand >= len(hands):
            self._slice_knife_hand = -1
            return []
        hand = hands[self._slice_knife_hand]
        if not hand.detected or not hand.gripped:
            # dropped — allow re-grab at current knife position
            self._try_reattach(hands, self._slice_knife_pos, 'slice_knife')
            if self._held_tool == 'slice_knife':
                self._slice_knife_hand = self._held_by
                self._held_tool = None   # don't interfere with other grabs
            else:
                self._slice_knife_hand = -1
            self._slice_ref_y = None
            self._slice_dir   = None
            return []

        # ── Step C: knife follows hand ────────────────────────────────────────
        self._slice_knife_pos = [hand.screen_x, hand.screen_y]

        # Knife must be over the onion (x near center 640)
        if abs(hand.screen_x - 640) > _ONB_R + 55:
            self._slice_ref_y = None
            self._slice_dir   = None
            return []

        # ── Step D: detect up-down stroke ────────────────────────────────────
        sy = hand.screen_y
        if self._slice_ref_y is None:
            self._slice_ref_y = sy
            return []
        delta = sy - self._slice_ref_y

        if self._slice_dir is None:
            if delta >= _SLICE_PX:
                self._slice_dir   = 'down'
                self._slice_ref_y = sy
            elif delta <= -_SLICE_PX:
                self._slice_dir   = 'up'
                self._slice_ref_y = sy
        elif self._slice_dir == 'down' and delta <= -_SLICE_PX:
            self._slice_dir   = 'up'
            self._slice_ref_y = sy
            self._slice_done += 1
            self._cooldown    = 6
            self._flash_event('SLICE!', (0, 255, 200), 12)
            if self._slice_done >= _SLICE_NEEDED:
                self._phase = 'dice_onion'
                self._dice_knife_pos = [_DICE_KNIFE_X, _DICE_KNIFE_Y]
                self._dice_knife_hand = -1
            return ['cut']
        elif self._slice_dir == 'up' and delta >= _SLICE_PX:
            self._slice_dir   = 'down'
            self._slice_ref_y = sy
            self._slice_done += 1
            self._cooldown    = 6
            self._flash_event('SLICE!', (0, 255, 200), 12)
            if self._slice_done >= _SLICE_NEEDED:
                self._phase = 'dice_onion'
                self._dice_knife_pos = [_DICE_KNIFE_X, _DICE_KNIFE_Y]
                self._dice_knife_hand = -1
            return ['cut']
        return []

    def _do_dice_onion(self, hands):
        """Knife on the right side, up-down strokes slice from right to left."""
        for i, hand in enumerate(hands):
            if not hand.detected or not hand.gripped:
                continue
            # Step A: grab knife (on the right side)
            if self._dice_knife_hand == -1:
                dx = hand.screen_x - self._dice_knife_pos[0]
                dy = hand.screen_y - self._dice_knife_pos[1]
                if dx*dx + dy*dy < 90**2:
                    self._dice_knife_hand = i
                continue
            if i != self._dice_knife_hand:
                continue
            # Step B: knife follows hand
            self._dice_knife_pos = [hand.screen_x, hand.screen_y]
            # Must be near the current cut line x (±60px) and within y range
            _n_s  = _SLICE_NEEDED + 1
            _dcw  = (2 * _ONB_R) // _n_s
            cut_line_x = 640 + _ONB_R - self._dice_done * _dcw
            if not (abs(hand.screen_x - cut_line_x) < 65 and
                    abs(hand.screen_y - _ONB_BOARD_Y) < _ONB_R + 90):
                self._dice_ref_y = None
                self._dice_dir   = None
                continue
            # Step C: detect up-down stroke
            sy = hand.screen_y
            if self._dice_ref_y is None:
                self._dice_ref_y = sy
                continue
            delta = sy - self._dice_ref_y
            if self._dice_dir is None:
                if delta >= _SLICE_PX:
                    self._dice_dir = 'down'; self._dice_ref_y = sy
                elif delta <= -_SLICE_PX:
                    self._dice_dir = 'up';   self._dice_ref_y = sy
            elif self._dice_dir == 'down' and delta <= -_SLICE_PX:
                self._dice_dir = 'up';   self._dice_ref_y = sy
                self._dice_done += 1;    self._cooldown = 6
                self._flash_event('DICE!', (80, 255, 200), 12)
                if self._dice_done >= _DICE_NEEDED:
                    self._phase = 'bowl_drop'
                    self._init_bowl_pieces()
                return ['cut']
            elif self._dice_dir == 'up' and delta >= _SLICE_PX:
                self._dice_dir = 'down'; self._dice_ref_y = sy
                self._dice_done += 1;    self._cooldown = 6
                self._flash_event('DICE!', (80, 255, 200), 12)
                if self._dice_done >= _DICE_NEEDED:
                    self._phase = 'bowl_drop'
                    self._init_bowl_pieces()
                return ['cut']
        # Drop check
        for i, hand in enumerate(hands):
            if self._dice_knife_hand == i and (not hand.detected or not hand.gripped):
                self._dice_knife_hand = -1
                self._dice_ref_y = None
                self._dice_dir   = None
        return []

    def _draw_diced_onion(self, frame, cx, cy, peel_frac, v_slices, h_done):
        """Rotated 90°: horizontal strip spread (horizontal gaps) + diced pieces on right."""
        r   = _ONB_R
        fh, fw = frame.shape[:2]
        n   = v_slices + 1           # number of strips
        sh  = (2 * r) // n           # strip HEIGHT (horizontal strips)
        gap = max(4, int(r * 0.18 * v_slices / _SLICE_NEEDED))
        total_spread = gap * (n - 1)

        # Draw full onion onto canvas
        canvas = frame.copy()
        cv2.circle(canvas, (cx, cy), r, (220, 235, 242), -1)
        cv2.circle(canvas, (cx, cy), int(r * 0.68), (205, 222, 232), 2)
        cv2.circle(canvas, (cx, cy), int(r * 0.36), (188, 210, 222), 2)
        cv2.circle(canvas, (cx, cy), max(4, int(r * 0.10)), (165, 196, 212), -1)
        cv2.circle(canvas, (cx, cy), r, (28, 68, 108), 4)

        x1 = max(0, cx - r - 2);  x2 = min(fw, cx + r + 2)

        # Each dice cut removes a vertical column from the RIGHT of all strips
        dice_cut_w  = (2 * r) // (n)              # width removed per cut
        right_clip  = cx + r - h_done * dice_cut_w  # remaining right boundary

        # Blit all horizontal strips, clipped to remaining right boundary
        for i in range(n):
            src_y1 = cy - r + i * sh
            src_y2 = cy - r + (i + 1) * sh
            shift  = i * gap - total_spread // 2
            dst_y1 = src_y1 + shift
            dst_y2 = src_y2 + shift

            s1  = max(0, src_y1);  d1 = max(0, dst_y1)
            row = min(min(fh, src_y2) - s1, min(fh, dst_y2) - d1)
            xa  = max(0, cx - r - 2)
            xb  = min(fw, right_clip)          # clip right side
            if row > 0 and xb > xa:
                frame[d1:d1 + row, xa:xb] = canvas[s1:s1 + row, xa:xb]

        # Fill HORIZONTAL gaps between strips (within remaining width)
        for i in range(n - 1):
            shift_i  = i * gap - total_spread // 2
            gy1      = cy - r + (i + 1) * sh + shift_i
            gy2      = gy1 + gap
            orig_cut = cy - r + (i + 1) * sh
            dist     = abs(orig_cut - cy)
            if dist < r:
                chord_w = int(math.sqrt(r * r - dist * dist)) * 2
                gx1c = max(0, cx - chord_w // 2)
                gx2c = min(fw, min(cx + chord_w // 2, right_clip))
                gy1c = max(0, gy1);  gy2c = min(fh, gy2)
                if gy2c > gy1c and gx2c > gx1c:
                    cv2.rectangle(frame, (gx1c, gy1c), (gx2c, gy2c), (245, 252, 255), -1)
                    cv2.line(frame, (gx1c, gy1c), (gx2c, gy1c), (180, 210, 225), 2)
                    cv2.line(frame, (gx1c, gy2c-1), (gx2c, gy2c-1), (180, 210, 225), 2)

        # Diced pieces: one column per cut, each column has n pieces (one per strip)
        for cut_i in range(h_done):
            col_cx = right_clip + cut_i * dice_cut_w + dice_cut_w // 2
            for strip_i in range(n):
                shift_i  = strip_i * gap - total_spread // 2
                piece_cy = cy - r + strip_i * sh + sh // 2 + shift_i
                rng2 = np.random.default_rng(seed=cut_i * 7 + strip_i)
                px   = col_cx + int(rng2.integers(-8, 8))
                py   = piece_cy + int(rng2.integers(-sh // 4, sh // 4))
                ang  = float(rng2.uniform(0, math.pi))
                w2, h2 = 16, 12
                ca, sa = math.cos(ang), math.sin(ang)
                pts = np.array([
                    [px + int(-w2*ca + h2*sa), py + int(-w2*sa - h2*ca)],
                    [px + int( w2*ca + h2*sa), py + int( w2*sa - h2*ca)],
                    [px + int( w2*ca - h2*sa), py + int( w2*sa + h2*ca)],
                    [px + int(-w2*ca - h2*sa), py + int(-w2*sa + h2*ca)],
                ], dtype=np.int32)
                cv2.fillPoly(frame, [pts + 2], (22, 20, 18))
                cv2.fillPoly(frame, [pts], (215, 235, 245))
                cv2.polylines(frame, [pts], True, (155, 175, 185), 1)

        # Blue guideline at the next cut position
        if h_done < n:
            guide_x = right_clip   # = cx + r - h_done * dice_cut_w
            top_y   = cy - r - total_spread // 2 - 8
            bot_y   = cy + r + total_spread // 2 + 8
            # Pulsing brightness
            import time as _t
            pulse = int(80 * abs(math.sin(_t.time() * 4)))
            gcol  = (200 + pulse, 120, 30)   # bright blue (BGR)
            cv2.line(frame, (guide_x, max(0, top_y)),
                     (guide_x, min(fh, bot_y)), gcol, 3, cv2.LINE_AA)
            # Arrows pointing to the line
            for ay in [cy - r//2, cy, cy + r//2]:
                cv2.arrowedLine(frame, (guide_x + 40, ay), (guide_x + 4, ay),
                                gcol, 2, tipLength=0.4)

        # Green shoots on the TOP strip
        for ox, ly2 in [(-14, -40), (0, -52), (18, -44)]:
            cv2.line(frame, (cx, cy - r + sh // 2 - total_spread // 2 + 8),
                     (cx + ox, cy - r + sh // 2 - total_spread // 2 + ly2),
                     (28, 140, 38), 4)

    def _init_bowl_pieces(self):
        """Scatter onion piece objects on the cutting board (right side)."""
        rng = np.random.default_rng(seed=13)
        self._bd_piece_pos  = []
        self._bd_piece_vel  = []
        self._bd_piece_ang  = []
        self._bd_piece_avel = []
        self._bd_in_bowl    = []
        for i in range(_BOWL_PIECE_COUNT):
            px = _PIECES_CX + int(rng.integers(-140, 140))
            py = _PIECES_CY + int(rng.integers(-80, 80))
            self._bd_piece_pos.append([float(px), float(py)])
            self._bd_piece_vel.append([0.0, 0.0])
            self._bd_piece_ang.append(float(rng.uniform(0, 6.28)))
            self._bd_piece_avel.append(0.0)
            self._bd_in_bowl.append(False)
        self._bd_prev_knife  = None
        self._bowl_knife_pos  = [_PIECES_CX + 120, _PIECES_CY - 20]
        self._bowl_knife_hand = -1

    def _do_bowl_drop(self, hands):
        # ── Step A: grab the knife ────────────────────────────────────────────
        if self._bowl_knife_hand == -1:
            for i, hand in enumerate(hands):
                if not hand.detected or not hand.gripped:
                    continue
                dx = hand.screen_x - self._bowl_knife_pos[0]
                dy = hand.screen_y - self._bowl_knife_pos[1]
                if dx * dx + dy * dy < GRAB_RADIUS ** 2:
                    self._bowl_knife_hand = i
                    self._bd_prev_knife   = None
                    self._flash_event('GRAB!', (0, 220, 255), 8)
                    return ['grab']
            # Update piece physics even without knife (friction slows them)
            self._update_bd_physics(0.0, 0.0)
            return []

        # ── Step B: verify hand still gripping ───────────────────────────────
        if self._bowl_knife_hand >= len(hands):
            self._bowl_knife_hand = -1
            return []
        hand = hands[self._bowl_knife_hand]
        if not hand.detected or not hand.gripped:
            self._try_reattach(hands, self._bowl_knife_pos, 'bowl_knife')
            if self._held_tool == 'bowl_knife':
                self._bowl_knife_hand = self._held_by
                self._held_tool = None
            else:
                self._bowl_knife_hand = -1
                self._bd_prev_knife   = None
            return []

        # ── Step C: knife follows hand ────────────────────────────────────────
        kx = float(hand.screen_x)
        ky = float(hand.screen_y)
        self._bowl_knife_pos = [kx, ky]

        # Knife velocity this frame
        if self._bd_prev_knife is not None:
            dkx = (kx - self._bd_prev_knife[0])
            dky = (ky - self._bd_prev_knife[1])
        else:
            dkx = dky = 0.0
        self._bd_prev_knife = (kx, ky)

        # ── Step D: push nearby pieces with knife ────────────────────────────
        self._update_bd_physics(kx, ky, dkx, dky)

        # ── Step E: check completion ─────────────────────────────────────────
        n_in = sum(self._bd_in_bowl)
        if n_in >= _BOWL_PIECE_GOAL:
            self._flash_event('DONE!', (255, 200, 50), 22)
            self._phase    = 'intro_fry'
            self._inter_t0 = 0.0
            return ['grab']
        return []

    def _update_bd_physics(self, kx=0.0, ky=0.0, dkx=0.0, dky=0.0):
        """Apply knife push force to nearby pieces, then integrate physics."""
        for i in range(len(self._bd_piece_pos)):
            if self._bd_in_bowl[i]:
                continue
            px, py = self._bd_piece_pos[i]

            # Knife push: if knife close enough and moving
            if (abs(dkx) + abs(dky)) > 0.5:
                dx   = px - kx
                dy   = py - ky
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < _KNIFE_PUSH_R:
                    falloff = 1.0 - dist / _KNIFE_PUSH_R
                    self._bd_piece_vel[i][0]  += dkx * falloff * 0.7
                    self._bd_piece_vel[i][1]  += dky * falloff * 0.7
                    self._bd_piece_avel[i]    += (dkx - dky) * falloff * 0.04

            # Friction
            self._bd_piece_vel[i][0]  *= 0.88
            self._bd_piece_vel[i][1]  *= 0.88
            self._bd_piece_avel[i]    *= 0.85

            # Integrate
            self._bd_piece_pos[i][0]  += self._bd_piece_vel[i][0]
            self._bd_piece_pos[i][1]  += self._bd_piece_vel[i][1]
            self._bd_piece_ang[i]     += self._bd_piece_avel[i]

            # Board boundary: clamp y and x (right wall = 1220, bounce back)
            self._bd_piece_pos[i][1] = max(355.0, min(self._bd_piece_pos[i][1], 510.0))
            if self._bd_piece_pos[i][0] > 1210.0:
                self._bd_piece_pos[i][0] = 1210.0
                self._bd_piece_vel[i][0] = -abs(self._bd_piece_vel[i][0]) * 0.6

            # Bowl collection: crosses the left boundary
            if self._bd_piece_pos[i][0] < _BOWL_COLLECT_X:
                self._bd_in_bowl[i] = True

    # ── Drawing: Onion cut scene ──────────────────────────────────────────────

    def _draw_big_onion(self, frame, cx, cy, peel_frac=0.0, slices=0):
        r  = _ONB_R
        fh, fw = frame.shape[:2]

        # ── Step 1: draw complete onion onto a canvas ─────────────────────────
        canvas = frame.copy()
        cv2.circle(canvas, (cx, cy), r, (220, 235, 242), -1)
        cv2.circle(canvas, (cx, cy), int(r * 0.68), (205, 222, 232), 2)
        cv2.circle(canvas, (cx, cy), int(r * 0.36), (188, 210, 222), 2)
        cv2.circle(canvas, (cx, cy), max(4, int(r * 0.10)), (165, 196, 212), -1)
        if peel_frac < 1.0:
            alpha_s = 1.0 - peel_frac
            skin_r  = int(r * 0.97)
            for i in range(10):
                a0 = i * 36 - 5
                a1 = a0 + int(32 * alpha_s)
                if a1 > a0:
                    cv2.ellipse(canvas, (cx, cy), (skin_r, skin_r),
                                0, a0, a1, (48, 98, 148), int(skin_r * 0.28))
            cv2.circle(canvas, (cx, cy), skin_r, (32, 72, 115), 3)
        cv2.circle(canvas, (cx, cy), r, (28, 68, 108), 4)

        if slices == 0:
            # No cuts: blit the onion region straight to frame
            y1 = max(0, cy - r - 2);  y2 = min(fh, cy + r + 2)
            x1 = max(0, cx - r - 2);  x2 = min(fw, cx + r + 2)
            frame[y1:y2, x1:x2] = canvas[y1:y2, x1:x2]
            shoots_cx = cx
        else:
            # ── Step 2: cut canvas into vertical strips and spread them ───────
            n   = slices + 1
            sw  = (2 * r) // n                                   # strip width
            gap = max(4, int(r * 0.18 * slices / _SLICE_NEEDED)) # gap per cut
            total_spread = gap * (n - 1)
            y1  = max(0, cy - r - 2);  y2 = min(fh, cy + r + 2)

            for i in range(n):
                src_x1 = cx - r + i * sw
                src_x2 = cx - r + (i + 1) * sw
                shift  = i * gap - total_spread // 2
                dst_x1 = src_x1 + shift
                dst_x2 = src_x2 + shift

                s1 = max(0, src_x1);  s2 = min(fw, src_x2)
                d1 = max(0, dst_x1);  d2 = min(fw, dst_x2)
                col = min(s2 - s1, d2 - d1)
                if col > 0 and y2 > y1:
                    frame[y1:y2, d1:d1 + col] = canvas[y1:y2, s1:s1 + col]

            # ── Step 3: fill gaps with white cut-face ─────────────────────────
            for i in range(n - 1):
                shift_i  = i * gap - total_spread // 2
                gx1      = cx - r + (i + 1) * sw + shift_i        # right edge of slice i
                gx2      = gx1 + gap
                orig_cut = cx - r + (i + 1) * sw                  # cut x in original
                dist     = abs(orig_cut - cx)
                if dist < r:
                    h_half = int(math.sqrt(r * r - dist * dist))
                    gx1c   = max(0, gx1);  gx2c = min(fw, gx2)
                    gy1    = max(0, cy - h_half);  gy2 = min(fh, cy + h_half)
                    if gx2c > gx1c and gy2 > gy1:
                        cv2.rectangle(frame, (gx1c, gy1), (gx2c, gy2), (245, 252, 255), -1)
                        cv2.line(frame, (gx1c, gy1), (gx1c, gy2), (180, 210, 225), 2)
                        cv2.line(frame, (gx2c-1, gy1), (gx2c-1, gy2), (180, 210, 225), 2)

            # shoots on the middle strip
            mid_i     = n // 2
            sw_       = (2 * r) // n
            shift_mid = mid_i * gap - total_spread // 2
            shoots_cx = cx - r + mid_i * sw_ + sw_ // 2 + shift_mid

        # ── Green shoots ──────────────────────────────────────────────────────
        for ox, ly2_off in [(-14, -50), (0, -65), (18, -55)]:
            cv2.line(frame, (shoots_cx, cy - r + 12),
                     (shoots_cx + ox, cy - r + ly2_off), (28, 140, 38), 4)

    def _draw_onion_cut(self, frame):
        p  = self._phase
        fw = frame.shape[1]

        fh = frame.shape[0]
        board_top = 345

        # Floor background — warm orange/wood tone with horizontal grain
        cv2.rectangle(frame, (0, board_top), (fw, fh), (55, 130, 210), -1)
        for gy in range(board_top + 20, fh, 30):
            cv2.line(frame, (0, gy), (fw, gy), (45, 115, 195), 1)
        cv2.line(frame, (0, board_top), (fw, board_top), (35, 100, 178), 3)

        if p == 'bowl_drop':
            # bowl_drop: only right-side cutting board (left area is open floor for bowl)
            bx, by = 460, board_top + 12
            bw, bh = fw - bx - 30, fh - by - 12
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (235, 242, 248), -1)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (180, 195, 210), 3)
            # Hanging hole on right edge
            cv2.ellipse(frame, (bx + bw - 28, by + bh // 2),
                        (10, 28), 0, 0, 360, (180, 195, 210), -1)

            # Bowl on the LEFT side
            self._draw_bowl_right(frame)
        else:
            # place_onion / peel_skin / slice_onion: full cutting board across bottom
            cb_x1, cb_y1 = 80, board_top + 12
            cb_x2, cb_y2 = fw - 80, fh - 12
            cv2.rectangle(frame, (cb_x1, cb_y1), (cb_x2, cb_y2), (235, 242, 248), -1)
            cv2.rectangle(frame, (cb_x1, cb_y1), (cb_x2, cb_y2), (180, 195, 210), 3)
            # Hanging hole on right edge
            cv2.ellipse(frame, (cb_x2 - 28, (cb_y1 + cb_y2) // 2),
                        (10, 28), 0, 0, 360, (180, 195, 210), -1)

            # Onion body
            onion_cx  = fw // 2
            onion_cy  = int(self._onb_y) if p == 'place_onion' else _ONB_BOARD_Y
            peel_frac = min(self._peel_done / _PEEL_NEEDED, 1.0)
            if p == 'dice_onion':
                self._draw_diced_onion(frame, onion_cx, onion_cy, peel_frac,
                                       _SLICE_NEEDED, self._dice_done)
            else:
                slices = self._slice_done if p == 'slice_onion' else 0
                self._draw_big_onion(frame, onion_cx, onion_cy, peel_frac, slices)

        # Knife (slice phase)
        if p == 'slice_onion':
            kx = int(self._slice_knife_pos[0])
            ky = int(self._slice_knife_pos[1])
            overlay(frame, self._knife_spr, kx, ky, size=195)
            if self._slice_knife_hand == -1:
                _grab_ring(frame, self._slice_knife_pos)

        if p == 'dice_onion':
            kx = int(self._dice_knife_pos[0])
            ky = int(self._dice_knife_pos[1])
            overlay(frame, self._knife_spr, kx, ky, size=195)
            if self._dice_knife_hand == -1:
                _grab_ring(frame, self._dice_knife_pos)

        # Pieces + knife (bowl_drop phase)
        if p == 'bowl_drop':

            # Individual piece objects at their physics positions
            for i in range(len(self._bd_piece_pos)):
                if self._bd_in_bowl[i]:
                    continue
                px = int(self._bd_piece_pos[i][0])
                py = int(self._bd_piece_pos[i][1])
                ang = float(self._bd_piece_ang[i])
                w2, h2 = 22, 16
                ca, sa = math.cos(ang), math.sin(ang)
                pts = np.array([
                    [px + int(-w2*ca + h2*sa), py + int(-w2*sa - h2*ca)],
                    [px + int( w2*ca + h2*sa), py + int( w2*sa - h2*ca)],
                    [px + int( w2*ca - h2*sa), py + int( w2*sa + h2*ca)],
                    [px + int(-w2*ca - h2*sa), py + int(-w2*sa + h2*ca)],
                ], dtype=np.int32)
                cv2.fillPoly(frame, [pts + 2], (22, 20, 18))
                cv2.fillPoly(frame, [pts], (215, 235, 245))
                cv2.polylines(frame, [pts], True, (155, 175, 185), 1)

            # Progress label
            n_in = sum(self._bd_in_bowl)
            _shadow_text(frame, f'{n_in}/{_BOWL_PIECE_GOAL}',
                         _BOWL2_CX, _BOWL2_CY - 130, 1.0, (255, 230, 100), 2, center=True)

            # Sweep arrow hint (when knife is grabbed)
            if self._bowl_knife_hand != -1:
                cv2.arrowedLine(frame, (780, 390), (420, 390),
                                (255, 200, 50), 3, tipLength=0.06)

            # Knife
            kx = int(self._bowl_knife_pos[0])
            ky = int(self._bowl_knife_pos[1])
            overlay(frame, self._knife_spr, kx, ky, size=195)
            if self._bowl_knife_hand == -1:
                _grab_ring(frame, self._bowl_knife_pos)

        # place_onion: grab ring + down arrow
        if p == 'place_onion':
            if not self._onb_grabbed:
                _grab_ring(frame, [onion_cx, int(self._onb_y)])
            else:
                ay = int(self._onb_y) - 85
                cv2.arrowedLine(frame, (onion_cx, ay), (onion_cx, ay + 70),
                                (0, 220, 255), 4, tipLength=0.28)

    def _draw_bowl_right(self, frame):
        """Large side-view blue bowl (Cooking Mama style)."""
        cx, cy = _BOWL2_CX, _BOWL2_CY
        r = 110   # bowl radius

        # Drop shadow
        cv2.ellipse(frame, (cx + 12, cy + 16), (r, int(r * 0.38)),
                    0, 0, 360, (28, 26, 22), -1)
        # Outer bowl shape (slightly squashed ellipse, deep blue)
        cv2.ellipse(frame, (cx, cy), (r, int(r * 0.72)), 0, 0, 360, (158, 120, 60), -1)
        # Bowl interior (lighter sky blue)
        cv2.ellipse(frame, (cx, cy - int(r * 0.12)), (int(r * 0.84), int(r * 0.55)),
                    0, 0, 360, (210, 185, 110), -1)
        # Inner depth circle
        cv2.ellipse(frame, (cx, cy - int(r * 0.18)), (int(r * 0.52), int(r * 0.33)),
                    0, 0, 360, (190, 162, 90), -1)
        # Rim highlight
        cv2.ellipse(frame, (cx, cy), (r, int(r * 0.72)),
                    0, 210, 330, (220, 200, 155), 5)
        # Dark outline
        cv2.ellipse(frame, (cx, cy), (r, int(r * 0.72)),
                    0, 0, 360, (100, 72, 28), 3)

    def _draw_onion_pieces(self, frame, cx, cy, scale=1.0):
        offsets = [(-50, -30), (40, -20), (0, 15), (-30, 35),
                   (55, 25), (10, -45), (-55, 10), (38, 45)]
        n = max(1, int(len(offsets) * scale))
        for i, (ox, oy) in enumerate(offsets[:n]):
            px, py = cx + ox, cy + oy
            ang    = i * 0.7
            w2, h2 = 18, 13
            ca, sa = math.cos(ang), math.sin(ang)
            pts = np.array([
                [px + int(-w2*ca + h2*sa), py + int(-w2*sa - h2*ca)],
                [px + int( w2*ca + h2*sa), py + int( w2*sa - h2*ca)],
                [px + int( w2*ca - h2*sa), py + int( w2*sa + h2*ca)],
                [px + int(-w2*ca - h2*sa), py + int(-w2*sa + h2*ca)],
            ], dtype=np.int32)
            cv2.fillPoly(frame, [pts + 2], (22, 20, 18))
            cv2.fillPoly(frame, [pts], (215, 235, 245))
            cv2.polylines(frame, [pts], True, (155, 175, 185), 1)

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


    def _draw_intermission(self, frame):
        from .game_manager import _draw_cooking_papa
        h, w = frame.shape[:2]
        _panel(frame, 0, 0, w, h, color=(20, 18, 16), alpha=0.62)

        title, sub = _INTRO_TEXT.get(self._phase, ('', ''))
        _draw_cooking_papa(frame, w // 4, h // 2 + 20, size=150)

        tcx = w * 9 // 16
        _shadow_text(frame, title, tcx, h // 2 - 26,
                     1.55, (80, 220, 255), 3, center=True)
        _shadow_text(frame, sub, tcx, h // 2 + 28,
                     0.68, (225, 225, 240), 1, center=True)

        # Progress dots show the title card's countdown
        n_dots = 5
        elapsed = 0.0 if self._inter_t0 == 0.0 else time.time() - self._inter_t0
        filled = int(n_dots * min(1.0, elapsed / _INTRO_DURATION))
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

        # ── Plate (under everything) ──────────────────────────────────────────
        px, py = cx, cy + 35
        cv2.ellipse(frame, (px, py), (340, 150), 0, 0, 360, (90, 90, 95), -1)
        cv2.ellipse(frame, (px, py), (340, 150), 0, 0, 360, (60, 60, 65), 3)
        cv2.ellipse(frame, (px, py - 8), (278, 118), 0, 0, 360, (235, 235, 238), -1)
        cv2.ellipse(frame, (px, py - 8), (278, 118), 0, 0, 360, (200, 200, 205), 2)
        cv2.ellipse(frame, (px, py - 8), (190, 78), 0, 0, 360, (210, 212, 218), 1)

        # ── Steak (center of plate) ───────────────────────────────────────────
        s = 140
        sx, sy = px + 35, py - 25
        pts = np.array([
            [sx - s,              sy + int(s*0.2)],
            [sx - int(s*0.75),    sy - int(s*0.6)],
            [sx - int(s*0.15),    sy - int(s*0.75)],
            [sx + int(s*0.45),    sy - int(s*0.65)],
            [sx + s,              sy - int(s*0.15)],
            [sx + int(s*1.05),    sy + int(s*0.3)],
            [sx + int(s*0.5),     sy + int(s*0.65)],
            [sx - int(s*0.25),    sy + int(s*0.7)],
            [sx - int(s*0.8),     sy + int(s*0.5)],
        ], np.int32)
        cv2.fillPoly(frame, [pts + np.array([4, 5])], (18, 15, 12))
        cv2.fillPoly(frame, [pts], (55, 78, 158))
        # Grill marks
        for gx in range(sx - int(s*0.7), sx + int(s*0.9), int(s*0.38)):
            cv2.line(frame, (gx - int(s*0.18), sy + int(s*0.3)),
                     (gx + int(s*0.18), sy - int(s*0.3)), (30, 48, 100), 5)
        cv2.polylines(frame, [pts], True, (35, 55, 110), 3)

        # ── Paprika & lemon, side by side on the left of the plate ───────────
        gx0, gy0 = px - 235, py + 5

        # Paprika (bell pepper), behind/left
        ppx, ppy, pps = gx0 - 20, gy0 + 10, 78
        body = np.array([
            [ppx - int(pps*0.62), ppy - int(pps*0.05)],
            [ppx - int(pps*0.74), ppy + int(pps*0.55)],
            [ppx - int(pps*0.30), ppy + int(pps*0.92)],
            [ppx + int(pps*0.18), ppy + int(pps*0.95)],
            [ppx + int(pps*0.66), ppy + int(pps*0.50)],
            [ppx + int(pps*0.58), ppy - int(pps*0.10)],
            [ppx + int(pps*0.18), ppy - int(pps*0.40)],
            [ppx - int(pps*0.18), ppy - int(pps*0.38)],
        ], np.int32)
        cv2.fillPoly(frame, [body + np.array([3, 4])], (15, 20, 60))
        cv2.fillPoly(frame, [body], (40, 55, 215))
        # glossy highlight
        cv2.ellipse(frame, (ppx - int(pps*0.28), ppy + int(pps*0.05)),
                    (int(pps*0.16), int(pps*0.30)), 20, 0, 360, (110, 130, 245), -1)
        cv2.polylines(frame, [body], True, (25, 35, 150), 2)
        # stem
        stem = np.array([
            [ppx - int(pps*0.16), ppy - int(pps*0.36)],
            [ppx + int(pps*0.16), ppy - int(pps*0.36)],
            [ppx + int(pps*0.10), ppy - int(pps*0.62)],
            [ppx - int(pps*0.10), ppy - int(pps*0.62)],
        ], np.int32)
        cv2.fillPoly(frame, [stem], (35, 110, 60))
        cv2.polylines(frame, [stem], True, (20, 80, 42), 2)

        # Lemon slice, slightly forward/right of the paprika
        lx, ly, lr = gx0 + 95, gy0 + 35, 62
        cv2.circle(frame, (lx+2, ly+3), lr+3, (12, 14, 12), -1)
        cv2.circle(frame, (lx, ly), lr, (0, 220, 240), -1)           # yellow peel
        cv2.circle(frame, (lx, ly), int(lr*0.82), (15, 240, 255), -1) # bright flesh
        for k in range(6):
            a = math.pi / 3 * k
            cv2.line(frame,
                     (int(lx + 6*math.cos(a)), int(ly + 6*math.sin(a))),
                     (int(lx + lr*0.80*math.cos(a)), int(ly + lr*0.80*math.sin(a))),
                     (8, 195, 230), 2)
        cv2.circle(frame, (lx, ly), int(lr*0.82), (5, 170, 215), 3)
        cv2.circle(frame, (lx, ly), lr, (0, 155, 190), 4)

        # ── Title text ────────────────────────────────────────────────────────
        _shadow_text(frame, 'Good Job!', cx, cy - 268,
                     1.0, (255, 255, 255), 2, center=True)
        _shadow_text(frame, 'Salisbury Steak!', cx, cy - 210,
                     1.4, (30, 190, 255), 3, center=True)


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


def _draw_dir_arrow(frame, cx, cy, direction, size=65, color=(80, 220, 255)):
    """Draw a filled directional arrow centered at (cx, cy)."""
    s = size
    h = s // 2   # shaft half-width
    n = s // 4   # neck half-width
    if direction == 'up':
        pts = [(cx, cy - s), (cx - h, cy), (cx - n, cy), (cx - n, cy + s),
               (cx + n, cy + s), (cx + n, cy), (cx + h, cy)]
    elif direction == 'down':
        pts = [(cx, cy + s), (cx + h, cy), (cx + n, cy), (cx + n, cy - s),
               (cx - n, cy - s), (cx - n, cy), (cx - h, cy)]
    elif direction == 'left':
        pts = [(cx - s, cy), (cx, cy - h), (cx, cy - n), (cx + s, cy - n),
               (cx + s, cy + n), (cx, cy + n), (cx, cy + h)]
    else:  # right
        pts = [(cx + s, cy), (cx, cy + h), (cx, cy + n), (cx - s, cy + n),
               (cx - s, cy - n), (cx, cy - n), (cx, cy - h)]
    arr = np.array(pts, dtype=np.int32)
    cv2.fillPoly(frame, [arr + np.array([3, 4])], (20, 18, 16))   # shadow
    cv2.fillPoly(frame, [arr], color)
    cv2.polylines(frame, [arr], True, (255, 255, 255), 2)


def _draw_spatula(frame, cx, cy):
    """Draw a slotted turner (black plastic + steel neck) centered at (cx, cy), pointing upward."""
    cx, cy = int(cx), int(cy)
    PLASTIC    = (42, 36, 32)     # near-black plastic (BGR)
    PLASTIC_HI = (74, 64, 58)
    PLASTIC_DK = (22, 18, 16)
    STEEL      = (212, 212, 216)
    STEEL_DK   = (150, 150, 156)

    # Handle (bottom): dark plastic body tapering toward the butt, with a hang-hole
    hw, hh = 12, 72
    hx0, hy0 = cx - hw, cy - hh
    hx1, hy1 = cx + hw, cy + hh
    pts_h = np.array([
        [hx0 + 3, hy0], [hx1 - 3, hy0],
        [hx1,     hy1 - 18], [cx + hw - 2, hy1],
        [cx - hw + 2, hy1], [hx0,         hy1 - 18],
    ], np.int32)
    cv2.fillPoly(frame, [pts_h], PLASTIC)
    cv2.polylines(frame, [pts_h], True, PLASTIC_DK, 2)
    cv2.line(frame, (hx0 + 4, hy0 + 8), (hx0 + 4, hy1 - 22), PLASTIC_HI, 2)
    cv2.ellipse(frame, (cx, hy1 - 15), (6, 11), 0, 0, 360, (250, 250, 250), -1)
    cv2.ellipse(frame, (cx, hy1 - 15), (6, 11), 0, 0, 360, PLASTIC_DK, 2)

    # Steel neck connecting handle to paddle
    nw, nh = 6, 22
    cv2.rectangle(frame, (cx - nw, hy0 - nh), (cx + nw, hy0 + 6), STEEL, -1)
    cv2.rectangle(frame, (cx - nw, hy0 - nh), (cx + nw, hy0 + 6), STEEL_DK, 1)
    cv2.line(frame, (cx - 2, hy0 - nh + 2), (cx - 2, hy0 + 4), STEEL_DK, 1)
    cv2.line(frame, (cx + 2, hy0 - nh + 2), (cx + 2, hy0 + 4), (236, 236, 240), 1)

    # Paddle head: wide flat slotted blade with one chamfered corner
    pw, ph = 36, 56
    py1 = hy0 - nh
    py0 = py1 - ph
    pts_p = np.array([
        [cx - pw,      py1], [cx - pw,      py0 + 12],
        [cx - pw + 14, py0], [cx + pw,      py0],
        [cx + pw,      py1],
    ], np.int32)
    cv2.fillPoly(frame, [pts_p], PLASTIC)
    cv2.polylines(frame, [pts_p], True, PLASTIC_DK, 2)
    cv2.line(frame, (cx - pw + 5, py0 + 18), (cx - pw + 5, py1 - 6), PLASTIC_HI, 2)
    # Long slots cut through the blade
    n_slots, slot_w = 5, 5
    span = (pw * 2) - 24
    step = span / n_slots
    for k in range(n_slots):
        sx = cx - pw + 12 + int(k * step + step * 0.5) - slot_w // 2
        cv2.rectangle(frame, (sx, py0 + 14), (sx + slot_w, py1 - 7), (244, 244, 246), -1)


def _demo_fist(frame, cx, cy, size=58, openness=0.0):
    """White cartoon-style hand — openness 0.0=fist, 1.0=fingers fully extended."""
    cx, cy = int(cx), int(cy)
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
