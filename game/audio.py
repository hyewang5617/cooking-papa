"""Sound-effect playback through pygame.mixer.

Each SFX first looks for an audio file in game/sounds/ (e.g. success.wav,
success.mp3, success.ogg — first match wins) and falls back to a synthesized
tone if no file is present. If no audio device is available, every public
function becomes a silent no-op so the game keeps running.

To use your own sounds, just drop files into game/sounds/ named:
  success.*, fail.*, grab.*, stage_clear.*, tick.*
"""
import os
import numpy as np

_SOUND_DIR = os.path.join(os.path.dirname(__file__), 'sounds')
_EXTS = ('.wav', '.ogg', '.mp3')

try:
    import pygame
    pygame.mixer.init()
    _ENABLED = True
    _SAMPLE_RATE = abs(pygame.mixer.get_init()[0])
    print(f'[audio] mixer ready: {pygame.mixer.get_init()}')
except Exception as _exc:
    _ENABLED = False
    _SAMPLE_RATE = 44100
    print(f'[audio] disabled (no sound device?): {_exc}')

_cache = {}


def _find_file(key):
    for ext in _EXTS:
        path = os.path.join(_SOUND_DIR, key + ext)
        if os.path.isfile(path):
            return path
    return None


def _tone(freq, duration, wave='sine', volume=1.0, fade=0.012):
    n = int(_SAMPLE_RATE * duration)
    t = np.arange(n) / _SAMPLE_RATE
    if wave == 'sine':
        w = np.sin(2 * np.pi * freq * t)
    elif wave == 'square':
        w = np.sign(np.sin(2 * np.pi * freq * t))
    elif wave == 'triangle':
        w = 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
    else:
        w = np.sin(2 * np.pi * freq * t)
    fade_n = max(1, int(_SAMPLE_RATE * fade))
    env = np.ones(n)
    env[:fade_n]  = np.linspace(0.0, 1.0, fade_n)
    env[-fade_n:] = np.linspace(1.0, 0.0, fade_n)
    return w * env * volume


def _make_tone_sound(segments):
    """segments: list of (freq, duration, wave, volume) tuples, concatenated."""
    parts = [_tone(*seg) for seg in segments]
    wave = np.concatenate(parts) if len(parts) > 1 else parts[0]
    mono = np.clip(wave * 32767, -32768, 32767).astype(np.int16)
    channels = pygame.mixer.get_init()[2]
    pcm = np.repeat(mono[:, None], channels, axis=1) if channels > 1 else mono
    return pygame.sndarray.make_sound(pcm)


def _sound(key, segments):
    if key not in _cache:
        path = _find_file(key)
        if path is not None:
            try:
                _cache[key] = pygame.mixer.Sound(path)
                print(f'[audio] loaded {key} <- {path}')
            except Exception as exc:
                print(f'[audio] failed to load {path}: {exc} (using synthesized tone)')
                _cache[key] = _make_tone_sound(segments)
        else:
            _cache[key] = _make_tone_sound(segments)
    return _cache[key]


def _play(key, segments):
    if not _ENABLED:
        return
    try:
        _sound(key, segments).play()
    except Exception:
        pass


def play_success():
    _play('success', [
        (523.25, 0.11, 'sine', 0.175),
        (659.25, 0.11, 'sine', 0.175),
        (783.99, 0.20, 'sine', 0.1875),
    ])


def play_fail():
    _play('fail', [
        (220.0, 0.13, 'square', 0.1375),
        (164.81, 0.22, 'square', 0.1375),
    ])


def play_grab():
    _play('grab', [
        (392.0, 0.11, 'triangle', 0.15),
    ])


def play_stage_clear():
    _play('stage_clear', [
        (523.25, 0.13, 'sine', 0.175),
        (659.25, 0.13, 'sine', 0.175),
        (783.99, 0.13, 'sine', 0.175),
        (1046.50, 0.30, 'sine', 0.2),
    ])


def play_tick():
    _play('tick', [
        (880.0, 0.05, 'square', 0.1),
    ])


# ── File-based scene sounds (굽는소리/배경음악/섞는소리/소스짜는소리/짜잔/칼질효과음/휘핑기소리) ──

_loops = {}


def _file_sound(key):
    if key not in _cache:
        path = _find_file(key)
        if path is not None:
            try:
                _cache[key] = pygame.mixer.Sound(path)
                print(f'[audio] loaded {key} <- {path}')
            except Exception as exc:
                print(f'[audio] failed to load {path}: {exc}')
                _cache[key] = None
        else:
            _cache[key] = None
            print(f'[audio] no file found for "{key}" in {_SOUND_DIR}')
    return _cache[key]


def _speed_sound(base_key, speed):
    """Return (and cache) `base_key`'s sound resampled to play at `speed`x (pitch shifts too)."""
    cache_key = (base_key, round(speed, 2))
    if cache_key not in _cache:
        base = _file_sound(base_key)
        if base is None or abs(speed - 1.0) < 1e-6:
            _cache[cache_key] = base
        else:
            try:
                arr = pygame.sndarray.array(base)
                n = arr.shape[0]
                new_n = max(1, int(n / speed))
                idx = np.clip((np.arange(new_n) * speed).astype(np.int64), 0, n - 1)
                _cache[cache_key] = pygame.sndarray.make_sound(arr[idx].copy())
            except Exception as exc:
                print(f'[audio] failed to resample {base_key}@{speed}: {exc}')
                _cache[cache_key] = base
    return _cache[cache_key]


def _trimmed_sound(base_key, skip_seconds):
    """Return (and cache) `base_key`'s sound with the first `skip_seconds` cut off."""
    cache_key = (base_key, 'skip', round(skip_seconds, 2))
    if cache_key not in _cache:
        base = _file_sound(base_key)
        if base is None or skip_seconds <= 0:
            _cache[cache_key] = base
        else:
            try:
                arr = pygame.sndarray.array(base)
                skip_n = int(_SAMPLE_RATE * skip_seconds)
                skip_n = min(skip_n, max(0, arr.shape[0] - 1))
                _cache[cache_key] = pygame.sndarray.make_sound(arr[skip_n:].copy())
            except Exception as exc:
                print(f'[audio] failed to trim {base_key}@{skip_seconds}s: {exc}')
                _cache[cache_key] = base
    return _cache[cache_key]


def _loop(key, active, volume=0.6):
    """Start/stop a looping sound so it plays exactly while `active` is True."""
    if not _ENABLED:
        return
    if active:
        if key not in _loops:
            snd = _file_sound(key)
            if snd is not None:
                snd.set_volume(volume)
                snd.play(loops=-1)
                _loops[key] = snd
    else:
        snd = _loops.pop(key, None)
        if snd is not None:
            snd.stop()


def _play_file(key, volume=0.8):
    if not _ENABLED:
        return
    snd = _file_sound(key)
    if snd is not None:
        snd.set_volume(volume)
        snd.play()


def loop_cooking(active):
    """굽는소리 — 스테이크/팬케이크를 굽는 동안 반복 재생."""
    _loop('굽는소리', active, volume=0.6)


def loop_mixing(active):
    """섞는소리 — 다짐육에 양파를 넣고 섞는 동안 반복 재생."""
    _loop('섞는소리', active, volume=0.6)


def loop_grinding(active):
    """가는소리 — 고기를 가는 동안 반복 재생."""
    _loop('가는 소리', active, volume=0.6)


_knife_loop_active = False
_KNIFE_LOOP_KEY    = '칼질효과음@skip1'


def loop_knife(active):
    """칼질효과음 — 칼이 실제로 움직이며 양파를 손질하는 동안만 반복 재생.

    음원의 1초 지점부터 재생하고, 음량은 기존(0.6)의 1.5배(0.9)로 키운다.
    """
    global _knife_loop_active
    if not _ENABLED:
        return
    if active:
        if not _knife_loop_active:
            snd = _trimmed_sound('칼질효과음', 1.0)
            if snd is not None:
                snd.set_volume(0.9)
                snd.play(loops=-1)
                _loops[_KNIFE_LOOP_KEY] = snd
            _knife_loop_active = True
    else:
        if _knife_loop_active:
            old = _loops.pop(_KNIFE_LOOP_KEY, None)
            if old is not None:
                old.stop()
            _knife_loop_active = False


def loop_syrup(active):
    """소스짜는소리 — 팬케이크 위에 시럽을 그리는 순간에만 반복 재생."""
    _loop('소스짜는소리', active, volume=0.6)


_whisk_loop_key = None


def loop_whisk(active, speed=1.0):
    """휘핑기소리 — 믹싱기가 등장하는 동안 항상 반복 재생.

    speed: 신호등 색에 따른 재생 배속 (초록 1.0 / 노랑 1.5 / 빨강 2.0).
    배속이 바뀌면 기존 루프를 멈추고 새 배속의 사운드로 교체한다.
    """
    global _whisk_loop_key
    if not _ENABLED:
        return
    if active:
        key = ('휘핑기소리', round(speed, 2))
        if _whisk_loop_key != key:
            old = _loops.pop(_whisk_loop_key, None)
            if old is not None:
                old.stop()
            snd = _speed_sound('휘핑기소리', speed)
            if snd is not None:
                snd.set_volume(0.6)
                snd.play(loops=-1)
                _loops[key] = snd
            _whisk_loop_key = key
    else:
        old = _loops.pop(_whisk_loop_key, None)
        if old is not None:
            old.stop()
        _whisk_loop_key = None


def play_tada():
    """짜잔 — 마지막 결과 공개(reveal) 장면 시작 시 한 번 재생."""
    _play_file('짜잔', volume=0.85)


def start_bgm(volume=0.15):
    """배경음악 — 게임 내내 낮은 볼륨으로 반복 재생."""
    if not _ENABLED:
        return
    path = _find_file('배경음악')
    if path is None:
        print(f'[audio] no background music file found in {_SOUND_DIR}')
        return
    try:
        pygame.mixer.music.load(path)
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play(loops=-1)
        print(f'[audio] bgm loaded <- {path}')
    except Exception as exc:
        print(f'[audio] failed to start bgm: {exc}')
