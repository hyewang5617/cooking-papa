"""Synthesized sound-effect playback (no external audio files needed).

All SFX are generated on the fly with numpy waveform synthesis and played
through pygame.mixer. If no audio device is available, every public function
becomes a silent no-op so the game keeps running.
"""
import numpy as np

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


def _make_sound(segments):
    """segments: list of (freq, duration, wave, volume) tuples, concatenated."""
    parts = [_tone(*seg) for seg in segments]
    wave = np.concatenate(parts) if len(parts) > 1 else parts[0]
    mono = np.clip(wave * 32767, -32768, 32767).astype(np.int16)
    channels = pygame.mixer.get_init()[2]
    pcm = np.repeat(mono[:, None], channels, axis=1) if channels > 1 else mono
    return pygame.sndarray.make_sound(pcm)


def _sound(key, segments):
    if key not in _cache:
        _cache[key] = _make_sound(segments)
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
        (523.25, 0.11, 'sine', 0.7),
        (659.25, 0.11, 'sine', 0.7),
        (783.99, 0.20, 'sine', 0.75),
    ])


def play_fail():
    _play('fail', [
        (220.0, 0.13, 'square', 0.55),
        (164.81, 0.22, 'square', 0.55),
    ])


def play_grab():
    _play('grab', [
        (392.0, 0.11, 'triangle', 0.6),
    ])


def play_stage_clear():
    _play('stage_clear', [
        (523.25, 0.13, 'sine', 0.7),
        (659.25, 0.13, 'sine', 0.7),
        (783.99, 0.13, 'sine', 0.7),
        (1046.50, 0.30, 'sine', 0.8),
    ])


def play_tick():
    _play('tick', [
        (880.0, 0.05, 'square', 0.4),
    ])
