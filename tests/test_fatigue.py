import numpy as np
import pytest

from src.detectors.face_landmarks import MOUTH_CORNERS, MOUTH_PAIRS
from src.detectors.fatigue import FatigueMonitor, PerclosMeter
from src.detectors.mouth import YawnDetector, mouth_aspect_ratio


def blinking_pattern(seconds, closed_fraction, fps, period=2.0):
    """Olhos fechados `closed_fraction` do tempo, em ciclos de `period` segundos."""
    frames_per_cycle = round(period * fps)
    closed_frames = round(closed_fraction * frames_per_cycle)
    return [(i % frames_per_cycle) < closed_frames for i in range(round(seconds * fps))]


def make_monitor(**overrides):
    params = dict(perclos_window=60, perclos_threshold=0.15, perclos_hysteresis=0.03,
                  min_observed_seconds=30, yawn_window=300, yawn_count=3)
    params.update(overrides)
    return FatigueMonitor(**params)


def feed(monitor, eyes, fps=30, start=0.0, yawns_at=()):
    status = None
    for i, closed in enumerate(eyes):
        t = start + i / fps
        status = monitor.update(closed, any(abs(t - y) < 0.5 / fps for y in yawns_at), t)
    return status


# --- PERCLOS ---------------------------------------------------------------

@pytest.mark.parametrize("fps", [10, 30])
def test_perclos_is_fraction_of_time_and_independent_of_fps(fps):
    meter = PerclosMeter(window_seconds=60)
    for i, closed in enumerate(blinking_pattern(60, 0.20, fps)):
        meter.update(closed, i / fps)
    assert meter.value == pytest.approx(0.20, abs=0.02)


def test_perclos_forgets_samples_outside_window():
    meter = PerclosMeter(window_seconds=60)
    eyes = [True] * (30 * 30) + [False] * (90 * 30)  # 30 s fechados, depois 90 s abertos
    for i, closed in enumerate(eyes):
        meter.update(closed, i / 30)
    assert meter.value == 0.0
    assert meter.observed_seconds == pytest.approx(60, abs=0.1)


def test_perclos_ignores_frames_without_face():
    meter = PerclosMeter(window_seconds=60)
    eyes = [True] * 60 + [None] * 600 + [False] * 60  # 2 s fechados, 20 s sem rosto, 2 s abertos
    for i, closed in enumerate(eyes):
        meter.update(closed, i / 30)
    assert meter.value == pytest.approx(0.5, abs=0.05)
    assert meter.observed_seconds == pytest.approx(4, abs=0.2)


# --- Fadiga ------------------------------------------------------------------

def test_no_fatigue_before_enough_data():
    status = feed(make_monitor(), [True] * (10 * 30))  # 10 s de olhos fechados
    assert status.perclos == pytest.approx(1.0) and not status.ready and not status.fatigued


def test_high_perclos_triggers_fatigue():
    status = feed(make_monitor(), blinking_pattern(60, 0.20, 30))
    assert status.ready and status.fatigued


def test_normal_blinking_is_not_fatigue():
    status = feed(make_monitor(), blinking_pattern(60, 0.05, 30))
    assert status.ready and not status.fatigued


def test_perclos_hysteresis():
    monitor = make_monitor()
    feed(monitor, blinking_pattern(60, 0.20, 30))
    status = feed(monitor, blinking_pattern(60, 0.13, 30), start=60)  # entre 12% e 15%
    assert status.fatigued  # continua: só sai abaixo de 15% - 3%
    status = feed(monitor, blinking_pattern(60, 0.05, 30), start=120)
    assert not status.fatigued


def test_yawns_trigger_fatigue_and_expire():
    monitor = make_monitor(yawn_window=300, yawn_count=3)
    status = feed(monitor, [False] * (100 * 30), yawns_at=(10, 40, 70))
    assert status.yawns == 3 and status.fatigued
    status = feed(monitor, [False] * (300 * 30), start=100)
    assert status.yawns == 0 and not status.fatigued


# --- Boca e bocejo -----------------------------------------------------------

def test_mar_uses_inner_lips():
    landmarks = np.zeros((478, 2), dtype=np.float32)
    left, right = MOUTH_CORNERS
    landmarks[left], landmarks[right] = (0, 0), (40, 0)
    for k, (top, bottom) in enumerate(MOUTH_PAIRS):
        landmarks[top], landmarks[bottom] = (10 + 10 * k, -12), (10 + 10 * k, 12)  # 24 de altura
    assert mouth_aspect_ratio(landmarks) == pytest.approx(0.6)


def test_talking_is_not_a_yawn_and_each_yawn_counts_once():
    detector = YawnDetector(mar_threshold=0.6, min_seconds=1.0)
    talking = ([0.7] * 6 + [0.2] * 6) * 10            # aberturas de 0,2 s
    yawn = [0.2] * 15 + [0.9] * 75 + [0.1] * 15       # boca aberta por 2,5 s
    statuses = [detector.update(mar, i / 30) for i, mar in enumerate(talking + yawn)]
    assert sum(s.yawn_started for s in statuses) == 1
    assert not any(s.yawning for s in statuses[:len(talking)])
