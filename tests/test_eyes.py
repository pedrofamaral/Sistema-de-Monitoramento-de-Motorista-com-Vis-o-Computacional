import numpy as np
import pytest

from src.detectors.eyes import DrowsinessDetector, EARCalibrator, average_ear, eye_aspect_ratio
from src.detectors.face_landmarks import LEFT_EYE, RIGHT_EYE

FPS = 30
OPEN, CLOSED = 0.30, 0.10


def eye_points(height, width=10.0):
    """p1..p6 de um olho com a altura e a largura dadas (EAR = altura / largura)."""
    return np.array([(0, 0), (3, -height / 2), (7, -height / 2),
                     (width, 0), (7, height / 2), (3, height / 2)], dtype=np.float32)


def run(detector, ears, start=0.0):
    """Alimenta o detector com uma lista de EARs a 30 FPS; retorna os estados."""
    return [detector.update(ear, start + i / FPS) for i, ear in enumerate(ears)]


def seconds(value, duration):
    return [value] * round(duration * FPS)


def make_detector(**overrides):
    params = dict(ear_threshold=0.20, hysteresis=0.03, closed_seconds=2.0, face_lost_grace=1.0)
    params.update(overrides)
    return DrowsinessDetector(**params)


# --- EAR -------------------------------------------------------------------

def test_ear_is_height_over_width():
    assert eye_aspect_ratio(eye_points(height=4)) == pytest.approx(0.4)
    assert eye_aspect_ratio(eye_points(height=0)) == pytest.approx(0.0)


def test_average_ear_uses_both_eyes():
    landmarks = np.zeros((478, 2), dtype=np.float32)
    landmarks[list(RIGHT_EYE)] = eye_points(height=2)          # EAR 0,2
    landmarks[list(LEFT_EYE)] = eye_points(height=4) + (100, 0)  # EAR 0,4
    assert average_ear(landmarks) == pytest.approx(0.3)


# --- Sonolência ------------------------------------------------------------

def test_blink_does_not_trigger():
    statuses = run(make_detector(), seconds(OPEN, 1) + seconds(CLOSED, 0.3) + seconds(OPEN, 1))
    assert not any(s.drowsy for s in statuses)
    assert any(s.eyes_closed for s in statuses)  # a piscada foi vista, só não virou alerta


def test_long_closure_triggers_after_threshold_and_stops_when_eyes_open():
    statuses = run(make_detector(), seconds(OPEN, 1) + seconds(CLOSED, 2.5) + seconds(OPEN, 0.5))
    closed = statuses[30:105]
    first_alert = next(i for i, s in enumerate(closed) if s.drowsy)
    assert first_alert / FPS == pytest.approx(2.0, abs=1 / FPS)
    assert not statuses[-1].drowsy and statuses[-1].closed_for == 0


def test_hysteresis_keeps_eyes_closed_near_threshold():
    # Depois de fechar, um EAR oscilando entre 0,19 e 0,21 (abaixo de 0,20 + 0,03)
    # não reabre o olho, então o tempo continua contando.
    oscillating = [0.19, 0.21] * FPS  # 2 s
    statuses = run(make_detector(), seconds(CLOSED, 0.2) + oscillating)
    assert all(s.eyes_closed for s in statuses)
    assert statuses[-1].drowsy


def test_ear_above_hysteresis_band_reopens():
    statuses = run(make_detector(), seconds(CLOSED, 0.5) + [0.24])
    assert not statuses[-1].eyes_closed


def test_short_face_loss_keeps_counting():
    # Olhos fechados por 1,5 s e o rosto some por 0,8 s (cabeça caiu): o alerta sai.
    statuses = run(make_detector(), seconds(CLOSED, 1.5) + seconds(None, 0.8))
    assert statuses[-1].drowsy


def test_long_face_loss_resets():
    statuses = run(make_detector(), seconds(CLOSED, 1.5) + seconds(None, 2.0))
    assert not statuses[-1].eyes_closed and not statuses[-1].drowsy


# --- Calibração ------------------------------------------------------------

def test_calibration_uses_median_of_open_eyes():
    calibrator = EARCalibrator(seconds=3.0, ratio=0.65)
    ears = seconds(0.36, 1.4) + seconds(0.05, 0.2) + seconds(None, 0.5) + seconds(0.36, 1.8)
    results = [calibrator.update(ear, i / FPS) for i, ear in enumerate(ears)]
    threshold = next(r for r in results if r is not None)
    assert threshold == pytest.approx(0.65 * 0.36)  # a piscada (0,05) não afeta a mediana
    assert calibrator.progress == 1.0


def test_calibration_waits_for_a_face():
    calibrator = EARCalibrator(seconds=1.0, ratio=0.65)
    assert all(calibrator.update(None, i / FPS) is None for i in range(90))
    assert calibrator.progress == 0.0
