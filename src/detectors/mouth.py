"""Abertura da boca (MAR) e detecção de bocejos."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .face_landmarks import MOUTH_CORNERS, MOUTH_PAIRS
from .temporal import PersistenceFilter


def mouth_aspect_ratio(landmarks: np.ndarray) -> float:
    """MAR: abertura média dos lábios internos dividida pela largura da boca."""
    left, right = MOUTH_CORNERS
    width = np.linalg.norm(landmarks[left] - landmarks[right])
    if width == 0:
        return 0.0
    height = np.mean([np.linalg.norm(landmarks[top] - landmarks[bottom])
                      for top, bottom in MOUTH_PAIRS])
    return float(height / width)


@dataclass
class MouthStatus:
    mar: float | None    # MAR no frame (None sem rosto)
    yawning: bool        # bocejo em andamento
    yawn_started: bool   # True só no frame em que o bocejo foi confirmado


class YawnDetector:
    """Boca muito aberta (MAR > limiar) por pelo menos `min_seconds` = bocejo.

    A fala abre a boca por frações de segundo e não chega ao tempo mínimo.
    Cada bocejo é contado uma única vez, no instante em que é confirmado.
    """

    def __init__(self, mar_threshold: float, min_seconds: float, gap_tolerance: float = 0.0):
        self.mar_threshold = mar_threshold
        self._timer = PersistenceFilter(min_seconds, gap_tolerance)
        self._was_yawning = False

    def update(self, mar: float | None, t: float) -> MouthStatus:
        yawning = self._timer.update(mar is not None and mar > self.mar_threshold, t)
        started = yawning and not self._was_yawning
        self._was_yawning = yawning
        return MouthStatus(mar=mar, yawning=yawning, yawn_started=started)
